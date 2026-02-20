"""Inference pipeline for diarization with speaker embeddings and clustering.

Based on DiariZen and pyannote.audio inference approach.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.signal import medfilt
from scipy.ndimage import binary_dilation, binary_erosion

from .hmm_smoothing import hmm_forward_backward_log, hmm_viterbi_log, load_hmm_transition_matrix

try:
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
except ImportError:
    AgglomerativeClustering = None
    SpectralClustering = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def multilabel_to_powerset_probs(
    multilabel: np.ndarray,
    mapping: List[Tuple[int, ...]],
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert soft multilabel (T, C) to powerset probabilities (T, K) under independence.

    P(powerset class k) = prod_{i in mapping[k]} p_i * prod_{i not in mapping[k]} (1 - p_i),
    then normalize per frame. Avoids boosting empty in overlap regions when OA is done in
    multilabel space first.

    Args:
        multilabel: (T, num_speakers) soft activations in [0, 1]
        mapping: list of K tuples; mapping[k] = speaker indices for powerset class k
        eps: clamp multilabel to [eps, 1-eps] to avoid log(0)

    Returns:
        (T, K) probabilities summing to 1 per frame
    """
    T, C = multilabel.shape
    K = len(mapping)
    p = np.clip(multilabel.astype(np.float64), eps, 1.0 - eps)
    log_probs = np.zeros((T, K), dtype=np.float64)
    for k in range(K):
        in_set = set(mapping[k])
        for i in range(C):
            if i in in_set:
                log_probs[:, k] += np.log(p[:, i])
            else:
                log_probs[:, k] += np.log(1.0 - p[:, i])
    # Normalize: subtract logsumexp per frame
    max_log = np.max(log_probs, axis=1, keepdims=True)
    log_probs = log_probs - (max_log + np.log(np.sum(np.exp(log_probs - max_log), axis=1, keepdims=True)))
    return np.exp(log_probs).astype(np.float32)


class DiarizationInference:
    """Inference pipeline for speaker diarization.

    Pipeline:
    1. Segmentation model: Get speaker activities (powerset-based)
    2. Optional median filtering: Smooth predictions
    3. Binarization: Convert probabilities to binary labels
    4. Speaker counting: Estimate number of speakers
    5. (Optional) Speaker embeddings: Extract embeddings for active speakers
    6. Clustering: Assign global speaker IDs
    7. Reconstruction: Generate final diarization output

    Args:
        model: Trained diarization model
        device: Device for inference ("cuda" or "cpu")
        apply_median_filtering: Whether to apply median filtering
        median_filter_size: Median filter kernel size (frames)
        binarization_threshold: Threshold for binarizing speaker activities
        speaker_embedding_model: Path to speaker embedding model (ESPnet2)
        embedding_exclude_overlap: Exclude overlapping speech for embeddings
        embedding_median_filter: Median filter kernel size (frames) for per-chunk; 0 = not used (when smoothing_embedding="median")
        embedding_dilation_size: Dilation kernel size (frames) after median; 0 = not used
        embedding_erosion_size: Erosion kernel size (frames) after dilation; 0 = not used
        embedding_min_speaker_duration_sec: Discard (chunk, speaker) with total single-speaker duration below this (merge then filter)
        post_oa_median_filter: Median filter kernel size (frames) after overlap-add; 0 = not used (when smoothing_post_oa="median")
        post_oa_dilation_size: Dilation kernel size (frames) after post-OA median; 0 = not used
        post_oa_erosion_size: Erosion kernel size (frames) after post-OA dilation; 0 = not used
        smoothing_embedding: "median" or "hmm" for per-chunk smoothing before embedding extraction (hmm uses powerset)
        smoothing_post_oa: "median" or "hmm" for post overlap-add smoothing (hmm uses powerset)
        hmm_transition_matrix_path: Path to .npy (K, K) row-stochastic matrix; required when either smoothing is "hmm"
        clustering_backend: "ahc" (agglomerative), "spectral", or "vbx" (variational bayes)
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        apply_median_filtering: bool = True,
        median_filter_size: int = 11,
        binarization_threshold: float = 0.5,
        speaker_embedding_model: Optional[str] = None,
        speaker_embedding_model_tag: Optional[str] = None,
        embedding_exclude_overlap: bool = True,
        embedding_median_filter: int = 11,
        embedding_dilation_size: int = 0,
        embedding_erosion_size: int = 0,
        embedding_min_speaker_duration_sec: float = 1.0,
        post_oa_median_filter: int = 11,
        post_oa_dilation_size: int = 0,
        post_oa_erosion_size: int = 0,
        smoothing_embedding: str = "median",  # "median" | "hmm"
        smoothing_post_oa: str = "median",  # "median" | "hmm"
        hmm_transition_matrix_path: Optional[str] = None,
        clustering_backend: str = "ahc",
        min_speakers: int = 1,
        max_speakers: int = 20,
        # DiariZen-style: run segmentation on chunks, then cluster on full recording
        chunk_duration_sec: Optional[float] = 30.0,
        chunk_shift_sec: Optional[float] = 20.0,
        frame_shift_sec: float = 0.02,
        sample_rate: int = 16000,
        **clustering_kwargs,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.apply_median_filtering = apply_median_filtering
        self.median_filter_size = median_filter_size
        self.binarization_threshold = binarization_threshold

        self.speaker_embedding_model_path = speaker_embedding_model
        self.speaker_embedding_model_tag = speaker_embedding_model_tag
        self.embedding_exclude_overlap = embedding_exclude_overlap
        self.embedding_median_filter = max(0, int(embedding_median_filter))
        self.embedding_dilation_size = max(0, int(embedding_dilation_size))
        self.embedding_erosion_size = max(0, int(embedding_erosion_size))
        self.embedding_min_speaker_duration_sec = max(0.0, float(embedding_min_speaker_duration_sec))
        self.post_oa_median_filter = max(0, int(post_oa_median_filter))
        self.post_oa_dilation_size = max(0, int(post_oa_dilation_size))
        self.post_oa_erosion_size = max(0, int(post_oa_erosion_size))
        self.smoothing_embedding = (smoothing_embedding or "median").strip().lower()
        self.smoothing_post_oa = (smoothing_post_oa or "median").strip().lower()
        if self.smoothing_embedding not in ("median", "hmm") or self.smoothing_post_oa not in ("median", "hmm"):
            raise ValueError('smoothing_embedding and smoothing_post_oa must be "median" or "hmm"')
        self._log_A: Optional[np.ndarray] = None
        if self.smoothing_embedding == "hmm" or self.smoothing_post_oa == "hmm":
            if not hmm_transition_matrix_path:
                raise ValueError("hmm_transition_matrix_path is required when smoothing_embedding or smoothing_post_oa is 'hmm'")
            self._log_A = load_hmm_transition_matrix(hmm_transition_matrix_path)
            logger.info("Loaded HMM transition matrix from %s (shape %s)", hmm_transition_matrix_path, self._log_A.shape)
        self._powerset = getattr(model, "powerset", None)

        self.clustering_backend = clustering_backend
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.clustering_kwargs = clustering_kwargs

        self.chunk_duration_sec = chunk_duration_sec
        self.chunk_shift_sec = chunk_shift_sec if chunk_shift_sec is not None else chunk_duration_sec
        self.frame_shift_sec = frame_shift_sec
        self.sample_rate = sample_rate

        # Speaker embedding model is required (no no-embedding path)
        has_embedding_spec = (
            (self.speaker_embedding_model_tag is not None and self.speaker_embedding_model_tag != "")
            or (self.speaker_embedding_model_path is not None and self.speaker_embedding_model_path != "")
        )
        if not has_embedding_spec:
            raise ValueError(
                "Speaker embedding model is required. Set inference.speaker_embedding_model_tag "
                "(e.g. espnet/voxcelebs12_rawnet3) or inference.speaker_embedding_model path."
            )
        self._load_speaker_embedding_model()

    def _load_speaker_embedding_model(self):
        """Load ESPnet2 speaker embedding model (for clustering across chunks, DiariZen/pyannote-style)."""
        has_tag = self.speaker_embedding_model_tag and self.speaker_embedding_model_tag.strip() != ""
        has_path = self.speaker_embedding_model_path and self.speaker_embedding_model_path.strip() != ""
        if not has_tag and not has_path:
            raise ValueError(
                "Set speaker_embedding_model_tag (e.g. espnet/voxcelebs12_rawnet3) or speaker_embedding_model path."
            )

        from espnet2.bin.spk_inference import Speech2Embedding

        if has_tag:
            logger.info("Loading speaker embedding model from tag: %s", self.speaker_embedding_model_tag)
            self.speaker_embedding_model = Speech2Embedding.from_pretrained(
                model_tag=self.speaker_embedding_model_tag,
                device=self.device,
            )
        else:
            logger.info("Loading speaker embedding model from path: %s", self.speaker_embedding_model_path)
            self.speaker_embedding_model = Speech2Embedding.from_pretrained(
                model_file=self.speaker_embedding_model_path,
                device=self.device,
            )
        logger.info("Speaker embedding model loaded successfully")

    def get_segmentations(
        self,
        waveform: torch.Tensor,
        soft: bool = True,
    ) -> np.ndarray:
        """Get speaker segmentations from waveform.

        DiariZen-style: if chunk_duration_sec is set, runs the segmentation model
        on overlapping chunks (to avoid OOM and match DiariZen), then stitches
        per-chunk segmentations into one full (num_frames, num_speakers) array.
        Clustering is applied later in __call__ to this full segmentation.

        Args:
            waveform: Input waveform (1D tensor or 2D with batch dim)
            soft: If True, return soft probabilities; else return binary labels

        Returns:
            Segmentations: (num_frames, num_speakers) array
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        total_samples = waveform.shape[1]
        duration_sec = total_samples / float(self.sample_rate)
        num_frames_total = int(round(duration_sec / self.frame_shift_sec))

        # No chunking: single forward (can OOM on long recordings)
        if self.chunk_duration_sec is None or duration_sec <= self.chunk_duration_sec:
            with torch.no_grad():
                speaker_activities, _ = self.model.inference(waveform)
            segmentations = speaker_activities.cpu().numpy()[0]
            # Debug: log raw outputs before binarization
            logger.debug(f"Raw speaker_activities shape: {segmentations.shape}, min={segmentations.min():.6f}, max={segmentations.max():.6f}, mean={segmentations.mean():.6f}")
            if not soft:
                segmentations = (segmentations >= self.binarization_threshold).astype(np.float32)
            return segmentations

        # Chunked segmentation (DiariZen-style): process in chunks, then stitch
        chunk_samples = int(self.chunk_duration_sec * self.sample_rate)
        shift_samples = int(self.chunk_shift_sec * self.sample_rate)
        num_speakers = getattr(self.model, "num_speakers", 4)
        stitched = np.zeros((num_frames_total, num_speakers), dtype=np.float32)
        counts = np.zeros(num_frames_total, dtype=np.float32)

        start_sample = 0
        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = waveform[:, start_sample:end_sample]
            with torch.no_grad():
                activities, _ = self.model.inference(chunk)
            seg = activities.cpu().numpy()[0]  # (frames_chunk, num_speakers)
            # Debug first chunk only
            if start_sample == 0:
                logger.debug(f"Chunk 0: activities shape={seg.shape}, min={seg.min():.6f}, max={seg.max():.6f}, mean={seg.mean():.6f}")
            start_sec = start_sample / float(self.sample_rate)
            start_frame = int(round(start_sec / self.frame_shift_sec))
            n_f = seg.shape[0]
            end_frame = min(start_frame + n_f, num_frames_total)
            n_f = end_frame - start_frame
            stitched[start_frame:end_frame] += seg[:n_f]
            counts[start_frame:end_frame] += 1.0
            start_sample += shift_samples
            if start_sample >= total_samples:
                break

        counts = np.maximum(counts, 1e-8)
        segmentations = (stitched / counts[:, np.newaxis]).astype(np.float32)
        if not soft:
            segmentations = (segmentations >= self.binarization_threshold).astype(np.float32)
        return segmentations

    def get_segmentations_and_chunks(
        self,
        waveform: torch.Tensor,
        soft: bool = True,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, np.ndarray, Optional[np.ndarray]]]]:
        """Get stitched segmentations and per-chunk data for embedding extraction and post-OA.

        Returns:
            segmentations: (num_frames, num_speakers) stitched array
            per_chunk_list: list of (start_frame, end_frame, start_sample, end_sample, seg, logits_ps)
                seg = (n_frames_chunk, num_speakers) multilabel; logits_ps = (n_frames_chunk, K) powerset logits
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        total_samples = waveform.shape[1]
        duration_sec = total_samples / float(self.sample_rate)
        num_frames_total = int(round(duration_sec / self.frame_shift_sec))
        num_speakers = getattr(self.model, "num_speakers", 4)
        powerset = getattr(self.model, "powerset", None)

        def _logits_to_seg(logits: torch.Tensor) -> np.ndarray:
            seg = powerset.to_multilabel(logits).cpu().numpy()[0] if powerset is not None else torch.softmax(logits, dim=-1).cpu().numpy()[0]
            return seg

        # No chunking
        if self.chunk_duration_sec is None or duration_sec <= self.chunk_duration_sec:
            with torch.no_grad():
                lengths = waveform.new_full((waveform.shape[0],), waveform.shape[1], dtype=torch.long)
                logits, _ = self.model.forward(waveform, lengths)
            seg = _logits_to_seg(logits)
            logits_ps = logits[0].cpu().numpy()
            if not soft:
                seg = (seg >= self.binarization_threshold).astype(np.float32)
            return seg, [(0, num_frames_total, 0, total_samples, seg, logits_ps)]

        # Chunked: run model.forward per chunk to get powerset logits and multilabel
        chunk_samples = int(self.chunk_duration_sec * self.sample_rate)
        shift_samples = int(self.chunk_shift_sec * self.sample_rate)
        stitched = np.zeros((num_frames_total, num_speakers), dtype=np.float32)
        counts = np.zeros(num_frames_total, dtype=np.float32)
        per_chunk_list = []
        start_sample = 0
        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = waveform[:, start_sample:end_sample]
            with torch.no_grad():
                lengths_chunk = chunk.new_full((chunk.shape[0],), chunk.shape[1], dtype=torch.long)
                logits, _ = self.model.forward(chunk, lengths_chunk)
            seg = _logits_to_seg(logits)
            logits_ps = logits[0].cpu().numpy()
            start_sec = start_sample / float(self.sample_rate)
            start_frame = int(round(start_sec / self.frame_shift_sec))
            n_f = seg.shape[0]
            end_frame = min(start_frame + n_f, num_frames_total)
            n_f_use = end_frame - start_frame
            stitched[start_frame:end_frame] += seg[:n_f_use]
            counts[start_frame:end_frame] += 1.0
            per_chunk_list.append((start_frame, end_frame, start_sample, end_sample, seg, logits_ps))
            start_sample += shift_samples
            if start_sample >= total_samples:
                break

        counts = np.maximum(counts, 1e-8)
        segmentations = (stitched / counts[:, np.newaxis]).astype(np.float32)
        if not soft:
            segmentations = (segmentations >= self.binarization_threshold).astype(np.float32)
        return segmentations, per_chunk_list

    def overlap_add_reorder(
        self,
        per_chunk_list: List[Tuple[int, int, int, int, np.ndarray]],
        chunk_speaker_to_global: Dict[Tuple[int, int], int],
        num_frames_total: int,
        num_speakers: int,
        hamming: bool = True,
        skip_average: bool = False,
    ) -> np.ndarray:
        """Overlap-add of chunk logits in global speaker space (pyannote-style).

        Follows pyannote.audio: for each chunk, reorder local speaker dimensions to
        global speaker IDs (max over locals mapping to same global), weight by Hamming
        window, then add into buffer. With skip_average=False, normalize by weights
        so values stay in [0,1] for thresholding; with skip_average=True return raw
        sum (pyannote uses this and then top-k per frame).

        See: pyannote.audio Inference.aggregate (hamming, skip_average),
             SpeakerDiarization.reconstruct (reorder with max).
        """
        stitched = np.zeros((num_frames_total, num_speakers), dtype=np.float32)
        weights = np.zeros((num_frames_total, num_speakers), dtype=np.float32)

        for chunk_idx, chunk_data in enumerate(per_chunk_list):
            start_frame, end_frame, _start_s, _end_s, seg = chunk_data[0], chunk_data[1], chunk_data[2], chunk_data[3], chunk_data[4]
            n_f, num_local = seg.shape
            end_frame_use = min(start_frame + n_f, num_frames_total)
            n_f_use = end_frame_use - start_frame
            # Reorder: reordered[f, g] = max over seg[f, s] for s mapping to g (pyannote reconstruct)
            reordered = np.zeros((n_f_use, num_speakers), dtype=np.float32)
            for s in range(num_local):
                g = chunk_speaker_to_global.get((chunk_idx, s), 0)
                g = min(g, num_speakers - 1)
                reordered[:, g] = np.maximum(reordered[:, g], seg[:n_f_use, s])
            # Hamming window (pyannote uses it in Inference.aggregate for overlap-add)
            if hamming and n_f_use > 1:
                win = np.hamming(n_f_use).astype(np.float32).reshape(-1, 1)
                reordered = reordered * win
                w = win
            else:
                w = np.ones((n_f_use, 1), dtype=np.float32)
            stitched[start_frame:end_frame_use] += reordered
            weights[start_frame:end_frame_use] += w

        if skip_average:
            segmentations = stitched.astype(np.float32)
        else:
            weights = np.maximum(weights, 1e-8)
            segmentations = (stitched / weights).astype(np.float32)
        return segmentations

    def overlap_add_reorder_powerset(
        self,
        per_chunk_list: List[Tuple[int, int, int, int, np.ndarray, Optional[np.ndarray]]],
        chunk_speaker_to_global: Dict[Tuple[int, int], int],
        num_frames_total: int,
        num_powerset_classes: int,
        mapping: List[Tuple[int, ...]],
        hamming: bool = True,
    ) -> np.ndarray:
        """Overlap-add per-chunk powerset logits in global powerset space. Returns (T, K) probs."""
        K = num_powerset_classes
        # Inverse: (tuple of sorted speaker indices) -> powerset index
        inv = {tuple(sorted(mapping[i])): i for i in range(K)}
        stitched = np.zeros((num_frames_total, K), dtype=np.float64)
        weights = np.zeros((num_frames_total,), dtype=np.float64)

        for chunk_idx, chunk_data in enumerate(per_chunk_list):
            start_frame, end_frame, _, _, _seg, logits_ps = chunk_data[0], chunk_data[1], chunk_data[2], chunk_data[3], chunk_data[4], chunk_data[5]
            if logits_ps is None or logits_ps.size == 0:
                continue
            n_f, _K = logits_ps.shape
            if _K != K:
                continue
            # logits -> probs
            logits_ps = logits_ps.astype(np.float64)
            max_log = np.max(logits_ps, axis=1, keepdims=True)
            probs = np.exp(logits_ps - max_log)
            probs = probs / probs.sum(axis=1, keepdims=True)
            # Local powerset index k -> global powerset index g
            local_to_global = np.zeros(K, dtype=np.int64)
            for k in range(K):
                speakers_local = mapping[k]
                global_speakers = tuple(sorted(chunk_speaker_to_global.get((chunk_idx, s), s) for s in speakers_local))
                local_to_global[k] = inv.get(global_speakers, 0)
            reordered = np.zeros((n_f, K), dtype=np.float64)
            for k in range(K):
                g = local_to_global[k]
                reordered[:, g] += probs[:, k]
            end_frame_use = min(start_frame + n_f, num_frames_total)
            n_f_use = end_frame_use - start_frame
            if hamming and n_f_use > 1:
                win = np.hamming(n_f_use).astype(np.float64)
            else:
                win = np.ones(n_f_use, dtype=np.float64)
            stitched[start_frame:end_frame_use] += reordered[:n_f_use]
            weights[start_frame:end_frame_use] += win
        weights = np.maximum(weights, 1e-8)
        stitched = (stitched / weights[:, np.newaxis]).astype(np.float32)
        return stitched

    def overlap_add_reorder_powerset_logits(
        self,
        per_chunk_list: List[Tuple[int, int, int, int, np.ndarray, Optional[np.ndarray]]],
        chunk_speaker_to_global: Dict[Tuple[int, int], int],
        num_frames_total: int,
        num_powerset_classes: int,
        mapping: List[Tuple[int, ...]],
        hamming: bool = True,
    ) -> np.ndarray:
        """Overlap-add per-chunk powerset logits (in logit space). Returns (T, K) logits.

        Reorder each chunk's logits to global powerset indices, add weighted logits (no softmax
        before adding). Caller then softmax to get probs for HMM. Averaging logits avoids
        the flattening effect of averaging probabilities in overlap regions.
        """
        K = num_powerset_classes
        inv = {tuple(sorted(mapping[i])): i for i in range(K)}
        stitched = np.zeros((num_frames_total, K), dtype=np.float64)
        weights = np.zeros((num_frames_total,), dtype=np.float64)

        for chunk_idx, chunk_data in enumerate(per_chunk_list):
            start_frame, end_frame, _, _, _seg, logits_ps = chunk_data[0], chunk_data[1], chunk_data[2], chunk_data[3], chunk_data[4], chunk_data[5]
            if logits_ps is None or logits_ps.size == 0:
                continue
            n_f, _K = logits_ps.shape
            if _K != K:
                continue
            logits_ps = logits_ps.astype(np.float64)
            local_to_global = np.zeros(K, dtype=np.int64)
            for k in range(K):
                speakers_local = mapping[k]
                global_speakers = tuple(sorted(chunk_speaker_to_global.get((chunk_idx, s), s) for s in speakers_local))
                local_to_global[k] = inv.get(global_speakers, 0)
            reordered = np.zeros((n_f, K), dtype=np.float64)
            for k in range(K):
                g = local_to_global[k]
                reordered[:, g] = logits_ps[:, k]
            end_frame_use = min(start_frame + n_f, num_frames_total)
            n_f_use = end_frame_use - start_frame
            if hamming and n_f_use > 1:
                win = np.hamming(n_f_use).astype(np.float64)
            else:
                win = np.ones(n_f_use, dtype=np.float64)
            stitched[start_frame:end_frame_use] += reordered[:n_f_use]
            weights[start_frame:end_frame_use] += win
        weights = np.maximum(weights, 1e-8)
        stitched_logits = (stitched / weights[:, np.newaxis]).astype(np.float32)
        return stitched_logits

    def apply_median_filter(
        self,
        segmentations: np.ndarray,
        kernel_size: int = 11,
    ) -> np.ndarray:
        """Apply median filter to smooth segmentations.

        Args:
            segmentations: (num_frames, num_speakers) array
            kernel_size: Median filter kernel size

        Returns:
            Filtered segmentations
        """
        num_frames, num_speakers = segmentations.shape
        filtered = np.zeros_like(segmentations)

        for spk_idx in range(num_speakers):
            filtered[:, spk_idx] = medfilt(
                segmentations[:, spk_idx],
                kernel_size=kernel_size
            )

        return filtered

    def apply_morphology(
        self,
        binary_segmentations: np.ndarray,
        dilation_size: int,
        erosion_size: int,
    ) -> np.ndarray:
        """Apply dilation followed by erosion (morphological closing) to binary segmentations.

        Args:
            binary_segmentations: (num_frames, num_speakers) binary array
            dilation_size: Dilation kernel size (frames); 0 = skip dilation
            erosion_size: Erosion kernel size (frames); 0 = skip erosion

        Returns:
            Morphologically processed binary segmentations
        """
        if dilation_size == 0 and erosion_size == 0:
            return binary_segmentations

        num_frames, num_speakers = binary_segmentations.shape
        result = binary_segmentations.copy().astype(bool)

        for spk_idx in range(num_speakers):
            if dilation_size > 0:
                # Create 1D structuring element for dilation
                structure = np.ones(dilation_size, dtype=bool)
                result[:, spk_idx] = binary_dilation(result[:, spk_idx], structure=structure)
            if erosion_size > 0:
                # Create 1D structuring element for erosion
                structure = np.ones(erosion_size, dtype=bool)
                result[:, spk_idx] = binary_erosion(result[:, spk_idx], structure=structure)

        return result.astype(np.float32)

    def count_speakers(
        self,
        segmentations: np.ndarray,
        min_duration_frames: Optional[int] = None,
    ) -> int:
        """Estimate number of speakers from segmentations.

        Only counts a speaker if they have at least min_duration_frames of activity
        (avoids counting spurious single-frame activations; default ~1s at 20ms frame shift).

        Args:
            segmentations: Soft or binary segmentations (num_frames, num_speakers)
            min_duration_frames: Minimum frames to count as active; if None, use
                embedding_min_speaker_duration_sec / frame_shift_sec (default ~50 frames).

        Returns:
            Estimated number of speakers
        """
        if min_duration_frames is None:
            min_duration_frames = max(
                1,
                int(self.embedding_min_speaker_duration_sec / self.frame_shift_sec),
            )
        binary = (segmentations >= self.binarization_threshold).astype(np.float64)
        speaker_activity = binary.sum(axis=0)  # Total frames per speaker
        active_speakers = (speaker_activity >= min_duration_frames).sum()

        # Clip to valid range
        active_speakers = int(np.clip(active_speakers, self.min_speakers, self.max_speakers))
        return active_speakers

    def extract_speaker_embeddings(
        self,
        waveform: torch.Tensor,
        segmentations: np.ndarray,
        frame_shift: float = 0.02,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Extract speaker embeddings for each speaker.

        Args:
            waveform: Input waveform (1D tensor)
            segmentations: Binary segmentations (num_frames, num_speakers)
            frame_shift: Frame shift in seconds
            sample_rate: Sample rate in Hz

        Returns:
            Embeddings: (num_speakers, embedding_dim) array
        """
        # Embedding model is always loaded (required at init)

        num_frames, num_speakers = segmentations.shape
        embeddings_list = []

        # Convert frame indices to sample indices
        samples_per_frame = int(frame_shift * sample_rate)

        for spk_idx in range(num_speakers):
            # Get frames where this speaker is active
            active_frames = np.where(segmentations[:, spk_idx] > 0)[0]

            if len(active_frames) == 0:
                # No activity for this speaker
                embeddings_list.append(None)
                continue

            # Optionally exclude overlapping speech
            if self.embedding_exclude_overlap:
                # Keep only frames with single speaker
                single_speaker_frames = segmentations.sum(axis=1) == 1
                active_frames = active_frames[single_speaker_frames[active_frames]]

            if len(active_frames) == 0:
                # No non-overlapping segments
                embeddings_list.append(None)
                continue

            # Extract audio segments for this speaker
            segments = []
            for frame_idx in active_frames:
                start_sample = frame_idx * samples_per_frame
                end_sample = min(start_sample + samples_per_frame, len(waveform))
                segment = waveform[start_sample:end_sample]
                segments.append(segment)

            # Concatenate segments
            speaker_audio = torch.cat(segments)
            # Cap duration to avoid OOM (embedding models expect short utterances, e.g. <= 30s)
            max_embedding_samples = int(30.0 * self.sample_rate)  # 30 seconds
            if speaker_audio.shape[0] > max_embedding_samples:
                speaker_audio = speaker_audio[:max_embedding_samples]

            # Extract embedding (Speech2Embedding expects 1D waveform and adds batch dim internally)
            with torch.no_grad():
                out = self.speaker_embedding_model(speaker_audio.squeeze() if speaker_audio.dim() > 1 else speaker_audio)
                embedding = out[0] if out.dim() > 1 else out

            embeddings_list.append(embedding.cpu().numpy())

        # Stack embeddings (use zeros for inactive speakers)
        if embeddings_list[0] is not None:
            embedding_dim = embeddings_list[0].shape[0]
        else:
            embedding_dim = 256  # Default

        embeddings = np.zeros((num_speakers, embedding_dim))
        for spk_idx, emb in enumerate(embeddings_list):
            if emb is not None:
                embeddings[spk_idx] = emb

        return embeddings

    def _get_contiguous_segments(
        self,
        segmentations: np.ndarray,
        min_frames: int = 15,
    ) -> List[Tuple[int, int, int]]:
        """Extract contiguous (start_frame, end_frame, local_speaker_idx) from binary segmentation.

        Each segment is one contiguous run of active frames for one channel.
        Used for segment-level embedding extraction (one embedding per segment, then cluster).
        """
        num_frames, num_speakers = segmentations.shape
        segments = []
        for spk_idx in range(num_speakers):
            active = segmentations[:, spk_idx] > 0
            in_run = False
            start_f = 0
            for f in range(num_frames):
                if active[f]:
                    if not in_run:
                        start_f = f
                        in_run = True
                else:
                    if in_run:
                        end_f = f - 1
                        if end_f - start_f + 1 >= min_frames:
                            segments.append((start_f, end_f, spk_idx))
                        in_run = False
            if in_run:
                end_f = num_frames - 1
                if end_f - start_f + 1 >= min_frames:
                    segments.append((start_f, end_f, spk_idx))
        return segments

    def extract_segment_embeddings(
        self,
        waveform: torch.Tensor,
        segments: List[Tuple[int, int, int]],
        frame_shift: float = 0.02,
        sample_rate: int = 16000,
        min_duration_sec: float = 0.3,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """Extract one embedding per contiguous segment. Returns (embeddings, valid_segments).

        Segments that are too short are skipped. Used for segment-level clustering.
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        samples_per_frame = int(frame_shift * sample_rate)
        min_samples = int(min_duration_sec * sample_rate)
        total_samples = waveform.shape[0]
        embeddings_list = []
        valid_segments = []
        for (start_f, end_f, _) in segments:
            start_s = start_f * samples_per_frame
            end_s = min((end_f + 1) * samples_per_frame, total_samples)
            if end_s - start_s < min_samples:
                continue
            seg_audio = waveform[start_s:end_s]
            # Cap to avoid OOM (e.g. 30s per segment)
            max_s = int(30.0 * sample_rate)
            if seg_audio.shape[0] > max_s:
                seg_audio = seg_audio[:max_s]
            with torch.no_grad():
                out = self.speaker_embedding_model(seg_audio.squeeze() if seg_audio.dim() > 1 else seg_audio)
                emb = out[0] if out.dim() > 1 else out
            embeddings_list.append(emb.cpu().numpy())
            valid_segments.append((start_f, end_f, _))
        if not embeddings_list:
            return np.zeros((0, 1), dtype=np.float32), []
        return np.stack(embeddings_list), valid_segments

    def extract_embeddings_pyannote_style(
        self,
        waveform: torch.Tensor,
        per_chunk_list: List[Tuple[int, int, int, int, np.ndarray]],
        frame_shift: float = 0.02,
        sample_rate: int = 16000,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Pyannote-style: per chunk, median→binarize, exclude overlap; merge all segments from same speaker;
        discard speakers with total duration < embedding_min_speaker_duration_sec; one embedding per (chunk, speaker).
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        samples_per_frame = int(frame_shift * sample_rate)
        min_samples = int(self.embedding_min_speaker_duration_sec * sample_rate)
        max_samples = int(30.0 * sample_rate)  # cap per (chunk, speaker) to avoid OOM
        embeddings_list = []
        chunk_speaker_pairs = []  # (chunk_idx, local_speaker_idx)

        # Suppress per-call "speech length" logs from ESPnet2 speaker model; log summary only
        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.setLevel(logging.WARNING)
        try:
            for chunk_idx, chunk_data in enumerate(per_chunk_list):
                start_frame, end_frame, start_sample, end_sample, seg, logits_ps = (
                    chunk_data[0], chunk_data[1], chunk_data[2], chunk_data[3], chunk_data[4],
                    chunk_data[5] if len(chunk_data) > 5 else None,
                )
                if self.smoothing_embedding == "hmm" and self._log_A is not None and logits_ps is not None and self._powerset is not None:
                    # HMM on powerset probs → Viterbi best sequence for this chunk → expand to binary for embedding extraction
                    x = logits_ps.astype(np.float64) - np.max(logits_ps, axis=1, keepdims=True)
                    log_B = x - np.log(np.sum(np.exp(x), axis=1, keepdims=True) + 1e-15)
                    best_path = hmm_viterbi_log(log_B, self._log_A)
                    mapping_np = self._powerset.mapping_matrix.cpu().numpy()
                    seg = mapping_np[best_path]
                    seg_smooth = seg
                elif self.embedding_median_filter > 0 and seg.shape[0] > 1:
                    seg_smooth = self.apply_median_filter(seg, kernel_size=self.embedding_median_filter)
                else:
                    seg_smooth = seg
                binary = (seg_smooth >= self.binarization_threshold).astype(np.float32)
                # Apply morphology (dilation then erosion) on top of median filtering
                if self.embedding_dilation_size > 0 or self.embedding_erosion_size > 0:
                    binary = self.apply_morphology(binary, self.embedding_dilation_size, self.embedding_erosion_size)
                num_frames_chunk, num_speakers = binary.shape
                # Single-speaker frames only (ignore overlap)
                single_speaker = (binary.sum(axis=1) == 1)

                for spk_idx in range(num_speakers):
                    active = (binary[:, spk_idx] > 0) & single_speaker
                    if not np.any(active):
                        continue
                    # Concatenate audio for frames where this speaker is active and no overlap
                    segments_audio = []
                    for f in np.where(active)[0]:
                        s_start = start_sample + f * samples_per_frame
                        s_end = min(s_start + samples_per_frame, end_sample)
                        segments_audio.append(waveform[s_start:s_end])
                    if not segments_audio:
                        continue
                    concat = torch.cat(segments_audio)
                    if concat.shape[0] < min_samples:
                        continue
                    if concat.shape[0] > max_samples:
                        concat = concat[:max_samples]
                    with torch.no_grad():
                        out = self.speaker_embedding_model(concat.squeeze() if concat.dim() > 1 else concat)
                        emb = out[0] if out.dim() > 1 else out
                    embeddings_list.append(emb.cpu().numpy())
                    chunk_speaker_pairs.append((chunk_idx, spk_idx))
        finally:
            root_logger.setLevel(old_level)

        if not embeddings_list:
            return np.zeros((0, 1), dtype=np.float32), []
        logger.info(
            "Speaker embeddings: extracted %d (chunk, speaker) embeddings (single-speaker merged, min_duration=%.2fs)",
            len(embeddings_list),
            self.embedding_min_speaker_duration_sec,
        )
        return np.stack(embeddings_list), chunk_speaker_pairs

    def cluster_speakers(
        self,
        segmentations: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        num_speakers: Optional[int] = None,
    ) -> np.ndarray:
        """Cluster speakers to assign global speaker IDs.

        Args:
            segmentations: Binary segmentations (num_frames, num_speakers)
            embeddings: Speaker embeddings (num_speakers, embedding_dim)
            num_speakers: Number of speakers (if None, auto-detect)

        Returns:
            Cluster assignments: (num_speakers,) array
                Maps local speaker index to global cluster ID
        """
        if num_speakers is None:
            num_speakers = self.count_speakers(segmentations)

        num_local_speakers = segmentations.shape[1]

        if embeddings is None:
            return np.arange(num_local_speakers)

        # Cluster based on embeddings
        if self.clustering_backend == "ahc":
            return self._cluster_ahc(embeddings, num_speakers)
        elif self.clustering_backend == "spectral":
            return self._cluster_spectral(embeddings, num_speakers)
        elif self.clustering_backend == "vbx":
            return self._cluster_vbx(embeddings, num_speakers)
        else:
            raise ValueError(f"Unknown clustering backend: {self.clustering_backend}")

    def _cluster_ahc(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """Agglomerative hierarchical clustering (pyannote-style).

        Uses min_speakers and max_speakers; when threshold is set in ahc config,
        uses distance_threshold to cut the dendrogram (n_clusters then ignored for fit).
        Otherwise n_clusters is clamped to [min_speakers, max_speakers].

        Returns:
            Cluster assignments: (n_embeddings,)
        """
        if AgglomerativeClustering is None:
            raise ImportError("scikit-learn is required for AHC clustering")

        ahc_cfg = self.clustering_kwargs.get("ahc", self.clustering_kwargs)
        if not isinstance(ahc_cfg, dict):
            ahc_cfg = {}
        metric = ahc_cfg.get("metric", "cosine")
        linkage = ahc_cfg.get("linkage", "average")
        threshold = ahc_cfg.get("threshold")

        use_threshold = threshold is not None and threshold != "auto" and isinstance(threshold, (int, float))
        if use_threshold:
            distance_threshold = float(threshold)
            n_clusters_param = None
            logger.info(
                "Clustering: AHC n_embeddings=%d distance_threshold=%s metric=%s linkage=%s (min_speakers=%d max_speakers=%d)",
                embeddings.shape[0],
                distance_threshold,
                metric,
                linkage,
                self.min_speakers,
                self.max_speakers,
            )
        else:
            n_clusters_param = int(np.clip(n_clusters, self.min_speakers, self.max_speakers))
            distance_threshold = None
            logger.info(
                "Clustering: AHC n_embeddings=%d n_clusters=%d metric=%s linkage=%s (min=%d max=%d)",
                embeddings.shape[0],
                n_clusters_param,
                metric,
                linkage,
                self.min_speakers,
                self.max_speakers,
            )

        # Cosine affinity cannot handle zero vectors
        if metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            zero_mask = (norms.ravel() < 1e-8)
            if zero_mask.any():
                non_zero = np.where(~zero_mask)[0]
                fill = embeddings[non_zero[0]] if len(non_zero) else np.ones(embeddings.shape[1], dtype=embeddings.dtype) * 1e-6
                embeddings = np.copy(embeddings)
                embeddings[zero_mask] = fill

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters_param,
            distance_threshold=distance_threshold,
            metric=metric,
            linkage=linkage,
        )
        labels = clustering.fit_predict(embeddings)
        n_result = getattr(clustering, "n_clusters_", len(np.unique(labels)))
        # Pyannote-style: when using distance_threshold, clamp to [min_speakers, max_speakers]
        if use_threshold and (n_result < self.min_speakers or n_result > self.max_speakers):
            n_clamp = int(np.clip(n_result, self.min_speakers, self.max_speakers))
            logger.info(
                "AHC threshold gave %d clusters; re-running with n_clusters=%d (min=%d max=%d)",
                n_result, n_clamp, self.min_speakers, self.max_speakers,
            )
            clustering = AgglomerativeClustering(
                n_clusters=n_clamp,
                distance_threshold=None,
                metric=metric,
                linkage=linkage,
            )
            labels = clustering.fit_predict(embeddings)
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(
            "Clustering done: labels %s -> counts per cluster %s",
            unique.tolist(),
            counts.tolist(),
        )
        return labels

    def _cluster_spectral(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """Spectral clustering (alternative to AHC; can handle non-convex clusters).

        Uses min_speakers and max_speakers to constrain n_clusters.
        Spectral clustering builds an affinity matrix from embeddings, then uses
        eigen decomposition to find clusters. Can work better than AHC when clusters
        are non-convex or have complex manifolds.

        Returns:
            Cluster assignments: (n_embeddings,)
        """
        if SpectralClustering is None:
            raise ImportError("scikit-learn is required for spectral clustering")

        spectral_cfg = self.clustering_kwargs.get("spectral", self.clustering_kwargs)
        if not isinstance(spectral_cfg, dict):
            spectral_cfg = {}
        affinity = spectral_cfg.get("affinity", "rbf")  # "rbf", "nearest_neighbors", or "cosine"
        gamma = spectral_cfg.get("gamma", 1.0)  # RBF kernel parameter
        n_neighbors = spectral_cfg.get("n_neighbors", 10)  # For nearest_neighbors affinity
        # Use min/max from spectral config if provided, else fall back to instance defaults
        min_spk = spectral_cfg.get("min_speakers", self.min_speakers)
        max_spk = spectral_cfg.get("max_speakers", self.max_speakers)
        n_clusters_param = int(np.clip(n_clusters, min_spk, max_spk))

        logger.info(
            "Clustering: Spectral n_embeddings=%d n_clusters=%d affinity=%s (min=%d max=%d)",
            embeddings.shape[0],
            n_clusters_param,
            affinity,
            min_spk,
            max_spk,
        )

        # Handle zero vectors for cosine-based affinity
        if affinity == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            zero_mask = (norms.ravel() < 1e-8)
            if zero_mask.any():
                non_zero = np.where(~zero_mask)[0]
                fill = embeddings[non_zero[0]] if len(non_zero) else np.ones(embeddings.shape[1], dtype=embeddings.dtype) * 1e-6
                embeddings = np.copy(embeddings)
                embeddings[zero_mask] = fill

        clustering = SpectralClustering(
            n_clusters=n_clusters_param,
            affinity=affinity,
            gamma=gamma if affinity == "rbf" else None,
            n_neighbors=n_neighbors if affinity == "nearest_neighbors" else None,
            random_state=42,  # For reproducibility
        )
        labels = clustering.fit_predict(embeddings)
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(
            "Clustering done: labels %s -> counts per cluster %s",
            unique.tolist(),
            counts.tolist(),
        )
        return labels

    def _cluster_vbx(
        self,
        embeddings: np.ndarray,
        num_speakers: int,
    ) -> np.ndarray:
        """Variational Bayes clustering (VBx) with cosine similarity.

        VBx models speaker turns with an HMM and uses variational inference to refine
        speaker assignments. This implementation uses cosine similarity instead of PLDA.

        Algorithm:
        1. Initialize with AHC labels
        2. Build HMM transition matrix (self-loops with loopP)
        3. Compute cosine similarity as log-likelihoods (scaled by fb)
        4. Run forward-backward to get speaker posteriors
        5. Update assignments and iterate until convergence

        Args:
            embeddings: (n_embeddings, embedding_dim) - one per (chunk, speaker)
            num_speakers: Target number of clusters (clamped to [min_speakers, max_speakers])

        Returns:
            Cluster assignments: (n_embeddings,)
        """
        n_embeddings, emb_dim = embeddings.shape
        n_clusters = int(np.clip(num_speakers, self.min_speakers, self.max_speakers))

        vbx_cfg = self.clustering_kwargs.get("vbx", self.clustering_kwargs)
        if not isinstance(vbx_cfg, dict):
            vbx_cfg = {}
        fa = float(vbx_cfg.get("fa", 0.3))  # False alarm probability
        fb = float(vbx_cfg.get("fb", 17.0))  # Scaling factor for cosine similarity (PLDA factor)
        loopP = float(vbx_cfg.get("loopP", 0.99))  # Self-loop probability
        max_iter = int(vbx_cfg.get("max_iter", 10))  # Max VB iterations

        logger.info(
            "Clustering: VBx n_embeddings=%d n_clusters=%d fa=%.2f fb=%.1f loopP=%.3f (min=%d max=%d)",
            n_embeddings,
            n_clusters,
            fa,
            fb,
            loopP,
            self.min_speakers,
            self.max_speakers,
        )

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        zero_mask = (norms.ravel() < 1e-8)
        if zero_mask.any():
            non_zero = np.where(~zero_mask)[0]
            fill = embeddings[non_zero[0]] if len(non_zero) else np.ones(emb_dim, dtype=embeddings.dtype) * 1e-6
            embeddings = np.copy(embeddings)
            embeddings[zero_mask] = fill
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / np.maximum(norms, 1e-8)

        # Step 1: Initialize with AHC (use same metric/linkage as config ahc when backend=ahc)
        if AgglomerativeClustering is None:
            raise ImportError("scikit-learn is required for VBx initialization (AHC)")
        ahc_cfg = self.clustering_kwargs.get("ahc", self.clustering_kwargs)
        if not isinstance(ahc_cfg, dict):
            ahc_cfg = {}
        ahc_metric = str(ahc_cfg.get("metric", "cosine"))
        ahc_linkage = str(ahc_cfg.get("linkage", "average"))
        ahc_init = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=ahc_metric,
            linkage=ahc_linkage,
        )
        labels = ahc_init.fit_predict(embeddings_norm)

        # Step 2: Build HMM transition matrix (K x K)
        # Self-loop: loopP, transitions: (1-loopP)/(K-1)
        log_A = np.full((n_clusters, n_clusters), np.log((1.0 - loopP) / max(1, n_clusters - 1)), dtype=np.float64)
        np.fill_diagonal(log_A, np.log(loopP))
        # Uniform initial state distribution
        log_init = np.log(np.ones(n_clusters, dtype=np.float64) / n_clusters)

        # Step 3: Compute cosine similarity matrix (n_embeddings x n_clusters)
        # For each embedding, compute similarity to each cluster centroid
        # Use as log-likelihood: log P(embedding | speaker_k) = fb * cosine_sim(embedding, centroid_k)
        cluster_centroids = np.zeros((n_clusters, emb_dim), dtype=np.float64)
        for k in range(n_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                cluster_centroids[k] = embeddings_norm[mask].mean(axis=0)
            else:
                # Fallback: use random embedding
                cluster_centroids[k] = embeddings_norm[np.random.randint(n_embeddings)]
        # Normalize centroids
        centroid_norms = np.linalg.norm(cluster_centroids, axis=1, keepdims=True)
        cluster_centroids = cluster_centroids / np.maximum(centroid_norms, 1e-8)

        # Cosine similarity: embeddings_norm @ centroids^T -> (n_embeddings, n_clusters)
        cosine_sim = np.dot(embeddings_norm, cluster_centroids.T)  # (n_embeddings, n_clusters)
        # Log-likelihood: fb * cosine_sim (scaled cosine similarity as PLDA substitute)
        # Add false alarm term: fa * uniform
        log_B = fb * cosine_sim.astype(np.float64)  # (n_embeddings, n_clusters)
        log_B += np.log(fa / n_clusters)  # False alarm term (uniform background)

        # Step 4: Variational Bayes iterations
        for iteration in range(max_iter):
            # Forward-backward to get posteriors
            posteriors = hmm_forward_backward_log(log_B, log_A, log_init)  # (n_embeddings, n_clusters)

            # Update cluster assignments (hard assignment from posteriors)
            labels_new = np.argmax(posteriors, axis=1)

            # Check convergence
            if np.array_equal(labels, labels_new):
                logger.info("VBx converged at iteration %d", iteration + 1)
                break

            labels = labels_new

            # Update cluster centroids
            for k in range(n_clusters):
                mask = (labels == k)
                if mask.sum() > 0:
                    cluster_centroids[k] = embeddings_norm[mask].mean(axis=0)
            centroid_norms = np.linalg.norm(cluster_centroids, axis=1, keepdims=True)
            cluster_centroids = cluster_centroids / np.maximum(centroid_norms, 1e-8)

            # Update log-likelihoods
            cosine_sim = np.dot(embeddings_norm, cluster_centroids.T)
            log_B = fb * cosine_sim.astype(np.float64)
            log_B += np.log(fa / n_clusters)

        unique, counts = np.unique(labels, return_counts=True)
        logger.info(
            "VBx clustering done: labels %s -> counts per cluster %s",
            unique.tolist(),
            counts.tolist(),
        )
        return labels

    def reconstruct_diarization(
        self,
        segmentations: np.ndarray,
        cluster_labels: np.ndarray,
        frame_shift: float = 0.02,
    ) -> List[Tuple[float, float, int]]:
        """Reconstruct final diarization output.

        Args:
            segmentations: Binary segmentations (num_frames, num_speakers)
            cluster_labels: Cluster assignments (num_speakers,)
            frame_shift: Frame shift in seconds

        Returns:
            List of (start_time, end_time, speaker_id) tuples
        """
        num_frames, num_speakers = segmentations.shape
        diarization = []

        # Create global speaker activities
        # Map local speakers to global clusters
        num_clusters = cluster_labels.max() + 1
        global_activities = np.zeros((num_frames, num_clusters))

        for local_spk in range(num_speakers):
            global_spk = cluster_labels[local_spk]
            global_activities[:, global_spk] = np.maximum(
                global_activities[:, global_spk],
                segmentations[:, local_spk]
            )

        # Extract segments for each global speaker
        for global_spk in range(num_clusters):
            active_frames = np.where(global_activities[:, global_spk] > 0)[0]

            if len(active_frames) == 0:
                continue

            # Group consecutive frames into segments
            segments = []
            start_frame = active_frames[0]
            prev_frame = active_frames[0]

            for frame in active_frames[1:]:
                if frame > prev_frame + 1:
                    # Gap detected, end current segment
                    end_frame = prev_frame
                    segments.append((start_frame, end_frame))
                    start_frame = frame
                prev_frame = frame

            # Add last segment
            segments.append((start_frame, prev_frame))

            # Convert frame indices to time
            for start_frame, end_frame in segments:
                start_time = start_frame * frame_shift
                end_time = (end_frame + 1) * frame_shift
                diarization.append((start_time, end_time, global_spk))

        # Sort by start time
        diarization.sort(key=lambda x: x[0])

        return diarization

    def reconstruct_diarization_from_segments(
        self,
        segments: List[Tuple[int, int, int]],
        segment_cluster_labels: np.ndarray,
        frame_shift: float = 0.02,
    ) -> List[Tuple[float, float, int]]:
        """Build diarization from segment list and per-segment cluster IDs. Merges adjacent same-speaker segments."""
        if len(segments) == 0:
            return []
        out = []
        for (start_f, end_f, _), global_id in zip(segments, segment_cluster_labels):
            start_t = start_f * frame_shift
            end_t = (end_f + 1) * frame_shift
            out.append((start_t, end_t, int(global_id)))
        out.sort(key=lambda x: x[0])
        # Merge adjacent segments with same speaker
        merged = []
        for start_t, end_t, spk in out:
            if merged and merged[-1][2] == spk and merged[-1][1] >= start_t - 0.01:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end_t), spk)
            else:
                merged.append((start_t, end_t, spk))
        return merged

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> List[Tuple[float, float, int]]:
        """Run inference on waveform.

        Args:
            waveform: Input waveform (1D tensor)
            sample_rate: Sample rate in Hz

        Returns:
            Diarization output: List of (start_time, end_time, speaker_id)
        """
        # 1. Per-chunk segmentations only (no full-recording stitch yet; stitch after clustering
        #    so speaker dimensions are reordered consistently — pyannote-style).
        segmentations, per_chunk_list = self.get_segmentations_and_chunks(waveform, soft=True)
        num_frames_total = segmentations.shape[0]
        # Estimate number of speakers from segmentations (duration-based) so we don't always get model.num_speakers
        num_speakers = self.count_speakers(segmentations)

        # 2. Per-chunk embedding extraction, then clustering
        segment_embeddings, chunk_speaker_pairs = self.extract_embeddings_pyannote_style(
            waveform,
            per_chunk_list,
            frame_shift=self.frame_shift_sec,
            sample_rate=sample_rate,
        )
        if len(chunk_speaker_pairs) == 0:
            logger.warning("No (chunk, speaker) embeddings extracted; returning empty diarization")
            return []

        n_clusters = max(1, min(num_speakers, segment_embeddings.shape[0]))
        logger.info(
            "Speaker clustering: %d (chunk, speaker) embeddings -> %d global speakers (backend=%s)",
            len(chunk_speaker_pairs),
            n_clusters,
            self.clustering_backend,
        )
        segment_labels = self._cluster_ahc(segment_embeddings, n_clusters)
        # (chunk_idx, local_speaker_idx) -> global_speaker_id
        chunk_speaker_to_global = {}
        for (chunk_idx, local_s), gid in zip(chunk_speaker_pairs, segment_labels):
            chunk_speaker_to_global[(chunk_idx, local_s)] = int(gid)

        # 3. Stitch only here: overlap-add with reordering by global speaker IDs
        if self.smoothing_post_oa == "hmm" and self._log_A is not None and self._powerset is not None:
            # Overlap-add on powerset probabilities (reordered via global speaker IDs from embeddings)
            stitched_ps = self.overlap_add_reorder_powerset(
                per_chunk_list,
                chunk_speaker_to_global,
                num_frames_total,
                self._powerset.num_powerset_classes,
                self._powerset.mapping,
            )
            # Reapply HMM smoothing on the whole sequence (Viterbi for best path)
            log_B = np.log(np.clip(stitched_ps.astype(np.float64), 1e-9, 1.0))
            best_path = hmm_viterbi_log(log_B, self._log_A)
            mapping_np = self._powerset.mapping_matrix.cpu().numpy()
            segmentations_global = mapping_np[best_path]
        else:
            segmentations_global = self.overlap_add_reorder(
                per_chunk_list,
                chunk_speaker_to_global,
                num_frames_total,
                num_speakers,
            )
            if self.post_oa_median_filter > 0:
                segmentations_global = self.apply_median_filter(
                    segmentations_global,
                    kernel_size=self.post_oa_median_filter,
                )
        if self.smoothing_post_oa == "hmm":
            segmentations_for_reconstruct = segmentations_global.astype(np.float32)
            num_speakers_reconstruct = segmentations_for_reconstruct.shape[1]
        else:
            segmentations_for_reconstruct = (segmentations_global >= self.binarization_threshold).astype(np.float32)
            # Apply morphology (dilation then erosion) on top of median filtering
            if self.post_oa_dilation_size > 0 or self.post_oa_erosion_size > 0:
                segmentations_for_reconstruct = self.apply_morphology(
                    segmentations_for_reconstruct,
                    self.post_oa_dilation_size,
                    self.post_oa_erosion_size,
                )
            num_speakers_reconstruct = num_speakers
        diarization = self.reconstruct_diarization(segmentations_for_reconstruct, np.arange(num_speakers_reconstruct))
        return diarization


def save_rttm(
    diarization: List[Tuple[float, float, int]],
    output_path: Path,
    recording_id: str,
):
    """Save diarization output in RTTM format.

    Args:
        diarization: List of (start_time, end_time, speaker_id)
        output_path: Output file path
        recording_id: Recording ID
    """
    with open(output_path, "w") as f:
        for start_time, end_time, speaker_id in diarization:
            duration = end_time - start_time
            # RTTM format:
            # SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker-id> <NA> <NA>
            f.write(
                f"SPEAKER {recording_id} 1 {start_time:.3f} {duration:.3f} "
                f"<NA> <NA> speaker_{speaker_id} <NA> <NA>\n"
            )


if __name__ == "__main__":
    # Test inference pipeline
    import sys

    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <audio_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    audio_path = sys.argv[2]

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location="cpu")

    # Load audio
    import torchaudio
    logger.info(f"Loading audio from {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform[0]

    # Run inference (speaker embedding model required)
    inference = DiarizationInference(
        model=model,
        device="cpu",
        apply_median_filtering=True,
        speaker_embedding_model_tag="espnet/voxcelebs12_rawnet3",
    )

    diarization = inference(waveform, sample_rate=sample_rate)

    # Print results
    print("\nDiarization results:")
    for start, end, speaker in diarization:
        print(f"  {start:.2f} - {end:.2f}: Speaker {speaker}")

    # Save RTTM
    output_path = Path(audio_path).with_suffix(".rttm")
    recording_id = Path(audio_path).stem
    save_rttm(diarization, output_path, recording_id)
    print(f"\nSaved RTTM to {output_path}")
