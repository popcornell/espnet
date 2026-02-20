"""Fit HMM transition matrix from training set frame labels (powerset) and save as .npy.

Run once before inference when using smoothing_embedding=hmm or smoothing_post_oa=hmm.
Example:
  python fit_hmm_transitions.py \\
    --train_manifest data/ami_lhotse/train_cuts.jsonl.gz \\
    --num_speakers 4 --max_speakers_per_frame 2 \\
    --output exp/full_training/hmm_transition.npy \\
    --pseudo_count 1.0
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from espnet3.components.diarization.powerset import Powerset

from src.dataset import DiarizationDataset
from src.hmm_smoothing import fit_transition_matrix_from_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fit HMM transition matrix on train labels")
    parser.add_argument("--train_manifest", type=str, required=True, help="Train cuts manifest (e.g. train_cuts.jsonl.gz)")
    parser.add_argument("--num_speakers", type=int, default=4)
    parser.add_argument("--max_speakers_per_frame", type=int, default=2)
    parser.add_argument("--output", type=str, required=True, help="Output .npy path (K, K) row-stochastic")
    parser.add_argument("--pseudo_count", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_duration", type=float, default=20.0)
    parser.add_argument("--chunk_shift", type=float, default=5.0)
    parser.add_argument("--frame_shift", type=float, default=0.02)
    args = parser.parse_args()

    powerset = Powerset(num_classes=args.num_speakers, max_set_size=args.max_speakers_per_frame)
    K = powerset.num_powerset_classes
    logger.info("Powerset: %d classes", K)

    dataset = DiarizationDataset(
        manifest_path=args.train_manifest,
        chunk_duration=args.chunk_duration,
        chunk_shift=args.chunk_shift,
        frame_shift=args.frame_shift,
        max_speakers=args.num_speakers,
    )
    from src.dataset import collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    all_labels = []
    all_lengths = []
    for batch in dataloader:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            _, batch = batch[0], batch[1]
        labels = batch["labels"]  # (B, T, num_speakers)
        lengths = batch.get("labels_lengths")
        if lengths is None:
            lengths = labels.new_full((labels.shape[0],), labels.shape[1], dtype=torch.long)
        ps = powerset.to_powerset(labels)  # (B, T)
        for b in range(labels.shape[0]):
            L = lengths[b].item()
            all_labels.append(ps[b, :L].cpu().numpy())
            all_lengths.append(L)
    if not all_labels:
        raise RuntimeError("No labels collected from dataset")
    labels_concat = np.concatenate(all_labels, axis=0)
    lengths_arr = np.array(all_lengths, dtype=np.int64)
    # fit_transition_matrix_from_labels expects (batch, T) or we can pass flat and lengths
    # It expects labels (batch, T); we have list of (T,) so we need to pad to same T or pass as (1, total_frames) with one length. Simpler: pass each chunk and accumulate counts ourselves, or pass (N, T_max) with lengths.
    T_max = max(all_lengths)
    batch_labels = np.zeros((len(all_labels), T_max), dtype=np.int64)
    for i, (arr, L) in enumerate(zip(all_labels, all_lengths)):
        batch_labels[i, :L] = arr
    A = fit_transition_matrix_from_labels(
        batch_labels,
        lengths_arr,
        num_states=K,
        pseudo_count=args.pseudo_count,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, A)
    logger.info("Saved HMM transition matrix (shape %s) to %s", A.shape, out_path)


if __name__ == "__main__":
    main()
