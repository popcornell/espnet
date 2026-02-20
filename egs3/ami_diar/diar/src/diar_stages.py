"""Diarization infer and metric stages for run.py pipeline."""
import logging
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def _load_model(checkpoint_path, train_cfg):
    """Load PowersetDiarizationModel from Lightning checkpoint. Model is built from train config or checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # ESPnet Lightning: state_dict keys are "model.model.xxx" (wrapper.model = PowersetDiarizationModel)
    state_dict = ckpt.get("state_dict", ckpt)
    prefix = "model.model."
    state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not state:
        prefix = "model."
        state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    
    # Build model from checkpoint hyper_parameters (preferred) or train config
    # Checkpoint hyper_parameters match the actual trained model architecture
    if "hyper_parameters" in ckpt and "model" in ckpt["hyper_parameters"]:
        model_cfg = ckpt["hyper_parameters"]["model"]
        logger.info("Using model config from checkpoint hyper_parameters")
    elif train_cfg and hasattr(train_cfg, "model"):
        model_cfg = train_cfg.model
        logger.info("Using model config from train_cfg (checkpoint hyper_parameters not available)")
    else:
        # Try to load from config file in exp directory
        exp_dir = Path(checkpoint_path).parent.parent
        # Try multiple possible config file names
        config_paths = [
            exp_dir / "config.yaml",
            exp_dir / "train_config.yaml",
            Path(checkpoint_path).parent / "config.yaml",
        ]
        model_cfg = None
        for config_path in config_paths:
            if config_path.exists():
                cfg = OmegaConf.load(config_path)
                if hasattr(cfg, "model"):
                    model_cfg = cfg.model
                    break
                # Also check if it's a train config with model inside
                if hasattr(cfg, "train_config") and hasattr(cfg.train_config, "model"):
                    model_cfg = cfg.train_config.model
                    break
        
        if model_cfg is None:
            # Last resort: try to load from debug.yaml or similar in conf/tuning/
            # This is a fallback - ideally train_config should be provided
            raise ValueError(
                f"Cannot determine model config. Please provide train_config or ensure checkpoint has hyper_parameters. "
                f"Checked: {config_paths}"
            )
    
    if model_cfg is None:
        raise ValueError("Cannot determine model config from checkpoint or train_cfg")
    
    # For inference: if XEUS ssl_model_path is None, try XEUS_CHECKPOINT env var
    # (During inference, SSL weights are in checkpoint, but model instantiation still needs the path)
    if (hasattr(model_cfg, "ssl_model_name") and 
        str(model_cfg.ssl_model_name).lower() == "xeus" and
        (not hasattr(model_cfg, "ssl_model_path") or 
         model_cfg.ssl_model_path is None or 
         str(model_cfg.ssl_model_path).lower() == "null")):
        xeus_checkpoint = os.environ.get("XEUS_CHECKPOINT")
        if xeus_checkpoint:
            OmegaConf.update(model_cfg, "ssl_model_path", xeus_checkpoint)
            logger.info("Using XEUS_CHECKPOINT env var for ssl_model_path: %s", xeus_checkpoint)
        else:
            # Try common location
            common_paths = [
                Path("data/xeus/xeus_checkpoint_new.pth"),
                Path("data/xeus/checkpoint.pth"),
                Path.home() / "models" / "xeus" / "checkpoint.pth",
            ]
            for path in common_paths:
                if path.exists():
                    OmegaConf.update(model_cfg, "ssl_model_path", str(path))
                    logger.info("Found XEUS checkpoint at common location: %s", path)
                    break
            else:
                logger.warning(
                    "XEUS ssl_model_path is None and XEUS_CHECKPOINT not set. "
                    "Model instantiation may fail. Set XEUS_CHECKPOINT env var or provide ssl_model_path in config."
                )
    
    wrapper = instantiate(model_cfg)
    
    # Safety check: detect SSL layer count mismatch from weight_sum.weight shape
    # This can happen if checkpoint was trained with different ssl_num_layers than config
    if "ssl_frontend.weight_sum.weight" in state:
        checkpoint_layers = state["ssl_frontend.weight_sum.weight"].shape[1]
        if hasattr(wrapper.model, "ssl_frontend") and hasattr(wrapper.model.ssl_frontend, "weight_sum"):
            if wrapper.model.ssl_frontend.weight_sum is not None:
                config_layers = wrapper.model.ssl_frontend.weight_sum.weight.shape[1]
                if checkpoint_layers != config_layers:
                    logger.warning(
                        f"SSL layer count mismatch: checkpoint has {checkpoint_layers} layers, "
                        f"but config specifies {config_layers}. Attempting to adjust..."
                    )
                    # Recreate weight_sum with correct size
                    import torch.nn as nn
                    wrapper.model.ssl_frontend.weight_sum = nn.Linear(checkpoint_layers, 1, bias=False)
                    logger.info(f"Adjusted ssl_frontend.weight_sum to {checkpoint_layers} layers")
    
    wrapper.model.load_state_dict(state, strict=True)
    return wrapper.model.eval()

def _write_ref_rttm_from_cuts(cuts, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cut in cuts:
        rttm_path = out / f"{cut.id}.rttm"
        with open(rttm_path, "w") as f:
            for sup in cut.supervisions:
                if getattr(sup, "speaker", None):
                    f.write(
                        f"SPEAKER {cut.id} 1 {sup.start:.3f} {sup.duration:.3f} <NA> <NA> {sup.speaker} <NA> <NA>\n"
                    )
    logger.info("Wrote %d reference RTTMs to %s", len(cuts), out)

def run_diar_infer(system):
    from src.dataset import DiarizationDataset, collate_fn
    from src.inference import DiarizationInference, save_rttm
    infer_cfg = system.infer_config
    train_cfg = system.train_config
    # When running with only --infer_config, train_cfg is None; load from train_config_path if set
    if train_cfg is None:
        train_config_path = OmegaConf.select(infer_cfg, "train_config_path", default=None)
        if train_config_path:
            path = Path(train_config_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                train_cfg = OmegaConf.load(path)
                logger.info("Loaded train config from %s for model loading", path)
            else:
                logger.warning("train_config_path %s not found; model loading may fail if checkpoint has no hyper_parameters", path)
    infer_dir = Path(OmegaConf.select(infer_cfg, "infer_dir", default=None) or (train_cfg.exp_dir + "/infer" if train_cfg and hasattr(train_cfg, "exp_dir") else "exp/infer"))
    model_path = OmegaConf.select(infer_cfg, "model_path", default=None) or (str(Path(train_cfg.exp_dir) / "checkpoints" / "last.ckpt") if train_cfg and hasattr(train_cfg, "exp_dir") else None)
    if model_path is None:
        raise ValueError("model_path must be specified in infer_config when train_config is not available")
    p = Path(model_path)
    if not p.exists():
        # ESPnet3 trainer may save to exp_dir/ directly (e.g. last.ckpt)
        if train_cfg and hasattr(train_cfg, "exp_dir"):
            alt = Path(train_cfg.exp_dir) / "last.ckpt"
            if alt.exists():
                model_path = str(alt)
            else:
                raise FileNotFoundError("Checkpoint not found: " + str(p))
        else:
            # Try parent directory
            alt = p.parent.parent / "last.ckpt"
            if alt.exists():
                model_path = str(alt)
            else:
                raise FileNotFoundError("Checkpoint not found: " + str(p))
    test_manifest = OmegaConf.select(infer_cfg, "dataset.test.0.dataset.manifest_path", default=None) or (str(Path(train_cfg.data_dir) / "test_cuts.jsonl.gz") if train_cfg and hasattr(train_cfg, "data_dir") else None)
    if test_manifest is None:
        raise ValueError("test manifest path must be specified in infer_config when train_config is not available")
    device = OmegaConf.select(infer_cfg, "device", default="cuda") if torch.cuda.is_available() else "cpu"
    pred_dir = infer_dir / "predictions"
    ref_dir = infer_dir / "reference"
    pred_dir.mkdir(parents=True, exist_ok=True)
    # Load full waveforms (chunk_duration=None); DiarizationInference does chunked segmentation + clustering internally (DiariZen-style)
    dataset = DiarizationDataset(manifest_path=test_manifest, chunk_duration=None, chunk_shift=None, frame_shift=0.02, max_speakers=4, sample_rate=16000)
    _write_ref_rttm_from_cuts(dataset.cuts, ref_dir)
    model = _load_model(model_path, train_cfg)
    # DiariZen-style: chunked segmentation inside inference, then clustering on full recording
    inf = infer_cfg.get("inference", {})
    chunk_dur = OmegaConf.select(infer_cfg, "inference.chunk_duration", default=30.0)
    chunk_shift = OmegaConf.select(infer_cfg, "inference.chunk_shift", default=20.0)
    spk_model = OmegaConf.select(infer_cfg, "inference.speaker_embedding_model", default=None)
    spk_tag = OmegaConf.select(infer_cfg, "inference.speaker_embedding_model_tag", default=None)
    clustering_kwargs = {}
    if "vbx" in inf:
        clustering_kwargs["vbx"] = dict(inf.vbx)
    if "ahc" in inf:
        clustering_kwargs["ahc"] = dict(inf.ahc)
    if "spectral" in inf:
        clustering_kwargs["spectral"] = dict(inf.spectral)
    inference = DiarizationInference(
        model=model,
        device=device,
        apply_median_filtering=OmegaConf.select(infer_cfg, "inference.apply_median_filtering", default=True),
        median_filter_size=OmegaConf.select(infer_cfg, "inference.median_filter_size", default=11),
        binarization_threshold=OmegaConf.select(infer_cfg, "inference.binarization_threshold", default=0.5),
        speaker_embedding_model=spk_model if spk_model else None,
        speaker_embedding_model_tag=spk_tag if spk_tag else None,
        embedding_exclude_overlap=OmegaConf.select(infer_cfg, "inference.embedding_exclude_overlap", default=True),
        embedding_min_speaker_duration_sec=OmegaConf.select(infer_cfg, "inference.embedding_min_speaker_duration_sec", default=1.0),
        smoothing_embedding=OmegaConf.select(infer_cfg, "inference.smoothing_embedding", default="median"),
        smoothing_post_oa=OmegaConf.select(infer_cfg, "inference.smoothing_post_oa", default="median"),
        embedding_median_filter=OmegaConf.select(infer_cfg, "inference.embedding_median_filter", default=11),
        embedding_dilation_size=OmegaConf.select(infer_cfg, "inference.embedding_dilation_size", default=0),
        embedding_erosion_size=OmegaConf.select(infer_cfg, "inference.embedding_erosion_size", default=0),
        post_oa_median_filter=OmegaConf.select(infer_cfg, "inference.post_oa_median_filter", default=11),
        post_oa_dilation_size=OmegaConf.select(infer_cfg, "inference.post_oa_dilation_size", default=0),
        post_oa_erosion_size=OmegaConf.select(infer_cfg, "inference.post_oa_erosion_size", default=0),
        hmm_transition_matrix_path=OmegaConf.select(infer_cfg, "inference.hmm_transition_matrix_path", default=None),
        clustering_backend=OmegaConf.select(infer_cfg, "inference.clustering_backend", default="ahc"),
        min_speakers=OmegaConf.select(infer_cfg, "inference.ahc.min_speakers", default=OmegaConf.select(infer_cfg, "inference.min_speakers", default=1)),
        max_speakers=OmegaConf.select(infer_cfg, "inference.ahc.max_speakers", default=OmegaConf.select(infer_cfg, "inference.max_speakers", default=10)),
        chunk_duration_sec=chunk_dur,
        chunk_shift_sec=chunk_shift,
        frame_shift_sec=0.02,
        sample_rate=16000,
        **clustering_kwargs,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for batch in dataloader:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            utt_ids, batch = batch[0], batch[1]
            utt_id = utt_ids[0] if isinstance(utt_ids, (list, tuple)) else utt_ids
        else:
            utt_id = batch.get("utt_ids", [None])[0] if isinstance(batch.get("utt_ids"), list) else batch.get("utt_id")
        wav = batch["waveform"][0] if "waveform" in batch else batch["speech"][0]
        # Reference is written as cut.id (e.g. EN2002a-4). Dataset uses utt_id = "{cut_id}_{start:06d}_{end:06d}".
        # Derive recording id so pred filename matches ref: strip trailing _start_end from utt_id.
        rec_id = utt_id.rsplit("_", 2)[0] if utt_id.count("_") >= 2 else utt_id
        logger.info("Running inference on %s (waveform length: %.2fs)", rec_id, len(wav) / 16000)
        diar = inference(wav, sample_rate=16000)
        logger.info("Inference result for %s: %d speaker segments", rec_id, len(diar))
        if len(diar) == 0:
            logger.warning("No speaker segments detected for %s - model may be predicting all empty sets", rec_id)
        save_rttm(diar, pred_dir / f"{rec_id}.rttm", rec_id)
    logger.info("Inference done: %s", pred_dir)
    return pred_dir

def run_diar_metric(system):
    infer_dir = Path(OmegaConf.select(system.metric_config, "infer_dir", default=None) or OmegaConf.select(system.infer_config, "infer_dir", default=None))
    if system.train_config:
        infer_dir = infer_dir or Path(system.train_config.exp_dir) / "infer"
    ref_dir = infer_dir / "reference"
    pred_dir = infer_dir / "predictions"
    if not pred_dir.exists() or not ref_dir.exists():
        raise FileNotFoundError(f"Run infer first: {pred_dir} / {ref_dir}")
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        raise ImportError("pip install pyannote.core pyannote.metrics")
    def load_rttm(path):
        ann = Annotation()
        with open(path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 8 and p[0] == "SPEAKER":
                    start, dur, spk = float(p[3]), float(p[4]), p[7]
                    ann[Segment(start, start + dur)] = spk
        return ann
    der = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    hyp_rttms = list(pred_dir.glob("*.rttm"))
    num_compared = 0
    for hyp_path in hyp_rttms:
        ref_path = ref_dir / (hyp_path.stem + ".rttm")
        if not ref_path.exists():
            logger.warning("No reference for %s (expected %s)", hyp_path.name, ref_path.name)
            continue
        ref_ann = load_rttm(ref_path)
        hyp_ann = load_rttm(hyp_path)
        der(ref_ann, hyp_ann, uem=None)
        num_compared += 1
    if num_compared == 0:
        logger.error("No ref/hyp pairs matched (ref stem must equal hyp stem). DER undefined.")
        global_der = float("nan")
    else:
        global_der = abs(der)
        logger.info("Compared %d ref/hyp RTTM pairs", num_compared)
    result = {"DER": global_der, "DER_percent": global_der * 100 if num_compared else None, "num_compared": num_compared}
    out = infer_dir / "metrics.json"
    import json
    with open(out, "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    if num_compared:
        logger.info("Global DER (AMI test): %.2f%%", global_der * 100)
    else:
        logger.info("Global DER (AMI test): N/A (no ref/hyp pairs matched)")
    return result
