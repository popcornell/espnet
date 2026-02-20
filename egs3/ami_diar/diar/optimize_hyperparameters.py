#!/usr/bin/env python3
"""Hyperparameter optimization for diarization inference using DER.

Runs inference with different parameter combinations and optimizes based on DER.
Uses Optuna for Bayesian optimization or grid search as fallback.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Add paths
recipe_dir = Path(__file__).parent
sys.path.insert(0, str(recipe_dir))
sys.path.insert(0, str(recipe_dir.parent.parent.parent))  # espnet root

from src.dataset import DiarizationDataset, collate_fn
from src.inference import DiarizationInference, save_rttm
from src.diar_stages import _load_model, _write_ref_rttm_from_cuts

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
except ImportError:
    raise ImportError("pip install pyannote.core pyannote.metrics")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    raise ImportError("Optuna is required. Install with: pip install optuna")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_rttm_as_annotation(rttm_path: Path) -> Annotation:
    """Load RTTM file as pyannote Annotation."""
    ann = Annotation()
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]
                ann[Segment(start, start + duration)] = speaker_id
    return ann


def compute_der(hypothesis_dir: Path, reference_dir: Path, collar: float = 0.25) -> float:
    """Compute DER between hypothesis and reference RTTM files.

    Args:
        hypothesis_dir: Directory with hypothesis RTTM files
        reference_dir: Directory with reference RTTM files
        collar: Collar in seconds for DER computation

    Returns:
        Global DER (0.0 to 1.0, lower is better)
    """
    der_metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    hyp_rttms = sorted(hypothesis_dir.glob("*.rttm"))
    num_compared = 0

    for hyp_path in hyp_rttms:
        ref_path = reference_dir / hyp_path.name
        if not ref_path.exists():
            logger.warning("No reference for %s (expected %s)", hyp_path.name, ref_path.name)
            continue

        ref_ann = load_rttm_as_annotation(ref_path)
        hyp_ann = load_rttm_as_annotation(hyp_path)
        der_metric(ref_ann, hyp_ann, uem=None)
        num_compared += 1

    if num_compared == 0:
        logger.error("No ref/hyp pairs matched. DER undefined.")
        return float("nan")

    global_der = abs(der_metric)
    logger.info("DER: %.4f (%.2f%%) from %d files", global_der, global_der * 100, num_compared)
    return global_der


def run_inference_with_params(
    model: torch.nn.Module,
    dataset: DiarizationDataset,
    output_dir: Path,
    params: Dict[str, Any],
    device: str = "cuda",
) -> Path:
    """Run inference with given parameters and save RTTM files.

    Args:
        model: Trained diarization model
        dataset: Test dataset
        output_dir: Output directory for RTTM files
        params: Parameter dictionary for DiarizationInference (may include nested ahc/spectral/vbx)
        device: Device for inference

    Returns:
        Path to predictions directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract clustering kwargs (nested configs)
    clustering_kwargs = {}
    for key in ["ahc", "spectral", "vbx"]:
        if key in params:
            clustering_kwargs[key] = params.pop(key)

    # Handle nested params like "ahc.min_speakers" -> {"ahc": {"min_speakers": ...}}
    nested_keys = [k for k in list(params.keys()) if "." in k]
    for key in nested_keys:
        outer, inner = key.split(".", 1)
        if outer not in clustering_kwargs:
            clustering_kwargs[outer] = {}
        clustering_kwargs[outer][inner] = params.pop(key)

    # Create inference pipeline
    inference = DiarizationInference(
        model=model,
        device=device,
        **params,
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
        rec_id = utt_id.rsplit("_", 2)[0] if utt_id.count("_") >= 2 else utt_id

        diarization = inference(wav, sample_rate=16000)
        save_rttm(diarization, output_dir / f"{rec_id}.rttm", rec_id)

    return output_dir


def objective_optuna(
    trial: "optuna.Trial",
    model: torch.nn.Module,
    dataset: DiarizationDataset,
    reference_dir: Path,
    base_output_dir: Path,
    device: str,
    param_space: Dict[str, Any],
) -> float:
    """Optuna objective function: run inference with trial parameters and return DER."""
    # Sample parameters from trial
    params = {}
    for key, space in param_space.items():
        if isinstance(space, dict):
            if space.get("type") == "categorical":
                params[key] = trial.suggest_categorical(key, space["choices"])
            elif space.get("type") == "int":
                params[key] = trial.suggest_int(key, space["low"], space["high"], step=space.get("step", 1))
            elif space.get("type") == "float":
                params[key] = trial.suggest_float(key, space["low"], space["high"], log=space.get("log", False))
            elif space.get("type") == "dict":  # Nested dict (e.g., ahc, vbx configs)
                nested = {}
                for nkey, nspace in space["params"].items():
                    trial_key = f"{key}.{nkey}"
                    if nspace.get("type") == "categorical":
                        nested[nkey] = trial.suggest_categorical(trial_key, nspace["choices"])
                    elif nspace.get("type") == "int":
                        nested[nkey] = trial.suggest_int(trial_key, nspace["low"], nspace["high"], step=nspace.get("step", 1))
                    elif nspace.get("type") == "float":
                        nested[nkey] = trial.suggest_float(trial_key, nspace["low"], nspace["high"], log=nspace.get("log", False))
                params[key] = nested
        else:
            params[key] = space  # Fixed value

    # Create unique output dir for this trial
    trial_id = trial.number
    trial_output_dir = base_output_dir / f"trial_{trial_id}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run inference
        run_inference_with_params(model, dataset, trial_output_dir, params, device)

        # Compute DER
        der = compute_der(trial_output_dir, reference_dir, collar=0.25)

        # Cleanup trial directory (optional - comment out to keep for debugging)
        # shutil.rmtree(trial_output_dir)

        return der if not np.isnan(der) else 1.0  # Return worst DER if NaN
    except Exception as e:
        logger.error("Trial %d failed: %s", trial_id, str(e))
        return 1.0  # Return worst DER on error




def main():
    parser = argparse.ArgumentParser(description="Optimize diarization hyperparameters")
    parser.add_argument("--train_config", type=str, help="Training config (for model loading)")
    parser.add_argument("--infer_config", type=str, required=True, help="Inference config")
    parser.add_argument("--test_manifest", type=str, required=True, help="Test dataset manifest")
    parser.add_argument("--reference_dir", type=str, required=True, help="Reference RTTM directory")
    parser.add_argument("--output_dir", type=str, default="exp/hyperopt", help="Output directory for optimization")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--param_config", type=str, help="JSON file with parameter search space (optional, should contain 'optuna_space' key)")
    args = parser.parse_args()

    # Load configs
    infer_cfg = OmegaConf.load(args.infer_config)
    
    # Load train_config if available (same logic as diar_stages.py)
    train_cfg = None
    if args.train_config:
        train_cfg = OmegaConf.load(args.train_config)
    else:
        # Try loading from train_config_path in infer_config
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

    # Load model (same logic as diar_stages.py)
    logger.info("Loading model...")
    model_path = OmegaConf.select(infer_cfg, "model_path", default=None)
    if model_path is None and train_cfg:
        model_path = str(Path(train_cfg.exp_dir) / "checkpoints" / "last.ckpt")
    if model_path is None:
        raise ValueError("model_path must be specified in infer_config")
    
    # Check if model path exists, try alternatives
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
    
    model = _load_model(model_path, train_cfg)
    model.eval()

    # Load dataset
    logger.info("Loading dataset from %s", args.test_manifest)
    dataset = DiarizationDataset(
        manifest_path=args.test_manifest,
        chunk_duration=None,
        chunk_shift=None,
        frame_shift=0.02,
        max_speakers=4,
        sample_rate=16000,
    )

    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure reference RTTMs exist (write from dataset cuts if needed)
    if not reference_dir.exists() or len(list(reference_dir.glob("*.rttm"))) == 0:
        logger.info("Writing reference RTTMs from dataset cuts to %s", reference_dir)
        _write_ref_rttm_from_cuts(dataset.cuts, reference_dir)

    # Check Optuna availability
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required. Install with: pip install optuna")

    # Define parameter search space - only VBx and erosion/dilation parameters
    if args.param_config:
        with open(args.param_config) as f:
            config_data = json.load(f)
            param_space = config_data.get("optuna_space", config_data)
    else:
        # Default parameter search space: VBx + erosion/dilation + AHC threshold (when backend=ahc)
        param_space = {
            "embedding_dilation_size": {"type": "int", "low": 0, "high": 10},
            "embedding_erosion_size": {"type": "int", "low": 0, "high": 10},
            "post_oa_dilation_size": {"type": "int", "low": 0, "high": 10},
            "post_oa_erosion_size": {"type": "int", "low": 0, "high": 10},
            "clustering_backend": {"type": "categorical", "choices": ["ahc", "vbx"]},
            "ahc": {
                "type": "dict",
                "params": {
                    "threshold": {"type": "categorical", "choices": ["auto", 0.5, 0.55, 0.6, 0.65, 0.7]},
                },
            },
            "vbx": {
                "type": "dict",
                "params": {
                    "fa": {"type": "float", "low": 0.1, "high": 0.5},
                    "fb": {"type": "float", "low": 10.0, "high": 25.0},
                    "loopP": {"type": "float", "low": 0.95, "high": 0.99},
                },
            },
        }

    # Base parameters from config (fixed, not optimized)
    base_params = {
        "apply_median_filtering": OmegaConf.select(infer_cfg, "inference.apply_median_filtering", default=True),
        "median_filter_size": OmegaConf.select(infer_cfg, "inference.median_filter_size", default=11),
        "speaker_embedding_model_tag": OmegaConf.select(infer_cfg, "inference.speaker_embedding_model_tag", default="espnet/voxcelebs12_rawnet3"),
        "embedding_exclude_overlap": OmegaConf.select(infer_cfg, "inference.embedding_exclude_overlap", default=True),
        "embedding_min_speaker_duration_sec": OmegaConf.select(infer_cfg, "inference.embedding_min_speaker_duration_sec", default=1.0),
        "smoothing_embedding": OmegaConf.select(infer_cfg, "inference.smoothing_embedding", default="median"),
        "smoothing_post_oa": OmegaConf.select(infer_cfg, "inference.smoothing_post_oa", default="median"),
        "embedding_median_filter": OmegaConf.select(infer_cfg, "inference.embedding_median_filter", default=0),
        "post_oa_median_filter": OmegaConf.select(infer_cfg, "inference.post_oa_median_filter", default=0),
        "binarization_threshold": OmegaConf.select(infer_cfg, "inference.binarization_threshold", default=0.5),
        "min_speakers": OmegaConf.select(infer_cfg, "inference.ahc.min_speakers", default=OmegaConf.select(infer_cfg, "inference.min_speakers", default=1)),
        "max_speakers": OmegaConf.select(infer_cfg, "inference.ahc.max_speakers", default=OmegaConf.select(infer_cfg, "inference.max_speakers", default=10)),
        "chunk_duration_sec": OmegaConf.select(infer_cfg, "inference.chunk_duration", default=20.0),
        "chunk_shift_sec": OmegaConf.select(infer_cfg, "inference.chunk_shift", default=5.0),
        "frame_shift_sec": 0.02,
        "sample_rate": 16000,
    }

    # Optuna optimization
    logger.info("Starting Optuna optimization with %d trials", args.n_trials)
    logger.info("Optimizing parameters: clustering_backend (ahc/vbx), AHC threshold, VBx (fa, fb, loopP), erosion/dilation sizes")
    study = optuna.create_study(direction="minimize", study_name="diarization_hyperopt")
    study.optimize(
        lambda trial: objective_optuna(trial, model, dataset, reference_dir, output_dir, args.device, {**base_params, **param_space}),
        n_trials=args.n_trials,
    )

    best_der = study.best_value
    
    # Reconstruct nested structure from Optuna's flat best_params
    # Optuna stores nested params as "vbx.fa", but we need {"vbx": {"fa": ...}}
    flat_best_params = study.best_params
    best_params = {}
    nested_groups = {}
    for key, value in flat_best_params.items():
        if "." in key:
            outer, inner = key.split(".", 1)
            if outer not in nested_groups:
                nested_groups[outer] = {}
            nested_groups[outer][inner] = value
        else:
            best_params[key] = value
    best_params.update(nested_groups)

    logger.info("=" * 80)
    logger.info("Optuna optimization complete")
    logger.info("=" * 80)
    logger.info("Best DER: %.4f (%.2f%%)", best_der, best_der * 100)
    logger.info("Best parameters:")
    for key, value in sorted(best_params.items()):
        logger.info("  %s: %s", key, value)

    # Save results (include both flat and nested formats)
    results = {
        "best_der": best_der,
        "best_params": best_params,
        "best_params_flat": flat_best_params,  # Optuna's original format
        "n_trials": len(study.trials),
        "all_trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
            }
            for trial in study.trials
        ],
    }
    with open(output_dir / "optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Optimization results saved to %s", output_dir / "optimization_results.json")


if __name__ == "__main__":
    main()
