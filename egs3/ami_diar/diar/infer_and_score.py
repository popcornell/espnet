#!/usr/bin/env python3
"""Inference and scoring script for diarization."""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import json

# Add paths
recipe_dir = Path(__file__).parent
sys.path.insert(0, str(recipe_dir))
sys.path.insert(0, str(recipe_dir.parent.parent.parent))  # espnet root

# Imports
from src.dataset import DiarizationDataset, collate_fn
from src.inference import DiarizationInference, save_rttm
from espnet3.components.diarization.segmentation_model import PowersetDiarizationModel


def load_model_from_checkpoint(checkpoint_path):
    """Load model from Lightning checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract model config from hyperparameters
    model_config = checkpoint["hyper_parameters"]["model_config"]

    # Create model
    model = PowersetDiarizationModel(**model_config)

    # Load state dict (remove 'model.' prefix from Lightning)
    state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("model."):
            new_key = key[6:]  # Remove 'model.' prefix
            state_dict[new_key] = value

    model.load_state_dict(state_dict)
    model.eval()

    return model


def run_inference(model, dataset, output_dir, device="cpu"):
    """Run inference on dataset and save RTTM files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create inference pipeline
    inference = DiarizationInference(
        model=model,
        device=device,
        apply_median_filtering=True,
        median_filter_size=11,
        binarization_threshold=0.5,
        speaker_embedding_model_tag="espnet/voxcelebs12_rawnet3",
        clustering_backend="ahc",
        min_speakers=1,
        max_speakers=10,
    )

    print(f"\nRunning inference on dataset...")

    # Use DataLoader to iterate properly
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Run inference on each recording
    for batch in dataloader:
        waveform = batch["speech"][0]  # Remove batch dim
        utt_id = batch["utt_ids"][0]

        # Extract recording ID (remove chunk info)
        recording_id = utt_id if not "_" in utt_id else "_".join(utt_id.split("_")[:2])

        print(f"  Processing {recording_id} ({len(waveform)} samples, {len(waveform)/16000:.1f}s)...")

        # Run diarization
        diarization = inference(waveform, sample_rate=16000)

        # Save RTTM
        rttm_path = output_dir / f"{recording_id}.rttm"
        save_rttm(diarization, rttm_path, recording_id)

        print(f"    Saved {len(diarization)} segments to {rttm_path}")

    print(f"\nInference complete! RTTM files saved to {output_dir}")


def compute_metrics_pyannote(hypothesis_dir, reference_dir):
    """Compute DER and JER using pyannote.metrics with multiple collar settings."""
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    except ImportError:
        print("Error: pyannote.core not installed")
        print("Install with: pip install pyannote.core pyannote.metrics")
        return

    hypothesis_dir = Path(hypothesis_dir)
    reference_dir = Path(reference_dir)

    print("\n" + "=" * 80)
    print("Computing Diarization Metrics with pyannote.metrics")
    print("=" * 80)

    # Find matching RTTM files
    hyp_rttms = list(hypothesis_dir.glob("*.rttm"))
    matched_files = []

    for hyp_rttm in hyp_rttms:
        recording_id = hyp_rttm.stem
        ref_rttm = reference_dir / f"{recording_id}.rttm"

        if not ref_rttm.exists():
            print(f"Warning: No reference found for {recording_id}, skipping")
            continue

        matched_files.append((recording_id, hyp_rttm, ref_rttm))

    if len(matched_files) == 0:
        print("Error: No matching reference files found")
        return

    print(f"Found {len(matched_files)} matching recordings\n")

    # Collar settings to evaluate
    collar_settings = [
        (0.0, "No collar"),
        (0.25, "250ms collar"),
        (0.5, "500ms collar"),
    ]

    results = {}

    for collar, collar_name in collar_settings:
        print(f"\n{'=' * 80}")
        print(f"Evaluating with {collar_name} (collar={collar}s)")
        print(f"{'=' * 80}")

        # Initialize metrics
        der_metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
        jer_metric = JaccardErrorRate(collar=collar)

        # Process each file
        for recording_id, hyp_rttm, ref_rttm in matched_files:
            # Load RTTM files
            hyp = load_rttm_as_annotation(hyp_rttm)
            ref = load_rttm_as_annotation(ref_rttm)

            # Compute metrics
            der_metric(ref, hyp, uem=None)
            jer_metric(ref, hyp, uem=None)

            print(f"  Processed: {recording_id}")

        # Get aggregated results
        der_total = abs(der_metric)
        jer_total = abs(jer_metric)

        # Get component values from the internal state
        der_components = {
            'false alarm': der_metric['false alarm'] / der_metric['total'] if der_metric['total'] > 0 else 0.0,
            'missed detection': der_metric['missed detection'] / der_metric['total'] if der_metric['total'] > 0 else 0.0,
            'confusion': der_metric['confusion'] / der_metric['total'] if der_metric['total'] > 0 else 0.0,
        }

        # Store results
        results[collar] = {
            'der': der_total,
            'der_components': der_components,
            'jer': jer_total,
        }

        # Print detailed results
        print(f"\n{'-' * 80}")
        print(f"Results for {collar_name}:")
        print(f"{'-' * 80}")
        print(f"DER: {der_total * 100:.2f}%")
        print(f"  False Alarm:   {der_components['false alarm'] * 100:.2f}%")
        print(f"  Missed Speech: {der_components['missed detection'] * 100:.2f}%")
        print(f"  Speaker Error: {der_components['confusion'] * 100:.2f}%")
        print(f"\nJER: {jer_total * 100:.2f}%")
        print(f"{'-' * 80}")

    # Print summary comparison
    print(f"\n{'=' * 80}")
    print("Summary - Impact of Collar")
    print(f"{'=' * 80}")
    print(f"{'Collar':<15} {'DER':<10} {'JER':<10} {'FA':<10} {'Miss':<10} {'Conf':<10}")
    print(f"{'-' * 80}")
    for collar, collar_name in collar_settings:
        r = results[collar]
        comps = r['der_components']
        print(f"{collar_name:<15} {r['der']*100:>8.2f}% {r['jer']*100:>8.2f}% "
              f"{comps['false alarm']*100:>8.2f}% {comps['missed detection']*100:>8.2f}% "
              f"{comps['confusion']*100:>8.2f}%")
    print(f"{'=' * 80}\n")

    return results


def load_rttm_as_annotation(rttm_path):
    """Load RTTM file as pyannote Annotation."""
    from pyannote.core import Annotation, Segment

    annotation = Annotation()

    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]

            segment = Segment(start, start + duration)
            annotation[segment] = speaker

    return annotation


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inference and scoring for diarization")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="exp/debug_run/checkpoints/last.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ami_debug",
        help="Data directory with lhotse manifests",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/debug_run/infer",
        help="Output directory for RTTM files",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default=None,
        help="Reference RTTM directory for scoring (if None, will extract from lhotse)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Diarization Inference and Scoring")
    print("=" * 80)

    # Load model
    model = load_model_from_checkpoint(args.checkpoint)

    # Create dataset
    print("\nCreating dataset...")
    dataset = DiarizationDataset(
        manifest_path=Path(args.data_dir) / "dev_cuts.jsonl.gz",
        chunk_duration=None,  # Whole recordings
        chunk_shift=None,
        frame_shift=0.02,
        max_speakers=4,
        sample_rate=16000,
    )

    print(f"Dataset: {len(dataset)} recordings")

    # Run inference
    run_inference(model, dataset, args.output_dir, device=args.device)

    # Compute metrics (DER and JER)
    if args.reference_dir:
        compute_metrics_pyannote(args.output_dir, args.reference_dir)
    else:
        print("\nNo reference directory provided, skipping metrics computation")
        print("To compute DER/JER, provide --reference-dir with ground truth RTTM files")


if __name__ == "__main__":
    main()
