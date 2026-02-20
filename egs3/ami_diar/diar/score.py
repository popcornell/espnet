#!/usr/bin/env python3
"""Standalone scoring script for diarization using pyannote.metrics.

Computes both DER and JER with multiple collar settings (0.0, 0.25, 0.5).
"""

import argparse
from pathlib import Path


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


def compute_metrics_detailed(hypothesis_dir, reference_dir, collar_values=None):
    """Compute DER and JER with detailed breakdown.

    Args:
        hypothesis_dir: Directory containing hypothesis RTTM files
        reference_dir: Directory containing reference RTTM files
        collar_values: List of collar values to evaluate (default: [0.0, 0.25, 0.5])

    Returns:
        dict: Results for each collar value
    """
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    except ImportError:
        print("Error: pyannote.metrics not installed")
        print("Install with: pip install pyannote.core pyannote.metrics")
        return None

    if collar_values is None:
        collar_values = [0.0, 0.25, 0.5]

    hypothesis_dir = Path(hypothesis_dir)
    reference_dir = Path(reference_dir)

    print("\n" + "=" * 80)
    print("Diarization Scoring with pyannote.metrics")
    print("=" * 80)

    # Find matching RTTM files
    hyp_rttms = sorted(hypothesis_dir.glob("*.rttm"))
    matched_files = []

    for hyp_rttm in hyp_rttms:
        recording_id = hyp_rttm.stem
        ref_rttm = reference_dir / f"{recording_id}.rttm"

        if not ref_rttm.exists():
            print(f"Warning: No reference found for {recording_id}, skipping")
            continue

        matched_files.append((recording_id, hyp_rttm, ref_rttm))

    if len(matched_files) == 0:
        print("\nError: No matching reference files found")
        print(f"  Hypothesis dir: {hypothesis_dir}")
        print(f"  Reference dir:  {reference_dir}")
        return None

    print(f"\nFound {len(matched_files)} matching recording(s):")
    for recording_id, _, _ in matched_files:
        print(f"  - {recording_id}")

    # Collar settings
    collar_names = {
        0.0: "No collar",
        0.25: "250ms collar",
        0.5: "500ms collar",
    }

    results = {}

    for collar in collar_values:
        collar_name = collar_names.get(collar, f"{int(collar*1000)}ms collar")

        print(f"\n{'=' * 80}")
        print(f"Evaluating: {collar_name} (collar={collar}s)")
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

            print(f"  ✓ {recording_id}")

        # Get aggregated results
        # Note: pyannote.metrics accumulates across all files when you call metric(ref, hyp)
        # The total is retrieved with abs(metric), components are stored in metric.components
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
            'collar_name': collar_name,
        }

        # Print detailed results
        print(f"\n{'-' * 80}")
        print(f"Results for {collar_name}:")
        print(f"{'-' * 80}")
        print(f"DER:           {der_total * 100:>6.2f}%")
        print(f"  False Alarm:   {der_components['false alarm'] * 100:>6.2f}%")
        print(f"  Missed Speech: {der_components['missed detection'] * 100:>6.2f}%")
        print(f"  Speaker Error: {der_components['confusion'] * 100:>6.2f}%")
        print(f"\nJER:           {jer_total * 100:>6.2f}%")
        print(f"{'-' * 80}")

    # Print summary comparison
    if len(collar_values) > 1:
        print(f"\n{'=' * 80}")
        print("Summary - Impact of Collar")
        print(f"{'=' * 80}")
        print(f"{'Collar':<15} {'DER':>8} {'JER':>8} {'FA':>8} {'Miss':>8} {'Spk Err':>10}")
        print(f"{'-' * 80}")
        for collar in collar_values:
            r = results[collar]
            comps = r['der_components']
            print(f"{r['collar_name']:<15} "
                  f"{r['der']*100:>7.2f}% "
                  f"{r['jer']*100:>7.2f}% "
                  f"{comps['false alarm']*100:>7.2f}% "
                  f"{comps['missed detection']*100:>7.2f}% "
                  f"{comps['confusion']*100:>9.2f}%")
        print(f"{'=' * 80}")

    # Print single-line summary for easy parsing
    print(f"\nQuick Summary (250ms collar):")
    r_025 = results.get(0.25, results.get(collar_values[0]))
    print(f"  DER={r_025['der']*100:.2f}%  JER={r_025['jer']*100:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Score diarization output using pyannote.metrics (DER and JER)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score with default collars (0.0, 0.25, 0.5)
  python score.py --hypothesis exp/debug_run/infer --reference data/ami_debug/ground_truth

  # Score with custom collar
  python score.py --hypothesis exp/debug_run/infer --reference data/ami_debug/ground_truth --collar 0.25

  # Score with multiple custom collars
  python score.py --hypothesis exp/debug_run/infer --reference data/ami_debug/ground_truth --collar 0.0 0.1 0.25 0.5
        """
    )

    parser.add_argument(
        "--hypothesis",
        type=str,
        required=True,
        help="Directory containing hypothesis RTTM files",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Directory containing reference/ground truth RTTM files",
    )
    parser.add_argument(
        "--collar",
        type=float,
        nargs="+",
        default=None,
        help="Collar value(s) in seconds (default: 0.0, 0.25, 0.5)",
    )

    args = parser.parse_args()

    # Compute metrics
    results = compute_metrics_detailed(
        args.hypothesis,
        args.reference,
        collar_values=args.collar,
    )

    if results is None:
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
