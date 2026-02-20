#!/usr/bin/env python3
"""Simple standalone inference script."""

import sys
from pathlib import Path
import torch
from lhotse import CutSet

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd().parent.parent.parent))

from src.inference import DiarizationInference, save_rttm
from espnet3.components.diarization.segmentation_model import PowersetDiarizationModel

# Load model
checkpoint_path = "exp/debug_run/checkpoints/last.ckpt"
print(f"Loading model from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

model_config = checkpoint["hyper_parameters"]["model_config"]
model = PowersetDiarizationModel(**model_config)

# Load state dict
state_dict = {}
for key, value in checkpoint["state_dict"].items():
    if key.startswith("model."):
        new_key = key[6:]
        state_dict[new_key] = value

model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully")

# Load cuts
cuts = CutSet.from_file("data/ami_debug/dev_cuts.jsonl.gz")
print(f"Processing {len(cuts)} recordings...")

# Create inference pipeline
inference = DiarizationInference(
    model=model,
    device="cpu",
    apply_median_filtering=True,
    median_filter_size=11,
    binarization_threshold=0.5,
    speaker_embedding_model_tag="espnet/voxcelebs12_rawnet3",
    clustering_backend="ahc",
    min_speakers=1,
    max_speakers=10,
)

# Create output directory
output_dir = Path("exp/debug_run/infer")
output_dir.mkdir(parents=True, exist_ok=True)

# Process each recording
for cut in cuts:
    recording_id = cut.id
    print(f"\nProcessing {recording_id}...")
    print(f"  Duration: {cut.duration:.1f}s")

    # Load audio
    audio = cut.load_audio()
    print(f"  Audio shape: {audio.shape}")

    # Convert to mono if multi-channel
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
        print(f"  Converted to mono: {audio.shape}")

    # Convert to tensor
    waveform = torch.from_numpy(audio).float()

    # Run diarization
    print(f"  Running diarization...")
    diarization = inference(waveform, sample_rate=16000)

    # Save RTTM
    rttm_path = output_dir / f"{recording_id}.rttm"
    save_rttm(diarization, rttm_path, recording_id)

    print(f"  Saved {len(diarization)} segments to {rttm_path}")

print(f"\nInference complete! Files saved to {output_dir}")
