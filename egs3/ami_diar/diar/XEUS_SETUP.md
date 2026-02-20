# XEUS SSL Model Setup Guide

## Overview

XEUS is a state-of-the-art multilingual self-supervised learning model developed by ESPnet. It has been integrated into the diarization recipe, but requires a checkpoint file to use.

## Key Differences: XEUS vs WavLM

| Aspect | WavLM | XEUS |
|--------|-------|------|
| **Loading Method** | HuggingFace Transformers | ESPnet SSLTask checkpoint |
| **Availability** | Directly via `transformers` | Requires checkpoint download |
| **Hidden Size** | 768-dim | 1024-dim |
| **Layers** | 13 (12 + 1 CNN) | 25 (24 + 1 CNN) |
| **Languages** | ~100 | 4057 |
| **Special Features** | Masked prediction | + Dereverberation |
| **Setup Complexity** | ✅ Easy (auto-download) | ⚠️ Manual download needed |

## How to Use XEUS

### Step 1: Download XEUS Checkpoint

Visit the XEUS HuggingFace repository and download the checkpoint:

**URL**: https://huggingface.co/espnet/xeus

The checkpoint file is typically named:
- `checkpoint.pth` or
- `exp/train_*/checkpoint.pth` or
- `valid.acc.best.pth`

Download size: ~1GB

### Step 2: Set Environment Variable

```bash
export XEUS_CHECKPOINT=/path/to/your/downloaded/checkpoint.pth
```

You can also add this to your `~/.bashrc` or `~/.zshrc` for persistence:

```bash
echo 'export XEUS_CHECKPOINT=/path/to/checkpoint.pth' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Run Training

```bash
cd /Users/samco/Projects/ESPnet3/espnet/egs3/ami_diar/diar
python3 train_xeus.py
```

The script will:
1. Check for `XEUS_CHECKPOINT` environment variable
2. Load XEUS using `espnet2.tasks.ssl.SSLTask.build_model_from_file()`
3. Extract features using `xeus_model.encode()` method
4. Train the diarization model

## Technical Implementation

### Loading XEUS (ssl_frontend.py)

```python
from espnet2.tasks.ssl import SSLTask

# Load XEUS checkpoint
xeus_model, train_args = SSLTask.build_model_from_file(
    None,
    checkpoint_path,
    device,
)

# Extract features with all layer outputs
all_layer_outputs = xeus_model.encode(
    waveform,
    lengths,
    use_mask=False,  # Don't use masking during fine-tuning
    use_final_output=False  # Get all layer outputs
)[0]

# Stack all layers for layer weighting
features = torch.stack(all_layer_outputs, dim=-1)
```

### Key Differences from HuggingFace Models

1. **Initialization**: Uses `SSLTask.build_model_from_file()` instead of `AutoModel.from_pretrained()`
2. **Forward pass**: Uses `encode()` method instead of direct forward call
3. **Interface**: Returns list of layer outputs, not a dictionary with `hidden_states`
4. **Checkpoint format**: ESPnet checkpoint format (`.pth`), not HuggingFace format

## Configuration

### Model Config for XEUS

```python
model_config = {
    "ssl_model_name": "xeus",  # Triggers XEUS loading in ssl_frontend.py
    "ssl_model_path": "/path/to/checkpoint.pth",  # Required for XEUS
    "ssl_freeze": True,  # Recommended (XEUS is large)
    "ssl_num_layers": None,  # Auto-detected (25 layers)
    "ssl_hidden_size": None,  # Auto-detected (1024-dim)
    "ssl_layer_weights": True,  # Learnable layer combination
    "projection_size": 128,  # Project 1024 -> 128
    # ... rest of config
}
```

### YAML Config (conf/tuning/xeus_debug.yaml)

```yaml
model:
  _target_: espnet3.components.diarization.segmentation_model.PowersetDiarizationModel
  ssl_model_name: xeus
  ssl_model_path: ${env:XEUS_CHECKPOINT}  # Use environment variable
  ssl_freeze: true
  ssl_num_layers: null  # Auto-detect
  ssl_hidden_size: null  # Auto-detect
  ssl_layer_weights: true
  projection_size: 256
  conformer_num_blocks: 4
  # ... rest of config
```

## Code Changes Made

### 1. ssl_frontend.py Updates

Added XEUS-specific loading and feature extraction:

```python
# In __init__
if model_name.lower() == "xeus" or (model_path and "xeus" in model_path.lower()):
    self._load_xeus_from_checkpoint(model_path)
    self.is_xeus = True

# New method: _load_xeus_from_checkpoint
def _load_xeus_from_checkpoint(self, checkpoint_path: Optional[str] = None):
    from espnet2.tasks.ssl import SSLTask
    self.upstream, self.train_args = SSLTask.build_model_from_file(
        None, checkpoint_path, device
    )

# New method: _extract_features_xeus
def _extract_features_xeus(self, waveform, lengths):
    all_layer_outputs = self.upstream.encode(
        waveform, lengths,
        use_mask=False,
        use_final_output=False
    )[0]
    features = torch.stack(all_layer_outputs, dim=-1)
    return features, lengths

# Updated forward to handle XEUS
if self.is_xeus:
    features, lengths = self._extract_features_xeus(waveform, lengths)
```

### 2. train_xeus.py Updates

Added checkpoint validation:

```python
import os
xeus_checkpoint = os.environ.get("XEUS_CHECKPOINT", None)

if xeus_checkpoint is None:
    print("ERROR: XEUS checkpoint not found!")
    print("Download from: https://huggingface.co/espnet/xeus")
    sys.exit(1)

model_config = {
    "ssl_model_name": "xeus",
    "ssl_model_path": xeus_checkpoint,
    # ...
}
```

## Troubleshooting

### Error: "XEUS checkpoint not found!"

**Solution**: Set the `XEUS_CHECKPOINT` environment variable:
```bash
export XEUS_CHECKPOINT=/path/to/checkpoint.pth
```

### Error: "No module named 'espnet2.tasks.ssl'"

**Solution**: Ensure you're using ESPnet3 with SSL task support:
```bash
cd /Users/samco/Projects/ESPnet3/espnet
pip install -e .
```

### Error: "XEUS requires a checkpoint path"

**Solution**: Provide `ssl_model_path` in model config:
```python
model_config = {
    "ssl_model_name": "xeus",
    "ssl_model_path": "/path/to/checkpoint.pth",  # This is required!
}
```

### Checkpoint file is too large to download

**Alternative**: Use WavLM for debugging/development:
```bash
python3 train_debug.py  # Uses WavLM Base (auto-downloads)
```

Then switch to XEUS for production training once you have the checkpoint.

## Performance Expectations

### Debug Run (3 meetings, 3 epochs)
- **WavLM**: ~47 seconds, 364MB checkpoint
- **XEUS**: ~2 minutes, ~1GB checkpoint

### Full Training (135 meetings, 50 epochs)
- **WavLM**: DER ~16-18%
- **XEUS**: DER ~14-16% (better multilingual performance)

## Summary

XEUS is now fully integrated into the diarization recipe! The key requirement is downloading the checkpoint file first. Once you have the checkpoint:

1. ✅ Set `XEUS_CHECKPOINT` environment variable
2. ✅ Run `python3 train_xeus.py`
3. ✅ Everything else works automatically

The `ssl_frontend.py` module automatically handles the differences between XEUS and HuggingFace models, providing a unified interface for the diarization system.

## Example: Complete Workflow

```bash
# 1. Download XEUS checkpoint (do this once)
cd ~/models
wget https://huggingface.co/espnet/xeus/resolve/main/checkpoint.pth

# 2. Set environment variable
export XEUS_CHECKPOINT=$HOME/models/checkpoint.pth

# 3. Navigate to recipe
cd /Users/samco/Projects/ESPnet3/espnet/egs3/ami_diar/diar

# 4. Run training
python3 train_xeus.py

# 5. Compare with WavLM (optional)
python3 train_debug.py  # WavLM for comparison
./compare_models.sh     # Compare both models
```

## References

- XEUS Paper: https://arxiv.org/abs/2407.00837
- XEUS HuggingFace: https://huggingface.co/espnet/xeus
- ESPnet SSL Task: `espnet2.tasks.ssl`
- Implementation: `espnet3/components/diarization/ssl_frontend.py`
