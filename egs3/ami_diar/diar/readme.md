# AMI Diarization Recipe with XEUS/WavLM and Powerset Encoding

This recipe implements a state-of-the-art speaker diarization system based on the DiariZen architecture, using:

- **SSL Frontend**: XEUS or WavLM for robust feature extraction
- **Conformer Encoder**: Temporal modeling with self-attention and convolution
- **Powerset Encoding**: Multi-label to multi-class conversion for better overlap handling
- **Lhotse**: Efficient data loading and processing

## Architecture

```
Waveform → SSL (XEUS/WavLM) → Projection → Layer Norm →
Conformer Encoder → Powerset Classifier → Segmentation →
(Optional) Speaker Embeddings → Clustering → Diarization
```

### Key Components

1. **SSL Frontend** (`espnet3/components/diarization/ssl_frontend.py`):
   - Supports multiple SSL models: XEUS, WavLM, HuBERT, Wav2Vec2
   - Learnable layer weighting for combining SSL layers
   - Gradient scaling for fine-tuning

2. **Powerset Encoding** (`espnet3/components/diarization/powerset.py`):
   - Converts multi-label (each speaker independently) to multi-class (all combinations)
   - Better handles overlapping speech
   - Example: For 3 speakers, max 2 overlapping → 7 classes: {}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}

3. **Segmentation Model** (`espnet3/components/diarization/segmentation_model.py`):
   - Conformer-based temporal modeling
   - Dual optimizer for SSL vs. non-SSL parameters
   - Frame-level speaker activity prediction

4. **Inference Pipeline** (`src/inference.py`):
   - Median filtering for smoothing
   - Optional speaker embedding extraction
   - Clustering (AHC or VBx) for speaker assignment

## Directory Structure

```
egs3/ami_diar/diar/
├── conf/
│   ├── tuning/
│   │   └── train_xeus_conformer_powerset.yaml  # Training config
│   ├── inference.yaml                           # Inference config
│   └── metric.yaml                              # Metric config
├── src/
│   ├── create_dataset.py   # Lhotse manifest creation
│   ├── dataset.py          # Diarization dataset with lhotse
│   └── inference.py        # Inference pipeline
├── run.py                  # Main entry point
├── run.sh                  # Shell script wrapper
├── path.sh                 # Environment setup
└── readme.md               # This file
```

## Prerequisites

### Software Requirements

```bash
# ESPnet3 with dependencies
cd espnet/tools
make

# Additional packages
pip install lhotse
pip install transformers  # For HuggingFace SSL models
pip install scikit-learn  # For clustering
```

### Data Preparation

Your data should be organized in Kaldi-style format:

```
data/
├── train/
│   ├── wav.scp  # <recording-id> <path-to-audio>
│   └── rttm     # RTTM format annotations
├── dev/
│   ├── wav.scp
│   └── rttm
└── test/
    ├── wav.scp
    └── rttm
```

## Usage

### Step 1: Create Lhotse Manifests

```bash
# Set up environment
. ./path.sh

# Create manifests from your data
python src/create_dataset.py \
  --data-dir /path/to/your/data \
  --output-dir data/lhotse_manifests \
  --dataset-name ami

# This creates:
# - data/lhotse_manifests/train_cuts.jsonl.gz
# - data/lhotse_manifests/dev_cuts.jsonl.gz
# - data/lhotse_manifests/test_cuts.jsonl.gz
```

### Step 2: Training

#### Full Pipeline (Recommended)

```bash
./run.sh --stages all
```

#### Stage-by-Stage

```bash
# Create dataset
python run.py --stages create_dataset

# Collect statistics (if using normalization)
python run.py --stages collect_stats

# Train model
python run.py --stages train

# Inference
python run.py --stages infer

# Compute metrics
python run.py --stages measure
```

### Step 3: Inference on New Data

```bash
# Edit conf/inference.yaml to set paths
# Then run inference
python run.py --stages infer --infer_config conf/inference.yaml
```

## Configuration

### Training Configuration

Key hyperparameters in `conf/tuning/train_xeus_conformer_powerset.yaml`:

**SSL Model Selection:**
```yaml
model:
  ssl_model_name: wavlm_base  # Options: "xeus", "wavlm_base", "wavlm_large", etc.
  ssl_freeze: false           # Set true to freeze SSL parameters
  ssl_feature_grad_mult: 0.1  # Gradient scaling for SSL
```

**Model Architecture:**
```yaml
model:
  projection_size: 256         # Hidden dimension
  conformer_num_blocks: 4      # Number of Conformer layers
  conformer_attention_heads: 4
  conformer_ffn_units: 1024
  num_speakers: 4              # Max speakers in dataset
  max_speakers_per_frame: 2    # Max simultaneous speakers
```

**Training Settings:**
```yaml
dataloader:
  train:
    batch_size: 16    # Adjust based on GPU memory

trainer:
  max_epochs: 100
  precision: 16-mixed  # Use mixed precision for faster training
  gradient_clip_val: 5.0
```

**Optimizer (Dual optimizer for SSL vs. non-SSL):**
```yaml
optimizer:
  lr: 0.001          # Learning rate for Conformer/Classifier

optimizer_ssl:
  lr: 0.00002        # Lower LR for SSL fine-tuning
```

### Using XEUS Instead of WavLM

To use XEUS model:

1. Download XEUS checkpoint from [WavLab](https://www.wavlab.org/activities/2024/xeus/)

2. Update configuration:
```yaml
model:
  ssl_model_name: xeus
  ssl_model_path: /path/to/xeus-checkpoint.pt
  ssl_num_layers: 25  # XEUS-Large has 25 layers
  ssl_hidden_size: 1024
```

### Inference Configuration

In `conf/inference.yaml`:

**Basic Settings:**
```yaml
inference:
  apply_median_filtering: true
  median_filter_size: 11
  binarization_threshold: 0.5
```

**With Speaker Embeddings:**
```yaml
inference:
  speaker_embedding_model: /path/to/espnet2/spk/model.pth
  clustering_backend: vbx  # or "ahc"
  min_speakers: 1
  max_speakers: 20
```

## Expected Results

### Benchmark (AMI Corpus)

| Model | DER (%) | JER (%) | Notes |
|-------|---------|---------|-------|
| WavLM-Base + Conformer | ~15-20 | ~25-30 | Baseline |
| XEUS + Conformer | ~14-18 | ~23-28 | Better multilingual |
| + Speaker Embeddings | ~12-16 | ~20-25 | With clustering |

*Note: Actual results depend on data preparation, hyperparameters, and training time*

## Differences from DiariZen

This implementation follows DiariZen's architecture but integrates with ESPnet3:

1. **Framework**: Uses PyTorch Lightning instead of custom training loop
2. **Data Loading**: Uses lhotse instead of custom dataset
3. **Configuration**: Uses Hydra/OmegaConf for configuration
4. **Modular**: SSL frontend, powerset, and model are separate components
5. **Inference**: Simplified inference pipeline (full VBx clustering can be added)

## Advanced Usage

### Multi-Channel Support

To add multi-channel support (following DiariZen-MC):

1. Update `SSLFrontend` to process channels separately
2. Add channel fusion module in segmentation model
3. Modify dataset to load multi-channel audio

### Custom SSL Models

To use a custom SSL model:

1. Option A: Use HuggingFace transformers:
```python
model:
  ssl_model_name: "your-org/your-ssl-model"
```

2. Option B: Use s3prl:
```yaml
model:
  _target_: espnet3.components.diarization.ssl_frontend.SSLFrontend
  use_s3prl: true
  model_name: "your_s3prl_model"
```

### Distributed Training

```bash
# Multi-GPU training
python run.py --stages train --train_config conf/tuning/train_xeus_conformer_powerset.yaml

# Edit trainer config:
trainer:
  num_device: 4  # Number of GPUs
  strategy: ddp  # Distributed data parallel
```

## Troubleshooting

### Out of Memory

1. Reduce batch size:
```yaml
dataloader:
  train:
    batch_size: 8  # or 4
```

2. Use gradient accumulation:
```yaml
trainer:
  accumulate_grad_batches: 2
```

3. Freeze SSL parameters:
```yaml
model:
  ssl_freeze: true
```

### Poor Performance

1. Check data quality (RTTM annotations)
2. Increase training epochs
3. Tune learning rates (especially SSL vs non-SSL)
4. Try different SSL models
5. Adjust `max_speakers_per_frame` based on overlap in data

### Dataset Loading Issues

Ensure lhotse manifests are created correctly:
```bash
# Verify manifests
python -c "from lhotse import CutSet; cuts = CutSet.from_file('data/lhotse_manifests/train_cuts.jsonl.gz'); print(f'Loaded {len(cuts)} cuts')"
```

## Citation

If you use this recipe, please cite:

```bibtex
@inproceedings{diarizen2024,
  title={DiariZen: Multi-channel Speaker Diarization with Self-Supervised Learning},
  author={...},
  booktitle={...},
  year={2024}
}

@inproceedings{xeus2024,
  title={XEUS: Towards Robust Speech Representation Learning for Thousands of Languages},
  author={Chen, Wanchi and others},
  booktitle={arXiv preprint arXiv:2407.00837},
  year={2024}
}

@inproceedings{espnet3,
  title={ESPnet3: A Complete Rewrite Towards Modular Speech Processing},
  author={...},
  booktitle={...},
  year={2025}
}
```

## Contributing

To add new features or improvements:

1. Add new components in `espnet3/components/diarization/`
2. Update configuration templates in `conf/`
3. Add tests and documentation
4. Submit PR to ESPnet repository

## References

- [DiariZen GitHub](https://github.com/your-org/DiariZen)
- [XEUS Paper](https://arxiv.org/abs/2407.00837)
- [ESPnet3 Documentation](https://espnet.github.io/espnet/)
- [Lhotse Documentation](https://lhotse.readthedocs.io/)
- [PyAnnote.audio](https://github.com/pyannote/pyannote-audio)

## License

Apache 2.0 (same as ESPnet)

## Contact

For questions or issues, please open an issue on the ESPnet GitHub repository.
