# Hyperparameter Optimization for Diarization

This script optimizes diarization inference hyperparameters by running inference with different parameter combinations and selecting the best based on DER (Diarization Error Rate).

## Installation

Install required dependencies:

```bash
pip install optuna pyannote.core pyannote.metrics
```

## Usage

### Basic Usage (Optuna - Recommended)

```bash
python optimize_hyperparameters.py \
    --infer_config conf/inference.yaml \
    --test_manifest data/ami_lhotse/test_cuts.jsonl.gz \
    --reference_dir exp/full_training/infer/reference \
    --output_dir exp/hyperopt \
    --n_trials 50 \
    --method optuna \
    --device cuda
```

### Grid Search (Fallback)

If Optuna is not available, use grid search:

```bash
python optimize_hyperparameters.py \
    --infer_config conf/inference.yaml \
    --test_manifest data/ami_lhotse/test_cuts.jsonl.gz \
    --reference_dir exp/full_training/infer/reference \
    --output_dir exp/hyperopt \
    --method grid \
    --device cuda
```

### Custom Parameter Space

Create a JSON file with your parameter search space (see `conf/hyperopt_params.json` for example):

```bash
python optimize_hyperparameters.py \
    --infer_config conf/inference.yaml \
    --test_manifest data/ami_lhotse/test_cuts.jsonl.gz \
    --reference_dir exp/full_training/infer/reference \
    --output_dir exp/hyperopt \
    --param_config conf/hyperopt_params.json \
    --n_trials 100 \
    --method optuna
```

## Parameters

- `--infer_config`: Path to inference config YAML (required)
- `--train_config`: Path to training config YAML (optional, for model loading)
- `--test_manifest`: Path to test dataset manifest (required)
- `--reference_dir`: Directory with reference RTTM files (required)
- `--output_dir`: Output directory for optimization results (default: `exp/hyperopt`)
- `--n_trials`: Number of Optuna trials (default: 50)
- `--method`: Optimization method: `optuna` or `grid` (default: `optuna`)
- `--device`: Device for inference: `cuda` or `cpu` (default: `cuda`)
- `--param_config`: JSON file with custom parameter search space (optional)

## Parameter Search Space

### Optuna Format

For Bayesian optimization with Optuna, define parameters as:

```json
{
  "param_name": {
    "type": "float|int|categorical",
    "low": 0.0,
    "high": 1.0,
    "step": 1,  // for int only
    "log": false,  // for float only
    "choices": ["a", "b"]  // for categorical
  },
  "nested_config": {
    "type": "dict",
    "params": {
      "nested_param": {"type": "int", "low": 2, "high": 4}
    }
  }
}
```

### Grid Search Format

For grid search, provide lists of values:

```json
{
  "param_name": [value1, value2, value3],
  "nested.param": [value1, value2]
}
```

## Output

Results are saved to `{output_dir}/optimization_results.json`:

```json
{
  "best_der": 0.1234,
  "best_params": {
    "binarization_threshold": 0.5,
    "embedding_median_filter": 11,
    ...
  },
  "n_trials": 50
}
```

For grid search, `all_results` contains DER for all combinations.

## Optimized Parameters

Common parameters optimized:

- **Binarization**: `binarization_threshold` (0.3-0.7)
- **Embedding smoothing**: `embedding_median_filter`, `embedding_dilation_size`, `embedding_erosion_size`
- **Post-OA smoothing**: `post_oa_median_filter`, `post_oa_dilation_size`, `post_oa_erosion_size`
- **Clustering**: `clustering_backend` (ahc/spectral/vbx), `ahc.min_speakers`, `ahc.max_speakers`, `ahc.threshold`
- **VBx**: `vbx.fa`, `vbx.fb`, `vbx.loopP`

## Tips

1. **Start with fewer trials** (20-30) to get a sense of the search space
2. **Use Optuna** for efficient Bayesian optimization (finds good parameters faster)
3. **Grid search** is useful for small, discrete parameter spaces
4. **Reference RTTMs** are automatically generated from dataset cuts if missing
5. **Trial outputs** are saved in `{output_dir}/trial_{n}/` for debugging (can be cleaned up)

## Example Workflow

1. Run initial optimization with default search space:
   ```bash
   python optimize_hyperparameters.py \
       --infer_config conf/inference.yaml \
       --test_manifest data/ami_lhotse/test_cuts.jsonl.gz \
       --reference_dir exp/full_training/infer/reference \
       --n_trials 30
   ```

2. Review results and narrow search space around best parameters

3. Run fine-tuning with focused search space:
   ```bash
   python optimize_hyperparameters.py \
       --infer_config conf/inference.yaml \
       --test_manifest data/ami_lhotse/test_cuts.jsonl.gz \
       --reference_dir exp/full_training/infer/reference \
       --param_config conf/hyperopt_fine_tune.json \
       --n_trials 50
   ```

4. Update `inference.yaml` with best parameters
