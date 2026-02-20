#!/usr/bin/env bash
# AMI diarization recipe (egs3/ami_diar/diar)
# Use: source path.sh && ./run.sh
# AMI lhotse manifests: set ami_manifests_dir in conf/tuning/debug.yaml (default: /raid/users/popcornell/AMI)

set -euo pipefail

# Source path configuration (activates environment)
. ./path.sh

# Default: debug config and stages that do not require publish_config
train_config=conf/tuning/debug.yaml
infer_config=conf/inference.yaml
metric_config=conf/metric.yaml
stages="create_dataset train_tokenizer train infer metric"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --stages)
      stages="$2"
      shift 2
      ;;
    --train-config)
      train_config="$2"
      shift 2
      ;;
    --infer-config)
      infer_config="$2"
      shift 2
      ;;
    --metric-config)
      metric_config="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--stages STAGES] [--train-config CONFIG] [--infer-config CONFIG] [--metric-config CONFIG]"
      echo "  Default stages: create_dataset train_tokenizer train infer metric"
      echo "  For full e2e + DER: use stages above (infer writes RTTM, metric prints global DER for AMI test set)."
      exit 1
      ;;
  esac
done

python run.py \
  --stages ${stages} \
  --train_config "${train_config}" \
  --infer_config "${infer_config}" \
  --metric_config "${metric_config}"
