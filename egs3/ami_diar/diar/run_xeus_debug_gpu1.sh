#!/bin/bash
# XEUS debug run on GPU 1. Uses same structure as full_training (WavLM) with XEUS frontend.
# Set XEUS_CHECKPOINT to your XEUS checkpoint path, or pass model.ssl_model_path=/path/to/xeus.pt
# Example: export XEUS_CHECKPOINT=/path/to/xeus/checkpoint.pth && ./run_xeus_debug_gpu1.sh

set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=1
EXTRA="${*}"
python run.py --stages create_dataset,train --train_config conf/tuning/train_xeus_debug.yaml $EXTRA
