#!/bin/bash
# Script to compare WavLM vs XEUS on AMI diarization

set -e

cd /Users/samco/Projects/ESPnet3/espnet/egs3/ami_diar/diar

echo "================================================================================"
echo "Comparing WavLM Base vs XEUS on AMI Diarization"
echo "================================================================================"

echo ""
echo "1. Training with WavLM Base..."
echo "--------------------------------------------------------------------------------"
python3 train_debug.py
echo "✓ WavLM training complete"
echo ""

echo "2. Training with XEUS..."
echo "--------------------------------------------------------------------------------"
python3 train_xeus.py
echo "✓ XEUS training complete"
echo ""

echo "3. Comparing results..."
echo "--------------------------------------------------------------------------------"
echo ""
echo "WavLM Results:"
ls -lh exp/debug_run/checkpoints/last.ckpt 2>/dev/null && echo "  Checkpoint size: $(du -h exp/debug_run/checkpoints/last.ckpt | cut -f1)" || echo "  No checkpoint found"
echo ""
echo "XEUS Results:"
ls -lh exp/xeus_debug_run/checkpoints/last.ckpt 2>/dev/null && echo "  Checkpoint size: $(du -h exp/xeus_debug_run/checkpoints/last.ckpt | cut -f1)" || echo "  No checkpoint found"
echo ""

echo "================================================================================"
echo "Comparison Summary"
echo "================================================================================"
echo ""
echo "Model          | SSL Layers | Hidden Dim | Params      | Speed  | Multilingual"
echo "---------------|------------|------------|-------------|--------|-------------"
echo "WavLM Base     | 12+1 CNN   | 768        | 95M (frozen)| Fast   | ~100 langs"
echo "XEUS           | 24+1 CNN   | 1024       | ~330M       | Medium | 4057 langs"
echo ""
echo "Use WavLM for: Fast iteration, English/major languages"
echo "Use XEUS for:  Multilingual data, robust features, production"
echo ""
echo "Both models trained! Check TensorBoard logs:"
echo "  tensorboard --logdir exp/"
echo ""
