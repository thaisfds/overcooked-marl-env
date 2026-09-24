#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=================================================="
echo "STARTING FULL EXPERIMENT CYCLE (TRAIN + EVALUATE)"
echo "=================================================="

# 1. Train model
venv/bin/python experiments/training/train_ppo.py --num-agents 1 --workers 8

echo "=================================================="
echo "TRAINING COMPLETED - STARTING EVALUATION"
echo "=================================================="

# 2. Evaluate model
venv/bin/python experiments/analysis/evaluate_ppo.py --num-episodes 5

echo "=================================================="
echo "EXPERIMENT CYCLE COMPLETED SUCCESSFULLY"
echo "=================================================="
