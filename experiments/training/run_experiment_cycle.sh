#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ "$HOME" = "/" ] || [ -z "$HOME" ]; then
    export HOME="$PROJECT_ROOT"
fi

echo "=================================================="
echo "STARTING FULL EXPERIMENT CYCLE (TRAIN + EVALUATE)"
echo "=================================================="

if [ -n "$PYTHON_CMD" ]; then
    # Respect explicit PYTHON_CMD environment variable
    :
elif [ -f /.dockerenv ]; then
    # Inside Docker container, use system python3 where requirements were installed
    PYTHON_CMD="python3"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

WORKERS=${WORKERS:-4}

echo "Using Python executable: $PYTHON_CMD"
echo "Workers count: $WORKERS"

# 1. Train model
$PYTHON_CMD experiments/training/train_ppo.py --num-agents 1 --workers $WORKERS "$@"

echo "=================================================="
echo "TRAINING COMPLETED - STARTING EVALUATION"
echo "=================================================="

# 2. Evaluate model
$PYTHON_CMD experiments/analysis/evaluate_ppo.py --num-episodes 5

echo "=================================================="
echo "EXPERIMENT CYCLE COMPLETED SUCCESSFULLY"
echo "=================================================="
