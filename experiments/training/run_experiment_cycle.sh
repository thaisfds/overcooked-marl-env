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
CONCURRENT=${CONCURRENT:-4}
EVAL_EPISODES=${EVAL_EPISODES:-20}

echo "Using Python executable: $PYTHON_CMD"
echo "Workers count: $WORKERS"
echo "Concurrent trials: $CONCURRENT"
echo "Evaluation episodes: $EVAL_EPISODES"
if [ $# -gt 0 ]; then
    echo "Custom arguments: $@"
fi

# 1. Train model (passes through all custom CLI arguments e.g. --map, --num-agents, --episodes, etc.)
$PYTHON_CMD experiments/training/train_ppo.py --workers $WORKERS --concurrent $CONCURRENT "$@"

echo "=================================================="
echo "TRAINING COMPLETED - STARTING EVALUATION OF BEST MODEL"
echo "=================================================="

# 2. Evaluate best model found in the training session (auto-detects map, agents, and best checkpoint)
$PYTHON_CMD experiments/analysis/evaluate_ppo.py --num-episodes $EVAL_EPISODES

echo "=================================================="
echo "EXPERIMENT CYCLE COMPLETED SUCCESSFULLY"
echo "=================================================="

# 3. Plot learning curves (saved to experiments/visualization/learning_curves.png)
$PYTHON_CMD experiments/visualization/plot_curves.py 2>/dev/null || true

# 4. Print persistent results summary table
$PYTHON_CMD -c "from experiments.utils.results_tracker import generate_markdown_summary_table; print('\n' + generate_markdown_summary_table())"

echo "=================================================="
echo "To visualize the rollout in Pygame, run:"
echo "  $PYTHON_CMD experiments/visualization/visualize_model.py"
echo "  $PYTHON_CMD experiments/visualization/visualize_all_models.py"
echo "=================================================="
