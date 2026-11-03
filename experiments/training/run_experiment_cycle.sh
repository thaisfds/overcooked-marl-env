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
SKIP_EVAL=${SKIP_EVAL:-0}
GENERATE_CHARTS=${GENERATE_CHARTS:-false}
ALGO=${ALGO:-ppo}

# Parse --algo from CLI arguments if provided
TRAIN_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --algo)
            ALGO="$2"
            shift 2
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

ALGO_LOWER="$(echo "$ALGO" | tr '[:upper:]' '[:lower:]')"
case "$ALGO_LOWER" in
    dqn)
        TRAIN_SCRIPT="experiments/training/train_dqn.py"
        ;;
    impala)
        TRAIN_SCRIPT="experiments/training/train_impala.py"
        ;;
    appo)
        TRAIN_SCRIPT="experiments/training/train_appo.py"
        ;;
    bc)
        TRAIN_SCRIPT="experiments/training/train_bc.py"
        ;;
    ppo|*)
        TRAIN_SCRIPT="experiments/training/train_ppo.py"
        ;;
esac

echo "Algorithm selected: $ALGO (using $TRAIN_SCRIPT)"
echo "Using Python executable: $PYTHON_CMD"
echo "Workers count: $WORKERS"
echo "Concurrent trials: $CONCURRENT"
if [ "$SKIP_EVAL" = "1" ] || [ "$EVAL_EPISODES" = "0" ] || [ "$ALGO_LOWER" = "bc" ]; then
    echo "Evaluation: SKIPPED (SKIP_EVAL=$SKIP_EVAL, EVAL_EPISODES=$EVAL_EPISODES, ALGO=$ALGO)"
else
    echo "Evaluation episodes: $EVAL_EPISODES"
fi
if [ ${#TRAIN_ARGS[@]} -gt 0 ]; then
    echo "Custom arguments: ${TRAIN_ARGS[*]}"
fi

# 1. Train model (passes through all custom CLI arguments e.g. --map, --num-agents, --episodes, etc.)
if [ "$ALGO_LOWER" = "bc" ]; then
    $PYTHON_CMD $TRAIN_SCRIPT "${TRAIN_ARGS[@]}"
else
    $PYTHON_CMD $TRAIN_SCRIPT --workers $WORKERS --concurrent $CONCURRENT "${TRAIN_ARGS[@]}"
fi

echo "=================================================="
echo "TRAINING COMPLETED"
echo "=================================================="

# 2. Evaluate best model found in the training session (auto-detects map, agents, and best checkpoint)
if [ "$SKIP_EVAL" != "1" ] && [ "$EVAL_EPISODES" != "0" ]; then
    echo "STARTING EVALUATION OF BEST MODEL ($EVAL_EPISODES episodes)..."
    EVAL_FLAGS=("--num-episodes" "$EVAL_EPISODES")
    if [ "$LOG_VISION" = "1" ]; then
        EVAL_FLAGS+=("--log-vision")
    fi
    $PYTHON_CMD experiments/analysis/evaluate_ppo.py "${EVAL_FLAGS[@]}" || echo "[WARNING] Evaluation completed with warnings."
else
    echo "Skipping post-training evaluation as requested."
fi

# 3. Optional: Only generate scientific report figures if explicitly requested via GENERATE_CHARTS=true
if [ "$GENERATE_CHARTS" = "true" ] || [ "$GENERATE_CHARTS" = "1" ]; then
    echo "Generating scientific report figures..."
    $PYTHON_CMD experiments/analysis/generate_report_charts.py 2>/dev/null || true
fi

# 4. Print persistent results summary table
$PYTHON_CMD -c "from experiments.utils.results_tracker import generate_markdown_summary_table; print('\n' + generate_markdown_summary_table())" 2>/dev/null || true

# 5. Extract evaluated best checkpoint for direct copy-paste Pygame visualization command
BEST_CP=$($PYTHON_CMD -c "import json, pathlib; p = pathlib.Path('experiments/logs/evaluation_results.json'); print(json.loads(p.read_text()).get('checkpoint_path', '') if p.exists() else '')" 2>/dev/null || true)

echo "=================================================="
echo "EXPERIMENT CYCLE COMPLETED SUCCESSFULLY"
echo "=================================================="
if [ -n "$BEST_CP" ]; then
    echo "Para reproduzir visualmente o melhor modelo no Pygame:"
    echo "  $PYTHON_CMD experiments/visualization/visualize_model.py $BEST_CP"
else
    echo "Para reproduzir visualmente no Pygame:"
    echo "  $PYTHON_CMD experiments/visualization/visualize_model.py"
fi
echo "  $PYTHON_CMD experiments/visualization/visualize_all_models.py"
echo "=================================================="
