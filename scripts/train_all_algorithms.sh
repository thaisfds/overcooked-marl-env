#!/usr/bin/env bash
# ==============================================================================
# MULTI-AGENT COMPARATIVE TRAINING — ALL 5 ALGORITHMS (LOCAL / DOCKER)
# ==============================================================================
# Sequentially trains the 5 available algorithms in Overcooked MARL:
#   1. PPO    (Proximal Policy Optimization — Top-4 Hyperparameter Baseline)
#   2. DQN    (Deep Q-Network with Prioritized Replay Buffer & Dueling Architecture)
#   3. IMPALA (Importance Weighted Actor-Learner Architecture with V-Trace)
#   4. APPO   (Asynchronous Proximal Policy Optimization)
#   5. BC     (Behavioral Cloning / Supervised Imitation Learning)
#
# Default configuration: 4 agents on tutorial.txt
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Select the appropriate Python executable (Docker container or local venv)
if [ -n "$PYTHON_CMD" ]; then
    :
elif [ -f /.dockerenv ]; then
    export PYTHON_CMD="python3"
elif [ -f "venv/bin/python" ]; then
    export PYTHON_CMD="venv/bin/python"
elif command -v python3 &>/dev/null; then
    export PYTHON_CMD="python3"
else
    export PYTHON_CMD="python"
fi

# Verify that required Python dependencies are installed
if ! $PYTHON_CMD -c "import numpy" &>/dev/null; then
    echo "❌ Error: Selected Python ($PYTHON_CMD) is missing required dependencies (e.g., numpy)."
    echo ""
    echo "💡 How to fix:"
    echo "  1. On a GPU server with Docker (recommended):"
    echo "     ALGOS=\"${ALGOS:-dqn impala appo bc}\" ./scripts/start_server.sh"
    echo "     or:"
    echo "     ALGOS=\"${ALGOS:-dqn impala appo bc}\" ./scripts/run_server_docker.sh"
    echo ""
    echo "  2. To run directly on the host without Docker:"
    echo "     python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    echo ""
    exit 1
fi

# 2. CLI Help
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: bash scripts/train_all_algorithms.sh [OPTIONS]"
    echo ""
    echo "Runs all 5 training pipelines (PPO, DQN, IMPALA, APPO, BC) with 4 agents."
    echo ""
    echo "Optional flags forwarded to training scripts:"
    echo "  --episodes <N>       Override total training episodes"
    echo "  --map <map.txt>      Map name (default: tutorial.txt)"
    echo "  --num-agents <N>     Number of agents (default: 4)"
    echo "  --fail-fast          Stop immediately if any algorithm fails"
    echo ""
    echo "Supported environment variables:"
    echo "  ALGOS=\"ppo dqn ...\"  Custom space-separated list of algorithms to run"
    echo "  WORKERS=<N>          Number of parallel rollout workers"
    echo "  CONCURRENT=<N>       Number of concurrent Ray Tune trials"
    echo "  EVAL_EPISODES=<N>    Post-training evaluation episodes (default: 20)"
    exit 0
fi

# 3. Parse global flags
FAIL_FAST=0
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--fail-fast" ]; then
        FAIL_FAST=1
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done

HAS_EPISODES=0
for arg in "${PASSTHROUGH_ARGS[@]}"; do
    if [ "$arg" = "--episodes" ]; then
        HAS_EPISODES=1
    fi
done
if [ "$HAS_EPISODES" -eq 0 ]; then
    PASSTHROUGH_ARGS+=("--episodes" "1000")
fi

# 4. Select algorithms list
DEFAULT_ALGOS=("ppo" "dqn" "impala" "appo" "bc")
if [ -n "$ALGOS" ]; then
    read -r -a SELECTED_ALGOS <<< "$ALGOS"
else
    SELECTED_ALGOS=("${DEFAULT_ALGOS[@]}")
fi

# 5. Auto-tune CPU/RAM concurrency to avoid OOM crashes
TOTAL_RAM_GB=$(awk '/MemTotal/ { printf "%.0f", $2 / 1024 / 1024 }' /proc/meminfo 2>/dev/null || echo 16)
CPU_COUNT=$($PYTHON_CMD -c "import os; print(os.cpu_count() or 4)" 2>/dev/null || echo 4)

if [ "$TOTAL_RAM_GB" -le 16 ]; then
    DEFAULT_CONCURRENT=2
    MAX_W_BY_CPU=$(( (CPU_COUNT / DEFAULT_CONCURRENT - 1) * DEFAULT_CONCURRENT ))
    [ "$MAX_W_BY_CPU" -lt 2 ] && MAX_W_BY_CPU=2
    DEFAULT_WORKERS=$(( MAX_W_BY_CPU < 6 ? MAX_W_BY_CPU : 6 ))
elif [ "$TOTAL_RAM_GB" -le 32 ]; then
    DEFAULT_CONCURRENT=2
    MAX_W_BY_CPU=$(( (CPU_COUNT / DEFAULT_CONCURRENT - 1) * DEFAULT_CONCURRENT ))
    [ "$MAX_W_BY_CPU" -lt 2 ] && MAX_W_BY_CPU=2
    DEFAULT_WORKERS=$(( MAX_W_BY_CPU < 8 ? MAX_W_BY_CPU : 8 ))
else
    DEFAULT_CONCURRENT=2
    MAX_W_BY_CPU=$(( (CPU_COUNT / DEFAULT_CONCURRENT - 1) * DEFAULT_CONCURRENT ))
    [ "$MAX_W_BY_CPU" -lt 4 ] && MAX_W_BY_CPU=4
    DEFAULT_WORKERS=$(( MAX_W_BY_CPU < 10 ? MAX_W_BY_CPU : 10 ))
fi

export WORKERS=${WORKERS:-$DEFAULT_WORKERS}
export CONCURRENT=${CONCURRENT:-$DEFAULT_CONCURRENT}
export EVAL_EPISODES=${EVAL_EPISODES:-20}

TOTAL_ALGOS=${#SELECTED_ALGOS[@]}
START_ALL_TIME=$(date +%s)

echo "======================================================================"
echo "🚀 STARTING MULTI-AGENT BENCHMARK SUITE (5 ALGORITHMS)"
echo "======================================================================"
echo "  Python Executable: $PYTHON_CMD"
echo "  Detected CPUs:     $CPU_COUNT"
echo "  Rollout Workers:   $WORKERS"
echo "  Ray Concurrency:   $CONCURRENT"
echo "  Default Agents:    4"
echo "  Default Map:       tutorial.txt"
echo "  Algorithms:        ${SELECTED_ALGOS[*]} (Total: $TOTAL_ALGOS)"
if [ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]; then
    echo "  Extra Arguments:   ${PASSTHROUGH_ARGS[*]}"
fi
echo "======================================================================"
echo ""

declare -A ALGO_STATUS
declare -A ALGO_DURATION

INDEX=1
for ALGO in "${SELECTED_ALGOS[@]}"; do
    ALGO_UPPER="$(echo "$ALGO" | tr '[:lower:]' '[:upper:]')"
    echo ""
    echo "======================================================================"
    echo "▶ [$INDEX/$TOTAL_ALGOS] RUNNING TRAINING: $ALGO_UPPER"
    echo "======================================================================"
    START_ALGO_TIME=$(date +%s)

    set +e
    bash experiments/training/run_experiment_cycle.sh --algo "$ALGO" "${PASSTHROUGH_ARGS[@]}"
    EXIT_CODE=$?
    set -e

    END_ALGO_TIME=$(date +%s)
    DURATION=$((END_ALGO_TIME - START_ALGO_TIME))
    ALGO_DURATION["$ALGO"]=$DURATION

    if [ $EXIT_CODE -eq 0 ]; then
        ALGO_STATUS["$ALGO"]="✅ SUCCESS"
        echo "🎉 [$INDEX/$TOTAL_ALGOS] $ALGO_UPPER training completed successfully in $((DURATION / 60))m $((DURATION % 60))s."
    else
        ALGO_STATUS["$ALGO"]="❌ FAILED (Code $EXIT_CODE)"
        echo "⚠️ [$INDEX/$TOTAL_ALGOS] $ALGO_UPPER training exited with code $EXIT_CODE."
        if [ "$FAIL_FAST" -eq 1 ]; then
            echo "⛔ --fail-fast flag enabled. Aborting training suite."
            exit $EXIT_CODE
        fi
    fi
    INDEX=$((INDEX + 1))
done

END_ALL_TIME=$(date +%s)
TOTAL_DURATION=$((END_ALL_TIME - START_ALL_TIME))

echo ""
echo "======================================================================"
echo "🏁 FINAL TRAINING SUITE REPORT (5 ALGORITHMS)"
echo "======================================================================"
echo "Total Duration: $((TOTAL_DURATION / 60))m $((TOTAL_DURATION % 60))s"
echo ""
echo "| Algorithm | Status | Duration |"
echo "| :--- | :--- | :--- |"
for ALGO in "${SELECTED_ALGOS[@]}"; do
    ALGO_UPPER="$(echo "$ALGO" | tr '[:lower:]' '[:upper:]')"
    DUR=${ALGO_DURATION["$ALGO"]}
    echo "| **$ALGO_UPPER** | ${ALGO_STATUS["$ALGO"]} | $((DUR / 60))m $((DUR % 60))s |"
done
echo ""

echo "📊 UPDATED MODEL COMPARISON TABLE:"
$PYTHON_CMD -c "from experiments.utils.results_tracker import generate_markdown_summary_table; print(generate_markdown_summary_table())" 2>/dev/null || true

echo ""
echo "======================================================================"
echo "💡 Visualize any trained model in Pygame:"
echo "   $PYTHON_CMD experiments/visualization/visualize_best_model.py"
echo "   $PYTHON_CMD experiments/visualization/visualize_all_models.py"
echo "======================================================================"
