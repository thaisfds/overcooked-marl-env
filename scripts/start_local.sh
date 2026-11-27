#!/usr/bin/env bash
# ==============================================================================
# Direct Local Execution (Without Docker) Using Python Virtualenv
# Trains 4 agents on tutorial.txt with the top-4 champion hyperparameter configs
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Select the appropriate Python executable (Docker container or local venv)
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

# Detect available RAM and CPUs to configure safe concurrency (avoids OOM)
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

echo "=================================================="
echo "Starting Local Training (Without Docker)"
echo "  Python:       $PYTHON_CMD"
echo "  CPUs:         $CPU_COUNT (Workers: $WORKERS, Concurrent: $CONCURRENT)"
echo "  Agents:       4"
echo "  Map:          tutorial.txt"
echo "  Mode:         Top-4 Champion Hyperparameters"
echo "=================================================="

# Run the full training + post-training evaluation cycle
bash experiments/training/run_experiment_cycle.sh "$@"
