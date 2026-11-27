#!/usr/bin/env bash
# ==============================================================================
# Server Deployment & Docker Execution Script for Overcooked MARL
# ==============================================================================
# Usage examples on a GPU server:
#
# 1. Default (4 agents on tutorial.txt):
#    ./scripts/run_server_docker.sh
#
# 2. Run all algorithms (or a custom subset):
#    ALGOS="dqn impala appo bc" ./scripts/run_server_docker.sh
#    ./scripts/run_server_docker.sh --all-algos
#
# 3. Specify map and number of agents:
#    ./scripts/run_server_docker.sh --map tutorial.txt --num-agents 4
#    ./scripts/run_server_docker.sh --map default_map.txt --num-agents 1
#
# 4. Customize training episodes or GPU allocation:
#    ./scripts/run_server_docker.sh --map tutorial.txt --num-agents 4 --episodes 1500
#    ./scripts/run_server_docker.sh --map tutorial.txt --num-agents 4 --gpus 1.0
#
# 5. Adjust concurrency and evaluation via environment variables:
#    WORKERS=8 CONCURRENT=4 ./scripts/run_server_docker.sh
#    EVAL_EPISODES=50 ./scripts/run_server_docker.sh
#
# 6. Run in the background (recommended over SSH):
#    tmux new -s training "./scripts/run_server_docker.sh"
#    # Or via nohup:
#    nohup ./scripts/run_server_docker.sh > training.log 2>&1 &
# ==============================================================================
set -e

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "Usage: ./scripts/run_server_docker.sh [OPTIONS]"
  echo ""
  echo "Examples:"
  echo "  ./scripts/run_server_docker.sh                                            # Default: 4 agents on tutorial.txt"
  echo "  ./scripts/run_server_docker.sh --map tutorial.txt --num-agents 2          # 2 agents on tutorial.txt"
  echo "  ./scripts/run_server_docker.sh --map default_map.txt --num-agents 1       # 1 agent on default_map.txt"
  echo "  ./scripts/run_server_docker.sh --episodes 3000                            # 3000 training episodes"
  echo "  WORKERS=8 CONCURRENT=4 ./scripts/run_server_docker.sh                     # Custom parallelism via env"
  echo "  EVAL_EPISODES=50 ./scripts/run_server_docker.sh                           # 50 evaluation episodes"
  echo ""
  echo "Server tip (background execution):"
  echo "  tmux new -s training './scripts/run_server_docker.sh'"
  echo "  nohup ./scripts/run_server_docker.sh > training.log 2>&1 &"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Remove any stale container with the same name
docker rm -f overcooked_container 2>/dev/null || true

# Detect TTY so this works both interactively and inside tmux/non-interactive scripts
DOCKER_TTY="-i"
if [ -t 0 ]; then
  DOCKER_TTY="-it"
fi

echo "=================================================="
echo "1. Pulling latest code changes from git..."
echo "=================================================="
git fetch
git pull

echo "=================================================="
echo "2. Building Docker image (overcooked_marl:latest)..."
echo "=================================================="

# Wait for the Docker daemon to respond (useful for Rootless Docker)
if ! docker info >/dev/null 2>&1; then
  echo "Waiting for Docker daemon socket to initialize..."
  for i in {1..30}; do
    if docker info >/dev/null 2>&1; then
      echo "✅ Docker daemon connected successfully!"
      break
    fi
    sleep 1
  done
fi

if ! docker info >/dev/null 2>&1; then
  echo "❌ Error: Could not connect to the Docker daemon after 30 seconds."
  echo "   Tip: run './scripts/start_server.sh' to auto-recover and start the daemon."
  exit 1
fi

if ! docker build -t overcooked_marl:latest .; then
  echo "⚠️ Cached Docker build failed. Pruning builder cache and rebuilding..."
  docker builder prune -a -f || true
  docker build --no-cache -t overcooked_marl:latest .
fi

# Detect whether the user wants the full multi-algorithm suite or a single algorithm
RUN_ALL_ALGOS=0
FILTERED_ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--all-algos" ]; then
    RUN_ALL_ALGOS=1
  else
    FILTERED_ARGS+=("$arg")
  fi
done

if [ -n "$ALGOS" ] || [ "$RUN_ALL_ALGOS" -eq 1 ]; then
  ENTRY_SCRIPT="scripts/train_all_algorithms.sh"
else
  ENTRY_SCRIPT="experiments/training/run_experiment_cycle.sh"
fi

ARGS=("${FILTERED_ARGS[@]}")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(--map tutorial.txt --num-agents 4)
fi

# Detect NVIDIA GPU support in host Docker
GPU_FLAGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  if docker run --rm --gpus all overcooked_marl:latest true 2>/dev/null; then
    GPU_FLAGS=(--gpus all)
    echo "🎮 NVIDIA GPU detected and enabled in Docker (--gpus all)."
  else
    echo "⚠️ nvidia-smi found, but NVIDIA Container Toolkit is not configured for Docker. Falling back to CPU."
  fi
else
  echo "ℹ️ No NVIDIA GPU detected on host. Running container in CPU mode."
fi

TOTAL_RAM_GB=$(awk '/MemTotal/ { printf "%.0f", $2 / 1024 / 1024 }' /proc/meminfo 2>/dev/null || echo 16)
SHM_SIZE="16g"
if [ "$TOTAL_RAM_GB" -le 16 ]; then
  SHM_SIZE="4g"
elif [ "$TOTAL_RAM_GB" -le 32 ]; then
  SHM_SIZE="8g"
fi

echo "=================================================="
echo "3. Running Docker Container..."
echo "   Script:    $ENTRY_SCRIPT"
echo "   Arguments: ${ARGS[*]}"
if [ -n "$ALGOS" ]; then
  echo "   ALGOS:     $ALGOS"
fi
echo "   SHM Size:  $SHM_SIZE"
echo "=================================================="
docker run --rm $DOCKER_TTY \
  "${GPU_FLAGS[@]}" \
  --shm-size="$SHM_SIZE" \
  -e WORKERS \
  -e CONCURRENT \
  -e EVAL_EPISODES \
  -e ALGO \
  -e ALGOS \
  -e FAIL_FAST \
  --name overcooked_container \
  -v "$PWD":/app \
  overcooked_marl:latest \
  bash "$ENTRY_SCRIPT" "${ARGS[@]}"
