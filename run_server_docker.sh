#!/usr/bin/env bash
# ==============================================================================
# VeRLab Server Deployment & Execution Script for Overcooked MARL
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

# Remover container antigo de mesmo nome se estiver rodando/preso
docker rm -f overcooked_container 2>/dev/null || true

# Detectar TTY para funcionar tanto em terminal normal quanto em tmux/scripts
DOCKER_TTY="-i"
if [ -t 0 ]; then
  DOCKER_TTY="-it"
fi

echo "=================================================="
echo "1. Pulling latest code changes from git ..."
echo "=================================================="
git fetch
git pull

echo "=================================================="
echo "2. Building Docker image (overcooked_marl:latest)..."
echo "=================================================="
docker build -t overcooked_marl:latest .

# Se nenhum argumento for passado, usar por padrão 2 agentes no tutorial.txt
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(--map tutorial.txt --num-agents 2)
fi

echo "=================================================="
echo "3. Running Docker Container with GPU access..."
echo "   Arguments: ${ARGS[*]}"
echo "=================================================="
docker run --rm $DOCKER_TTY \
  --gpus all \
  --shm-size=8g \
  -e WORKERS \
  -e CONCURRENT \
  -e EVAL_EPISODES \
  --name overcooked_container \
  -v "$PWD":/app \
  overcooked_marl:latest \
  bash experiments/training/run_experiment_cycle.sh "${ARGS[@]}"
