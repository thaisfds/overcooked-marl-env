#!/usr/bin/env bash
# ==============================================================================
# VeRLab Server Deployment & Execution Script for Overcooked MARL
# ==============================================================================
# Exemplos de uso no servidor:
#
# 1. Padrão (4 agentes no mapa tutorial.txt):
#    ./run_server_docker.sh
#
# 2. Rodar bateria de todos os algoritmos (ou subconjunto):
#    ALGOS="dqn impala appo bc" ./run_server_docker.sh
#    ./run_server_docker.sh --all-algos
#
# 3. Especificar mapa e quantidade de agentes:
#    ./run_server_docker.sh --map tutorial.txt --num-agents 4
#    ./run_server_docker.sh --map default_map.txt --num-agents 1
#
# 4. Customizar iterações de treinamento ou GPU:
#    ./run_server_docker.sh --map tutorial.txt --num-agents 4 --episodes 1500
#    ./run_server_docker.sh --map tutorial.txt --num-agents 4 --gpus 1.0
#
# 5. Ajustar recursos de concorrência e avaliação via variáveis de ambiente:
#    WORKERS=8 CONCURRENT=4 ./run_server_docker.sh
#    EVAL_EPISODES=50 ./run_server_docker.sh
#
# 6. Executar em background no servidor (recomendado para não travar a conexão SSH):
#    tmux new -s treino "./run_server_docker.sh"
#    # Ou via nohup:
#    nohup ./run_server_docker.sh > treino.log 2>&1 &
# ==============================================================================
set -e

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "Uso: ./run_server_docker.sh [OPÇÕES]"
  echo ""
  echo "Exemplos:"
  echo "  ./run_server_docker.sh                                            # Padrão: 2 agentes no tutorial.txt"
  echo "  ./run_server_docker.sh --map tutorial.txt --num-agents 2          # 2 agentes no tutorial"
  echo "  ./run_server_docker.sh --map default_map.txt --num-agents 1       # 1 agente no default_map"
  echo "  ./run_server_docker.sh --training-iterations 150                  # 150 iterações"
  echo "  WORKERS=8 CONCURRENT=4 ./run_server_docker.sh                     # Paralelismo customizado via env"
  echo "  EVAL_EPISODES=50 ./run_server_docker.sh                           # 50 episódios de avaliação"
  echo ""
  echo "Dica para servidor (background):"
  echo "  tmux new -s treino './run_server_docker.sh'"
  echo "  nohup ./run_server_docker.sh > treino.log 2>&1 &"
  exit 0
fi

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

# Aguardar até o docker daemon responder (útil para Docker Rootless)
if ! docker info >/dev/null 2>&1; then
  echo "Aguardando Docker daemon inicializar o socket..."
  for i in {1..20}; do
    if docker info >/dev/null 2>&1; then
      echo "Docker daemon conectado com sucesso!"
      break
    fi
    sleep 1
  done
fi

if ! docker build -t overcooked_marl:latest .; then
  echo "⚠️ Falha no build usando cache. Limpando cache corrompido do builder e reconstruindo..."
  docker builder prune -a -f || true
  docker build --no-cache -t overcooked_marl:latest .
fi

# Detectar se o usuário quer rodar a bateria de algoritmos ou um algoritmo específico
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
  ENTRY_SCRIPT="treinar_todos_algoritmos.sh"
else
  ENTRY_SCRIPT="experiments/training/run_experiment_cycle.sh"
fi

ARGS=("${FILTERED_ARGS[@]}")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(--map tutorial.txt --num-agents 4)
fi

# Detectar suporte a GPU no Docker da máquina host
GPU_FLAGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  if docker run --rm --gpus all overcooked_marl:latest true 2>/dev/null; then
    GPU_FLAGS=(--gpus all)
    echo "🎮 GPU NVIDIA detectada e habilitada no Docker (--gpus all)."
  else
    echo "⚠️ nvidia-smi encontrado, mas o Docker não tem NVIDIA Container Toolkit configurado. Executando em CPU."
  fi
else
  echo "ℹ️ GPU NVIDIA não detectada na máquina host. Executando container em modo CPU."
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
