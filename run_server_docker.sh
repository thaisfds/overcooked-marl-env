#!/usr/bin/env bash
# ==============================================================================
# VeRLab Server Deployment & Execution Script for Overcooked MARL
# ==============================================================================
# Exemplos de uso no servidor:
#
# 1. Padrão (2 agentes no mapa tutorial.txt):
#    ./run_server_docker.sh
#
# 2. Especificar mapa e quantidade de agentes:
#    ./run_server_docker.sh --map tutorial.txt --num-agents 2
#    ./run_server_docker.sh --map default_map.txt --num-agents 1
#
# 3. Customizar iterações de treinamento ou GPU:
#    ./run_server_docker.sh --map tutorial.txt --num-agents 2 --training-iterations 150
#    ./run_server_docker.sh --map tutorial.txt --num-agents 2 --gpus 1.0
#
# 4. Ajustar recursos de concorrência e avaliação via variáveis de ambiente:
#    WORKERS=8 CONCURRENT=4 ./run_server_docker.sh
#    EVAL_EPISODES=50 ./run_server_docker.sh
#
# 5. Executar em background no servidor (recomendado para não travar a conexão SSH):
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

docker build -t overcooked_marl:latest .

# Se nenhum argumento for passado, usar por padrão 2 agentes no tutorial.txt com fine-tuning warm-start
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(--map tutorial.txt --num-agents 2 --restore-checkpoint auto)
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
