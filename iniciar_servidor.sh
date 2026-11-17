#!/usr/bin/env bash
# ==============================================================================
# Script de automação tmux para o servidor:
# 1. Inicia o daemon do Docker (rootless) no painel esquerdo via send-keys (para carregar aliases)
# 2. Divide a tela verticalmente (lado a lado)
# 3. Executa o treinamento (run_server_docker.sh) com os argumentos passados
# 4. Conecta à sessão tmux
# ==============================================================================

SESSION_NAME="treino"

# Mata sessão anterior com mesmo nome caso exista
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# 0. Auto-recuperação do Docker: se o daemon não responder, limpa travas e zumbis automaticamente
if ! docker info &>/dev/null; then
  echo "🧹 Docker não respondendo. Limpando processos antigos e travas órfãs..."
  pkill -9 -u "$USER" -f "dockerd|rootlesskit|containerd|docker-proxy" 2>/dev/null || true
  USER_DIR="${USER_DOCKER_DIR:-/draft-hdd/$USER/docker-rootless}"
  rm -f "$USER_DIR/.docker/run/docker.pid" "$USER_DIR/.docker/run/docker.sock" 2>/dev/null || true
  rm -f "/draft-hdd/$USER/docker-rootless/.docker/run/docker.pid" "/draft-hdd/$USER/docker-rootless/.docker/run/docker.sock" 2>/dev/null || true
fi

echo "=================================================="
echo "Iniciando sessão tmux: $SESSION_NAME"
echo "=================================================="

# 1. Cria a sessão com terminal interativo padrão (carrega ~/.bashrc e todos os aliases)
tmux new-session -d -s "$SESSION_NAME" -c "$PWD"

# 2. Inicia o daemon do docker no painel esquerdo (.0) de forma resiliente
DAEMON_CMD='if docker info &>/dev/null; then echo "✅ Docker daemon já está ativo e funcionando."; elif type docker-rootless-daemon-run &>/dev/null; then pkill -9 -u "$USER" -f "dockerd|rootlesskit|containerd|docker-proxy" 2>/dev/null || true; USER_DIR="${USER_DOCKER_DIR:-/draft-hdd/$USER/docker-rootless}"; rm -f "$USER_DIR/.docker/run/docker.pid" "$USER_DIR/.docker/run/docker.sock" 2>/dev/null || true; echo "🚀 Iniciando Docker Rootless Daemon..."; docker-rootless-daemon-run; else echo "ℹ️ Docker daemon do sistema pronto."; fi'
tmux send-keys -t "$SESSION_NAME.0" "$DAEMON_CMD" C-m

# 3. Divide a tela verticalmente (-h: lado a lado)
tmux split-window -h -t "$SESSION_NAME" -c "$PWD"

# Pausa para o shell carregar ~/.bashrc completamente antes de receber comandos
sleep 2

# 4. Monta os argumentos e repassa variáveis de ambiente para o treinamento
if [ $# -eq 0 ]; then
  TRAIN_ARGS="--map tutorial.txt --num-agents 4"
else
  TRAIN_ARGS="$*"
fi

ENV_PREFIX=""
[ -n "$WORKERS" ] && ENV_PREFIX="WORKERS=$WORKERS $ENV_PREFIX"
[ -n "$CONCURRENT" ] && ENV_PREFIX="CONCURRENT=$CONCURRENT $ENV_PREFIX"
[ -n "$ALGO" ] && ENV_PREFIX="ALGO=$ALGO $ENV_PREFIX"
[ -n "$ALGOS" ] && ENV_PREFIX="ALGOS=\"$ALGOS\" $ENV_PREFIX"
[ -n "$FAIL_FAST" ] && ENV_PREFIX="FAIL_FAST=$FAIL_FAST $ENV_PREFIX"
[ -n "$EVAL_EPISODES" ] && ENV_PREFIX="EVAL_EPISODES=$EVAL_EPISODES $ENV_PREFIX"

# Envia o comando no painel direito (.1) aguardando dinamicamente o Docker responder antes de rodar
TRAIN_CMD="echo '⏳ Aguardando Docker daemon responder...'; for i in {1..30}; do if docker info &>/dev/null; then echo '✅ Docker conectado com sucesso!'; break; fi; sleep 1; done; ${ENV_PREFIX}bash run_server_docker.sh $TRAIN_ARGS; echo ''; echo '🏁 Processo finalizado com status: '\$?; echo 'Pressione Enter para fechar este painel.'; read"
tmux send-keys -t "$SESSION_NAME.1" "$TRAIN_CMD" C-m

# 5. Anexa o usuário diretamente à sessão para acompanhar
tmux attach-session -t "$SESSION_NAME"
