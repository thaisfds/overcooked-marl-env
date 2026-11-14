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

echo "=================================================="
echo "Iniciando sessão tmux: $SESSION_NAME"
echo "=================================================="

# 1. Cria a sessão com terminal interativo padrão (carrega ~/.bashrc e todos os aliases)
tmux new-session -d -s "$SESSION_NAME" -c "$PWD"

# 2. Envia o comando para rodar o daemon do docker no painel esquerdo (.0) se necessário
if command -v docker-rootless-daemon-run >/dev/null 2>&1; then
  tmux send-keys -t "$SESSION_NAME.0" 'docker-rootless-daemon-run' C-m
else
  tmux send-keys -t "$SESSION_NAME.0" 'echo "ℹ️ Docker daemon já gerenciado pelo host (rootless daemon não necessário aqui)."' C-m
fi

# 3. Divide a tela verticalmente (-h: lado a lado)
tmux split-window -h -t "$SESSION_NAME" -c "$PWD"

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

# Envia o comando no painel direito (.1) aguardando 6s para o daemon subir
tmux send-keys -t "$SESSION_NAME.1" "sleep 6 && ${ENV_PREFIX}bash run_server_docker.sh $TRAIN_ARGS; echo ''; echo '🏁 Processo finalizado com status: '\$?; echo 'Pressione Enter para fechar este painel.'; read" C-m

# 5. Anexa o usuário diretamente à sessão para acompanhar
tmux attach-session -t "$SESSION_NAME"
