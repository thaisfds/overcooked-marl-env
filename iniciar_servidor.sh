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

# 2. Envia o comando para rodar o daemon do docker no painel esquerdo (.0)
tmux send-keys -t "$SESSION_NAME.0" 'docker-rootless-daemon-run' C-m

# 3. Divide a tela verticalmente (-h: lado a lado)
tmux split-window -h -t "$SESSION_NAME" -c "$PWD"

# 4. Monta os argumentos para o treinamento
if [ $# -eq 0 ]; then
  TRAIN_ARGS="--map tutorial.txt --num-agents 2"
else
  TRAIN_ARGS="$*"
fi

# Envia o comando no painel direito (.1) aguardando 6s para o daemon subir
tmux send-keys -t "$SESSION_NAME.1" "sleep 6 && bash run_server_docker.sh $TRAIN_ARGS" C-m

# 5. Anexa o usuário diretamente à sessão para acompanhar
tmux attach-session -t "$SESSION_NAME"
