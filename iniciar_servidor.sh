#!/usr/bin/env bash
# ==============================================================================
# Script de automação tmux para o servidor:
# 1. Inicia o daemon do Docker (rootless) em um painel
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

# 1. Cria a sessão em background rodando o daemon do docker no painel esquerdo
tmux new-session -d -s "$SESSION_NAME" -c "$PWD" 'docker-rootless-daemon-run; bash'

# 2. Divide verticalmente (-h: lado a lado) e aguarda 3s antes de rodar o treino
tmux split-window -h -t "$SESSION_NAME" -c "$PWD" "sleep 3 && bash run_server_docker.sh \"\$@\"; bash"

# 3. Anexa o usuário diretamente à sessão
tmux attach-session -t "$SESSION_NAME"
