#!/usr/bin/env bash
# ==============================================================================
# Execução local direta (sem Docker) utilizando o virtualenv do Python
# Treina 4 agentes no mapa tutorial.txt com as 4 melhores combinações de hiperparâmetros
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Garante uso do Python correto (container Docker ou venv local)
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

# Detecta memória e CPUs disponíveis para definir concorrência segura (evita OOM)
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
echo "Iniciando Treinamento Local (Sem Docker)"
echo "  Python:       $PYTHON_CMD"
echo "  CPUs:         $CPU_COUNT (Workers: $WORKERS, Concorrentes: $CONCURRENT)"
echo "  Agentes:      4"
echo "  Mapa:         tutorial.txt"
echo "  Modo:         TOP 4 Hiperparâmetros Campeões"
echo "=================================================="

# Executa o ciclo completo de treinamento + avaliação pós-treino
bash experiments/training/run_experiment_cycle.sh "$@"
