#!/usr/bin/env bash
# ==============================================================================
# TREINAMENTO COMPARATIVO MULTIAGENTE - TODOS OS 5 ALGORITMOS (LOCAL / PYTHON)
# ==============================================================================
# Executa sequencialmente o treinamento dos 5 algoritmos disponíveis no Overcooked MARL:
#   1. PPO    (Proximal Policy Optimization - Baseline Campeão com TOP-4 Hiperparâmetros)
#   2. DQN    (Deep Q-Network com Prioritized Replay Buffer & Dueling Architecture)
#   3. IMPALA (Importance Weighted Actor-Learner Architecture com V-trace)
#   4. APPO   (Asynchronous Proximal Policy Optimization)
#   5. BC     (Behavioral Cloning / Imitation Learning supervisionado)
#
# Configurações padrão: 4 agentes no mapa tutorial.txt
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Garante o uso do Python apropriado (container Docker ou venv local)
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

# 2. Ajuda CLI
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Uso: bash treinar_todos_algoritmos.sh [OPÇÕES]"
    echo ""
    echo "Executa os 5 treinamentos (PPO, DQN, IMPALA, APPO, BC) com 4 agentes localmente."
    echo ""
    echo "Opções opcionais repassadas aos treinamentos:"
    echo "  --episodes <N>       Sobrescreve o total de episódios de treino"
    echo "  --map <mapa.txt>     Nome do mapa (padrão: tutorial.txt)"
    echo "  --num-agents <N>     Quantidade de agentes (padrão: 4)"
    echo "  --fail-fast          Interrompe a bateria de testes se algum algoritmo falhar"
    echo ""
    echo "Variáveis de ambiente suportadas:"
    echo "  ALGOS=\"ppo dqn ...\"  Lista customizada de algoritmos a executar"
    echo "  WORKERS=<N>          Quantidade de workers paralelos de rollout"
    echo "  CONCURRENT=<N>       Quantidade de trials concorrentes no Ray Tune"
    echo "  EVAL_EPISODES=<N>    Episódios de avaliação pós-treino (padrão: 20)"
    exit 0
fi

# 3. Tratamento de flags globais
FAIL_FAST=0
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--fail-fast" ]; then
        FAIL_FAST=1
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done

HAS_EPISODES=0
for arg in "${PASSTHROUGH_ARGS[@]}"; do
    if [ "$arg" = "--episodes" ]; then
        HAS_EPISODES=1
    fi
done
if [ "$HAS_EPISODES" -eq 0 ]; then
    PASSTHROUGH_ARGS+=("--episodes" "1500")
fi

# 4. Definição da lista de algoritmos
DEFAULT_ALGOS=("ppo" "dqn" "impala" "appo" "bc")
if [ -n "$ALGOS" ]; then
    read -r -a SELECTED_ALGOS <<< "$ALGOS"
else
    SELECTED_ALGOS=("${DEFAULT_ALGOS[@]}")
fi

# 5. Otimização de recursos de CPU da máquina local
CPU_COUNT=$($PYTHON_CMD -c "import os; print(os.cpu_count() or 4)" 2>/dev/null || echo 4)
DEFAULT_WORKERS=$(awk -v c="$CPU_COUNT" 'BEGIN { w = int(c * 0.7); if (w < 4) w = 4; if (w > 16) w = 16; print w }')

export WORKERS=${WORKERS:-$DEFAULT_WORKERS}
export CONCURRENT=${CONCURRENT:-4}
export EVAL_EPISODES=${EVAL_EPISODES:-20}

TOTAL_ALGOS=${#SELECTED_ALGOS[@]}
START_ALL_TIME=$(date +%s)

echo "======================================================================"
echo "🚀 INICIANDO BATERIA DE TREINAMENTO DOS 5 ALGORITMOS MULTIAGENTE"
echo "======================================================================"
echo "  Python Executável: $PYTHON_CMD"
echo "  CPUs Detectadas:   $CPU_COUNT"
echo "  Workers Rollout:   $WORKERS"
echo "  Concorrência Ray:  $CONCURRENT"
echo "  Agentes Padrão:    4"
echo "  Mapa Padrão:       tutorial.txt"
echo "  Algoritmos:        ${SELECTED_ALGOS[*]} (Total: $TOTAL_ALGOS)"
if [ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]; then
    echo "  Argumentos Extras: ${PASSTHROUGH_ARGS[*]}"
fi
echo "======================================================================"
echo ""

declare -A ALGO_STATUS
declare -A ALGO_DURATION

INDEX=1
for ALGO in "${SELECTED_ALGOS[@]}"; do
    ALGO_UPPER="$(echo "$ALGO" | tr '[:lower:]' '[:upper:]')"
    echo ""
    echo "======================================================================"
    echo "▶ [$INDEX/$TOTAL_ALGOS] EXECUTANDO TREINAMENTO: $ALGO_UPPER"
    echo "======================================================================"
    START_ALGO_TIME=$(date +%s)

    set +e
    bash experiments/training/run_experiment_cycle.sh --algo "$ALGO" "${PASSTHROUGH_ARGS[@]}"
    EXIT_CODE=$?
    set -e

    END_ALGO_TIME=$(date +%s)
    DURATION=$((END_ALGO_TIME - START_ALGO_TIME))
    ALGO_DURATION["$ALGO"]=$DURATION

    if [ $EXIT_CODE -eq 0 ]; then
        ALGO_STATUS["$ALGO"]="✅ SUCESSO"
        echo "🎉 [$INDEX/$TOTAL_ALGOS] Treinamento do $ALGO_UPPER concluído com sucesso em $((DURATION / 60))m $((DURATION % 60))s."
    else
        ALGO_STATUS["$ALGO"]="❌ FALHA (Código $EXIT_CODE)"
        echo "⚠️ [$INDEX/$TOTAL_ALGOS] Treinamento do $ALGO_UPPER finalizou com código $EXIT_CODE."
        if [ "$FAIL_FAST" -eq 1 ]; then
            echo "⛔ Flag --fail-fast ativa. Abortando bateria de treinamentos."
            exit $EXIT_CODE
        fi
    fi
    INDEX=$((INDEX + 1))
done

END_ALL_TIME=$(date +%s)
TOTAL_DURATION=$((END_ALL_TIME - START_ALL_TIME))

echo ""
echo "======================================================================"
echo "🏁 RELATÓRIO FINAL DA BATERIA DE TREINAMENTOS (5 ALGORITMOS)"
echo "======================================================================"
echo "Duração Total: $((TOTAL_DURATION / 60))m $((TOTAL_DURATION % 60))s"
echo ""
echo "| Algoritmo | Status | Duração |"
echo "| :--- | :--- | :--- |"
for ALGO in "${SELECTED_ALGOS[@]}"; do
    ALGO_UPPER="$(echo "$ALGO" | tr '[:lower:]' '[:upper:]')"
    DUR=${ALGO_DURATION["$ALGO"]}
    echo "| **$ALGO_UPPER** | ${ALGO_STATUS["$ALGO"]} | $((DUR / 60))m $((DUR % 60))s |"
done
echo ""

echo "📊 TABELA COMPARATIVA ATUALIZADA DOS MODELOS:"
$PYTHON_CMD -c "from experiments.utils.results_tracker import generate_markdown_summary_table; print(generate_markdown_summary_table())" 2>/dev/null || true

echo ""
echo "======================================================================"
echo "💡 Visualização no Pygame de qualquer modelo treinado:"
echo "   $PYTHON_CMD experiments/visualization/visualize_best_model.py"
echo "   $PYTHON_CMD experiments/visualization/visualize_all_models.py"
echo "======================================================================"
