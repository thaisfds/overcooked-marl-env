"""
Simple Agent Test for Overcooked MARL Environment

Este teste simples cria alguns comportamentos previsíveis nos agentes 
e visualiza cada passo para melhor compreensão do comportamento.
"""

import time
import numpy as np
from overcooked_marl_env.env import OvercookedMARL

# Mapeia as ações para strings legíveis
ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT", 
    3: "RIGHT",
    4: "INTERACT"
}

def visualize_state(env, step_num, actions=None, rewards=None):
    """
    Visualiza o estado atual do ambiente com informações adicionais.
    """
    print("\n" + "=" * 40)
    print(f"PASSO {step_num}")
    print("=" * 40)
    
    # Mostra ações tomadas
    if actions:
        print("Ações:")
        for agent, action in actions.items():
            action_name = ACTION_NAMES.get(action, f"DESCONHECIDO({action})")
            print(f"  {agent}: {action_name}")
    
    # Mostra recompensas obtidas
    if rewards:
        print("Recompensas:")
        for agent, reward in rewards.items():
            print(f"  {agent}: {reward}")
    
    print("\nEstado Atual:")
    env.render()  # Renderiza o estado atual do jogo
    print("-" * 40)

def test_simple_movement():
    """
    Executa um teste de movimento e interação para os agentes seguindo um roteiro específico:
    1. Ambos fazem um movimento circular
    2. Tentam se mover para espaços ocupados
    3. Cada agente prepara um prato de salada completo
    """
    print("\n=== TESTE COMPLETO DE MOVIMENTO E INTERAÇÃO DOS AGENTES ===\n")
    
    # Cria o ambiente com renderização para humanos
    env = OvercookedMARL(render_mode="human", verbose=True)
    
    # Reseta o ambiente
    observations, info = env.reset(seed=42)
    
    # Visualiza o estado inicial
    print("\nEstado Inicial:")
    visualize_state(env, 0)
    time.sleep(1)  # Pausa para visualização
    
    print("\n=== PARTE 1: MOVIMENTOS CIRCULARES ===")
    print("Agente 0: esquerda, baixo, direita, cima (sentido horário)")
    print("Agente 1: direita, cima, esquerda, baixo (sentido anti-horário)")
    
    # Ações para movimentos circulares
    # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: INTERACT
    circular_movements = [
        # agent_0: esquerda, baixo, direita, cima
        # agent_1: direita, cima, esquerda, baixo
        {"agent_0": 2, "agent_1": 3},  # Agente 0 para esquerda, Agente 1 para direita
        {"agent_0": 1, "agent_1": 0},  # Agente 0 para baixo, Agente 1 para cima
        {"agent_0": 3, "agent_1": 2},  # Agente 0 para direita, Agente 1 para esquerda
        {"agent_0": 0, "agent_1": 1},  # Agente 0 para cima, Agente 1 para baixo
    ]
    
    # Total de recompensa acumulada
    total_rewards = {agent: 0 for agent in env.agents}
    
    # Executa os movimentos circulares
    for step, actions in enumerate(circular_movements):
        # Executa um passo
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Acumula recompensas
        for agent, reward in rewards.items():
            total_rewards[agent] += reward
        
        # Visualiza o estado após a ação
        visualize_state(env, step + 1, actions, rewards)
        print(f"Recompensa Total: {total_rewards}")
        time.sleep(0.8)  # Pausa para visualização
    
    print("\n=== PARTE 2: TENTATIVA DE COLISÃO ENTRE AGENTES ===")
    print("Os agentes tentarão se mover para o mesmo espaço ou para o espaço um do outro")
    
    # Reseta o ambiente para a parte 2
    observations, info = env.reset(seed=42)
    visualize_state(env, 0)
    time.sleep(1)
    
    # Sequência para gerar colisões entre agentes
    collision_sequence = [
        # Movem-se para ficar próximos
        {"agent_0": 3, "agent_1": 0},  # Agente 0 vai para direita, Agente 1 vai para cima
        {"agent_0": 1, "agent_1": 3},  # Agente 0 vai para baixo, Agente 1 vai para direita
        
        # Tentam se mover para o mesmo espaço
        {"agent_0": 3, "agent_1": 2},  # Ambos tentam ir para o mesmo espaço (colisão)
        
        # Tentam trocar de posição (um para o espaço do outro)
        {"agent_0": 0, "agent_1": 1},  # Agente 0 tenta ir para onde está o Agente 1 e vice-versa
    ]
    
    # Reinicia recompensas totais
    total_rewards = {agent: 0 for agent in env.agents}
    
    # Executa a sequência de colisões
    for step, actions in enumerate(collision_sequence):
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent, reward in rewards.items():
            total_rewards[agent] += reward
        
        visualize_state(env, step + 1, actions, rewards)
        print(f"Recompensa Total: {total_rewards}")
        time.sleep(0.8)
    
    print("\n=== PARTE 3: PREPARAÇÃO DE SALADA SIMPLIFICADA ===")
    print("Agente 0 prepara a salada, Agente 1 observa e depois entrega")
    
    # Reseta o ambiente para a parte 3
    observations, info = env.reset(seed=42)
    visualize_state(env, 0)
    
    # Mostrar as posições iniciais dos agentes
    agent_positions = env.state["agent_pos"]
    print("\nPosições iniciais dos agentes:")
    for agent_id, pos in agent_positions.items():
        print(f"{agent_id}: {pos}")
    
    time.sleep(1)
    
    # Sequência para preparar salada simplificada
    # Analisando o mapa: #C#P# (C=tábua de corte, P=dispenser de pratos)
    #       # 0 # (agente 0 está no meio)
    #       #   D (D=ponto de entrega)
    #       # 1 # (agente 1 está no meio)
    #       #IIT# (I=dispenser de ingredientes, T=mesa)
    
    salad_preparation = [
        # 1. Agente 0 vai para ingredientes, Agente 1 não faz nada
        {"agent_0": 1, "agent_1": None},  # Agente 0 para baixo
        {"agent_0": 1, "agent_1": None},  # Agente 0 para baixo
        {"agent_0": 2, "agent_1": None},  # Agente 0 para esquerda (dispenser)
        
        # 2. Agente 0 pega o vegetal
        {"agent_0": 4, "agent_1": None},  # Agente 0 pega o vegetal
        
        # 3. Agente 0 vai para a tábua de corte
        {"agent_0": 0, "agent_1": None},  # Agente 0 para cima
        {"agent_0": 0, "agent_1": None},  # Agente 0 para cima
        {"agent_0": 2, "agent_1": None},  # Agente 0 para esquerda (tábua)
        
        # 4. Agente 0 corta o vegetal
        {"agent_0": 4, "agent_1": None},  # Agente 0 corta o vegetal
        
        # 5. Agente 0 vai buscar um prato
        {"agent_0": 3, "agent_1": None},  # Agente 0 para direita
        {"agent_0": 3, "agent_1": None},  # Agente 0 para direita (prato)
        
        # 6. Agente 0 pega o prato
        {"agent_0": 4, "agent_1": None},  # Agente 0 pega o prato
        
        # 7. Agente 0 volta para a tábua de corte
        {"agent_0": 2, "agent_1": None},  # Agente 0 para esquerda
        {"agent_0": 2, "agent_1": None},  # Agente 0 para esquerda (tábua)
        
        # 8. Agente 0 pega o vegetal cortado com o prato
        {"agent_0": 4, "agent_1": None},  # Agente 0 pega o vegetal com o prato
        
        # 9. Agente 0 coloca o prato com salada em algum lugar para o Agente 1 pegar
        {"agent_0": 1, "agent_1": None},  # Agente 0 para baixo
        {"agent_0": 1, "agent_1": None},  # Agente 0 para baixo
        {"agent_0": 3, "agent_1": None},  # Agente 0 para direita (mesa)
        {"agent_0": 4, "agent_1": None},  # Agente 0 coloca o prato na mesa
        
        # 10. Agente 0 se afasta e Agente 1 começa a se mover
        {"agent_0": 0, "agent_1": 0},  # Agente 0 para cima, Agente 1 para cima
        
        # 11. Agente 1 pega o prato e vai para entrega
        {"agent_0": None, "agent_1": 0},  # Agente 1 para cima
        {"agent_0": None, "agent_1": 3},  # Agente 1 para direita
        {"agent_0": None, "agent_1": 4},  # Agente 1 pega o prato com salada
        
        # 12. Agente 1 vai para o ponto de entrega
        {"agent_0": None, "agent_1": 0},  # Agente 1 para cima
        {"agent_0": None, "agent_1": 3},  # Agente 1 para direita (entrega)
        
        # 13. Agente 1 entrega o prato
        {"agent_0": None, "agent_1": 4},  # Agente 1 entrega o prato
    ]
    
    # Reinicia recompensas totais
    total_rewards = {agent: 0 for agent in env.agents}
    
    # Executa a sequência de preparação da salada
    for step, actions in enumerate(salad_preparation):
        # Processa ações, removendo as que são None
        current_actions = {agent: action for agent, action in actions.items() if action is not None}
        
        observations, rewards, terminations, truncations, infos = env.step(current_actions)
        
        for agent, reward in rewards.items():
            total_rewards[agent] += reward
        
        visualize_state(env, step + 1, current_actions, rewards)
        
        # Mostra o que cada agente está carregando
        print("Inventário:")
        for agent, agent_info in infos.items():
            print(f"  {agent}: {agent_info.get('inventory', 'Vazio')}")
        
        print(f"Recompensa Total: {total_rewards}")
        
        if any(terminations.values()):
            print("\nObjetivo alcançado! Pratos entregues com sucesso!")
            break
            
        if any(truncations.values()):
            print("\nTempo esgotado!")
            break
        
        time.sleep(1)  # Pausa maior para este teste
    
    # Fecha o ambiente
    env.close()
    print("\n=== TESTE FINALIZADO ===")

if __name__ == "__main__":
    test_simple_movement()