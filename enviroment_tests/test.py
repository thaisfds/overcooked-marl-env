import gymnasium as gym
import numpy as np

# Importamos a classe direta para instanciação
from overcooked_marl_env.env import OvercookedMARL

def action_to_str(action):
    """Convert numeric action to descriptive string"""
    action_map = {
        0: "(Cima)",
        1: "(Baixo)",
        2: "(Esquerda)",
        3: "(Direita)",
        4: "(Interagir)"
    }
    # Handle numpy int types by converting to Python int
    if hasattr(action, 'item'):
        action = action.item()
    return action_map.get(action, f"Desconhecido ({action})")

def run_sanity_check(env_class, max_steps=100):
    """
    Executa um teste básico no ambiente multi-agente para verificar se a API está correta.
    """
    
    # 1. Criar o Ambiente
    print("--- 1. Inicializando Ambiente ---")
    
    env = env_class(verbose=False) 
    print(f"Ambiente criado via classe direta: {env.metadata['name']}")
        
    print(f"Agentes Ativos: {env.agents}")
    print(f"Espaço de Ação para 'agent_0': {env.action_space('agent_0')}")
    print(f"Espaço de Observação para 'agent_0': {env.observation_space('agent_0').shape}")
    print("-" * 30)

    # 2. Resetar o Ambiente
    print("--- 2. Reset (Início do Episódio) ---")
    observations, info = env.reset(seed=42)
    print(f"Observações Iniciais: {len(observations)} agentes")
    print(f"Info Inicial: {info.keys()}")
    print("-" * 30)
    
    total_reward = {agent: 0 for agent in env.agents}
    
    # 3. Executar Passos de Simulação
    print(f"--- 3. Executando {max_steps} Passos Aleatórios ---")
    for step in range(max_steps):
        # 3.1 Gerar Ações Aleatórias para CADA agente ativo
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        
        # 3.2 Executar o Passo (step)
        # O retorno é sempre 5 dicionários: obs, reward, terminated, truncated, info
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # 3.3 Processar o Resultado
        
        # Verifica se o episódio acabou para algum agente
        done = any(terminations.values()) or any(truncations.values())

        # Acumula recompensas
        for agent, reward in rewards.items():
            total_reward[agent] += reward
        
        if step % 10 == 0 or done:
            print(f"\n[Passo {step}]")
            
            # Format actions with more meaningful names
            readable_actions = {}
            for agent, action in actions.items():
                readable_actions[agent] = action_to_str(action)
            print(f"  Ações: {readable_actions}")
            
            print(f"  Recompensas: {rewards}")
            print(f"  Total Acumulado: {total_reward}")
            print(f"  Terminated: {terminations}")
            print(f"  Truncated: {truncations}")
        
        if done:
            print("\n!!! EPISÓDIO TERMINADO/TRUNCADO !!!")
            break

    # 4. Fechar o Ambiente
    env.close()
    print("-" * 30)
    print("Teste concluído com sucesso (Estrutura verificada).")

if __name__ == "__main__":
    # Certifique-se de que a classe OvercookedMARL está definida no seu env.py
    run_sanity_check(OvercookedMARL)