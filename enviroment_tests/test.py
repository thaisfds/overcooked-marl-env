import gymnasium as gym
import numpy as np

# Importe a classe do seu ambiente. 
# Se você seguiu o setup.py e o __init__.py, o nome do pacote deve funcionar.
# Se ainda não instalou, use a importação direta:
from overcooked_marl_env.env import OvercookedMARL

def run_sanity_check(env_class, max_steps=100):
    """
    Executa um teste básico no ambiente multi-agente para verificar se a API está correta.
    """
    
    # 1. Criar o Ambiente
    print("--- 1. Inicializando Ambiente ---")
    try:
        # Tenta usar gym.make() se o ambiente estiver registrado
        env = gym.make("OvercookedMARL-v0")
        print(f"Ambiente criado via gym.make(): {env.unwrapped.metadata['name']}")
    except Exception:
        # Caso o make() falhe, usa a classe diretamente
        env = env_class()
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
            print(f"  Ações: {actions}")
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