import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv # Importe a classe base ParallelEnv

class OvercookedMARL(ParallelEnv): # Use ParallelEnv para MARL
    metadata = {"render_modes": ["human", "rgb_array"], "name": "overcooked_marl_v0"}

    def __init__(self, render_mode=None, map_size=(10, 10), num_fixed_agents=2):
        super().__init__()
        
        # 2.1. Variáveis de Design (Seus Agentes)
        self.map_size = map_size
        self.agents = [f"agent_{i}" for i in range(num_fixed_agents)] # IDs dos agentes
        self.possible_agents = self.agents[:] # Lista de todos os IDs possíveis
        
        # 2.2. O mapa e o estado do ambiente
        # Exemplo: inicializar o estado do mapa, posições das estações, etc.
        self.state = self._initialize_game_state() 
        self._action_to_direction = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        
        # 2.3. Definir Spaces (O MAIS IMPORTANTE)

        # Observation Space (Visão Global 10x10)
        # Exemplo de camadas: Agentes, Estações, Ingredientes, Obstáculos, etc.
        observation_space_single = spaces.Box(
            low=0, 
            high=100, # Valor máximo (depende da codificação)
            shape=(map_size[0], map_size[1], 5), # 5 camadas (exemplo)
            dtype=np.int32
        )
        # O espaço de observação final é um DICIONÁRIO mapeando agentes
        self.observation_spaces = {agent: observation_space_single for agent in self.agents}
        
        # Action Space (Andar + Interagir)
        # 0-3: 4 Direções; 4: Interagir
        action_space_single = spaces.Discrete(5) 
        # O espaço de ação final é um DICIONÁRIO mapeando agentes
        self.action_spaces = {agent: action_space_single for agent in self.agents}

        # Inicializar variáveis de renderização
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    # OBRIGATÓRIO pela ParallelEnv
    def observe(self, agent: str):
        """Retorna a observação para um agente específico. Usado internamente."""
        # TODO: Implementar a lógica para extrair a observação do 'state' para o agente.
        return self._get_obs()[agent] 

    def state(self):
        """Retorna o estado global do ambiente. Opcional, mas útil."""
        return self.state

    # Funções Auxiliares - Não faz parte da API, mas é essencial
    def _initialize_game_state(self):
        # TODO: Crie a lógica de inicialização do mapa (paredes, estações).
        # Para começar, pode ser apenas um dicionário vazio ou um array numpy simples.
        return {"map": np.zeros(self.map_size), "agent_pos": {agent: (0, 0) for agent in self.agents}}

    def _get_obs(self):
        """Retorna o dicionário de observações para TODOS os agentes."""
        # TODO: Lógica para mapear o estado interno para o formato de observação.
        # Por enquanto, retorne observações de placeholder.
        obs = {agent: self.observation_spaces[agent].sample() for agent in self.agents}
        return obs

    def _get_info(self):
        """Retorna o dicionário de informações auxiliares (debugging)."""
        # TODO: Adicione a métrica de distância do Overcooked ou outros dados não essenciais.
        info = {agent: {"distance": 0} for agent in self.agents}
        return info

    # 2.4. Implementar a Função Essencial: RESET
    def reset(self, seed=None, options=None):
        # 1. Chamar o reset do super
        super().reset(seed=seed)
        
        # 2. Reinicializar o estado do jogo e as posições dos agentes
        self.state = self._initialize_game_state() 
        self.agents = self.possible_agents[:] # Reinicializa a lista de agentes ativos
        
        # 3. Retornar observações e informações
        observations = self._get_obs()
        info = self._get_info()
        
        return observations, info

    # 2.5. Implementar a Função Essencial: STEP
    def step(self, actions):
        # Este é o método mais complexo. Você deve processar o dicionário de ações.
        
        # Inicialização dos dicionários de retorno
        observations = {}
        rewards = {}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {}
        
        # TODO: Lógica de simulação, colisões e interações
        # Exemplo Básico de Movimento e Recompensa Híbrida
        for agent_id, action in actions.items():
            if action < 4: # É uma ação de movimento
                # Aplique a regra de colisão/restrição
                pass 
            else: # É uma ação de interacção
                # Aplique a lógica de interacção (cortar, pegar, colocar)
                pass

            # Exemplo de Recompensa
            rewards[agent_id] = 0.0
            # Pequena recompensa individual pela ação
            if action == 4: rewards[agent_id] += 0.1 

        # Lógica de Recompensa Global
        # TODO: Se um prato for entregue, calcule a grande recompensa e distribua para TODOS.
        # if dish_is_delivered:
        #    global_reward = 10.0
        #    for agent_id in self.agents:
        #        rewards[agent_id] += global_reward

        # Lógica de Conclusão do Episódio
        # 1. Se o tempo acabou:
        # if self.time_limit_reached():
        #    for agent_id in self.agents:
        #        truncations[agent_id] = True 
        
        # 2. Se todos os pratos foram entregues:
        # if self.all_dishes_delivered():
        #    for agent_id in self.agents:
        #        terminations[agent_id] = True
        
        # 3. Gerar Observação e Info Final
        observations = self._get_obs()
        infos = self._get_info()

        # Opcional: Remova agentes inativos (se houver)
        # self.agents = [agent for agent in self.possible_agents if not terminations[agent]]
        
        # Retorno OBRIGATÓRIO (5 dicionários)
        return observations, rewards, terminations, truncations, infos

    # 2.6. Implementar as Funções Opcionais (e de Boas Práticas)
    def render(self):
        # TODO: Lógica de renderização (desenhar o mapa e os agentes)
        pass

    def close(self):
        # TODO: Lógica para fechar a janela de renderização
        pass

# 2.7. Funções de Suporte ao PettingZoo (Para usar o módulo env_checker)
# Se você quiser que o PettingZoo cheque seu ambiente:
# from pettingzoo.test import parallel_api_test
# env = OvercookedMARL()
# parallel_api_test(env, num_cycles=100) # Opcional, mas altamente recomendado