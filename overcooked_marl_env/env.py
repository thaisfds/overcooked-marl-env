import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv

# =================================================================
# VARIÁVEIS E CONSTANTES GLOBAIS
# =================================================================

CHAR_TO_NUM = {
    # Células Transitáveis (Chão)
    ' ': 0,  # Espaço Livre (Chão)
    
    # Células Não Transitáveis (Estações / Obstáculos)
    '#': 1,  # Bancada Vazia (Obstáculo estático)
    'I': 2,  # Estação de Ingredientes
    'C': 3,  # Estação de Corte
    'F': 4,  # Fogão/Cozimento
    'P': 5,  # Estação de Pratos Limpos
    'D': 6,  # Ponto de Entrega (Delivery)
    'T': 7,  # Lixo (Trash)
}

ITEM_NAMES = {
    # CÓDIGOS BÁSICOS
    'none': 0,          
    'dish_empty': 1,    
    
    # CÓDIGOS FINAIS (IDs simples e sequenciais)
    'tomato_final': 2,  
    'lettuce_final': 3, 
    'meat_final': 4,    
    
    # CÓDIGOS DE ESTADOS INTERMEDIÁRIOS (Sequenciais logo após o último final)
    'tomato_raw': 5,      
    'lettuce_raw': 6,     
    'meat_raw': 7,        
    'meat_cut': 8,        
    
    # CÓDIGO FINAL DE RECEITA (Valor Bit-wise esperado na entrega)
    'RECIPE_FULL_CODE': 15, # 1 (Prato) + 2 (Tomate) + 4 (Alface) + 8 (Carne)
}

# Mapeia o estado RAW/INTERMEDIÁRIO para o PRÓXIMO estado
PROCESSING_MAP = {
    ITEM_NAMES['tomato_raw']: ITEM_NAMES['tomato_final'],  # Cru (5) -> Final (2)
    ITEM_NAMES['lettuce_raw']: ITEM_NAMES['lettuce_final'], # Cru (6) -> Final (3)
    
    ITEM_NAMES['meat_raw']: ITEM_NAMES['meat_cut'],        # Cru (7) -> Intermediário (8)
    ITEM_NAMES['meat_cut']: ITEM_NAMES['meat_final'],      # Intermediário (8) -> Final (4)
}

# Mapeia a estação de processamento para os itens que ela trata
STATION_PROCESSOR = {
    CHAR_TO_NUM['C']: [ITEM_NAMES['tomato_raw'], ITEM_NAMES['lettuce_raw'], ITEM_NAMES['meat_raw']],
    CHAR_TO_NUM['F']: [ITEM_NAMES['meat_cut']],
}

# Constantes para Observação Vetorial
MAX_PROCESS_STEPS = 3 
TOTAL_CHANNELS = 5 + MAX_PROCESS_STEPS # 5 canais base + 3 canais para o vetor de inventário

# Lista Ordenada dos ingredientes finais (usa IDs simples)
FINAL_INGREDIENTS_ORDER = [
    ITEM_NAMES['tomato_final'], # 2
    ITEM_NAMES['lettuce_final'], # 3
    ITEM_NAMES['meat_final'], # 4
]

def _generate_bit_mapping():
    """Calcula o mapeamento de ID Simples para o valor Bit-wise (2, 4, 8...)."""
    mapping = {ITEM_NAMES['dish_empty']: 1}
    all_codes = []
    bit_value = 2 
    
    # Mapeia (2 -> 2), (3 -> 4), (4 -> 8)
    for simple_code in FINAL_INGREDIENTS_ORDER:
        mapping[simple_code] = bit_value
        all_codes.append(simple_code)
        bit_value *= 2 
    return mapping, all_codes

FINAL_TO_BIT, ALL_FINAL_CODES = _generate_bit_mapping()

# Mapeamento de Códigos para o Vetor de Progresso (Visão da IA)
PROCESSING_VECTOR_MAP = {
    # Itens Crus/Intermediários
    ITEM_NAMES['tomato_raw']:    [1, 0, 0],  
    ITEM_NAMES['lettuce_raw']:   [1, 0, 0],  
    ITEM_NAMES['meat_raw']:      [1, 0, 0],  
    ITEM_NAMES['meat_cut']:      [1, 1, 0],  
    
    # Itens Finais
    ITEM_NAMES['tomato_final']:  [1, 1, 0],  
    ITEM_NAMES['lettuce_final']: [1, 1, 0],  
    ITEM_NAMES['meat_final']:    [1, 1, 1],  
    
    # Pratos e Outros (Sem Vetor de Progresso)
    ITEM_NAMES['dish_empty']:    [0, 0, 0],  
    ITEM_NAMES['none']:          [0, 0, 0],  
    
    # Pratos Combinados (Códigos Bit-wise)
    **{code: [0, 0, 0] for code in range(1, 16)}, 
}

# =================================================================
# FUNÇÕES BASICAS DO GYMNASIUM / PETTINGZOO
# =================================================================

# Classe Principal do Ambiente MARL
class OvercookedMARL(ParallelEnv):
    
    # Metadados (Necessário para Gymnasium/PettingZoo)
    metadata = {"render_modes": ["human", "rgb_array"], "name": "overcooked_marl_v0"}


    def __init__(self, render_mode=None, map_path="maps/default_map.txt", num_fixed_agents=2):
        super().__init__()
        
        # Variáveis de Setup
        self.map_path = map_path 
        self.num_fixed_agents = num_fixed_agents
        
        # Cria IDs de agentes fixos: 'agent_0', 'agent_1', etc. (Necessário antes de _initialize_game_state)
        self.possible_agents = [f"agent_{i}" for i in range(num_fixed_agents)]
        self.agents = self.possible_agents[:] # Lista de agentes ativos no momento
        
        # Inicializa o Estado
        self.state = self._initialize_game_state() 
        
        # Definição do Observation Space
        observation_space_single = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.map_size[0], self.map_size[1], TOTAL_CHANNELS), 
            dtype=np.uint8 
        )
        
        # Dicionário mapeando cada agente do espaço
        self.observation_spaces = {
            agent: observation_space_single
            for agent in self.possible_agents
        }
        
        # Definição do Action Space (0-3: Movimento, 4: Interagir)
        action_space_single = spaces.Discrete(5) 
        
        self.action_spaces = {
            agent: action_space_single
            for agent in self.possible_agents
        }
        
        # Mapeamento de Ações para Movimentos
        # 0: Cima, 1: Baixo, 2: Esquerda, 3: Direita, 4: Interagir
        self._action_to_direction = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}

        # Variáveis de Renderização (para a Fase 3)
        self.render_mode = render_mode
        self.window = None
        self.clock = None

# =================================================================
# FUNÇÕES AUXILIARES
# =================================================================

    def _load_map_from_file(self, map_path):
        """Lê o arquivo de texto, traduz os caracteres para números e encontra a posição inicial dos agentes."""
        static_map, agent_start_pos = [], {}
        with open(map_path, 'r') as f:
            for y, line in enumerate(f.readlines()):
                row = []
                for x, char in enumerate(line.strip()):
                    if '0' <= char <= str(self.num_fixed_agents - 1):
                        agent_id = f"agent_{char}"
                        agent_start_pos[agent_id] = (x, y)
                        row.append(CHAR_TO_NUM[' ']) 
                    elif char in CHAR_TO_NUM:
                        row.append(CHAR_TO_NUM[char])
                    else:
                        raise ValueError(f"Caractere '{char}' não reconhecido no mapa em ({x}, {y}).")
                static_map.append(row)
        return np.array(static_map, dtype=np.uint8), agent_start_pos
    
    def _initialize_game_state(self):
        """Carrega o mapa dinamicamente e inicializa o estado de jogo."""
        map_static_np, agent_start_pos = self._load_map_from_file(self.map_path)
        self.map_size = map_static_np.shape
        items_on_ground = np.zeros(self.map_size, dtype=np.uint8) 
        
        state = {
            "map_static": map_static_np,
            "agent_pos": agent_start_pos,
            "carrying": {agent: ITEM_NAMES['none'] for agent in self.possible_agents},
            "items_on_ground": items_on_ground,
            "dishes_pending": 3,
            "time_steps": 0
        }
        return state

    def _get_obs(self):
        """
        Gera as observações para todos os agentes, codificando o estado do jogo 
        em canais (N x M x TOTAL_CHANNELS).
        """
        H, W = self.map_size 
        NUM_BASE_CHANNELS = 5 
        NUM_VECTOR_CHANNELS = MAX_PROCESS_STEPS
        
        observations = {}
        base_observation = np.zeros((H, W, TOTAL_CHANNELS), dtype=np.uint8)

        # Camada 0: Mapa Estático
        base_observation[:, :, 0] = self.state["map_static"]

        # Camada 1: Itens no Chão/Bancada
        base_observation[:, :, 1] = self.state["items_on_ground"]
        
        # Camada 4: Ponto de Entrega (Binário)
        base_observation[:, :, 4] = np.where(self.state["map_static"] == CHAR_TO_NUM['D'], 1, 0)
        
        
        for agent_id in self.agents:
            obs = base_observation.copy()
            ax, ay = self.state["agent_pos"][agent_id]
            
            # Camada 2: Posição do Agente que Observa
            obs[ay, ax, 2] = 1 
            
            # Camada 3: Posição dos OUTROS Agentes
            for other_id, other_pos in self.state["agent_pos"].items():
                if other_id != agent_id:
                    ox, oy = other_pos
                    obs[oy, ox, 3] = 1
                    
            # -------------------------------------------------------------------
            # CANAIS 5-7: VETOR DE PROCESSAMENTO (Inventário)
            # -------------------------------------------------------------------
            
            carrying_item_code = self.state["carrying"][agent_id]
            
            # Usa PROCESSING_VECTOR_MAP para converter o ID Simples para o vetor de observação
            progress_vector = PROCESSING_VECTOR_MAP.get(carrying_item_code, [0, 0, 0])
            
            # Codifica o vetor nas camadas 5, 6 e 7 (na posição do agente)
            for i in range(NUM_VECTOR_CHANNELS):
                obs[ay, ax, 5 + i] = progress_vector[i] 
            
            observations[agent_id] = obs
            
        return observations

    def _get_info(self):
        """Retorna o dicionário de informações auxiliares."""
        # Adicicionar futuramente informações úteis para debugging (ex: pontuação atual)
        info = {agent: {"score": 0, "dishes_remaining": self.state["dishes_pending"]} for agent in self.agents}
        return info

    def reset(self, seed=None, options=None):
        """Inicia um novo episódio."""
        gym.Env.reset(self, seed=seed) 
        
        # Reinicializa o estado e a lista de agentes
        self.state = self._initialize_game_state()
        self.agents = self.possible_agents[:] 
        
        observations = self._get_obs()
        info = self._get_info()
        
        return observations, info

    def step(self, actions):
        """Executa um passo de simulação, processando o dicionário de ações."""
        
        # Inicialização dos dicionários de retorno
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        episode_terminated = False
        episode_truncated = False
        global_reward = 0.0 # Acumulador de recompensa global
        
        # Processamento das Ações e Lógica de Movimento/Interação
        
        for agent_id, action in actions.items():
            current_pos = self.state["agent_pos"][agent_id]
            current_x, current_y = current_pos
            
            rewards[agent_id] += -0.01 
            
            if action < 4:  # Ações 0, 1, 2, 3: Movimento
                dx, dy = self._action_to_direction[action]
                new_x, new_y = current_x + dx, current_y + dy
                max_x, max_y = self.map_size[1] - 1, self.map_size[0] - 1
                new_x, new_y = np.clip(new_x, 0, max_x), np.clip(new_y, 0, max_y)
                
                cell_type = self.state["map_static"][new_y, new_x]
                is_collision = (cell_type != CHAR_TO_NUM[' '])
                other_agent_pos = [pos for id, pos in self.state["agent_pos"].items() if id != agent_id]
                
                if (new_x, new_y) in other_agent_pos: is_collision = True
                
                if not is_collision: self.state["agent_pos"][agent_id] = (new_x, new_y)
                else: rewards[agent_id] += -0.005
            
            elif action == 4: # Ação 4: Interagir
                local_reward, interaction_global_reward = self._handle_interaction(agent_id)
                rewards[agent_id] += local_reward
                global_reward += interaction_global_reward # Acumula o bônus global

        # Distribui a Recompensa Global Acumulada
        if global_reward > 0:
            for agent_id in self.agents: rewards[agent_id] += global_reward
                
        # Atualização de Estado e Conclusão do Episódio
        
        self.state["time_steps"] += 1
        MAX_EPISODE_STEPS = 500 
        
        # Derrota por tempo esgotado
        if self.state["time_steps"] >= MAX_EPISODE_STEPS:
            episode_truncated = True

        # Vitória
        if self.state["dishes_pending"] <= 0:
            episode_terminated = True
            
        # Distribui o status de conclusão para cada agente
        for agent_id in self.agents:
            terminations[agent_id] = episode_terminated
            truncations[agent_id] = episode_truncated
        
        # Gerar Observação e Info Final
        observations = self._get_obs() 
        infos = self._get_info() 

        return observations, rewards, terminations, truncations, infos
    
    # Função exigida pela ParallelEnv
    def observe(self, agent: str):
        """Função exigida que retorna a observação de um agente (acessa o _get_obs)."""
        return self._get_obs()[agent] 

    # Função exigida pela ParallelEnv
    def state(self):
        """Função exigida que retorna o estado GLOBAL do ambiente."""
        return self.state

    # Função exigida pela ParallelEnv
    def render(self):
        # TODO: Implementação de renderização na Fase 3
        pass

    # Função exigida pela ParallelEnv
    def close(self):
        # TODO: Implementação para fechar a janela de renderização
        pass

    # Função exigida pela ParallelEnv
    def observation_space(self, agent):
        """Retorna o espaço de observação para um agente específico."""
        return self.observation_spaces[agent]
    
    # Função exigida pela ParallelEnv
    def action_space(self, agent):
        """Retorna o espaço de ação para um agente específico."""
        return self.action_spaces[agent]
    
    def _get_tile_in_front(self, agent_id):
        """Retorna a posição atual do agente. (Simplificado: Interage com o que está sob ele)."""
        return self.state["agent_pos"][agent_id]

    def _handle_interaction(self, agent_id):
        """Executa a lógica de Ação 4 (Interagir) usando o mapeamento genérico."""
        
        ix, iy = self._get_tile_in_front(agent_id)
        carrying_item = self.state["carrying"][agent_id]
        station_type = self.state["map_static"][iy, ix] 
        item_on_ground = self.state["items_on_ground"][iy, ix] 
        
        reward = -0.005
        global_reward_bonus = 0.0

        # --- 1. LÓGICA DE PROCESSAMENTO (Genérica) ---
        if station_type in STATION_PROCESSOR:
            if carrying_item in STATION_PROCESSOR[station_type]:
                
                # Usa o PROCESSING_MAP com os novos IDs sequenciais (7 -> 8 ou 8 -> 4)
                if carrying_item in PROCESSING_MAP:
                    next_state = PROCESSING_MAP[carrying_item]
                    self.state["carrying"][agent_id] = next_state
                    reward += 1.0 if next_state in ALL_FINAL_CODES else 0.5
                    return reward, 0.0
        
        # --- 2. ESTAÇÕES DE MATERIAIS ---
        
        elif station_type == CHAR_TO_NUM['I']: # Estação de Ingredientes (Tomate Cru - Exemplo)
            if carrying_item == ITEM_NAMES['none']:
                self.state["carrying"][agent_id] = ITEM_NAMES['tomato_raw']
                reward += 0.5
        
        elif station_type == CHAR_TO_NUM['P']: # Estação de Pratos Limpos
            if carrying_item == ITEM_NAMES['none']:
                self.state["carrying"][agent_id] = ITEM_NAMES['dish_empty']
                reward += 0.5
        
        # --- 3. BANCADA E COMBINAÇÃO (Lógica Bit-wise) ---
        
        elif station_type == CHAR_TO_NUM['#']:
            is_a_dish = item_on_ground != ITEM_NAMES['none'] and item_on_ground % 2 != 0 
            is_final_ingredient = carrying_item in ALL_FINAL_CODES
            
            # Combinação: Ingrediente Final + Prato na Bancada
            if is_final_ingredient and is_a_dish:
                # Usa FINAL_TO_BIT para converter o ID Simples (2, 3, 4) para Bit-wise (2, 4, 8)
                bit_value_ingrediente = FINAL_TO_BIT.get(carrying_item, 0)
                new_bit_code = item_on_ground + bit_value_ingrediente 
                
                self.state["carrying"][agent_id] = new_bit_code 
                self.state["items_on_ground"][iy, ix] = ITEM_NAMES['none'] 
                reward += 2.0
            
            # Colocar item na bancada
            elif carrying_item != ITEM_NAMES['none'] and item_on_ground == ITEM_NAMES['none']:
                self.state["items_on_ground"][iy, ix] = carrying_item
                self.state["carrying"][agent_id] = ITEM_NAMES['none']
                reward += 0.2
            
            # Pegar item da bancada
            elif carrying_item == ITEM_NAMES['none'] and item_on_ground != ITEM_NAMES['none']:
                self.state["carrying"][agent_id] = item_on_ground
                self.state["items_on_ground"][iy, ix] = ITEM_NAMES['none']
                reward += 0.2
        
        # --- 4. ENTREGA ('D') ---
        elif station_type == CHAR_TO_NUM['D']: 
            if carrying_item == ITEM_NAMES['RECIPE_FULL_CODE']: 
                self.state["carrying"][agent_id] = ITEM_NAMES['none'] 
                self.state["dishes_pending"] -= 1 
                
                global_reward_bonus = 20.0
                reward += 10.0
        
        return reward, global_reward_bonus