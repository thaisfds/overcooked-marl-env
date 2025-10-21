import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from .core.items import ItemType, ItemManager
from .core.recipes import Recipe, RecipeManager

# Constantes para observação
MAX_PROCESS_STEPS = 3 
TOTAL_CHANNELS = 5 + MAX_PROCESS_STEPS # 5 canais base + 3 canais para processamento

class OvercookedMARL(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "overcooked_marl_v0"}


    def __init__(self, render_mode=None, map_path="maps/default_map.txt", num_fixed_agents=2, verbose=False):
        super().__init__()
        
        self.map_path = map_path 
        self.num_fixed_agents = num_fixed_agents
        self.verbose = verbose
        
        self.item_manager = ItemManager()
        self.recipe_manager = RecipeManager(self.item_manager)
        
        self.possible_agents = [f"agent_{i}" for i in range(num_fixed_agents)]
        self.agents = self.possible_agents[:]
        
        self._already_loaded_maps = set()
        self.state = self._initialize_game_state() 
        
        if not hasattr(self, 'map_size') or self.map_size is None:
            raise ValueError("Map size was not properly initialized")
        
        # 0: Cima, 1: Baixo, 2: Esquerda, 3: Direita, 4: Interagir
        self._action_to_direction = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
        
        action_space_single = spaces.Discrete(5) 
        
        # Observation Space
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
        
        self.action_spaces = {
            agent: action_space_single
            for agent in self.possible_agents
        }

        self.render_mode = render_mode
        self.window = None
        self.clock = None

# =================================================================
# FUNÇÕES AUXILIARES
# =================================================================

    def _load_map_from_file(self, map_path):
        is_reload = map_path in self._already_loaded_maps
        self._already_loaded_maps.add(map_path)
        
        static_map = []
        agent_start_pos = {}
        dispenser_assignments = {}
        recipes_to_load = []
        
        import os
        if not os.path.isabs(map_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            map_path = os.path.join(base_dir, map_path)
        
        if not is_reload or self.verbose:
            print(f"Loading map from: {map_path}")
        
        if not os.path.exists(map_path):
            available_dirs = os.listdir(os.path.dirname(map_path))
            print(f"WARNING: Map file not found! Available files in directory: {available_dirs}")
            raise FileNotFoundError(f"Map file not found at {map_path}")
        
        with open(map_path, 'r') as f:
            lines = f.readlines()
            
            if not is_reload and self.verbose:
                print(f"Map file loaded with {len(lines)} lines")
                print(f"First few lines: {[line.strip() for line in lines[:3]]}")

        # --- 1. Processar Grid Map e Agentes ---
        # First, find the actual grid map section between headers
        grid_lines = []
        in_grid_section = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                if in_grid_section:
                    in_grid_section = False
                continue
                
            if line_stripped == "# Grid Map":
                in_grid_section = True
                continue
                
            if line_stripped.startswith("# Dispensers") or line_stripped.startswith("# Recipes"):
                in_grid_section = False
                continue
                
            if in_grid_section and not (line_stripped[0].isdigit() and ' ' in line_stripped):
                grid_lines.append(line)
        
        if not is_reload or self.verbose:
            print(f"Found {len(grid_lines)} grid lines in map file")
        
        # Fallback para grid sem marcadores explícitos
        if len(grid_lines) == 0:
            in_grid_section = True
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                if not line_stripped or (i == 0 and line_stripped.startswith('#')):
                    continue
                    
                if not line_stripped or (line_stripped[0].isdigit() and ' ' in line_stripped):
                    in_grid_section = False
                    break
                    
                if any(marker in line_stripped for marker in ["# Dispensers", "# Recipes"]):
                    in_grid_section = False
                    break
                    
                if not line_stripped.startswith('#'):
                    grid_lines.append(line)
                    
            if not is_reload or self.verbose:
                print(f"Found {len(grid_lines)} grid lines using fallback method")
        
        # Process the grid lines
        for y, line in enumerate(grid_lines):
            row = []
            if not is_reload or self.verbose:
                print(f"Processing map row {y}: '{line.strip()}'")
            
            for x, char in enumerate(line):
                # Handle agent positions (0, 1, etc.)
                if char.isdigit() and 0 <= int(char) < self.num_fixed_agents:
                    agent_id = f"agent_{char}"
                    agent_start_pos[agent_id] = (x, y)
                    if not is_reload or self.verbose:
                        print(f"Found agent {agent_id} at position ({x}, {y})")
                    row.append(self.item_manager.get_cell_code(' '))  # Agent stands on empty space
                    
                # Handle map cells
                elif char in self.item_manager.CHAR_TO_NUM:
                    row.append(self.item_manager.get_cell_code(char))
                    
                # Empty space
                elif char == ' ':
                    row.append(self.item_manager.get_cell_code(' '))
                    
                # Skip newline characters without warning
                elif char == '\n' or char == '\r':
                    continue
                    
                # Skip characters we don't understand (like '#' comment marker)
                else:
                    if not is_reload or self.verbose:
                        print(f"WARNING: Unknown character '{char}' at ({x}, {y}), treating as empty space")
                    row.append(self.item_manager.get_cell_code(' '))
            
            if row:  # Only add non-empty rows
                static_map.append(row)
                
        if not static_map:
            print("WARNING: No map data was successfully processed!")
            # Create a minimal default map to avoid crashes
            static_map = [
                [self.item_manager.get_cell_code('#'), self.item_manager.get_cell_code('#'), self.item_manager.get_cell_code('#')],
                [self.item_manager.get_cell_code('#'), self.item_manager.get_cell_code(' '), self.item_manager.get_cell_code('#')],
                [self.item_manager.get_cell_code('#'), self.item_manager.get_cell_code('#'), self.item_manager.get_cell_code('#')]
            ]
            if not is_reload or self.verbose:
                print("Created a minimal default map as a fallback")

        static_map_np = np.array(static_map, dtype=np.uint8)
        
        # Verificar se temos posições iniciais para todos os agentes
        for i in range(self.num_fixed_agents):
            agent_id = f"agent_{i}"
            if agent_id not in agent_start_pos:
                if not is_reload or self.verbose:
                    print(f"Note: No explicit position found for {agent_id} in map. Finding suitable position...")
                # Encontrar um espaço vazio para posicionar o agente
                found_position = False
                for y in range(len(static_map)):
                    for x in range(len(static_map[0])):
                        if static_map[y][x] == self.item_manager.get_cell_code(' '):
                            agent_start_pos[agent_id] = (x, y)
                            found_position = True
                            if not is_reload or self.verbose:
                                print(f"Assigned {agent_id} to position ({x}, {y})")
                            break
                    if found_position:
                        break
                        
                # Se ainda não encontrou posição, coloque em (0,0)
                if not found_position:
                    agent_start_pos[agent_id] = (0, 0)
                    if not is_reload or self.verbose:
                        print(f"No suitable position found for {agent_id}. Using default (0,0)")
                    
        if not is_reload or self.verbose:
            print(f"Final agent positions: {agent_start_pos}")
        
        # --- 2. Processar Atribuições do Dispenser (RAW_ITEM_ID X Y) e Receitas ---
        # Process the remaining lines for dispensers and recipes
        dispenser_section = False
        recipe_section = False
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
                
            if line_stripped.startswith('# Dispensers'):
                dispenser_section = True
                recipe_section = False
                continue
                
            if line_stripped.startswith('# Recipes'):
                dispenser_section = False
                recipe_section = True
                continue
                
            if line_stripped.startswith('#'):
                continue

            # Parse based on current section
            if dispenser_section and line_stripped[0].isdigit():
                parts = line_stripped.split()
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    raw_item_id = int(parts[0])
                    x, y = int(parts[1]), int(parts[2])
                    dispenser_assignments[(x, y)] = raw_item_id
                    if not is_reload or self.verbose:
                        print(f"Added dispenser with item ID {raw_item_id} at position ({x}, {y})")
                    
            elif recipe_section:
                # Recipes can be comma-separated values like "SALAD, BURGUER"
                recipe_names = [name.strip().upper() for name in line_stripped.split(',')]
                recipes_to_load.extend([r for r in recipe_names if r])
                if not is_reload or self.verbose:
                    print(f"Found recipes: {recipes_to_load}")
            
            # If line contains digits in any section, it might be dispenser data
            elif line_stripped[0].isdigit():
                parts = line_stripped.split()
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    raw_item_id = int(parts[0])
                    x, y = int(parts[1]), int(parts[2])
                    dispenser_assignments[(x, y)] = raw_item_id
                    if not is_reload or self.verbose:
                        print(f"Added dispenser with item ID {raw_item_id} at position ({x}, {y})")
                    
            # If line contains recipe names in any section
            elif any(keyword in line_stripped.upper() for keyword in ["SALAD", "BURGUER"]):
                recipe_names = [name.strip().upper() for name in line_stripped.split(',')]
                recipes_to_load.extend([r for r in recipe_names if r])
                if not is_reload or self.verbose:
                    print(f"Found recipes: {recipes_to_load}") 
                
        # Make sure we have valid data
        if len(static_map) == 0:
            print("WARNING: No valid grid map was found in the file. Creating a default map.")
            static_map = [
                [self.item_manager.get_cell_code('#'), self.item_manager.get_cell_code('C'), self.item_manager.get_cell_code('#')],
                [self.item_manager.get_cell_code(' '), self.item_manager.get_cell_code(' '), self.item_manager.get_cell_code('D')],
                [self.item_manager.get_cell_code('I'), self.item_manager.get_cell_code(' '), self.item_manager.get_cell_code('#')]
            ]
        
        # Initialize recipes
        self._initialize_recipes(recipes_to_load)
            
        return static_map_np, agent_start_pos, dispenser_assignments
        
    def _initialize_recipes(self, recipe_names):
        """Initialize recipes based on the names from the map file."""
        added_recipes = False
        
        for name in recipe_names:
            name = name.strip().upper()
            if name == "SALAD":
                recipe = Recipe(
                    name="SALAD",
                    ingredients=[ItemType.TOMATO, ItemType.LETTUCE],
                    reward=10.0
                )
                self.recipe_manager.add_recipe(recipe)
                added_recipes = True
                if self.map_path not in self._already_loaded_maps or self.verbose:
                    print(f"Added recipe: {name}")
            elif name == "BURGUER":
                recipe = Recipe(
                    name="BURGUER",
                    ingredients=[ItemType.LETTUCE, ItemType.MEAT],
                    reward=15.0
                )
                self.recipe_manager.add_recipe(recipe)
                added_recipes = True
                if self.map_path not in self._already_loaded_maps or self.verbose:
                    print(f"Added recipe: {name}")
        
        # Certifique-se de que pelo menos uma receita está definida (caso o mapa não especifique)
        if not added_recipes:
            if self.map_path not in self._already_loaded_maps or self.verbose:
                print("WARNING: No recipes were loaded from the map. Adding default SALAD recipe.")
            default_recipe = Recipe(
                name="SALAD",
                ingredients=[ItemType.TOMATO, ItemType.LETTUCE],
                reward=10.0
            )
            self.recipe_manager.add_recipe(default_recipe)
    
    def _initialize_game_state(self):
        """Carrega o mapa e inicializa o estado de jogo."""
        map_static_np, agent_start_pos, dispenser_assignments = self._load_map_from_file(self.map_path)
        
        if map_static_np.size == 0:
            raise ValueError(f"Map file at {self.map_path} resulted in an empty map")
            
        self.map_size = map_static_np.shape
        if self.map_path not in self._already_loaded_maps or self.verbose:
            print(f"Map loaded with size: {self.map_size}")
        
        items_on_ground = np.zeros(self.map_size, dtype=np.uint8) 
        
        carrying = {}
        for agent in self.possible_agents:
            if agent not in agent_start_pos:
                if self.map_path not in self._already_loaded_maps or self.verbose:
                    print(f"WARNING: Agent {agent} was not assigned a position in the map. Using default position.")
                # Encontra um espaço vazio
                for y in range(self.map_size[0]):
                    for x in range(self.map_size[1]):
                        if map_static_np[y, x] == self.item_manager.get_cell_code(' '):
                            agent_start_pos[agent] = (x, y)
                            break
                    if agent in agent_start_pos:
                        break
                if agent not in agent_start_pos:
                    agent_start_pos[agent] = (0, 0)
            
            carrying[agent] = ItemType.NONE.value
        
        state = {
            "map_static": map_static_np,
            "agent_pos": agent_start_pos,
            "carrying": carrying,
            "items_on_ground": items_on_ground,
            "dishes_pending": 3,
            "time_steps": 0,
            "dispensers": dispenser_assignments,
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
        delivery_code = self.item_manager.get_cell_code('D')
        base_observation[:, :, 4] = np.where(self.state["map_static"] == delivery_code, 1, 0)
        
        
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
            progress_vector = self.item_manager.PROCESSING_VECTOR_MAP.get(carrying_item_code, [0, 0, 0])
            
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
        
        # Make sure the map is marked as loaded to avoid duplicate logs
        if not self.map_path in self._already_loaded_maps:
            self._already_loaded_maps.add(self.map_path)
        
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
                is_collision = (cell_type != self.item_manager.get_cell_code(' '))
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
        """Executa a lógica de Ação 4 (Interagir) usando o mapeamento dinâmico de dispensers."""
        
        ix, iy = self._get_tile_in_front(agent_id)
        carrying_item = self.state["carrying"][agent_id]
        station_type = self.state["map_static"][iy, ix] 
        item_on_ground = self.state["items_on_ground"][iy, ix] 
        
        reward = -0.005
        global_reward_bonus = 0.0

        # Lógica de Processamento
        if station_type in self.item_manager.STATION_PROCESSOR:
            if carrying_item in self.item_manager.STATION_PROCESSOR[station_type]:
                if carrying_item in self.item_manager.PROCESSING_MAP:
                    next_state = self.item_manager.PROCESSING_MAP[carrying_item]
                    self.state["carrying"][agent_id] = next_state
                    reward += 1.0 if next_state in self.recipe_manager.ALL_FINAL_CODES else 0.5
                    return reward, 0.0
        
        # Lógica de Coleta de Itens
        if (ix, iy) in self.state["dispensers"]:
            if carrying_item == ItemType.NONE.value:
                item_to_dispense = self.state["dispensers"][(ix, iy)]
                self.state["carrying"][agent_id] = item_to_dispense
                reward += 0.5
                return reward, 0.0
        
        # Bancada e Combinação
        elif station_type == self.item_manager.get_cell_code('#'):
            is_a_dish = item_on_ground != ItemType.NONE.value and item_on_ground % 2 != 0 
            is_final_ingredient = carrying_item in self.recipe_manager.ALL_FINAL_CODES
            
            # Combinação: Ingrediente + Prato
            if is_final_ingredient and is_a_dish:
                bit_value_ingrediente = self.recipe_manager.FINAL_TO_BIT.get(carrying_item, 0)
                new_bit_code = item_on_ground + bit_value_ingrediente 
                self.state["carrying"][agent_id] = new_bit_code 
                self.state["items_on_ground"][iy, ix] = ItemType.NONE.value 
                reward += 2.0
                
            # Colocar item na bancada
            elif carrying_item != ItemType.NONE.value and item_on_ground == ItemType.NONE.value:
                self.state["items_on_ground"][iy, ix] = carrying_item
                self.state["carrying"][agent_id] = ItemType.NONE.value
                reward += 0.2
            
            # Pegar item da bancada
            elif carrying_item == ItemType.NONE.value and item_on_ground != ItemType.NONE.value:
                self.state["carrying"][agent_id] = item_on_ground
                self.state["items_on_ground"][iy, ix] = ItemType.NONE.value
                reward += 0.2
        
        # Entrega
        elif station_type == self.item_manager.get_cell_code('D'): 
            if carrying_item == ItemType.RECIPE_FULL.value: 
                self.state["carrying"][agent_id] = ItemType.NONE.value
                self.state["dishes_pending"] -= 1 
                global_reward_bonus = 20.0
                reward += 10.0
        
        return reward, global_reward_bonus