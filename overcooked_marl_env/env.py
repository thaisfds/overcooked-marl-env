import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from .core.items import ItemType, ItemManager
from .core.recipes import Recipe, RecipeManager, RECIPE_DEFINITIONS, get_available_recipes, get_default_recipe

# Constants for observation
MAX_PROCESS_STEPS = 3 
TOTAL_CHANNELS = 5 + MAX_PROCESS_STEPS # 5 base channels + 3 channels for processing

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
        
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Interact
        self._action_to_direction = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
        
        action_space_single = spaces.Discrete(5) 
        
        # Observation Space
        observation_space_single = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.map_size[0], self.map_size[1], TOTAL_CHANNELS), 
            dtype=np.uint8 
        )
        
        # Dictionary mapping each agent to the action space
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
# HELPER FUNCTIONS
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
        
        # Fallback for grid without explicit markers
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
        
        # Check if we have starting positions for all agents
        for i in range(self.num_fixed_agents):
            agent_id = f"agent_{i}"
            if agent_id not in agent_start_pos:
                if not is_reload or self.verbose:
                    print(f"Note: No explicit position found for {agent_id} in map. Finding suitable position...")
                # Find an empty space to position the agent
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
                        
                # If position still not found, place at (0,0)
                if not found_position:
                    agent_start_pos[agent_id] = (0, 0)
                    if not is_reload or self.verbose:
                        print(f"No suitable position found for {agent_id}. Using default (0,0)")
                    
        if not is_reload or self.verbose:
            print(f"Final agent positions: {agent_start_pos}")
        
        # --- 2. Process Dispenser Assignments (RAW_ITEM_ID X Y) and Recipes ---
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
            # Check if the line contains known recipes using the dynamic list of available recipes
            elif any(recipe_name in line_stripped.upper() for recipe_name in get_available_recipes()):
                recipe_names = [name.strip().upper() for name in line_stripped.split(',')]
                valid_recipes = [r for r in recipe_names if r in get_available_recipes()]
                recipes_to_load.extend(valid_recipes)
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
        
        # Store the order of recipes for bonus calculation
        self.recipe_order = []
        
        # Add recipes in the order specified in the map file
        for name in recipe_names:
            name = name.strip().upper()
            if name in RECIPE_DEFINITIONS:
                self.recipe_manager.add_recipe(RECIPE_DEFINITIONS[name])
                self.recipe_order.append(name)
                if self.map_path not in self._already_loaded_maps or self.verbose:
                    print(f"Added recipe: {name}")
        
        # Make sure at least one recipe is defined (in case the map doesn't specify any)
        if len(self.recipe_order) == 0:
            default_recipe = get_default_recipe()
            if self.map_path not in self._already_loaded_maps or self.verbose:
                print(f"WARNING: No recipes were loaded from the map. Adding default {default_recipe} recipe.")
            
            if default_recipe in RECIPE_DEFINITIONS:
                self.recipe_manager.add_recipe(RECIPE_DEFINITIONS[default_recipe])
                self.recipe_order.append(default_recipe)
    
    def _initialize_game_state(self):
        """Loads the map and initializes the game state."""
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
                # Find an empty space
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
            "expected_recipe_index": 0,  # Index of the next expected recipe
            "delivered_recipes": [],     # Recipes already delivered
        }
        return state

    def _get_obs(self):
        """
        Generates observations for all agents, encoding the game state
        in channels (N x M x TOTAL_CHANNELS).
        """
        H, W = self.map_size 
        NUM_BASE_CHANNELS = 5 
        NUM_VECTOR_CHANNELS = MAX_PROCESS_STEPS
        
        observations = {}
        base_observation = np.zeros((H, W, TOTAL_CHANNELS), dtype=np.uint8)

        # Layer 0: Static Map
        base_observation[:, :, 0] = self.state["map_static"]

        # Layer 1: Items on Ground/Counter
        base_observation[:, :, 1] = self.state["items_on_ground"]
        
        # Layer 4: Delivery Point (Binary)
        delivery_code = self.item_manager.get_cell_code('D')
        base_observation[:, :, 4] = np.where(self.state["map_static"] == delivery_code, 1, 0)
        
        
        for agent_id in self.agents:
            obs = base_observation.copy()
            ax, ay = self.state["agent_pos"][agent_id]
            
            # Layer 2: Position of the Observing Agent
            obs[ay, ax, 2] = 1 
            
            # Layer 3: Position of OTHER Agents
            for other_id, other_pos in self.state["agent_pos"].items():
                if other_id != agent_id:
                    ox, oy = other_pos
                    obs[oy, ox, 3] = 1
                    
            # -------------------------------------------------------------------
            # CHANNELS 5-7: PROCESSING VECTOR (Inventory)
            # -------------------------------------------------------------------
            
            carrying_item_code = self.state["carrying"][agent_id]
            
            # Uses PROCESSING_VECTOR_MAP to convert Simple ID to observation vector
            progress_vector = self.item_manager.PROCESSING_VECTOR_MAP.get(carrying_item_code, [0, 0, 0])
            
            # Encodes the vector in layers 5, 6 and 7 (at the agent's position)
            for i in range(NUM_VECTOR_CHANNELS):
                obs[ay, ax, 5 + i] = progress_vector[i] 
            
            observations[agent_id] = obs
            
        return observations

    def _get_info(self):
        """Returns the dictionary of auxiliary information."""
        info = {}
        for agent in self.agents:
            agent_info = {
                "score": 0, 
                "dishes_remaining": self.state["dishes_pending"],
                "delivered_recipes": self.state.get("delivered_recipes", []),
                "recipe_order": self.recipe_order,
                "next_expected_recipe": self.recipe_order[self.state["expected_recipe_index"] % len(self.recipe_order)] if len(self.recipe_order) > 0 else "None"
            }
            info[agent] = agent_info
            
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
        
        # Reset variables related to recipe order
        self.state["expected_recipe_index"] = 0
        self.state["delivered_recipes"] = []
        
        observations = self._get_obs()
        info = self._get_info()
        
        return observations, info

    def step(self, actions):
        """Executes a simulation step, processing the action dictionary."""
        
        # Initialization of return dictionaries
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        episode_terminated = False
        episode_truncated = False
        global_reward = 0.0 # Global reward accumulator
        
        # Processing Actions and Movement/Interaction Logic
        
        for agent_id, action in actions.items():
            current_pos = self.state["agent_pos"][agent_id]
            current_x, current_y = current_pos
            
            rewards[agent_id] += -0.01 
            
            if action < 4:  # Actions 0, 1, 2, 3: Movement
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
            
            elif action == 4: # Action 4: Interact
                local_reward, interaction_global_reward = self._handle_interaction(agent_id)
                rewards[agent_id] += local_reward
                global_reward += interaction_global_reward # Accumulate global bonus

        # Distribute the Accumulated Global Reward
        if global_reward > 0:
            for agent_id in self.agents: rewards[agent_id] += global_reward
                
        # State Update and Episode Conclusion
        
        self.state["time_steps"] += 1
        MAX_EPISODE_STEPS = 500 
        
        # Defeat due to time out
        if self.state["time_steps"] >= MAX_EPISODE_STEPS:
            episode_truncated = True

        # Victory
        if self.state["dishes_pending"] <= 0:
            episode_terminated = True
            
        # Distribute completion status to each agent
        for agent_id in self.agents:
            terminations[agent_id] = episode_terminated
            truncations[agent_id] = episode_truncated
        
        # Generate Final Observation and Info
        observations = self._get_obs() 
        infos = self._get_info() 

        return observations, rewards, terminations, truncations, infos
    
    # Function required by ParallelEnv
    def observe(self, agent: str):
        """Required function that returns an agent's observation (accesses _get_obs)."""
        return self._get_obs()[agent] 

    # Function required by ParallelEnv
    def state(self):
        """Required function that returns the GLOBAL state of the environment."""
        return self.state

    # Function required by ParallelEnv
    def render(self):
        # TODO: Implementation of rendering in Phase 3
        pass

    # Function required by ParallelEnv
    def close(self):
        # TODO: Implementation to close the rendering window
        pass

    # Function required by ParallelEnv
    def observation_space(self, agent):
        """Returns the observation space for a specific agent."""
        return self.observation_spaces[agent]
    
    # Function required by ParallelEnv
    def action_space(self, agent):
        """Returns the action space for a specific agent."""
        return self.action_spaces[agent]
    
    def _get_tile_in_front(self, agent_id):
        """Returns the current position of the agent. (Simplified: Interacts with what is under it)."""
        return self.state["agent_pos"][agent_id]

    def _handle_interaction(self, agent_id):
        """Executes the logic of Action 4 (Interact) using dynamic dispenser mapping."""
        
        ix, iy = self._get_tile_in_front(agent_id)
        carrying_item = self.state["carrying"][agent_id]
        station_type = self.state["map_static"][iy, ix] 
        item_on_ground = self.state["items_on_ground"][iy, ix] 
        
        reward = -0.005
        global_reward_bonus = 0.0

        # Processing Logic
        if station_type in self.item_manager.STATION_PROCESSOR:
            if carrying_item in self.item_manager.STATION_PROCESSOR[station_type]:
                if carrying_item in self.item_manager.PROCESSING_MAP:
                    next_state = self.item_manager.PROCESSING_MAP[carrying_item]
                    self.state["carrying"][agent_id] = next_state
                    reward += 1.0 if next_state in self.recipe_manager.ALL_FINAL_CODES else 0.5
                    return reward, 0.0
        
        # Item Collection Logic
        if (ix, iy) in self.state["dispensers"]:
            if carrying_item == ItemType.NONE.value:
                item_to_dispense = self.state["dispensers"][(ix, iy)]
                self.state["carrying"][agent_id] = item_to_dispense
                reward += 0.5
                return reward, 0.0
        
        # Counter and Combination
        elif station_type == self.item_manager.get_cell_code('#'):
            is_a_dish = item_on_ground != ItemType.NONE.value and item_on_ground % 2 != 0 
            is_final_ingredient = carrying_item in self.recipe_manager.ALL_FINAL_CODES
            
            # Combination: Ingredient + Dish
            if is_final_ingredient and is_a_dish:
                bit_value_ingrediente = self.recipe_manager.FINAL_TO_BIT.get(carrying_item, 0)
                new_bit_code = item_on_ground + bit_value_ingrediente 
                self.state["carrying"][agent_id] = new_bit_code 
                self.state["items_on_ground"][iy, ix] = ItemType.NONE.value 
                reward += 2.0
                
            # Place item on counter
            elif carrying_item != ItemType.NONE.value and item_on_ground == ItemType.NONE.value:
                self.state["items_on_ground"][iy, ix] = carrying_item
                self.state["carrying"][agent_id] = ItemType.NONE.value
                reward += 0.2
            
            # Pick up item from counter
            elif carrying_item == ItemType.NONE.value and item_on_ground != ItemType.NONE.value:
                self.state["carrying"][agent_id] = item_on_ground
                self.state["items_on_ground"][iy, ix] = ItemType.NONE.value
                reward += 0.2
        
        # Delivery
        elif station_type == self.item_manager.get_cell_code('D'): 
            if carrying_item == ItemType.RECIPE_FULL.value:
                delivered_recipe = ""
                if len(self.recipe_order) > 0:
                    delivered_recipe = self.recipe_order[len(self.state["delivered_recipes"]) % len(self.recipe_order)]
                
                # Check if delivery follows the expected order
                is_correct_order = False
                if len(self.recipe_order) > 0:
                    expected_index = self.state["expected_recipe_index"] % len(self.recipe_order)
                    expected_recipe = self.recipe_order[expected_index]
                    is_correct_order = (delivered_recipe == expected_recipe)
                    
                    # Advance to the next recipe in order
                    self.state["expected_recipe_index"] += 1
                
                # Register the delivery
                self.state["delivered_recipes"].append(delivered_recipe)
                self.state["carrying"][agent_id] = ItemType.NONE.value
                self.state["dishes_pending"] -= 1
                
                # Base reward for delivery (use specific value of the recipe)
                recipe_reward = RECIPE_DEFINITIONS.get(delivered_recipe, Recipe("", [])).reward
                base_reward = recipe_reward
                
                # Bonus if recipe was delivered in correct order
                order_bonus = recipe_reward if is_correct_order else 0.0
                
                # Apply rewards
                global_reward_bonus = recipe_reward + order_bonus
                reward += base_reward
                
                # Delivery information
                if self.verbose:
                    if is_correct_order:
                        print(f"Recipe {delivered_recipe} delivered in correct order! Reward: {base_reward} + Bonus: {order_bonus}")
                    else:
                        print(f"Recipe {delivered_recipe} delivered out of order. Reward: {base_reward} (no bonus). Expected: {expected_recipe}")
        
        return reward, global_reward_bonus