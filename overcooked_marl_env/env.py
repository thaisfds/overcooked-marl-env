import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
from pettingzoo.utils.env import ParallelEnv
from .core.items import (
    ItemState, ItemType, ItemManager,
    create_ingredient, create_plate, advance_processing,
    is_ready, can_combine, can_process, add_ingredient_to_plate,
    get_ingredient_display_name
)
from .core.recipes import (
    Recipe, RecipeManager, RECIPE_DEFINITIONS,
    get_available_recipes, get_default_recipe
)
from .map.map_loader import MapLoader
from .environment.game_state import MapData
from .environment.observation_builder import ObservationBuilder
from .mechanics import MovementSystem, CollisionResolver
from .mechanics.interaction_handler import InteractionHandler

# For rendering
import pygame
from pygame import gfxdraw

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
        self.recipe_manager = RecipeManager()
        self.map_loader = MapLoader()
        self.observation_builder = ObservationBuilder(max_process_steps=MAX_PROCESS_STEPS)
        
        self.possible_agents = [f"agent_{i}" for i in range(num_fixed_agents)]
        self.agents = self.possible_agents[:]
        
        self._already_loaded_maps = set()
        self.state = self._initialize_game_state() 
        
        if not hasattr(self, 'map_size') or self.map_size is None:
            raise ValueError("Map size was not properly initialized")
        
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Interact
        self._action_to_direction = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        
        # Movement and Collision Systems (Phase 5)
        self.movement_system = MovementSystem(self._action_to_direction)
        self.collision_resolver = CollisionResolver(collision_penalty=0.5)
        
        # Interaction System (Phase 6)
        self.interaction_handler = InteractionHandler(self.item_manager, self.recipe_manager)
        
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
        """
        Load map using the new MapLoader system.
        Returns tuple of (static_map, agent_positions, dispensers).
        """
        is_reload = map_path in self._already_loaded_maps
        self._already_loaded_maps.add(map_path)
        
        if not is_reload or self.verbose:
            print(f"[ENV] Loading map from: {map_path}")
        
        # Use MapLoader to load and validate the map
        try:
            map_data = self.map_loader.load_map(map_path, use_default_on_error=False)
        except Exception as e:
            if self.verbose:
                print(f"[ENV] Error loading map: {e}")
                print(f"[ENV] Falling back to default map")
            # Fallback to default map
            map_data = self.map_loader.load_map()
        
        # Initialize recipes from map data
        self._initialize_recipes(map_data.recipes)
        
        if not is_reload or self.verbose:
            print(f"[ENV] Map loaded successfully:")
            print(f"  - Size: {map_data.size}")
            print(f"  - Agents: {list(map_data.agent_positions.keys())}")
            print(f"  - Dispensers: {len(map_data.dispensers)}")
            print(f"  - Recipes: {map_data.recipes}")
        
        return map_data.static_map, map_data.agent_positions, map_data.dispensers
            
        return static_map_np, agent_start_pos, dispenser_assignments
        
    def _initialize_recipes(self, recipe_names):
        """Initialize recipes based on the names from the map file."""
        
        # Store the order of recipes for bonus calculation
        self.recipe_order = []
        
        # Add recipes in the order specified in the map file
        for name in recipe_names:
            name = name.strip().upper()
            # Check if the recipe exists in RecipeManager
            recipe = self.recipe_manager.get_recipe_by_name(name)
            if recipe:
                self.recipe_order.append(name)
                if self.map_path not in self._already_loaded_maps or self.verbose:
                    print(f"Added recipe: {name}")
            else:
                if self.map_path not in self._already_loaded_maps or self.verbose:
                    print(f"WARNING: Recipe '{name}' not found in RecipeManager")
        
        # Make sure at least one recipe is defined (in case the map doesn't specify any)
        if len(self.recipe_order) == 0:
            default_recipe = get_default_recipe()
            if self.map_path not in self._already_loaded_maps or self.verbose:
                print(f"WARNING: No recipes were loaded from the map. Adding default {default_recipe} recipe.")
            
            recipe = self.recipe_manager.get_recipe_by_name(default_recipe)
            if recipe:
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
            
            carrying[agent] = 0  # 0 = not carrying anything (for backwards compatibility)
        
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
            "item_instances": {},        # Mapeia item_instance_id -> item_dict
            "agent_holding_instance": {},  # Mapeia agent_id -> item_instance_id
            "ground_instances": {},      # Mapeia (x, y) -> item_instance_id
            "next_item_id": 0,           # Counter to generate unique item IDs
            "agent_orientation": {agent: (0, -1) for agent in self.possible_agents},  # All agents start facing North
            "accumulated_rewards": {agent: 0.0 for agent in self.possible_agents},
            "last_step_rewards": {agent: 0.0 for agent in self.possible_agents}
        }
        return state

    def _get_obs(self):
        """
        Generates observations for all agents, encoding the game state
        in channels (N x M x TOTAL_CHANNELS).
        
        Uses ObservationBuilder to construct multi-channel observations following SRP.
        """
        return self.observation_builder.build_observations(
            self.state, 
            self.agents, 
            self.item_manager
        )

    def _get_info(self):
        """Returns the dictionary of auxiliary information."""
        info = {}
        for agent in self.agents:
            # Get the item instance that the agent is carrying
            carrying_instance_id = self.state["agent_holding_instance"].get(agent, None)
            carrying_item = self._get_item(carrying_instance_id) if carrying_instance_id is not None else None
            
            # Extract item information
            if carrying_item:
                item_display = get_ingredient_display_name(carrying_item)
                # For backwards compatibility, create a "code" based on display name hash
                item_code = hash(item_display) % 1000  # Arbitrary number for compatibility
            else:
                item_display = "NONE"
                item_code = 0
            
            agent_info = {
                "score": 0, 
                "dishes_remaining": self.state["dishes_pending"],
                "delivered_recipes": self.state.get("delivered_recipes", []),
                "recipe_order": self.recipe_order,
                "next_expected_recipe": self.recipe_order[self.state["expected_recipe_index"] % len(self.recipe_order)] if len(self.recipe_order) > 0 else "None",
                "inventory": item_display,
                "carrying_item_code": item_code,
                "carrying_item_data": carrying_item  # Complete item data
            }
            info[agent] = agent_info
            
        return info

    def reset(self, seed=None, options=None):
        """Starts a new episode."""
        gym.Env.reset(self, seed=seed) 
        
        # Make sure the map is marked as loaded to avoid duplicate logs
        if not self.map_path in self._already_loaded_maps:
            self._already_loaded_maps.add(self.map_path)
        
        # Reinitialize state and agent list
        self.state = self._initialize_game_state()
        self.agents = self.possible_agents[:]
        
        # Reset variables related to recipe order
        self.state["expected_recipe_index"] = 0
        self.state["delivered_recipes"] = []
        
        # Ensure rewards are reset to zero
        self.state["accumulated_rewards"] = {agent: 0.0 for agent in self.possible_agents}
        self.state["last_step_rewards"] = {agent: 0.0 for agent in self.possible_agents}
        
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
        
        # Apply base penalty for any action
        for agent_id in actions.keys():
            rewards[agent_id] -= 0.5
        
        # ============================================================
        # MOVEMENT PROCESSING (Using MovementSystem and CollisionResolver)
        # ============================================================
        
        # Step 1: Process movement actions and get intended positions
        intended_positions = self.movement_system.process_movement_actions(
            actions,
            self.state,
            self.map_size,
            self.item_manager
        )
        
        # Step 2: Resolve collisions and conflicts
        current_positions = {agent: pos for agent, pos in self.state["agent_pos"].items()}
        final_positions, collision_penalties = self.collision_resolver.resolve_collisions(
            intended_positions,
            current_positions,
            verbose=self.verbose
        )
        
        # Step 3: Apply collision penalties
        for agent_id, penalty in collision_penalties.items():
            if penalty > 0:
                rewards[agent_id] -= penalty
        
        # Step 4: Update agent positions in state
        for agent_id, pos in final_positions.items():
            self.state["agent_pos"][agent_id] = pos
        
        # Movement debugging (when verbose=True)
        if self.verbose:
            print("\n--- Movement Information ---")
            for agent_id, action in actions.items():
                if action < 4:  # Movement action
                    old_pos = current_positions.get(agent_id, (0, 0))
                    new_pos = final_positions.get(agent_id, (0, 0))
                    
                    info = self.movement_system.get_movement_info(
                        agent_id, action, old_pos, new_pos,
                        self.state, self.item_manager
                    )
                    print(info)
        
        # Process interaction actions (4)
        for agent_id, action in actions.items():
            if action == 4:  # Action 4: Interact
                local_reward, interaction_global_reward = self.interaction_handler.handle_interaction(
                    agent_id, self.state, verbose=self.verbose
                )
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
        
        # Accumulate rewards
        for agent_id in self.agents:
            self.state["accumulated_rewards"][agent_id] += rewards[agent_id]
        
        # Save current step rewards
        self.state["last_step_rewards"] = rewards.copy()
        
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
        """
        Renders the environment. Supports two modes:
        - "human": renders ASCII representation to console
        - "rgb_array": returns RGB array for visualization
        """
        if self.render_mode is None:
            return
            
        if self.render_mode == "human":
            return self._render_text()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _get_orientation_name(self, direction):
        """Converts direction vector to readable name."""
        direction_map = {
            (0, -1): "North",
            (0, 1): "South",
            (1, 0): "East",
            (-1, 0): "West"
        }
        return direction_map.get(direction, "Unknown")
    
    def _get_station_name(self, cell_code):
        """Converts cell code to readable name."""
        station_map = {
            self.item_manager.get_cell_code('C'): "Chopping Board",
            self.item_manager.get_cell_code('P'): "Plate Dispenser",
            self.item_manager.get_cell_code('D'): "Delivery",
            self.item_manager.get_cell_code('I'): "Ingredient Dispenser",
            self.item_manager.get_cell_code('T'): "Trash",
            self.item_manager.get_cell_code('#'): "Counter",
            self.item_manager.get_cell_code(' '): "Empty"
        }
        return station_map.get(cell_code, "Unknown")
    
    def _get_agent_facing_info(self, agent_id):
        """Returns information about what the agent is facing."""
        orientation = self.state["agent_orientation"][agent_id]
        orientation_name = self._get_orientation_name(orientation)
        
        # Get cell in front
        x, y = self.state["agent_pos"][agent_id]
        dx, dy = orientation
        front_x, front_y = x + dx, y + dy
        
        # Check if within bounds
        if 0 <= front_x < self.map_size[1] and 0 <= front_y < self.map_size[0]:
            cell_code = self.state["map_static"][front_y, front_x]
            station_name = self._get_station_name(cell_code)
            return f"{orientation_name} -> {station_name}"
        else:
            return f"{orientation_name} -> Wall"
    
    def _render_text(self):
        """Renders the environment as ASCII art to the console."""
        H, W = self.map_size
        grid = [[' ' for _ in range(W)] for _ in range(H)]
        
        # Original character map
        char_map = {
            self.item_manager.get_cell_code(' '): ' ',
            self.item_manager.get_cell_code('#'): '#',
            self.item_manager.get_cell_code('C'): 'C',
            self.item_manager.get_cell_code('P'): 'P',
            self.item_manager.get_cell_code('D'): 'D',
            self.item_manager.get_cell_code('I'): 'I',
            self.item_manager.get_cell_code('T'): 'T',
            self.item_manager.get_cell_code('|'): '|',
            self.item_manager.get_cell_code('_'): '_',
        }
        
        def get_item_char(item_dict):
            """Get character representation for an item."""
            if not item_dict:
                return ' '
            
            item_type = item_dict.get("type")
            
            if item_type == ItemType.INGREDIENT:
                name = item_dict["name"]
                state = item_dict.get("state", ItemState.RAW)
                
                # Map ingredients to letters
                name_map = {'tomato': 't', 'lettuce': 'l', 'cucumber': 'c'}
                char = name_map.get(name, '?')
                
                # Uppercase if READY, lowercase if RAW, number if CHOPPING
                if state == ItemState.READY:
                    return char.upper()
                elif state == ItemState.RAW:
                    return char.lower()
                elif state == ItemState.CHOPPING_1:
                    return '1'
                elif state == ItemState.CHOPPING_2:
                    return '2'
                elif state == ItemState.CHOPPING_3:
                    return '3'
                return char
                
            elif item_type == ItemType.DISH:
                ingredients = item_dict.get("ingredients", [])
                if not ingredients:
                    return 'p'  # empty plate
                else:
                    # Show first letter of first ingredient (e.g., 't' for tomato in plate)
                    return ingredients[0][0].lower()
            
            return '?'
        
        # Draw static map elements
        for y in range(H):
            for x in range(W):
                cell_code = self.state["map_static"][y, x]
                grid[y][x] = char_map.get(cell_code, '?')
                
                # Draw items on ground
                ground_instance_id = self.state["ground_instances"].get((x, y), None)
                if ground_instance_id is not None:
                    ground_item = self._get_item(ground_instance_id)
                    grid[y][x] = get_item_char(ground_item)
        
        # Draw agents
        for agent_id, pos in self.state["agent_pos"].items():
            x, y = pos
            agent_num = int(agent_id.split('_')[1])
            
            # Show agent number
            grid[y][x] = str(agent_num)
            
            # Check if agent is carrying something
            carrying_instance_id = self.state["agent_holding_instance"].get(agent_id, None)
            if carrying_instance_id is not None:
                carrying_item = self._get_item(carrying_instance_id)
                item_char = get_item_char(carrying_item)
                # Print what the agent is carrying next to the grid
                print(f"Agent {agent_num} is carrying: {get_ingredient_display_name(carrying_item)}")
        
        # Print the grid to console
        print("\n" + "-" * (W + 2))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("-" * (W + 2))
        
        # Print game state information
        print(f"Step: {self.state['time_steps']}")
        print(f"Remaining Dishes: {self.state['dishes_pending']}")
        
        # Show orders and delivered
        if len(self.recipe_order) > 0:
            # Calculate pending orders: original order minus delivered
            pending_orders = list(self.recipe_order)
            for delivered_recipe in self.state['delivered_recipes']:
                if delivered_recipe in pending_orders:
                    pending_orders.remove(delivered_recipe)  # Remove first occurrence
            
            if len(pending_orders) > 0:
                orders_str = ", ".join(pending_orders)
                print(f"Orders: {orders_str}")
            else:
                print(f"Orders: (none)")
            
            if len(self.state['delivered_recipes']) > 0:
                delivered_str = ", ".join(self.state['delivered_recipes'])
                print(f"Delivered: {delivered_str}")
            else:
                print(f"Delivered: (none)")
        else:
            print(f"Orders: (none)")
        
        # Print all agents' information
        print("\n--- Agent Status ---")
        for agent_id in self.agents:
            agent_pos = self.state["agent_pos"][agent_id]
            x, y = int(agent_pos[0]), int(agent_pos[1])
            
            # Get orientation and what's in front
            facing_info = self._get_agent_facing_info(agent_id)
            
            # Get inventory
            carrying_instance_id = self.state["agent_holding_instance"].get(agent_id)
            if carrying_instance_id:
                carrying_item = self._get_item(carrying_instance_id)
                inventory_str = get_ingredient_display_name(carrying_item)
            else:
                inventory_str = "Empty"
            
            # Get rewards
            acc_reward = self.state["accumulated_rewards"].get(agent_id, 0.0)
            step_reward = self.state["last_step_rewards"].get(agent_id, 0.0)
            
            print(f"  {agent_id} at ({x},{y} {facing_info}) - Accumulated: {acc_reward:.2f} - Step: {step_reward:.2f} - Inventory: {inventory_str}")
        
        return None

    def _render_rgb_array(self):
        """
        Renders the environment as an RGB array using Pygame.
        Returns a numpy array of shape (H, W, 3) with RGB values.
        """
        if self.window is None:
            pygame.init()
            pygame.display.init()
            
            H, W = self.map_size
            cell_size = 60  # Size of each cell in pixels
            
            # Extra space for info panel
            # Space for agents info + recipe progress (2 lines)
            agent_lines = len(self.possible_agents)
            recipe_lines = 2  # Two lines: Orders and Delivered
            self.info_panel_height = (30 * agent_lines) + (25 * recipe_lines)
            self.window_size = (W * cell_size, H * cell_size + self.info_panel_height)
            # Create actual Pygame window for visualization
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Overcooked MARL Environment")
            self.cell_size = cell_size
            self.clock = pygame.time.Clock()
            
            # Load or create assets
            self.colors = {
                'wall': (50, 50, 50),       # Dark gray
                'floor': (200, 200, 200),    # Light gray
                'chopping': (220, 170, 110), # Light brown
                'plates': (150, 150, 220),   # Light blue
                'delivery': (50, 180, 50),   # Green
                'counter': (180, 180, 150),  # Beige
                'agent0': (0, 0, 0),         # Black
                'agent1': (0, 0, 0),         # Black
                'tomato': (220, 50, 50),     # Red
                'lettuce': (50, 180, 50),    # Green
                'patty': (139, 69, 19),      # Brown
                'plate': (250, 250, 250),    # White
                'salad': (120, 200, 80),     # Light green
                'burger': (200, 150, 100),   # Tan
                'text': (0, 0, 0)            # Black
            }
        
        # Clear the surface
        self.window.fill((200, 200, 200))
        
        H, W = self.map_size
        
        # Draw static map elements
        for y in range(H):
            for x in range(W):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                cell_code = self.state["map_static"][y, x]
                
                # Draw cell background based on type
                if cell_code == self.item_manager.get_cell_code('#'):
                    pygame.draw.rect(self.window, self.colors['counter'], rect)
                    pygame.draw.rect(self.window, (100, 100, 100), rect, 1)  # Border
                elif cell_code == self.item_manager.get_cell_code('|'):
                    pygame.draw.rect(self.window, self.colors['wall'], rect)
                elif cell_code == self.item_manager.get_cell_code('_'):
                    pygame.draw.rect(self.window, self.colors['wall'], rect)
                elif cell_code == self.item_manager.get_cell_code(' '):
                    pygame.draw.rect(self.window, self.colors['floor'], rect)
                    pygame.draw.rect(self.window, (150, 150, 150), rect, 1)  # Grid lines
                elif cell_code == self.item_manager.get_cell_code('C'):
                    pygame.draw.rect(self.window, self.colors['chopping'], rect)
                    pygame.draw.rect(self.window, (100, 100, 100), rect, 2)  # Border
                    # Draw knife symbol
                    pygame.draw.line(self.window, (100, 100, 100), 
                                    (x * self.cell_size + 15, y * self.cell_size + 30),
                                    (x * self.cell_size + self.cell_size - 15, y * self.cell_size + 30), 3)
                elif cell_code == self.item_manager.get_cell_code('P'):
                    pygame.draw.rect(self.window, self.colors['plates'], rect)
                    # Draw plate symbol (black)
                    pygame.draw.circle(self.window, (0, 0, 0), 
                                    (x * self.cell_size + self.cell_size//2, y * self.cell_size + self.cell_size//2), 
                                    self.cell_size//3, 2)
                elif cell_code == self.item_manager.get_cell_code('D'):
                    pygame.draw.rect(self.window, self.colors['delivery'], rect)
                elif cell_code in [self.item_manager.get_cell_code('T'), self.item_manager.get_cell_code('I')]:
                    pygame.draw.rect(self.window, self.colors['counter'], rect)
                    pygame.draw.rect(self.window, (100, 100, 100), rect, 1)  # Border
                    
                    # Draw X on trash station
                    if cell_code == self.item_manager.get_cell_code('T'):
                        padding = self.cell_size // 4
                        pygame.draw.line(self.window, (0, 0, 0),
                                       (x * self.cell_size + padding, y * self.cell_size + padding),
                                       (x * self.cell_size + self.cell_size - padding, 
                                        y * self.cell_size + self.cell_size - padding), 4)
                        pygame.draw.line(self.window, (0, 0, 0),
                                       (x * self.cell_size + self.cell_size - padding, y * self.cell_size + padding),
                                       (x * self.cell_size + padding, 
                                        y * self.cell_size + self.cell_size - padding), 4)
                
                # Draw visual indicators on stations
                center_x = x * self.cell_size + self.cell_size // 2
                center_y = y * self.cell_size + self.cell_size // 2
                
                # Dispenser indicators
                if (x, y) in self.state["dispensers"]:
                    dispenser_data = self.state["dispensers"][(x, y)]
                    
                    # Get ingredient name
                    if isinstance(dispenser_data, str):
                        ingredient_name = dispenser_data
                    else:
                        ingredient_map = {5: "tomato", 6: "lettuce", 7: "cucumber"}
                        ingredient_name = ingredient_map.get(dispenser_data, "tomato")
                    
                    # Color mapping for dispensers
                    dispenser_colors = {
                        "tomato": (220, 50, 50),      # Red
                        "lettuce": (50, 180, 50),     # Green
                        "cucumber": (255, 200, 0),    # Yellow
                    }
                    
                    indicator_color = dispenser_colors.get(ingredient_name, (150, 150, 150))
                    indicator_radius = self.cell_size // 4
                    
                    pygame.draw.circle(self.window, indicator_color, 
                                      (center_x, center_y), indicator_radius)
                
                # Plate dispenser indicator
                if cell_code == self.item_manager.get_cell_code('P'):
                    indicator_radius = self.cell_size // 4
                    pygame.draw.circle(self.window, (250, 250, 250), 
                                      (center_x, center_y), indicator_radius)
                
                # Chopping board with ingredient on it
                if cell_code == self.item_manager.get_cell_code('C'):
                    # Check if there's an item on this chopping board
                    ground_instance_id = self.state["ground_instances"].get((x, y), None)
                    if ground_instance_id is not None:
                        ground_item = self._get_item(ground_instance_id)
                        if ground_item and ground_item["type"] == ItemType.INGREDIENT:
                            # Draw ingredient on chopping board
                            item_color = self._get_item_color(ground_item)
                            item_radius = self.cell_size // 3
                            
                            pygame.draw.circle(self.window, item_color,
                                             (center_x, center_y), item_radius)
                            
                            # Draw progress indicator (small lines)
                            cuts_done = ground_item.get("cuts_done", 0)
                            for i in range(cuts_done):
                                line_x = center_x - 10 + (i * 10)
                                pygame.draw.line(self.window, (50, 50, 50),
                                               (line_x, center_y - 15),
                                               (line_x, center_y + 15), 2)
                
                # Delivery station arrow (simple triangle pointing right)
                if cell_code == self.item_manager.get_cell_code('D'):
                    arrow_size = self.cell_size // 4
                    
                    # Simple triangle arrow pointing right (→)
                    arrow_points = [
                        (center_x - arrow_size, center_y - arrow_size),  # Top left
                        (center_x - arrow_size, center_y + arrow_size),  # Bottom left
                        (center_x + arrow_size, center_y),               # Right tip
                    ]
                    
                    pygame.draw.polygon(self.window, (255, 255, 255), arrow_points)
                    pygame.draw.polygon(self.window, (100, 100, 100), arrow_points, 2)
                
                # Draw items on ground
                ground_instance_id = self.state["ground_instances"].get((x, y), None)
                if ground_instance_id is not None:
                    ground_item = self._get_item(ground_instance_id)
                    self._draw_item(x, y, ground_item)
        
        # Draw agents
        for agent_id, pos in self.state["agent_pos"].items():
            x, y = pos
            agent_num = int(agent_id.split('_')[1])  # Extract number from agent_id
            
            # Draw agent circle
            agent_center = (x * self.cell_size + self.cell_size//2, y * self.cell_size + self.cell_size//2)
            agent_radius = self.cell_size//3
            
            pygame.draw.circle(
                self.window, 
                self.colors[f'agent{agent_num}'], 
                agent_center,
                agent_radius
            )
            
            # Draw carried item as smaller circle inside agent (if any)
            carrying_instance_id = self.state["agent_holding_instance"].get(agent_id, None)
            if carrying_instance_id is not None:
                carrying_item = self._get_item(carrying_instance_id)
                if carrying_item:
                    item_radius = agent_radius // 2  # Small circle half the size of the agent
                    item_color = self._get_item_color(carrying_item)
                    
                    # Draw item circle inside the agent
                    pygame.draw.circle(
                        self.window,
                        item_color,
                        agent_center,
                        item_radius
                    )
            
            # Draw agent ID on top
            font = pygame.font.SysFont(None, 20)
            text = font.render(str(agent_num), True, (255, 255, 255))
            text_rect = text.get_rect(center=agent_center)
            self.window.blit(text, text_rect)
        
        # Draw game info at the bottom
        H, W = self.map_size
        info_panel_y = H * self.cell_size
        
        # Draw info panel background
        info_panel_rect = pygame.Rect(0, info_panel_y, W * self.cell_size, self.info_panel_height)
        pygame.draw.rect(self.window, (240, 240, 240), info_panel_rect)
        pygame.draw.line(self.window, (100, 100, 100), (0, info_panel_y), (W * self.cell_size, info_panel_y), 2)
        
        # Render agent information (one line per agent)
        font = pygame.font.SysFont(None, 18)
        line_height = 30
        
        for idx, agent_id in enumerate(sorted(self.agents)):
            # Get agent information
            step_reward = self.state["last_step_rewards"].get(agent_id, 0.0)
            total_reward = self.state["accumulated_rewards"].get(agent_id, 0.0)
            
            # Get inventory - compact version
            carrying_instance_id = self.state["agent_holding_instance"].get(agent_id, None)
            if carrying_instance_id is not None:
                carrying_item = self._get_item(carrying_instance_id)
                item_type = carrying_item.get("type")
                
                if item_type == ItemType.INGREDIENT:
                    # For ingredients: name + state
                    name = carrying_item["name"].capitalize()
                    state = carrying_item.get("state", ItemState.RAW)
                    if state == ItemState.RAW:
                        inventory = f"{name}(raw)"
                    elif state == ItemState.READY:
                        inventory = f"{name}(ok)"
                    else:
                        cuts = carrying_item["cuts_done"]
                        inventory = f"{name}({cuts}/3)"
                        
                elif item_type == ItemType.DISH:
                    # For plates: show ingredient initials
                    ingredients = carrying_item.get("ingredients", [])
                    if not ingredients:
                        inventory = "Plate(empty)"
                    else:
                        # Use ingredient initials: t=tomato, l=lettuce, m=meat, o=onion
                        initials = "".join([ing[0].upper() for ing in ingredients])
                        inventory = f"Plate({initials})"
                else:
                    inventory = str(carrying_item.get("name", "?"))
            else:
                inventory = "Empty"
            
            # Format text in compact form: "A0 | T: 10.00 | S: 0.50 | I: Tomato"
            agent_num = int(agent_id.split('_')[1])
            info_text = f"A{agent_num} | T:{total_reward:7.2f} | S:{step_reward:6.2f} | I: {inventory}"
            
            # Render text
            text_surface = font.render(info_text, True, (0, 0, 0))
            text_y = info_panel_y + 5 + (idx * line_height)
            self.window.blit(text_surface, (5, text_y))
        
        # Render recipe progress below agent info
        recipe_start_y = info_panel_y + (len(self.agents) * 30) + 5
        
        # Draw separator line
        separator_y = recipe_start_y - 3
        pygame.draw.line(self.window, (150, 150, 150), 
                        (0, separator_y), (W * self.cell_size, separator_y), 1)
        
        # Line 1: Orders (pending only)
        if len(self.recipe_order) > 0:
            # Calculate pending orders: original order minus delivered
            pending_orders = list(self.recipe_order)
            for delivered_recipe in self.state['delivered_recipes']:
                if delivered_recipe in pending_orders:
                    pending_orders.remove(delivered_recipe)  # Remove first occurrence
            
            if len(pending_orders) > 0:
                orders_str = ", ".join(pending_orders)
                orders_text = f"Orders: {orders_str}"
            else:
                orders_text = "Orders: (none)"
        else:
            orders_text = "Orders: (none)"
        
        text_surface = font.render(orders_text, True, (0, 0, 0))
        self.window.blit(text_surface, (5, recipe_start_y))
        
        # Line 2: Delivered
        if len(self.state['delivered_recipes']) > 0:
            delivered_str = ", ".join(self.state['delivered_recipes'])
            delivered_text = f"Delivered: {delivered_str}"
        else:
            delivered_text = "Delivered: (none)"
        
        text_surface = font.render(delivered_text, True, (0, 0, 0))
        self.window.blit(text_surface, (5, recipe_start_y + 20))
        
        # Update the Pygame window
        pygame.display.flip()
        
        # Process Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        # Convert Pygame surface to numpy array
        rgb_array = pygame.surfarray.array3d(self.window)
        return np.transpose(rgb_array, (1, 0, 2))
    
    def _get_item_color(self, item_dict):
        """Returns the color corresponding to the item dictionary."""
        if not item_dict:
            return (128, 128, 128)  # Gray for empty item
        
        item_type = item_dict.get("type")
        
        if item_type == ItemType.INGREDIENT:
            name = item_dict["name"]
            state = item_dict.get("state", ItemState.RAW)
            
            # Base colors for each ingredient
            base_colors = {
                "tomato": (220, 50, 50),    # Red
                "lettuce": (50, 180, 50),   # Green
                "cucumber": (255, 200, 0),  # Yellow
            }
            
            base_color = base_colors.get(name, (150, 150, 150))
            
            # Adjust brightness based on state
            if state == ItemState.RAW:
                # Darker for RAW
                return tuple(int(c * 0.6) for c in base_color)
            elif state in [ItemState.CHOPPING_1, ItemState.CHOPPING_2, ItemState.CHOPPING_3]:
                # Gradient from dark to light
                if state == ItemState.CHOPPING_1:
                    factor = 0.7
                elif state == ItemState.CHOPPING_2:
                    factor = 0.85
                else:
                    factor = 0.95
                return tuple(int(c * factor) for c in base_color)
            elif state == ItemState.READY:
                # Full color for READY
                return base_color
            
            return base_color
        
        elif item_type == ItemType.DISH:
            ingredients = item_dict.get("ingredients", [])
            if not ingredients:
                return (250, 250, 250)  # White (empty plate)
            elif len(ingredients) == 1:
                # Color based on first ingredient
                ing = ingredients[0]
                if ing == "tomato":
                    return (255, 150, 150)
                elif ing == "lettuce":
                    return (150, 255, 150)
                elif ing == "meat":
                    return (200, 150, 100)
                elif ing == "onion":
                    return (255, 255, 150)
            else:
                # Multiple ingredients - mixed color
                return (120, 200, 80)  # Green salad
        
        return (200, 200, 200)  # Default color
    
    def _draw_item(self, x, y, item_dict, offset=0):
        """Helper method to draw an item dict at the specified position."""
        if not item_dict:
            return
        
        item_size = self.cell_size // 3
        center_x = x * self.cell_size + self.cell_size//2
        center_y = y * self.cell_size + self.cell_size//2
        
        item_type = item_dict.get("type")
        item_color = self._get_item_color(item_dict)
        
        if item_type == ItemType.INGREDIENT:
            # Draw circle with color based on state
            pygame.draw.circle(self.window, item_color, 
                              (center_x, center_y - offset), item_size)
            
            # If not READY, add border indicating it needs processing
            state = item_dict.get("state")
            if state != ItemState.READY:
                pygame.draw.circle(self.window, (0, 0, 0), 
                                  (center_x, center_y - offset), item_size, 2)
                
                # Show cutting progress with small lines
                cuts_done = item_dict.get("cuts_done", 0)
                cuts_needed = item_dict.get("cuts_needed", 3)
                if cuts_done > 0:
                    for i in range(cuts_done):
                        angle_offset = (i - cuts_done/2) * 40  # Spread the lines
                        line_length = item_size // 2
                        start_x = center_x + int(line_length * 0.3 * (i - 1))
                        end_x = center_x + int(line_length * 0.3 * (i - 1))
                        pygame.draw.line(self.window, (50, 50, 50),
                                       (start_x, center_y-offset-line_length//2),
                                       (end_x, center_y-offset+line_length//2), 2)
        
        elif item_type == ItemType.DISH:
            ingredients = item_dict.get("ingredients", [])
            
            # Draw plate (white circle with border)
            pygame.draw.circle(self.window, (250, 250, 250), 
                              (center_x, center_y - offset), item_size)
            pygame.draw.circle(self.window, (200, 200, 200), 
                              (center_x, center_y - offset), item_size, 2)
            
            # If it has ingredients, draw small colored circles inside
            if ingredients:
                num_ingredients = len(ingredients)
                for i, ing_name in enumerate(ingredients):
                    # Create a fake ingredient dictionary to get the color
                    fake_item = {"name": ing_name, "type": ItemType.INGREDIENT, "state": ItemState.READY}
                    ing_color = self._get_item_color(fake_item)
                    
                    # Position ingredients in a circle inside the plate
                    if num_ingredients == 1:
                        ing_x, ing_y = center_x, center_y - offset
                    else:
                        angle = (i / num_ingredients) * 2 * 3.14159
                        radius = item_size // 3
                        ing_x = center_x + int(radius * (i - num_ingredients/2) * 0.6)
                        ing_y = center_y - offset
                    
                    pygame.draw.circle(self.window, ing_color, 
                                      (ing_x, ing_y), item_size//4)

    # Function required by ParallelEnv
    def close(self):
        """Closes any open rendering windows."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    # Function required by ParallelEnv
    def observation_space(self, agent):
        """Returns the observation space for a specific agent."""
        return self.observation_spaces[agent]
    
    # Function required by ParallelEnv
    def action_space(self, agent):
        """Returns the action space for a specific agent."""
        return self.action_spaces[agent]
    
    def _create_item_instance(self, item_dict):
        """
        Creates a new item instance with unique ID.
        
        Args:
            item_dict: Dictionary representing the item
        
        Returns:
            Unique ID of the created instance
        """
        item_id = self.state["next_item_id"]
        self.state["next_item_id"] += 1
        self.state["item_instances"][item_id] = item_dict.copy()
        return item_id
    
    def _get_item(self, item_instance_id):
        """
        Returns the complete item dictionary.
        
        Args:
            item_instance_id: Instance ID
        
        Returns:
            Item dictionary or None if it doesn't exist
        """
        return self.state["item_instances"].get(item_instance_id)
    
    def _update_item(self, item_instance_id, new_item_dict):
        """
        Updates an item completely.
        
        Args:
            item_instance_id: Instance ID
            new_item_dict: New item dictionary
        """
        if item_instance_id in self.state["item_instances"]:
            self.state["item_instances"][item_instance_id] = new_item_dict.copy()
    
    def _delete_item_instance(self, item_instance_id):
        """Removes an item instance from the game."""
        if item_instance_id in self.state["item_instances"]:
            del self.state["item_instances"][item_instance_id]
    
    def _get_tile_in_front(self, agent_id):
        """Returns the position in front of the agent based on its orientation."""
        agent_pos = self.state["agent_pos"][agent_id]
        x, y = agent_pos
        
        # First, check if there's an item at the agent's current position
        if self.state["items_on_ground"][y, x] > 0:
            return agent_pos
        
        # Get agent orientation
        dx, dy = self.state["agent_orientation"][agent_id]
        
        # Calculate position in front
        new_x, new_y = x + dx, y + dy
        
        # Check if within bounds
        if 0 <= new_x < self.map_size[1] and 0 <= new_y < self.map_size[0]:
            return (new_x, new_y)
        
        # If out of bounds, return current position
        return agent_pos
