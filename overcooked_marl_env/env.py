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
        
        self.possible_agents = [f"agent_{i}" for i in range(num_fixed_agents)]
        self.agents = self.possible_agents[:]
        
        self._already_loaded_maps = set()
        self.state = self._initialize_game_state() 
        
        if not hasattr(self, 'map_size') or self.map_size is None:
            raise ValueError("Map size was not properly initialized")
        
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Interact
        self._action_to_direction = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        
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

        # --- 1. Process Grid Map and Agents ---
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
            if dispenser_section:
                parts = line_stripped.split()
                if len(parts) >= 3:
                    # New format: name x y (e.g., "tomato 1 4")
                    if not parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                        ingredient_name = parts[0].lower()
                        x, y = int(parts[1]), int(parts[2])
                        dispenser_assignments[(x, y)] = ingredient_name
                        if not is_reload or self.verbose:
                            print(f"Added dispenser with ingredient '{ingredient_name}' at position ({x}, {y})")
                    # Legacy format: ID x y (e.g., "5 1 4")
                    elif all(p.isdigit() for p in parts[:3]):
                        raw_item_id = int(parts[0])
                        x, y = int(parts[1]), int(parts[2])
                        dispenser_assignments[(x, y)] = raw_item_id
                        if not is_reload or self.verbose:
                            print(f"Added dispenser with item ID {raw_item_id} at position ({x}, {y}) [legacy]")
                    
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
            
            # Get the item instance that the agent is carrying
            carrying_instance_id = self.state["agent_holding_instance"].get(agent_id, None)
            carrying_item = self._get_item(carrying_instance_id) if carrying_instance_id is not None else None
            
            # Convert item to progress vector
            progress_vector = [0, 0, 0]
            if carrying_item:
                item_type = carrying_item.get("type")
                if item_type == ItemType.INGREDIENT:
                    # Vector based on cutting progress
                    cuts_done = carrying_item.get("cuts_done", 0)
                    cuts_needed = carrying_item.get("cuts_needed", 3)
                    
                    if cuts_done == 0:
                        progress_vector = [1, 0, 0]  # RAW
                    elif cuts_done < cuts_needed:
                        progress_vector = [0, 1, 0]  # CHOPPING
                    else:
                        progress_vector = [0, 0, 1]  # READY
                        
                elif item_type == ItemType.DISH:
                    # Plate: vector based on number of ingredients
                    num_ingredients = len(carrying_item.get("ingredients", []))
                    if num_ingredients == 0:
                        progress_vector = [1, 0, 0]  # Empty
                    elif num_ingredients == 1:
                        progress_vector = [0, 1, 0]  # 1 ingredient
                    else:
                        progress_vector = [0, 0, 1]  # 2+ ingredients (recipe)
            
            # Encodes the vector in layers 5, 6 and 7 (at the agent's position)
            for i in range(NUM_VECTOR_CHANNELS):
                obs[ay, ax, 5 + i] = progress_vector[i] 
            
            observations[agent_id] = obs
            
        return observations

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
        
        # Simplified approach: process all movements in a simpler way
        # Step 1: Collect all movement intentions
        intended_positions = {}
        
        for agent_id, action in actions.items():
            rewards[agent_id] -= 0.01  # Base penalty for any action
            
            # Process only movement actions (0-3)
            if action < 4:  
                dx, dy = self._action_to_direction[action]
                
                # ALWAYS update orientation based on action
                self.state["agent_orientation"][agent_id] = (dx, dy)
                
                current_pos = self.state["agent_pos"][agent_id]
                current_x, current_y = current_pos
                
                new_x, new_y = current_x + dx, current_y + dy
                max_x, max_y = self.map_size[1] - 1, self.map_size[0] - 1
                new_x, new_y = np.clip(new_x, 0, max_x), np.clip(new_y, 0, max_y)
                
                # Check collisions with walls/stations
                cell_type = self.state["map_static"][new_y, new_x]
                space_code = self.item_manager.get_cell_code(' ')  # Code for empty space
                
                if cell_type == space_code:  # Can only walk on empty spaces
                    intended_positions[agent_id] = (new_x, new_y)
                else:
                    # Keep current position if collided with wall/station
                    intended_positions[agent_id] = current_pos
                    rewards[agent_id] -= 0.005  # Collision penalty
        
        # Step 2: Find and resolve conflicts
        current_positions = {agent: pos for agent, pos in self.state["agent_pos"].items()}
        final_positions = current_positions.copy()  # Start with current positions
        temp_positions = {}  # To store intermediate positions before resolving conflicts
        
        # First check if intended positions don't collide with other current agents
        for agent_id, intended_pos in intended_positions.items():
            # Check if another agent is already at the intended position
            other_agent_there = any(pos == intended_pos for other_id, pos in current_positions.items() 
                                   if other_id != agent_id)
            if not other_agent_there:
                temp_positions[agent_id] = intended_pos
            else:
                # If an agent already exists in the cell, we can't move there
                temp_positions[agent_id] = current_positions[agent_id]
                rewards[agent_id] -= 0.005  # Penalty for invalid movement attempt
                if self.verbose:
                    print(f"Agent {agent_id} cannot move to {intended_pos} because another agent is already at that position")
                
        # Detect conflicts - two agents trying to go to the same space
        position_agents = {}  # Map of position -> list of agents trying to go there
        
        for agent_id, pos in temp_positions.items():
            # Only consider conflicts for positions that are not current agent positions
            if pos != current_positions[agent_id]:
                if pos in position_agents:
                    position_agents[pos].append(agent_id)
                else:
                    position_agents[pos] = [agent_id]
        
        # Resolver conflitos
        for pos, agents in position_agents.items():
            if len(agents) == 1:  # No conflict
                agent_id = agents[0]
                final_positions[agent_id] = pos  # Allow movement
            else:
                # Conflict: no agent moves
                for agent_id in agents:
                    rewards[agent_id] -= 0.005  # Additional penalty for conflict
                    # Keep current position
                    final_positions[agent_id] = current_positions[agent_id]
                if self.verbose:
                    print(f"Movement conflict: agents {agents} tried to go to {pos}, none moved")
        
        # Step 3: Update agent positions
        for agent_id, pos in final_positions.items():
            self.state["agent_pos"][agent_id] = pos
        
        # Movement debugging (when verbose=True)
        if self.verbose:
            print("\n--- Movement Information ---")
            for agent_id, action in actions.items():
                if action < 4:
                    old_pos = current_positions.get(agent_id, (0, 0))
                    new_pos = final_positions.get(agent_id, (0, 0))
                    dx, dy = self._action_to_direction.get(action, (0, 0))
                    intended_pos = (old_pos[0] + dx, old_pos[1] + dy)
                    
                    # Convert to regular int before printing (to avoid np.int64 in logs)
                    old_pos_print = (int(old_pos[0]), int(old_pos[1]))
                    new_pos_print = (int(new_pos[0]), int(new_pos[1]))
                    intended_pos_print = (int(intended_pos[0]), int(intended_pos[1]))
                    
                    if old_pos != new_pos:
                        print(f"Agent {agent_id} moved from {old_pos_print} to {new_pos_print}")
                    else:
                        if intended_pos != old_pos:
                            # Check why it failed
                            cell_type = self.state["map_static"][intended_pos[1], intended_pos[0]]
                            if cell_type != self.item_manager.get_cell_code(' '):
                                print(f"Agent {agent_id} did not move to {intended_pos_print} because the cell is not empty space")
                            else:
                                other_agent_there = any(pos == intended_pos for other_id, pos in current_positions.items() if other_id != agent_id)
                                if other_agent_there:
                                    print(f"Agent {agent_id} did not move to {intended_pos_print} because another agent is already there")
                                else:
                                    print(f"Agent {agent_id} did not move to {intended_pos_print} due to movement conflict")
                        else:
                            print(f"Agent {agent_id} did not attempt to move (invalid action)")
        
        # Process interaction actions (4)
        for agent_id, action in actions.items():
            if action == 4:  # Action 4: Interact
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
                name_map = {'tomato': 't', 'lettuce': 'l', 'meat': 'm', 'onion': 'o'}
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
        print(f"Next Recipe: {self.recipe_order[self.state['expected_recipe_index']] if self.state['expected_recipe_index'] < len(self.recipe_order) else 'None'}")
        print(f"Delivered: {self.state['delivered_recipes']}")
        
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
            
            self.window_size = (W * cell_size, H * cell_size)
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
                    # Draw plate symbol
                    pygame.draw.circle(self.window, (250, 250, 250), 
                                    (x * self.cell_size + self.cell_size//2, y * self.cell_size + self.cell_size//2), 
                                    self.cell_size//3, 2)
                elif cell_code == self.item_manager.get_cell_code('D'):
                    pygame.draw.rect(self.window, self.colors['delivery'], rect)
                    # Draw serving symbol
                    pygame.draw.polygon(self.window, (250, 250, 250), [
                        (x * self.cell_size + 20, y * self.cell_size + 20),
                        (x * self.cell_size + self.cell_size//2, y * self.cell_size + 10),
                        (x * self.cell_size + self.cell_size - 20, y * self.cell_size + 20),
                        (x * self.cell_size + self.cell_size//2, y * self.cell_size + self.cell_size - 10)
                    ], 2)
                elif cell_code in [self.item_manager.get_cell_code('T'), self.item_manager.get_cell_code('I')]:
                    pygame.draw.rect(self.window, self.colors['counter'], rect)
                    pygame.draw.rect(self.window, (100, 100, 100), rect, 1)  # Border
                
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
        font = pygame.font.SysFont(None, 24)
        
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
                "meat": (139, 69, 19),      # Brown
                "onion": (200, 200, 100),   # Light yellow
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

    def _handle_interaction(self, agent_id):
        """Executes the logic of Action 4 (Interact) using new item system."""
        
        ix, iy = self._get_tile_in_front(agent_id)
        
        # Get the item instance that the agent is carrying
        carrying_instance_id = self.state["agent_holding_instance"].get(agent_id, None)
        carrying_item = self._get_item(carrying_instance_id) if carrying_instance_id is not None else None
        
        # Get the item instance on the ground
        ground_instance_id = self.state["ground_instances"].get((ix, iy), None)
        ground_item = self._get_item(ground_instance_id) if ground_instance_id is not None else None
        
        station_type = self.state["map_static"][iy, ix] 
        
        if self.verbose:
            ix_print, iy_print = int(ix), int(iy)
            print(f"Agent {agent_id} interacting at position ({ix_print}, {iy_print})")
            print(f"  Station type: {station_type}")
            if carrying_item:
                print(f"  Carrying: {get_ingredient_display_name(carrying_item)} (instance: {carrying_instance_id})")
            else:
                print(f"  Carrying: nothing")
            if ground_item:
                print(f"  Ground: {get_ingredient_display_name(ground_item)} (instance: {ground_instance_id})")
            else:
                print(f"  Ground: nothing")
        
        reward = -0.005
        global_reward_bonus = 0.0

        # Processing Logic (chopping board C)
        if station_type == self.item_manager.get_cell_code('C'):
            if carrying_item and can_process(carrying_item):
                # Advance processing
                updated_item = advance_processing(carrying_item)
                self._update_item(carrying_instance_id, updated_item)
                
                if self.verbose:
                    cuts_done = updated_item["cuts_done"]
                    cuts_needed = updated_item["cuts_needed"]
                    print(f"  Cutting... {cuts_done}/{cuts_needed} cuts")
                
                # Check if processing is complete
                if is_ready(updated_item):
                    reward += 1.0  # Completed processing
                    if self.verbose:
                        print(f"  Item processed and ready to use!")
                else:
                    reward += 0.1  # Partial progress
                
                return reward, 0.0
        
        # Plate Dispenser (P)
        if station_type == self.item_manager.get_cell_code('P'):
            # If not carrying anything, pick up an empty plate
            if not carrying_item:
                plate = create_plate()
                plate_instance = self._create_item_instance(plate)
                self.state["agent_holding_instance"][agent_id] = plate_instance
                reward += 0.5
                if self.verbose:
                    print(f"  Picked up an empty plate from dispenser (instance: {plate_instance})")
                return reward, 0.0
            
            # If carrying a READY ingredient, pick up plate and combine automatically
            elif carrying_item and is_ready(carrying_item):
                plate = create_plate()
                plate = add_ingredient_to_plate(plate, carrying_item["name"])
                plate_instance = self._create_item_instance(plate)
                
                # Delete the ingredient that was combined
                self._delete_item_instance(carrying_instance_id)
                del self.state["agent_holding_instance"][agent_id]
                
                # Now carry the plate
                self.state["agent_holding_instance"][agent_id] = plate_instance
                reward += 0.5 + 2.0  # Pick up + combine
                if self.verbose:
                    print(f"  Picked up plate and combined with {carrying_item['name']} (instance: {plate_instance})")
                return reward, 0.0
        
        # Item Collection Logic (dynamic dispensers - defined in map)
        # Only dispense if agent is not carrying anything
        if (ix, iy) in self.state["dispensers"] and not carrying_item:
            dispenser_data = self.state["dispensers"][(ix, iy)]
            
            # Support both name (string) and numeric ID (legacy)
            if isinstance(dispenser_data, str):
                # New system: ingredient name
                ingredient = self.item_manager.create_ingredient_by_name(dispenser_data)
            else:
                # Legacy: numeric ID - map to ingredient
                ingredient_map = {
                    5: "tomato",
                    6: "lettuce",
                    7: "onion",
                    8: "meat",
                }
                ingredient_name = ingredient_map.get(dispenser_data, "tomato")
                ingredient = self.item_manager.create_ingredient_by_name(ingredient_name)
            
            item_instance = self._create_item_instance(ingredient)
            self.state["agent_holding_instance"][agent_id] = item_instance
            reward += 0.5
            if self.verbose:
                print(f"  Picked up {ingredient['name']} from dispenser (instance: {item_instance})")
            return reward, 0.0
        
        # Counter and Combination (counters/tables)
        # Treat 'T', '#', and 'I' as counters to allow placing/picking up items
        elif station_type in [
            self.item_manager.get_cell_code('T'),
            self.item_manager.get_cell_code('#'),
            self.item_manager.get_cell_code('I')
        ]:
            is_carrying_dish = carrying_item and carrying_item.get("type") == ItemType.DISH
            is_carrying_ready = carrying_item and is_ready(carrying_item)
            is_ground_dish = ground_item and ground_item.get("type") == ItemType.DISH
            is_ground_ready = ground_item and is_ready(ground_item)
            
            # Scenario 1: Carrying READY item + plate on counter → pick up plate and combine
            if is_carrying_ready and is_ground_dish:
                # Add ingredient to plate
                updated_plate = add_ingredient_to_plate(ground_item, carrying_item["name"])
                
                # Deletar ingrediente que foi combinado
                self._delete_item_instance(carrying_instance_id)
                del self.state["agent_holding_instance"][agent_id]
                
                # Agent now carries the updated plate
                self._update_item(ground_instance_id, updated_plate)
                self.state["agent_holding_instance"][agent_id] = ground_instance_id
                del self.state["ground_instances"][(ix, iy)]
                
                reward += 2.0
                if self.verbose:
                    print(f"  Combined {carrying_item['name']} with plate from counter")
                return reward, 0.0
            
            # Scenario 2: Carrying plate + READY item on counter → pick up item and combine
            elif is_carrying_dish and is_ground_ready:
                # Add ingredient to plate
                updated_plate = add_ingredient_to_plate(carrying_item, ground_item["name"])
                self._update_item(carrying_instance_id, updated_plate)
                
                # Deletar ingrediente que foi combinado
                self._delete_item_instance(ground_instance_id)
                del self.state["ground_instances"][(ix, iy)]
                
                reward += 2.0
                if self.verbose:
                    print(f"  Combined {ground_item['name']} from counter with plate")
                return reward, 0.0
            
            # Scenario 3 & 4: Place item/plate on counter (only if counter is empty)
            elif carrying_item and not ground_item:
                # Validation: if placing plate, need to make sure it's allowed
                if is_carrying_dish:
                    # Plate can only go on empty counter
                    pass  # OK, counter is empty
                
                # Transfer instance to counter
                self.state["ground_instances"][(ix, iy)] = carrying_instance_id
                del self.state["agent_holding_instance"][agent_id]
                
                reward += 0.2
                if self.verbose:
                    print(f"  Placed {get_ingredient_display_name(carrying_item)} on counter")
                return reward, 0.0
            
            # Scenario 5: Pick up item/plate from counter (if hands are empty)
            elif not carrying_item and ground_item:
                # Transfer instance to agent
                self.state["agent_holding_instance"][agent_id] = ground_instance_id
                del self.state["ground_instances"][(ix, iy)]
                
                reward += 0.2
                if self.verbose:
                    print(f"  Picked up {get_ingredient_display_name(ground_item)} from counter")
                return reward, 0.0
        
        # Delivery
        elif station_type == self.item_manager.get_cell_code('D'):
            # Can only deliver if carrying a plate
            if not carrying_item or carrying_item.get("type") != ItemType.DISH:
                if self.verbose:
                    print(f"  Cannot deliver: not carrying a plate")
                return reward, 0.0
            
            # Check if plate has ingredients
            ingredients = carrying_item.get("ingredients", [])
            if not ingredients:
                if self.verbose:
                    print(f"  Cannot deliver: plate is empty")
                return reward, 0.0
            
            # Validate if ingredients form a valid recipe
            is_valid, recipe_name, recipe_reward = self.recipe_manager.validate_delivery(ingredients)
            
            if is_valid:
                # Valid recipe!
                # Check if delivered in correct order
                is_correct_order = False
                if len(self.recipe_order) > 0:
                    expected_index = self.state["expected_recipe_index"] % len(self.recipe_order)
                    expected_recipe = self.recipe_order[expected_index]
                    is_correct_order = (recipe_name == expected_recipe)
                    self.state["expected_recipe_index"] += 1
                
                # Register delivery
                self.state["delivered_recipes"].append(recipe_name)
                self.state["dishes_pending"] -= 1
                
                # Delete delivered plate instance
                self._delete_item_instance(carrying_instance_id)
                del self.state["agent_holding_instance"][agent_id]
                
                # Calculate rewards
                base_reward = recipe_reward
                order_bonus = recipe_reward if is_correct_order else 0.0
                global_reward_bonus = recipe_reward + order_bonus
                reward += base_reward
                
                if self.verbose:
                    if is_correct_order:
                        print(f"  Recipe {recipe_name} delivered in correct order!")
                        print(f"  Reward: {base_reward} + Bonus: {order_bonus}")
                    else:
                        print(f"  Recipe {recipe_name} delivered out of order")
                        print(f"  Reward: {base_reward} (no bonus)")
                        if len(self.recipe_order) > 0:
                            print(f"  Expected: {expected_recipe}")
                
                return reward, global_reward_bonus
            else:
                # Invalid recipe!
                if self.verbose:
                    ingredients_str = ", ".join(ingredients)
                    print(f"  Invalid recipe! Ingredients: [{ingredients_str}]")
                    print(f"  No score")
                
                # No reward (or penalty)
                return reward, 0.0
        
        # For debugging
        if self.verbose:
            print(f"  Reward: {reward}, Global Bonus: {global_reward_bonus}")
            
        return reward, global_reward_bonus