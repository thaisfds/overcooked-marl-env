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
from .core.constants import (
    MAX_PROCESS_STEPS,
    TOTAL_CHANNELS,
    MAX_EPISODE_STEPS
)
from .map.map_loader import MapLoader
from .environment.game_state import MapData
from .environment.observation_builder import ObservationBuilder
from .environment.featurized_observation_builder import FeaturizedObservationBuilder
from .mechanics import MovementSystem, CollisionResolver
from .mechanics.interaction_handler import InteractionHandler
from .rendering import Renderer

class OvercookedMARL(ParallelEnv):
    metadata = {"render_modes": ["human", "ascii"], "name": "overcooked_marl_v0"}

    # =================================================================
    # PETTINGZOO API METHODS (Public Interface)
    # Following PettingZoo conventions: https://pettingzoo.farama.org/
    # =================================================================

    def __init__(self, render_mode=None, map_path=None, num_fixed_agents=None, verbose=False, custom_recipes_path=None, featurized_obs=False):
        super().__init__()
        
        # Resolve map path: if None or just filename, construct full path
        self.map_path = self._resolve_map_path(map_path)
        self.verbose = verbose
        self.featurized_obs = featurized_obs
        
        self.item_manager = ItemManager()
        self.recipe_manager = RecipeManager()
        
        # Load custom recipes if provided before initializing other systems
        if custom_recipes_path:
            if verbose:
                print(f"[ENV] Loading custom recipes from: {custom_recipes_path}")
            self.recipe_manager.load_recipes_from_json(custom_recipes_path)
            self.item_manager.rebuild_ingredient_creators()
            
        self.map_loader = MapLoader()
        
        # Determine number of agents from map if not specified
        if num_fixed_agents is None:
            try:
                map_data = self.map_loader.load_map(self.map_path, use_default_on_error=False)
                num_fixed_agents = len(map_data.agent_positions)
                if num_fixed_agents == 0:
                    num_fixed_agents = 2
            except Exception:
                num_fixed_agents = 2
                
        self.num_fixed_agents = num_fixed_agents
        self.observation_builder = ObservationBuilder(max_process_steps=MAX_PROCESS_STEPS)
        self.featurized_observation_builder = FeaturizedObservationBuilder(self.recipe_manager)
        
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
        self.collision_resolver = CollisionResolver(collision_penalty=0.0)  # SEM PENALIDADE
        
        # Interaction System (Phase 6)
        self.interaction_handler = InteractionHandler(self.item_manager, self.recipe_manager)
        
        # Rendering System (Phase 7)
        self.renderer = Renderer(self.item_manager, render_mode=render_mode, recipe_manager=self.recipe_manager)
        
        action_space_single = spaces.Discrete(5) 
        
        # Observation Space
        if self.featurized_obs:
            observation_space_single = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(96,),
                dtype=np.float32
            )
        else:
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
        self.window = None  # Deprecated - kept for backwards compatibility
        self.clock = None   # Deprecated - kept for backwards compatibility

    def reset(self, seed=None, options=None):
        """
        Resets the environment to start a new episode.
        
        Args:
            seed (int, optional): Random seed for episode initialization
            options (dict, optional): Additional options for reset
            
        Returns:
            tuple: (observations, infos) for all agents
        """
        gym.Env.reset(self, seed=seed) 
        
        # Make sure the map is marked as loaded to avoid duplicate logs
        if not self.map_path in self._already_loaded_maps:
            self._already_loaded_maps.add(self.map_path)
        
        # Reinitialize state and agent list
        self.state = self._initialize_game_state()
        self.agents = self.possible_agents[:]
        
        # Reset variables related to recipe order
        # recipe_order is already set in _initialize_game_state from self.recipe_order
        self.state["expected_recipe_index"] = 0
        self.state["delivered_recipes"] = []
        
        # Ensure rewards and position histories are reset to zero
        self.state["accumulated_rewards"] = {agent: 0.0 for agent in self.possible_agents}
        self.state["last_step_rewards"] = {agent: 0.0 for agent in self.possible_agents}
        self.position_history = {agent: [] for agent in self.possible_agents}
        
        observations = self._get_obs()
        info = self._get_info()
        
        return observations, info

    def step(self, actions):
        """
        Executes one timestep of the environment.
        
        Args:
            actions (dict): Dictionary mapping agent_id to action
                Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Interact
        
        Returns:
            tuple: (observations, rewards, terminations, truncations, infos)
                - observations (dict): Agent observations
                - rewards (dict): Rewards for each agent
                - terminations (dict): Episode ended naturally
                - truncations (dict): Episode ended due to time limit
                - infos (dict): Additional information
        """
        
        # Initialization of return dictionaries
        step_cost = getattr(self.interaction_handler.rewards, 'step_cost', -0.01)
        rewards = {agent: step_cost for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        episode_terminated = False
        episode_truncated = False
        global_reward = 0.0 # Global reward accumulator
        
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
        
        # Step 3: Collision penalties REMOVIDAS
        # for agent_id, penalty in collision_penalties.items():
        #     if penalty > 0:
        #         rewards[agent_id] -= penalty
        
        # Step 4: Update agent positions in state
        for agent_id, pos in final_positions.items():
            self.state["agent_pos"][agent_id] = pos
        
        # Movement debugging (when verbose=True and actions taken)
        if self.verbose and actions:
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
    
    def render(self):
        """
        Renders the current state of the environment.
        
        Returns:
            np.ndarray (if render_mode="human"): RGB image array (Pygame graphical rendering)
            None (if render_mode="ascii"): Prints ASCII to console
            None (if render_mode=None): No rendering
        """
        if self.render_mode is None:
            return None
        
        return self.renderer.render(
            game_state=self.state,
            map_size=self.map_size,
            agents=self.agents,
            recipe_order=self.recipe_order
        )

    def observation_space(self, agent):
        """
        Returns the observation space for the specified agent.
        
        Args:
            agent (str): Agent identifier
            
        Returns:
            gym.spaces.Box: Observation space (H x W x C)
        """
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        """
        Returns the action space for the specified agent.
        
        Args:
            agent (str): Agent identifier
            
        Returns:
            gym.spaces.Discrete: Action space with 5 actions
        """
        return self.action_spaces[agent]

    # PettingZoo-specific methods
    def observe(self, agent: str):
        """
        Returns observation for a specific agent.
        
        Args:
            agent (str): Agent identifier
            
        Returns:
            np.ndarray: Agent's observation
        """
        return self._get_obs()[agent] 

    def get_state(self):
        """
        Returns the global state of the environment (PettingZoo API).
        
        Note: In PettingZoo, this typically returns a global observation.
        This implementation returns the internal state dictionary.
        
        This method provides the PettingZoo-compatible interface while
        maintaining backward compatibility with direct `self.state` access.
        
        Returns:
            dict: Complete environment state
        """
        return self.state

    def close(self):
        """
        Closes the environment and cleans up resources.
        """
        self.renderer.close()
        # Keep for backwards compatibility
        if self.window is not None:
            self.window = None

    # =================================================================
    # PRIVATE HELPER METHODS
    # =================================================================

    def _resolve_map_path(self, map_path):
        """
        Resolve map path from map name or full path.
        
        Args:
            map_path: Can be:
                - None or empty: uses default_map.txt
                - Just filename (e.g., "tutorial.txt"): constructs path to overcooked_marl_env/map/maps/
                - Full path (absolute or relative): uses as-is for backward compatibility
        
        Returns:
            str: Absolute path to map file
        """
        # If None or empty, use default map
        if not map_path or map_path.strip() == "":
            map_path = "default_map.txt"
        
        # If already an absolute path, use as-is
        if os.path.isabs(map_path):
            return map_path
        
        # If contains path separators (relative path), try to use as-is first
        # Otherwise assume it's just a filename
        if os.sep in map_path or '/' in map_path or '\\' in map_path:
            # It's a relative path, try to resolve it
            abs_path = os.path.abspath(map_path)
            if os.path.exists(abs_path):
                return abs_path
            # If doesn't exist, fall through to construct from maps/ directory
        
        # Assume it's just a filename, construct path to maps/ directory
        # Try multiple possible locations (same logic as MapLoader)
        possible_base_paths = [
            # Relative to this file (overcooked_marl_env/env.py)
            os.path.join(os.path.dirname(__file__), 'map', 'maps'),
            # Relative to package root
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'map', 'maps'),
            # From workspace root
            os.path.join('overcooked_marl_env', 'map', 'maps'),
        ]
        
        for base_path in possible_base_paths:
            full_path = os.path.join(base_path, map_path)
            abs_path = os.path.abspath(full_path)
            if os.path.exists(abs_path):
                return abs_path
        
        # If not found, return the constructed path anyway (will raise error later)
        # Use the first possible path as default
        default_path = os.path.join(possible_base_paths[0], map_path)
        return os.path.abspath(default_path)

    def _load_map_from_file(self, map_path):
        """
        Load map using the new MapLoader system.
        Returns tuple of (static_map, agent_positions, dispensers).
        """
        is_reload = map_path in self._already_loaded_maps
        self._already_loaded_maps.add(map_path)
        
        # Auto-load recipes.json in the same directory as the map if it exists
        if map_path:
            recipes_json_path = os.path.join(os.path.dirname(map_path), "recipes.json")
            if os.path.exists(recipes_json_path):
                if self.verbose:
                    print(f"[ENV] Auto-loading adjacent recipes from: {recipes_json_path}")
                self.recipe_manager.load_recipes_from_json(recipes_json_path)
                self.item_manager.rebuild_ingredient_creators()
        
        if self.verbose:
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
        
        if self.verbose:
            print(f"[ENV] Map loaded successfully:")
            print(f"  - Size: {map_data.size}")
            print(f"  - Agents: {list(map_data.agent_positions.keys())}")
            print(f"  - Dispensers: {len(map_data.dispensers)}")
            print(f"  - Recipes: {map_data.recipes}")
        
        return map_data.static_map, map_data.agent_positions, map_data.dispensers
        
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
                if self.verbose:
                    print(f"Added recipe: {name}")
            else:
                if self.verbose:
                    print(f"WARNING: Recipe '{name}' not found in RecipeManager")
        
        # Make sure at least one recipe is defined (in case the map doesn't specify any)
        if len(self.recipe_order) == 0:
            default_recipe = get_default_recipe()
            if self.verbose:
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
        if self.verbose:
            print(f"Map loaded with size: {self.map_size}")
        
        items_on_ground = np.zeros(self.map_size, dtype=np.uint8) 
        
        agent_start_pos = {k: v for k, v in agent_start_pos.items() if k in self.possible_agents}
        
        carrying = {}
        for agent in self.possible_agents:
            if agent not in agent_start_pos:
                if self.verbose:
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
        
        total_recipes_count = len(self.recipe_order) if hasattr(self, 'recipe_order') and len(self.recipe_order) > 0 else 3
        
        state = {
            "map_static": map_static_np,
            "agent_pos": agent_start_pos,
            "carrying": carrying,
            "items_on_ground": items_on_ground,
            "dishes_pending": total_recipes_count,
            "time_steps": 0,
            "dispensers": dispenser_assignments,
            "recipe_order": self.recipe_order.copy() if hasattr(self, 'recipe_order') else [],  # List of recipes in order
            "expected_recipe_index": 0,  # Index of the next expected recipe
            "delivered_recipes": [],     # Recipes already delivered
            "item_instances": {},        # Map item_instance_id -> item_dict
            "agent_holding_instance": {},  # Map agent_id -> item_instance_id
            "ground_instances": {},      # Map (x, y) -> item_instance_id
            "next_item_id": 0,           # Counter to generate unique item IDs
            "agent_orientation": {agent: (0, -1) for agent in self.possible_agents},  # All agents start facing North
            "accumulated_rewards": {agent: 0.0 for agent in self.possible_agents},
            "last_step_rewards": {agent: 0.0 for agent in self.possible_agents}
        }
        return state

    def _get_obs(self):
        """
        Generates observations for all agents.
        """
        if self.featurized_obs:
            return self.featurized_observation_builder.build_observations(
                self.state,
                self.agents,
                self.item_manager,
                self.recipe_manager
            )
        return self.observation_builder.build_observations(
            self.state, 
            self.agents, 
            self.item_manager,
            self.recipe_manager
        )

    def _get_info(self):
        """Returns the dictionary of auxiliary information."""
        info = {}
        for agent in self.agents:
            # Get the item instance that the agent is carrying
            carrying_instance_id = self.state["agent_holding_instance"].get(agent, None)
            carrying_item = self.interaction_handler._get_item(carrying_instance_id, self.state) if carrying_instance_id is not None else None
            
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
