"""
Observation Builder for Overcooked MARL Environment.

This module is responsible for building observations for agents,
following the Single Responsibility Principle.

Responsibilities:
- Build multi-channel observations for each agent
- Encode game state into observation space
- Provide agent-specific views of the environment
"""

import numpy as np
from typing import Dict, Tuple, Any
from ..core.items import ItemType, ItemState, INGREDIENTS
from .env_config import (
    EMPTY_PROCESSING_VECTOR,
    DEFAULT_CUTS_NEEDED,
    CHANNEL_STATIC_MAP,
    CHANNEL_ITEMS_GROUND,
    CHANNEL_AGENT_SELF,
    CHANNEL_AGENTS_OTHER,
    CHANNEL_DELIVERY_POINTS,
    CHANNEL_RECIPE_ORDER,
    CHANNEL_PROCESSING_START,
    NUM_BASE_CHANNELS,
    MAX_RECIPES_IN_ORDER,
)
from ..core.constants import MAX_PROCESS_STEPS


class ObservationBuilder:
    """
    Builds observations for agents in the Overcooked environment.
    
    The observation is a multi-channel tensor (H x W x C) where:
    - Channel 0: Static map (walls, counters, stations)
    - Channel 1: Items on ground/counters
    - Channel 2: Position of the observing agent
    - Channel 3: Positions of other agents
    - Channel 4: Delivery points (binary)
    - Channel 5: Active Recipe Orders queue
    - Channels 6-8: Processing vector (inventory state)
    
    This class follows SRP by focusing only on observation construction,
    delegating state queries to the environment state dictionary and ItemManager.
    """
    
    def __init__(self, max_process_steps: int = MAX_PROCESS_STEPS, recipe_manager=None):
        """
        Initialize the observation builder.
        
        Args:
            max_process_steps: Maximum number of processing steps (for inventory channels)
            recipe_manager: RecipeManager instance for encoding recipes (optional, can be passed in build_observations)
        """
        self.max_process_steps = max_process_steps
        self.num_base_channels = NUM_BASE_CHANNELS
        self.total_channels = self.num_base_channels + max_process_steps
        self.recipe_manager = recipe_manager
    
    def build_observations(
        self,
        game_state: Dict[str, Any],
        agent_ids: list,
        item_manager,
        recipe_manager=None
    ) -> Dict[str, np.ndarray]:
        """
        Build observations for all agents.
        
        Args:
            game_state: Complete game state dictionary
            agent_ids: List of active agent IDs
            item_manager: ItemManager instance for cell codes
            recipe_manager: RecipeManager instance for encoding recipes (optional)
        
        Returns:
            Dictionary mapping agent_id -> observation array
        """
        # Use provided recipe_manager or fall back to instance variable
        recipe_mgr = recipe_manager or self.recipe_manager
        
        map_size = game_state["map_static"].shape
        observations = {}
        
        # Build base observation (shared layers)
        base_obs = self._build_base_observation(game_state, map_size, item_manager, recipe_mgr)
        
        # Build agent-specific observations
        for agent_id in agent_ids:
            observations[agent_id] = self._build_agent_observation(
                agent_id,
                game_state,
                base_obs.copy(),
                item_manager
            )
        
        return observations
    
    def _build_base_observation(
        self,
        game_state: Dict[str, Any],
        map_size: Tuple[int, int],
        item_manager,
        recipe_manager=None
    ) -> np.ndarray:
        """
        Build the base observation shared by all agents.
        
        This includes:
        - Static map layer
        - Items on ground layer
        - Delivery points layer
        - Recipe order layer
        
        Args:
            game_state: Game state dictionary
            map_size: Tuple of (height, width)
            item_manager: ItemManager for cell codes
            recipe_manager: RecipeManager instance for encoding recipes
        
        Returns:
            Base observation array (H x W x TOTAL_CHANNELS)
        """
        H, W = map_size
        base_obs = np.zeros((H, W, self.total_channels), dtype=np.uint8)
        
        # Layer 0: Static Map
        base_obs[:, :, CHANNEL_STATIC_MAP] = game_state["map_static"]
        
        # Layer 1: Items on Ground/Counter (FIXED: uses ground_instances)
        base_obs[:, :, CHANNEL_ITEMS_GROUND] = self._build_ground_items_channel(
            game_state, map_size, item_manager
        )
        
        # Layer 4: Delivery Points (Binary)
        delivery_code = item_manager.get_cell_code('D')
        base_obs[:, :, CHANNEL_DELIVERY_POINTS] = np.where(
            game_state["map_static"] == delivery_code,
            1,
            0
        )
        
        # Layer 5: Recipe Order (NEW)
        if recipe_manager is not None:
            base_obs[:, :, CHANNEL_RECIPE_ORDER] = self._build_recipe_order_channel(
                game_state, map_size, recipe_manager
            )
        
        return base_obs
    
    def _build_agent_observation(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        base_obs: np.ndarray,
        item_manager
    ) -> np.ndarray:
        """
        Build observation for a specific agent.
        
        Adds agent-specific layers:
        - Agent's own position
        - Other agents' positions
        - Processing vector (inventory state)
        
        Args:
            agent_id: ID of the agent
            game_state: Game state dictionary
            base_obs: Base observation to build upon
            item_manager: ItemManager instance
        
        Returns:
            Complete observation for the agent
        """
        # Get agent position
        ax, ay = game_state["agent_pos"][agent_id]
        
        # Layer 2: Position of the Observing Agent
        base_obs[ay, ax, CHANNEL_AGENT_SELF] = 1
        
        # Layer 3: Position of OTHER Agents
        for other_id, (ox, oy) in game_state["agent_pos"].items():
            if other_id != agent_id:
                base_obs[oy, ox, CHANNEL_AGENTS_OTHER] = 1
        
        # Channels 5-7: Processing Vector (Inventory)
        progress_vector = self._build_processing_vector(agent_id, game_state)
        for i in range(self.max_process_steps):
            base_obs[ay, ax, CHANNEL_PROCESSING_START + i] = progress_vector[i]
        
        return base_obs
    
    def _build_processing_vector(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> list:
        """
        Build the processing vector for an agent's inventory.
        
        The vector represents the state of the item being carried:
        - For ingredients: [cuts_done, cuts_needed, is_ready]
        - For dishes: [num_ingredients, 0, 0]
        - For nothing: EMPTY_PROCESSING_VECTOR
        
        Args:
            agent_id: ID of the agent
            game_state: Game state dictionary
        
        Returns:
            List of MAX_PROCESS_STEPS integers representing the processing state
        """
        # Get the item instance that the agent is carrying
        carrying_instance_id = game_state.get("agent_holding_instance", {}).get(agent_id)
        
        if carrying_instance_id is None:
            return EMPTY_PROCESSING_VECTOR.copy()
        
        carrying_item = game_state["item_instances"].get(carrying_instance_id)
        if not carrying_item:
            return EMPTY_PROCESSING_VECTOR.copy()
        
        item_type = carrying_item.get("type")
        
        if item_type == ItemType.INGREDIENT:
            return self._ingredient_to_vector(carrying_item)
        elif item_type == ItemType.DISH:
            return self._dish_to_vector(carrying_item)
        
        return EMPTY_PROCESSING_VECTOR.copy()
    
    def _ingredient_to_vector(self, item: Dict[str, Any]) -> list:
        """
        Convert ingredient state to processing vector.
        
        Args:
            item: Ingredient item dictionary
        
        Returns:
            [cuts_done, cuts_needed, is_ready]
        """
        state = item.get("state", ItemState.RAW)
        cuts_done = item.get("cuts_done", 0)
        cuts_needed = item.get("cuts_needed", DEFAULT_CUTS_NEEDED)
        
        # Check if ingredient is ready
        is_ready = 1 if state == ItemState.READY else 0
        
        return [cuts_done, cuts_needed, is_ready]
    
    def _dish_to_vector(self, item: Dict[str, Any]) -> list:
        """
        Convert dish state to processing vector.
        
        Args:
            item: Dish item dictionary
        
        Returns:
            [num_ingredients, 0, 0]
        """
        ingredients = item.get("ingredients", [])
        num_ingredients = len(ingredients)
        
        return [num_ingredients, 0, 0]
    
    def get_observation_shape(self, map_size: Tuple[int, int]) -> Tuple[int, int, int]:
        """
        Get the shape of observations.
        
        Args:
            map_size: Tuple of (height, width)
        
        Returns:
            Observation shape (height, width, channels)
        """
        return (map_size[0], map_size[1], self.total_channels)
    
    def _build_ground_items_channel(
        self,
        game_state: Dict[str, Any],
        map_size: Tuple[int, int],
        item_manager
    ) -> np.ndarray:
        """
        Build Channel 1 with items on ground using ground_instances.
        
        This method correctly populates the channel by iterating over
        ground_instances and converting item_dicts to codes.
        
        Args:
            game_state: Game state dictionary
            map_size: Tuple of (height, width)
            item_manager: ItemManager instance
        
        Returns:
            Array of shape (H, W) with item codes
        """
        H, W = map_size
        channel = np.zeros((H, W), dtype=np.uint8)
        
        # Iterate over ground_instances to get items on ground
        ground_instances = game_state.get("ground_instances", {})
        item_instances = game_state.get("item_instances", {})
        
        for (x, y), instance_id in ground_instances.items():
            # Check bounds
            if 0 <= y < H and 0 <= x < W:
                item_dict = item_instances.get(instance_id)
                if item_dict:
                    code = self._item_to_code(item_dict)
                    channel[y, x] = code
        
        return channel
    
    def _item_to_code(self, item_dict: Dict[str, Any]) -> int:
        """
        Convert item_dict to numeric code for observation channel.
        
        Encoding:
        - Ingredientes: 10 + índice_ingrediente + (estado_offset * 10)
        - Pratos: 50 + número_ingredientes
        - Vazio/None: 0
        
        Args:
            item_dict: Item dictionary from item_instances
        
        Returns:
            Numeric code for the item
        """
        if not item_dict:
            return 0
        
        item_type = item_dict.get("type")
        
        if item_type == ItemType.INGREDIENT:
            # Get ingredient name
            ingredient_name = item_dict.get("name", "").lower()
            
            # Find index in INGREDIENTS dict
            ingredient_list = list(INGREDIENTS.keys())
            if ingredient_name in ingredient_list:
                ingredient_index = ingredient_list.index(ingredient_name)
            else:
                # Unknown ingredient, use 0
                ingredient_index = 0
            
            # Get state offset (0-4)
            state = item_dict.get("state", ItemState.RAW)
            state_offsets = {
                ItemState.RAW: 0,
                ItemState.CHOPPING_1: 1,
                ItemState.CHOPPING_2: 2,
                ItemState.CHOPPING_3: 3,
                ItemState.READY: 4,
            }
            state_offset = state_offsets.get(state, 0)
            
            # Code: 10 + índice + (estado_offset * 10)
            return 10 + ingredient_index + (state_offset * 10)
        
        elif item_type == ItemType.DISH:
            # Get number of ingredients
            ingredients = item_dict.get("ingredients", [])
            num_ingredients = len(ingredients)
            
            # Code: 50 + número_ingredientes
            return 50 + num_ingredients
        
        # Unknown type or empty
        return 0
    
    def _build_recipe_order_channel(
        self,
        game_state: Dict[str, Any],
        map_size: Tuple[int, int],
        recipe_manager
    ) -> np.ndarray:
        """
        Build Channel 5 with recipe order information.
        
        Uses the first row (y=0) of the grid to represent the list of orders.
        Each position in the row represents one order in sequence.
        
        Args:
            game_state: Game state dictionary
            map_size: Tuple of (height, width)
            recipe_manager: RecipeManager instance
        
        Returns:
            Array of shape (H, W) with recipe codes
        """
        H, W = map_size
        channel = np.zeros((H, W), dtype=np.uint8)
        
        # Get recipe order information from game state
        recipe_order = game_state.get("recipe_order", [])
        expected_recipe_index = game_state.get("expected_recipe_index", 0)
        delivered_recipes = game_state.get("delivered_recipes", [])
        
        # Fill first row (y=0) with recipe information
        # Each column (x position) represents one order in sequence
        for i, recipe_name in enumerate(recipe_order[:MAX_RECIPES_IN_ORDER]):
            if i < W:  # Make sure we don't exceed grid width
                # Count how many times this recipe appears up to position i
                recipes_up_to_i = recipe_order[:i+1]
                occurrences_in_order = recipes_up_to_i.count(recipe_name)
                
                # Count how many times this recipe has been delivered
                occurrences_delivered = delivered_recipes.count(recipe_name)
                
                # Check if this specific position has been delivered
                # (if delivered count >= occurrences up to this position)
                is_delivered = occurrences_delivered >= occurrences_in_order
                
                # Check if this is the next expected recipe
                is_next = (i == expected_recipe_index)
                
                # Encode recipe slot
                code = self._encode_recipe_slot(recipe_name, is_delivered, is_next, recipe_manager)
                channel[0, i] = code
        
        # Rest of the grid remains 0 (not used for recipes)
        return channel
    
    def _get_recipe_code(self, recipe_name: str, recipe_manager) -> int:
        """
        Get numeric code for a recipe name using RecipeManager.
        
        Uses RecipeManager to get the index of the recipe dynamically.
        Returns index + 1 to avoid 0 (which represents empty slot).
        
        Args:
            recipe_name: Name of the recipe (e.g., "SIMPLE_SALAD")
            recipe_manager: RecipeManager instance
        
        Returns:
            Numeric code for the recipe (1-based index)
        """
        if recipe_manager is None:
            return 0
        
        # Get all recipe names from RecipeManager
        all_recipe_names = recipe_manager.get_all_recipe_names()
        
        # Find index of this recipe
        recipe_name_upper = recipe_name.upper()
        if recipe_name_upper in all_recipe_names:
            recipe_index = all_recipe_names.index(recipe_name_upper)
            return recipe_index + 1  # 1-based to avoid 0
        
        # Recipe not found, return 0
        return 0
    
    def _get_recipe_status_code(self, is_delivered: bool, is_next: bool) -> int:
        """
        Get status code for a recipe slot.
        
        Args:
            is_delivered: Whether the recipe has been delivered
            is_next: Whether this is the next expected recipe
        
        Returns:
            Status code: 0 (not delivered, not next), 100 (not delivered, is next), 200 (delivered)
        """
        if is_delivered:
            return 200
        elif is_next:
            return 100
        else:
            return 0
    
    def _encode_recipe_slot(
        self,
        recipe_name: str,
        is_delivered: bool,
        is_next: bool,
        recipe_manager
    ) -> int:
        """
        Encode a recipe slot with recipe code and status.
        
        Args:
            recipe_name: Name of the recipe
            is_delivered: Whether the recipe has been delivered
            is_next: Whether this is the next expected recipe
            recipe_manager: RecipeManager instance
        
        Returns:
            Encoded value: recipe_code + status_code
        """
        recipe_code = self._get_recipe_code(recipe_name, recipe_manager)
        status_code = self._get_recipe_status_code(is_delivered, is_next)
        return recipe_code + status_code
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ObservationBuilder(channels={self.total_channels}, "
            f"base={self.num_base_channels}, "
            f"processing={self.max_process_steps})"
        )
