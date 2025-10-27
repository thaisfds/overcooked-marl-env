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
from ..core.items import ItemType, ItemState


class ObservationBuilder:
    """
    Builds observations for agents in the Overcooked environment.
    
    The observation is a multi-channel tensor (H x W x C) where:
    - Channel 0: Static map (walls, stations)
    - Channel 1: Items on ground/counters
    - Channel 2: Position of the observing agent
    - Channel 3: Positions of other agents
    - Channel 4: Delivery points (binary)
    - Channels 5-7: Processing vector (inventory state)
    
    This class follows SRP by focusing only on observation construction,
    delegating state management to GameState and item queries to ItemInstanceManager.
    """
    
    def __init__(self, max_process_steps: int = 3):
        """
        Initialize the observation builder.
        
        Args:
            max_process_steps: Maximum number of processing steps (for inventory channels)
        """
        self.max_process_steps = max_process_steps
        self.num_base_channels = 5
        self.total_channels = self.num_base_channels + max_process_steps
    
    def build_observations(
        self,
        game_state: Dict[str, Any],
        agent_ids: list,
        item_manager
    ) -> Dict[str, np.ndarray]:
        """
        Build observations for all agents.
        
        Args:
            game_state: Complete game state dictionary
            agent_ids: List of active agent IDs
            item_manager: ItemManager instance for cell codes
        
        Returns:
            Dictionary mapping agent_id -> observation array
        """
        map_size = game_state["map_static"].shape
        observations = {}
        
        # Build base observation (shared layers)
        base_obs = self._build_base_observation(game_state, map_size, item_manager)
        
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
        item_manager
    ) -> np.ndarray:
        """
        Build the base observation shared by all agents.
        
        This includes:
        - Static map layer
        - Items on ground layer
        - Delivery points layer
        
        Args:
            game_state: Game state dictionary
            map_size: Tuple of (height, width)
            item_manager: ItemManager for cell codes
        
        Returns:
            Base observation array (H x W x TOTAL_CHANNELS)
        """
        H, W = map_size
        base_obs = np.zeros((H, W, self.total_channels), dtype=np.uint8)
        
        # Layer 0: Static Map
        base_obs[:, :, 0] = game_state["map_static"]
        
        # Layer 1: Items on Ground/Counter
        base_obs[:, :, 1] = game_state["items_on_ground"]
        
        # Layer 4: Delivery Points (Binary)
        delivery_code = item_manager.get_cell_code('D')
        base_obs[:, :, 4] = np.where(
            game_state["map_static"] == delivery_code,
            1,
            0
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
        base_obs[ay, ax, 2] = 1
        
        # Layer 3: Position of OTHER Agents
        for other_id, (ox, oy) in game_state["agent_pos"].items():
            if other_id != agent_id:
                base_obs[oy, ox, 3] = 1
        
        # Channels 5-7: Processing Vector (Inventory)
        progress_vector = self._build_processing_vector(agent_id, game_state)
        for i in range(self.max_process_steps):
            base_obs[ay, ax, 5 + i] = progress_vector[i]
        
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
        - For nothing: [0, 0, 0]
        
        Args:
            agent_id: ID of the agent
            game_state: Game state dictionary
        
        Returns:
            List of 3 integers representing the processing state
        """
        # Get the item instance that the agent is carrying
        carrying_instance_id = game_state.get("agent_holding_instance", {}).get(agent_id)
        
        if carrying_instance_id is None:
            return [0, 0, 0]
        
        carrying_item = game_state["item_instances"].get(carrying_instance_id)
        if not carrying_item:
            return [0, 0, 0]
        
        item_type = carrying_item.get("type")
        
        if item_type == ItemType.INGREDIENT:
            return self._ingredient_to_vector(carrying_item)
        elif item_type == ItemType.DISH:
            return self._dish_to_vector(carrying_item)
        
        return [0, 0, 0]
    
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
        cuts_needed = item.get("cuts_needed", 3)
        
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
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ObservationBuilder(channels={self.total_channels}, "
            f"base={self.num_base_channels}, "
            f"processing={self.max_process_steps})"
        )
