"""
Game State Manager for Overcooked MARL Environment.

This module provides a centralized manager for the game state, following the
Single Responsibility Principle. It handles all game state data and provides
a clean interface for state manipulation.

The GameState class separates state management from environment logic,
making the code more maintainable, testable, and easier to understand.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from .env_config import (
    INITIAL_ORIENTATION,
    DEFAULT_DISHES_PENDING,
)
from ..core.constants import MAX_EPISODE_STEPS
from ..core.items import TILES


@dataclass
class MapData:
    """
    Container for map-related data.
    
    Attributes:
        static_map: Numpy array representing the static map
        agent_positions: Initial positions for each agent
        dispensers: Dispenser assignments {(x, y): ingredient_name}
        recipes: List of recipe names to be completed
        size: Tuple of (height, width)
    """
    static_map: np.ndarray
    agent_positions: Dict[str, Tuple[int, int]]
    dispensers: Dict[Tuple[int, int], str]
    recipes: List[str]
    size: Tuple[int, int]


class GameState:
    """
    Manages the complete game state for the Overcooked MARL environment.
    
    This class is responsible for:
    - Storing and managing all game state variables
    - Providing clean getters and setters
    - Validating state changes
    - Tracking episode progress
    - Managing timesteps and completion status
    
    The class encapsulates state complexity and provides a single
    source of truth for the game's current state.
    
    Attributes:
        map_static: Static map (walls, stations, etc.)
        map_size: Tuple of (height, width)
        agent_positions: Current positions of all agents
        agent_orientations: Current orientations of all agents
        items_on_ground: Numpy array of item codes on ground
        dispensers: Dispenser position -> ingredient mapping
        recipe_order: Ordered list of recipes to complete
        expected_recipe_index: Index of next expected recipe
        delivered_recipes: List of already delivered recipes
        dishes_pending: Number of dishes remaining
        time_steps: Current timestep
        accumulated_rewards: Total rewards per agent
        last_step_rewards: Rewards from the last step
        agents: List of active agent IDs
    """
    
    def __init__(self, map_data: MapData, agent_ids: List[str]):
        """
        Initialize game state from map data.
        
        Args:
            map_data: MapData object containing map information
            agent_ids: List of agent IDs (e.g., ["agent_0", "agent_1"])
        """
        # Map data
        self.map_static = map_data.static_map.copy()
        self.map_size = map_data.size
        self.items_on_ground = np.zeros(self.map_size, dtype=np.uint8)
        self.dispensers = map_data.dispensers.copy()
        
        # Agent data
        self.agents = agent_ids.copy()
        self.agent_positions = map_data.agent_positions.copy()
        self.agent_orientations = {agent_id: INITIAL_ORIENTATION for agent_id in agent_ids}
        
        # Recipe and delivery data
        self.recipe_order = map_data.recipes.copy()
        self.expected_recipe_index = 0
        self.delivered_recipes: List[str] = []
        self.dishes_pending = DEFAULT_DISHES_PENDING
        
        # Episode tracking
        self.time_steps = 0
        self.accumulated_rewards = {agent_id: 0.0 for agent_id in agent_ids}
        self.last_step_rewards = {agent_id: 0.0 for agent_id in agent_ids}
    
    def reset(self, map_data: MapData) -> None:
        """
        Reset the game state for a new episode.
        
        Args:
            map_data: MapData object with initial configuration
        """
        self.map_static = map_data.static_map.copy()
        self.map_size = map_data.size
        self.items_on_ground = np.zeros(self.map_size, dtype=np.uint8)
        self.dispensers = map_data.dispensers.copy()
        
        self.agent_positions = map_data.agent_positions.copy()
        self.agent_orientations = {agent_id: INITIAL_ORIENTATION for agent_id in self.agents}
        
        self.recipe_order = map_data.recipes.copy()
        self.expected_recipe_index = 0
        self.delivered_recipes.clear()
        self.dishes_pending = DEFAULT_DISHES_PENDING
        
        self.time_steps = 0
        self.accumulated_rewards = {agent_id: 0.0 for agent_id in self.agents}
        self.last_step_rewards = {agent_id: 0.0 for agent_id in self.agents}
    
    # =================================================================
    # AGENT POSITION AND ORIENTATION
    # =================================================================
    
    def get_agent_position(self, agent_id: str) -> Tuple[int, int]:
        """Get the current position of an agent."""
        return self.agent_positions[agent_id]
    
    def set_agent_position(self, agent_id: str, position: Tuple[int, int]) -> None:
        """Set the position of an agent."""
        self.agent_positions[agent_id] = position
    
    def update_agent_position(self, agent_id: str, new_position: Tuple[int, int]) -> None:
        """Update agent position (alias for set_agent_position)."""
        self.agent_positions[agent_id] = new_position
    
    def get_agent_orientation(self, agent_id: str) -> Tuple[int, int]:
        """Get the current orientation of an agent."""
        return self.agent_orientations[agent_id]
    
    def set_agent_orientation(self, agent_id: str, orientation: Tuple[int, int]) -> None:
        """Set the orientation of an agent."""
        self.agent_orientations[agent_id] = orientation
    
    def get_position_in_front(self, agent_id: str) -> Tuple[int, int]:
        """
        Get the position in front of an agent based on its orientation.
        
        Args:
            agent_id: The agent ID
        
        Returns:
            Tuple of (x, y) coordinates in front of the agent
        """
        x, y = self.agent_positions[agent_id]
        dx, dy = self.agent_orientations[agent_id]
        return (x + dx, y + dy)
    
    def is_position_valid(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is within map bounds.
        
        Args:
            position: Tuple of (x, y) coordinates
        
        Returns:
            bool: True if position is within bounds
        """
        x, y = position
        return 0 <= x < self.map_size[1] and 0 <= y < self.map_size[0]
    
    # =================================================================
    # GROUND ITEMS
    # =================================================================
    
    def get_ground_item_code(self, position: Tuple[int, int]) -> int:
        """
        Get the item code at a ground position.
        
        Args:
            position: Tuple of (x, y) coordinates
        
        Returns:
            int: Item code (0 if no item)
        """
        x, y = position
        if not self.is_position_valid(position):
            return TILES[' ']['code']
        return self.items_on_ground[y, x]
    
    def set_ground_item_code(self, position: Tuple[int, int], code: int) -> None:
        """
        Set the item code at a ground position.
        
        Args:
            position: Tuple of (x, y) coordinates
            code: Item code to set (TILES[' ']['code'] for no item)
        """
        x, y = position
        if self.is_position_valid(position):
            self.items_on_ground[y, x] = code
    
    def clear_ground_item(self, position: Tuple[int, int]) -> None:
        """Clear the item at a ground position."""
        self.set_ground_item_code(position, TILES[' ']['code'])
    
    def has_ground_item(self, position: Tuple[int, int]) -> bool:
        """Check if there's an item at a ground position."""
        return self.get_ground_item_code(position) > 0
    
    # =================================================================
    # MAP AND STATIONS
    # =================================================================
    
    def get_cell_code(self, position: Tuple[int, int]) -> int:
        """
        Get the static map cell code at a position.
        
        Args:
            position: Tuple of (x, y) coordinates
        
        Returns:
            int: Cell code
        """
        x, y = position
        if not self.is_position_valid(position):
            return TILES[' ']['code']
        return self.map_static[y, x]
    
    def is_dispenser_position(self, position: Tuple[int, int]) -> bool:
        """Check if a position is a dispenser."""
        return position in self.dispensers
    
    def get_dispenser_ingredient(self, position: Tuple[int, int]) -> Optional[str]:
        """Get the ingredient name for a dispenser at a position."""
        return self.dispensers.get(position)
    
    # =================================================================
    # RECIPES AND DELIVERY
    # =================================================================
    
    def get_current_expected_recipe(self) -> Optional[str]:
        """
        Get the name of the currently expected recipe.
        
        Returns:
            str: Recipe name, or None if all recipes delivered
        """
        if self.expected_recipe_index < len(self.recipe_order):
            return self.recipe_order[self.expected_recipe_index]
        return None
    
    def mark_recipe_delivered(self) -> None:
        """Mark the current recipe as delivered and advance to next."""
        if self.expected_recipe_index < len(self.recipe_order):
            recipe_name = self.recipe_order[self.expected_recipe_index]
            self.delivered_recipes.append(recipe_name)
            self.expected_recipe_index += 1
            self.dishes_pending = max(0, self.dishes_pending - 1)
    
    def get_delivered_count(self) -> int:
        """Get the number of recipes delivered."""
        return len(self.delivered_recipes)
    
    def get_pending_recipes(self) -> List[str]:
        """Get the list of pending recipes."""
        return self.recipe_order[self.expected_recipe_index:]
    
    def are_all_recipes_delivered(self) -> bool:
        """Check if all recipes have been delivered."""
        return self.expected_recipe_index >= len(self.recipe_order)
    
    # =================================================================
    # EPISODE MANAGEMENT
    # =================================================================
    
    def increment_timestep(self) -> None:
        """Increment the timestep counter."""
        self.time_steps += 1
    
    def get_timestep(self) -> int:
        """Get the current timestep."""
        return self.time_steps
    
    def is_episode_complete(self, max_steps: int = MAX_EPISODE_STEPS) -> Tuple[bool, bool]:
        """
        Check if the episode should end.
        
        Args:
            max_steps: Maximum steps before truncation
        
        Returns:
            Tuple of (terminated, truncated)
            - terminated: True if goal achieved (all recipes delivered)
            - truncated: True if max steps reached
        """
        terminated = self.dishes_pending <= 0
        truncated = self.time_steps >= max_steps
        return terminated, truncated
    
    # =================================================================
    # REWARDS
    # =================================================================
    
    def add_reward(self, agent_id: str, reward: float) -> None:
        """
        Add reward to an agent's accumulated total.
        
        Args:
            agent_id: The agent ID
            reward: Reward value to add
        """
        self.accumulated_rewards[agent_id] += reward
        self.last_step_rewards[agent_id] = reward
    
    def set_last_step_reward(self, agent_id: str, reward: float) -> None:
        """Set the reward for the last step."""
        self.last_step_rewards[agent_id] = reward
    
    def get_accumulated_reward(self, agent_id: str) -> float:
        """Get the total accumulated reward for an agent."""
        return self.accumulated_rewards[agent_id]
    
    def get_last_step_reward(self, agent_id: str) -> float:
        """Get the reward from the last step for an agent."""
        return self.last_step_rewards[agent_id]
    
    def clear_last_step_rewards(self) -> None:
        """Reset all last step rewards to zero."""
        self.last_step_rewards = {agent_id: 0.0 for agent_id in self.agents}
    
    # =================================================================
    # STATE SERIALIZATION
    # =================================================================
    
    def get_state_dict(self) -> dict:
        """
        Get the complete state as a dictionary.
        
        Returns:
            dict: Complete state for serialization or inspection
        """
        return {
            "map_static": self.map_static.copy(),
            "map_size": self.map_size,
            "items_on_ground": self.items_on_ground.copy(),
            "dispensers": self.dispensers.copy(),
            "agent_positions": self.agent_positions.copy(),
            "agent_orientations": self.agent_orientations.copy(),
            "recipe_order": self.recipe_order.copy(),
            "expected_recipe_index": self.expected_recipe_index,
            "delivered_recipes": self.delivered_recipes.copy(),
            "dishes_pending": self.dishes_pending,
            "time_steps": self.time_steps,
            "accumulated_rewards": self.accumulated_rewards.copy(),
            "last_step_rewards": self.last_step_rewards.copy(),
            "agents": self.agents.copy(),
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"GameState("
                f"timestep={self.time_steps}, "
                f"delivered={len(self.delivered_recipes)}/{len(self.recipe_order)}, "
                f"pending_dishes={self.dishes_pending})")


__all__ = ['GameState', 'MapData']
