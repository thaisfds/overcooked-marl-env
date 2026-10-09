"""
Base station handler interface for Overcooked MARL interactions.

Implements the Strategy pattern for stations (chopping board, counters, dispensers, delivery, trash).
"""

from typing import Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod


class StationHandler(ABC):
    """
    Abstract base class for all station interaction strategies.
    
    Provides access to environment managers, reward shaper, and state modification helpers.
    """

    def __init__(self, context: Any):
        """
        Initialize with reference to the parent InteractionHandler context.
        """
        self.context = context

    @property
    def rewards(self):
        return self.context.rewards

    @property
    def reward_shaper(self):
        return self.context.reward_shaper

    @property
    def item_manager(self):
        return self.context.item_manager

    @property
    def recipe_manager(self):
        return self.context.recipe_manager

    def get_item(self, item_instance_id: Optional[int], game_state: Dict) -> Optional[Dict]:
        return self.context._get_item(item_instance_id, game_state)

    def create_item_instance(self, item_dict: Dict, game_state: Dict) -> int:
        return self.context._create_item_instance(item_dict, game_state)

    def update_item(self, item_instance_id: int, new_item_dict: Dict, game_state: Dict):
        return self.context._update_item(item_instance_id, new_item_dict, game_state)

    def delete_item_instance(self, item_instance_id: int, game_state: Dict):
        return self.context._delete_item_instance(item_instance_id, game_state)

    @abstractmethod
    def handle(
        self,
        agent_id: str,
        tile_pos: Tuple[int, int],
        carrying_item: Optional[Dict],
        carrying_instance_id: Optional[int],
        ground_item: Optional[Dict],
        ground_instance_id: Optional[int],
        game_state: Dict,
        verbose: bool = False
    ) -> Tuple[float, float]:
        """
        Handle an agent interaction at this station.

        Returns:
            Tuple[float, float]: (local_reward, global_reward_bonus)
        """
        pass
