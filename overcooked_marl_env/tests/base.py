import unittest
import numpy as np
from overcooked_marl_env.core.items import ItemManager, create_ingredient

class OvercookedTestCase(unittest.TestCase):
    """Base test case class for Overcooked MARL unit tests, providing shared state builders."""
    
    def create_basic_game_state(self, width=10, height=10):
        """
        Create a standard mock game state dictionary for testing.
        
        Args:
            width: Width of the grid.
            height: Height of the grid.
            
        Returns:
            Dict: Initialized game state template.
        """
        return {
            "agent_pos": {
                "agent_0": (5, 5),
                "agent_1": (3, 3)
            },
            "agent_orientation": {
                "agent_0": (1, 0),   # Facing East (Right)
                "agent_1": (0, -1)   # Facing North (Up)
            },
            "agent_holding_instance": {},
            "ground_instances": {},
            "item_instances": {},
            "next_item_id": 1,
            "delivered_recipes": [],
            "dishes_pending": 3,
            "recipe_order": ["SIMPLE_SALAD", "GARDEN_SALAD"],
            "expected_recipe_index": 0,
            "dispensers": {},
            "map_static": np.zeros((height, width), dtype=np.uint8),
            "items_on_ground": np.zeros((height, width), dtype=np.uint8),
        }
        
    def create_item_in_state(self, game_state, item_dict):
        """
        Register a new item instance inside the test game state.
        
        Args:
            game_state: The game state dictionary to update.
            item_dict: The item properties.
            
        Returns:
            int: The unique item instance ID.
        """
        item_id = game_state["next_item_id"]
        game_state["next_item_id"] += 1
        game_state["item_instances"][item_id] = item_dict.copy()
        return item_id
