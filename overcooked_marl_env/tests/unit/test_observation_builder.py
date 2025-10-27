"""
Unit tests for ObservationBuilder.

Tests the observation construction logic, ensuring observations
are correctly built for different game states and agent configurations.
"""

import unittest
import numpy as np
from overcooked_marl_env.environment.observation_builder import ObservationBuilder
from overcooked_marl_env.core.items import ItemManager, ItemType, ItemState


class TestObservationBuilder(unittest.TestCase):
    """Test cases for ObservationBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = ObservationBuilder(max_process_steps=3)
        self.item_manager = ItemManager()
        
        # Create a simple game state for testing
        self.map_size = (5, 5)
        self.game_state = {
            "map_static": np.zeros(self.map_size, dtype=np.uint8),
            "items_on_ground": np.zeros(self.map_size, dtype=np.uint8),
            "agent_pos": {
                "agent_0": (1, 1),
                "agent_1": (3, 3)
            },
            "agent_holding_instance": {},
            "item_instances": {}
        }
        
        # Add a delivery point
        self.game_state["map_static"][2, 2] = self.item_manager.get_cell_code('D')
    
    def test_initialization(self):
        """Test ObservationBuilder initialization."""
        self.assertEqual(self.builder.max_process_steps, 3)
        self.assertEqual(self.builder.num_base_channels, 5)
        self.assertEqual(self.builder.total_channels, 8)
    
    def test_observation_shape(self):
        """Test that observations have the correct shape."""
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0", "agent_1"],
            self.item_manager
        )
        
        expected_shape = (5, 5, 8)
        self.assertEqual(obs["agent_0"].shape, expected_shape)
        self.assertEqual(obs["agent_1"].shape, expected_shape)
    
    def test_static_map_layer(self):
        """Test that static map is correctly encoded in channel 0."""
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0"],
            self.item_manager
        )
        
        # Channel 0 should match the static map
        np.testing.assert_array_equal(
            obs["agent_0"][:, :, 0],
            self.game_state["map_static"]
        )
    
    def test_items_on_ground_layer(self):
        """Test that items on ground are correctly encoded in channel 1."""
        # Add an item on ground
        self.game_state["items_on_ground"][2, 1] = 5
        
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0"],
            self.item_manager
        )
        
        # Channel 1 should match items on ground
        np.testing.assert_array_equal(
            obs["agent_0"][:, :, 1],
            self.game_state["items_on_ground"]
        )
    
    def test_agent_position_layer(self):
        """Test that agent's own position is marked in channel 2."""
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0"],
            self.item_manager
        )
        
        # Agent 0 is at (1, 1)
        self.assertEqual(obs["agent_0"][1, 1, 2], 1)
        
        # All other positions should be 0
        obs_copy = obs["agent_0"][:, :, 2].copy()
        obs_copy[1, 1] = 0
        self.assertEqual(np.sum(obs_copy), 0)
    
    def test_other_agents_layer(self):
        """Test that other agents are marked in channel 3."""
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0", "agent_1"],
            self.item_manager
        )
        
        # From agent_0's perspective, agent_1 at (3, 3) should be marked
        self.assertEqual(obs["agent_0"][3, 3, 3], 1)
        
        # From agent_1's perspective, agent_0 at (1, 1) should be marked
        self.assertEqual(obs["agent_1"][1, 1, 3], 1)
    
    def test_delivery_point_layer(self):
        """Test that delivery points are marked in channel 4."""
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0"],
            self.item_manager
        )
        
        # Delivery point at (2, 2) should be marked
        self.assertEqual(obs["agent_0"][2, 2, 4], 1)
        
        # All other positions should be 0
        obs_copy = obs["agent_0"][:, :, 4].copy()
        obs_copy[2, 2] = 0
        self.assertEqual(np.sum(obs_copy), 0)
    
    def test_empty_processing_vector(self):
        """Test processing vector when agent carries nothing."""
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0"],
            self.item_manager
        )
        
        # Agent at (1, 1) carries nothing
        # Channels 5-7 should all be 0
        self.assertEqual(obs["agent_0"][1, 1, 5], 0)
        self.assertEqual(obs["agent_0"][1, 1, 6], 0)
        self.assertEqual(obs["agent_0"][1, 1, 7], 0)
    
    def test_processing_vector_raw_ingredient(self):
        """Test processing vector for raw ingredient."""
        # Create a raw tomato
        item = {
            "type": ItemType.INGREDIENT,
            "name": "tomato",
            "state": ItemState.RAW,
            "cuts_done": 0,
            "cuts_needed": 3
        }
        
        # Add to game state
        self.game_state["item_instances"][0] = item
        self.game_state["agent_holding_instance"]["agent_0"] = 0
        
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0"],
            self.item_manager
        )
        
        # Agent at (1, 1) carries raw ingredient
        # Channel 5: cuts_done = 0
        # Channel 6: cuts_needed = 3
        # Channel 7: is_ready = 0
        self.assertEqual(obs["agent_0"][1, 1, 5], 0)
        self.assertEqual(obs["agent_0"][1, 1, 6], 3)
        self.assertEqual(obs["agent_0"][1, 1, 7], 0)
    
    def test_processing_vector_chopping_ingredient(self):
        """Test processing vector for ingredient being chopped."""
        # Create a tomato in chopping state
        item = {
            "type": ItemType.INGREDIENT,
            "name": "tomato",
            "state": ItemState.CHOPPING_2,
            "cuts_done": 2,
            "cuts_needed": 3
        }
        
        # Add to game state
        self.game_state["item_instances"][0] = item
        self.game_state["agent_holding_instance"]["agent_0"] = 0
        
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0"],
            self.item_manager
        )
        
        # Channel 5: cuts_done = 2
        # Channel 6: cuts_needed = 3
        # Channel 7: is_ready = 0
        self.assertEqual(obs["agent_0"][1, 1, 5], 2)
        self.assertEqual(obs["agent_0"][1, 1, 6], 3)
        self.assertEqual(obs["agent_0"][1, 1, 7], 0)
    
    def test_processing_vector_ready_ingredient(self):
        """Test processing vector for ready ingredient."""
        # Create a ready tomato
        item = {
            "type": ItemType.INGREDIENT,
            "name": "tomato",
            "state": ItemState.READY,
            "cuts_done": 3,
            "cuts_needed": 3
        }
        
        # Add to game state
        self.game_state["item_instances"][0] = item
        self.game_state["agent_holding_instance"]["agent_0"] = 0
        
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0"],
            self.item_manager
        )
        
        # Channel 5: cuts_done = 3
        # Channel 6: cuts_needed = 3
        # Channel 7: is_ready = 1
        self.assertEqual(obs["agent_0"][1, 1, 5], 3)
        self.assertEqual(obs["agent_0"][1, 1, 6], 3)
        self.assertEqual(obs["agent_0"][1, 1, 7], 1)
    
    def test_processing_vector_empty_dish(self):
        """Test processing vector for empty plate."""
        # Create an empty plate
        item = {
            "type": ItemType.DISH,
            "ingredients": []
        }
        
        # Add to game state
        self.game_state["item_instances"][0] = item
        self.game_state["agent_holding_instance"]["agent_0"] = 0
        
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0"],
            self.item_manager
        )
        
        # Channel 5: num_ingredients = 0
        # Channels 6-7: unused = 0
        self.assertEqual(obs["agent_0"][1, 1, 5], 0)
        self.assertEqual(obs["agent_0"][1, 1, 6], 0)
        self.assertEqual(obs["agent_0"][1, 1, 7], 0)
    
    def test_processing_vector_dish_with_ingredients(self):
        """Test processing vector for plate with ingredients."""
        # Create a plate with 2 ingredients
        item = {
            "type": ItemType.DISH,
            "ingredients": ["tomato", "lettuce"]
        }
        
        # Add to game state
        self.game_state["item_instances"][0] = item
        self.game_state["agent_holding_instance"]["agent_0"] = 0
        
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0"],
            self.item_manager
        )
        
        # Channel 5: num_ingredients = 2
        # Channels 6-7: unused = 0
        self.assertEqual(obs["agent_0"][1, 1, 5], 2)
        self.assertEqual(obs["agent_0"][1, 1, 6], 0)
        self.assertEqual(obs["agent_0"][1, 1, 7], 0)
    
    def test_multiple_agents_independent_observations(self):
        """Test that each agent gets independent observations."""
        # Agent 0 carries a tomato
        item0 = {
            "type": ItemType.INGREDIENT,
            "name": "tomato",
            "state": ItemState.RAW,
            "cuts_done": 0,
            "cuts_needed": 3
        }
        
        # Agent 1 carries a plate
        item1 = {
            "type": ItemType.DISH,
            "ingredients": ["lettuce"]
        }
        
        self.game_state["item_instances"][0] = item0
        self.game_state["item_instances"][1] = item1
        self.game_state["agent_holding_instance"]["agent_0"] = 0
        self.game_state["agent_holding_instance"]["agent_1"] = 1
        
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0", "agent_1"],
            self.item_manager
        )
        
        # Agent 0's processing vector (tomato)
        self.assertEqual(obs["agent_0"][1, 1, 5], 0)  # cuts_done
        self.assertEqual(obs["agent_0"][1, 1, 6], 3)  # cuts_needed
        
        # Agent 1's processing vector (plate with 1 ingredient)
        self.assertEqual(obs["agent_1"][3, 3, 5], 1)  # num_ingredients
        self.assertEqual(obs["agent_1"][3, 3, 6], 0)  # unused
    
    def test_get_observation_shape(self):
        """Test get_observation_shape method."""
        shape = self.builder.get_observation_shape((10, 15))
        self.assertEqual(shape, (10, 15, 8))
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.builder)
        self.assertIn("channels=8", repr_str)
        self.assertIn("base=5", repr_str)
        self.assertIn("processing=3", repr_str)


if __name__ == '__main__':
    unittest.main()
