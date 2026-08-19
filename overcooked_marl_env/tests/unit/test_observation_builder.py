"""
Unit tests for ObservationBuilder.

Tests the observation construction logic, ensuring observations
are correctly built for different game states and agent configurations.
"""

import numpy as np
from overcooked_marl_env.tests.base import OvercookedTestCase
from overcooked_marl_env.environment.observation_builder import ObservationBuilder
from overcooked_marl_env.core.items import ItemManager, ItemType, ItemState


class TestObservationBuilder(OvercookedTestCase):
    """Test cases for ObservationBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = ObservationBuilder(max_process_steps=3)
        self.item_manager = ItemManager()
        
        # Create a simple game state for testing using base class builder
        self.map_size = (5, 5)
        self.game_state = self.create_basic_game_state(width=5, height=5)
        self.game_state["agent_pos"] = {
            "agent_0": (1, 1),
            "agent_1": (3, 3)
        }
        
        # Add a delivery point
        self.game_state["map_static"][2, 2] = self.item_manager.get_cell_code('D')
    
    def test_initialization(self):
        """Test ObservationBuilder initialization."""
        self.assertEqual(self.builder.max_process_steps, 3)
        self.assertEqual(self.builder.num_base_channels, 6)
        self.assertEqual(self.builder.total_channels, 9)
    
    def test_observation_shape(self):
        """Test that observations have the correct shape."""
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0", "agent_1"],
            self.item_manager
        )
        
        expected_shape = (5, 5, 9)
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
        item = {
            "type": ItemType.INGREDIENT,
            "name": "tomato",
            "state": ItemState.RAW,
            "cuts_done": 0,
            "cuts_needed": 3
        }
        item_id = self.create_item_in_state(self.game_state, item)
        self.game_state["ground_instances"][(3, 3)] = item_id
        
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0"],
            self.item_manager
        )
        
        # Channel 1 should have item code at (3, 3)
        expected_code = self.builder._item_to_code(item)
        self.assertEqual(obs["agent_0"][3, 3, 1], expected_code)
    
    def test_agent_positions_layers(self):
        """Test that agent positions are correctly encoded in channels 2 and 3."""
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0", "agent_1"],
            self.item_manager
        )
        
        # For agent_0:
        # Channel 2: agent_0 pos (1, 1) -> 1
        # Channel 3: agent_1 pos (3, 3) -> 1
        self.assertEqual(obs["agent_0"][1, 1, 2], 1)
        self.assertEqual(obs["agent_0"][3, 3, 3], 1)
        self.assertEqual(obs["agent_0"][1, 1, 3], 0)
        self.assertEqual(obs["agent_0"][3, 3, 2], 0)
        
        # For agent_1:
        # Channel 2: agent_1 pos (3, 3) -> 1
        # Channel 3: agent_0 pos (1, 1) -> 1
        self.assertEqual(obs["agent_1"][3, 3, 2], 1)
        self.assertEqual(obs["agent_1"][1, 1, 3], 1)
        self.assertEqual(obs["agent_1"][3, 3, 3], 0)
        self.assertEqual(obs["agent_1"][1, 1, 2], 0)
    
    def test_delivery_points_layer(self):
        """Test that delivery points are correctly encoded in channel 4."""
        obs = self.builder.build_observations(
            self.game_state,
            ["agent_0"],
            self.item_manager
        )
        
        # Channel 4 should have 1 at delivery point (2, 2)
        self.assertEqual(obs["agent_0"][2, 2, 4], 1)
        self.assertEqual(obs["agent_0"][0, 0, 4], 0)
    
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
        # Channel 6: cuts_done = 0
        # Channel 7: cuts_needed = 3
        # Channel 8: is_ready = 0
        self.assertEqual(obs["agent_0"][1, 1, 6], 0)
        self.assertEqual(obs["agent_0"][1, 1, 7], 3)
        self.assertEqual(obs["agent_0"][1, 1, 8], 0)
    
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
        
        # Channel 6: cuts_done = 2
        # Channel 7: cuts_needed = 3
        # Channel 8: is_ready = 0
        self.assertEqual(obs["agent_0"][1, 1, 6], 2)
        self.assertEqual(obs["agent_0"][1, 1, 7], 3)
        self.assertEqual(obs["agent_0"][1, 1, 8], 0)
    
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
        
        # Channel 6: cuts_done = 3
        # Channel 7: cuts_needed = 3
        # Channel 8: is_ready = 1
        self.assertEqual(obs["agent_0"][1, 1, 6], 3)
        self.assertEqual(obs["agent_0"][1, 1, 7], 3)
        self.assertEqual(obs["agent_0"][1, 1, 8], 1)
    
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
        
        # Channel 6: num_ingredients = 0
        # Channels 7-8: unused = 0
        self.assertEqual(obs["agent_0"][1, 1, 6], 0)
        self.assertEqual(obs["agent_0"][1, 1, 7], 0)
        self.assertEqual(obs["agent_0"][1, 1, 8], 0)
    
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
        
        # Channel 6: num_ingredients = 2
        # Channels 7-8: unused = 0
        self.assertEqual(obs["agent_0"][1, 1, 6], 2)
        self.assertEqual(obs["agent_0"][1, 1, 7], 0)
        self.assertEqual(obs["agent_0"][1, 1, 8], 0)
    
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
        self.assertEqual(obs["agent_0"][1, 1, 6], 0)  # cuts_done
        self.assertEqual(obs["agent_0"][1, 1, 7], 3)  # cuts_needed
        
        # Agent 1's processing vector (plate with 1 ingredient)
        self.assertEqual(obs["agent_1"][3, 3, 6], 1)  # num_ingredients
        self.assertEqual(obs["agent_1"][3, 3, 7], 0)  # unused
    
    def test_get_observation_shape(self):
        """Test get_observation_shape method."""
        shape = self.builder.get_observation_shape((10, 15))
        self.assertEqual(shape, (10, 15, 9))
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.builder)
        self.assertIn("channels=9", repr_str)
        self.assertIn("base=6", repr_str)
        self.assertIn("processing=3", repr_str)


if __name__ == '__main__':
    import unittest
    unittest.main()
