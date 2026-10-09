"""Unit tests for GameState."""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from overcooked_marl_env.environment.game_state import GameState, MapData


class TestGameState(unittest.TestCase):
    """Test suite for GameState class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple test map
        static_map = np.array([
            [1, 2, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.uint8)
        
        agent_positions = {
            "agent_0": (1, 1),
            "agent_1": (2, 1)
        }
        
        dispensers = {
            (0, 1): "tomato",
            (2, 2): "lettuce"
        }
        
        recipes = ["SIMPLE_SALAD", "GARDEN_SALAD"]
        
        self.map_data = MapData(
            static_map=static_map,
            agent_positions=agent_positions,
            dispensers=dispensers,
            recipes=recipes,
            size=(3, 3)
        )
        
        self.agent_ids = ["agent_0", "agent_1"]
        self.game_state = GameState(self.map_data, self.agent_ids)
    
    def test_initialization(self):
        """Test that game state initializes correctly."""
        self.assertEqual(self.game_state.map_size, (3, 3))
        self.assertEqual(len(self.game_state.agents), 2)
        self.assertEqual(self.game_state.time_steps, 0)
        self.assertEqual(self.game_state.dishes_pending, 2)
        self.assertEqual(self.game_state.expected_recipe_index, 0)
    
    def test_agent_positions(self):
        """Test agent position management."""
        # Get initial position
        pos = self.game_state.get_agent_position("agent_0")
        self.assertEqual(pos, (1, 1))
        
        # Update position
        new_pos = (2, 2)
        self.game_state.set_agent_position("agent_0", new_pos)
        self.assertEqual(self.game_state.get_agent_position("agent_0"), new_pos)
        
        # Test update_agent_position (alias)
        self.game_state.update_agent_position("agent_0", (1, 1))
        self.assertEqual(self.game_state.get_agent_position("agent_0"), (1, 1))
    
    def test_agent_orientations(self):
        """Test agent orientation management."""
        # Get initial orientation (should be North)
        orientation = self.game_state.get_agent_orientation("agent_0")
        self.assertEqual(orientation, (0, -1))
        
        # Set new orientation
        self.game_state.set_agent_orientation("agent_0", (1, 0))  # East
        self.assertEqual(self.game_state.get_agent_orientation("agent_0"), (1, 0))
    
    def test_position_in_front(self):
        """Test getting position in front of agent."""
        # Agent at (1, 1) facing North (0, -1)
        front_pos = self.game_state.get_position_in_front("agent_0")
        self.assertEqual(front_pos, (1, 0))
        
        # Change orientation to East
        self.game_state.set_agent_orientation("agent_0", (1, 0))
        front_pos = self.game_state.get_position_in_front("agent_0")
        self.assertEqual(front_pos, (2, 1))
    
    def test_position_validation(self):
        """Test position validation."""
        self.assertTrue(self.game_state.is_position_valid((0, 0)))
        self.assertTrue(self.game_state.is_position_valid((2, 2)))
        self.assertFalse(self.game_state.is_position_valid((-1, 0)))
        self.assertFalse(self.game_state.is_position_valid((3, 0)))
        self.assertFalse(self.game_state.is_position_valid((0, 3)))
    
    def test_ground_items(self):
        """Test ground item management."""
        position = (1, 1)
        
        # Initially no item
        self.assertEqual(self.game_state.get_ground_item_code(position), 0)
        self.assertFalse(self.game_state.has_ground_item(position))
        
        # Set item code
        self.game_state.set_ground_item_code(position, 5)
        self.assertEqual(self.game_state.get_ground_item_code(position), 5)
        self.assertTrue(self.game_state.has_ground_item(position))
        
        # Clear item
        self.game_state.clear_ground_item(position)
        self.assertEqual(self.game_state.get_ground_item_code(position), 0)
        self.assertFalse(self.game_state.has_ground_item(position))
    
    def test_map_cells(self):
        """Test accessing map cells."""
        # Test valid position
        code = self.game_state.get_cell_code((1, 1))
        self.assertEqual(code, 0)  # From our test map
        
        code = self.game_state.get_cell_code((0, 0))
        self.assertEqual(code, 1)
        
        # Test invalid position
        code = self.game_state.get_cell_code((10, 10))
        self.assertEqual(code, 0)
    
    def test_dispensers(self):
        """Test dispenser management."""
        # Check if position is dispenser
        self.assertTrue(self.game_state.is_dispenser_position((0, 1)))
        self.assertFalse(self.game_state.is_dispenser_position((1, 1)))
        
        # Get dispenser ingredient
        ingredient = self.game_state.get_dispenser_ingredient((0, 1))
        self.assertEqual(ingredient, "tomato")
        
        ingredient = self.game_state.get_dispenser_ingredient((2, 2))
        self.assertEqual(ingredient, "lettuce")
        
        # Non-existent dispenser
        ingredient = self.game_state.get_dispenser_ingredient((1, 1))
        self.assertIsNone(ingredient)
    
    def test_recipes(self):
        """Test recipe management."""
        # Get current expected recipe
        recipe = self.game_state.get_current_expected_recipe()
        self.assertEqual(recipe, "SIMPLE_SALAD")
        
        # Get pending recipes
        pending = self.game_state.get_pending_recipes()
        self.assertEqual(len(pending), 2)
        self.assertEqual(pending[0], "SIMPLE_SALAD")
        
        # Mark recipe as delivered
        self.game_state.mark_recipe_delivered()
        self.assertEqual(self.game_state.get_delivered_count(), 1)
        self.assertEqual(self.game_state.dishes_pending, 1)
        self.assertEqual(self.game_state.get_current_expected_recipe(), "GARDEN_SALAD")
        
        # Deliver second recipe
        self.game_state.mark_recipe_delivered()
        self.assertEqual(self.game_state.get_delivered_count(), 2)
        self.assertTrue(self.game_state.are_all_recipes_delivered())
        self.assertIsNone(self.game_state.get_current_expected_recipe())
    
    def test_timesteps(self):
        """Test timestep management."""
        self.assertEqual(self.game_state.get_timestep(), 0)
        
        self.game_state.increment_timestep()
        self.assertEqual(self.game_state.get_timestep(), 1)
        
        self.game_state.increment_timestep()
        self.assertEqual(self.game_state.get_timestep(), 2)
    
    def test_episode_completion(self):
        """Test episode completion checking."""
        # Not complete initially
        terminated, truncated = self.game_state.is_episode_complete(max_steps=500)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        
        # Complete all recipes (terminated)
        self.game_state.dishes_pending = 0
        terminated, truncated = self.game_state.is_episode_complete(max_steps=500)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        
        # Reset and test truncation
        self.game_state.dishes_pending = 3
        self.game_state.time_steps = 500
        terminated, truncated = self.game_state.is_episode_complete(max_steps=500)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
    
    def test_rewards(self):
        """Test reward management."""
        # Initial rewards
        self.assertEqual(self.game_state.get_accumulated_reward("agent_0"), 0.0)
        self.assertEqual(self.game_state.get_last_step_reward("agent_0"), 0.0)
        
        # Add reward
        self.game_state.add_reward("agent_0", 10.0)
        self.assertEqual(self.game_state.get_accumulated_reward("agent_0"), 10.0)
        self.assertEqual(self.game_state.get_last_step_reward("agent_0"), 10.0)
        
        # Add more reward
        self.game_state.add_reward("agent_0", 5.0)
        self.assertEqual(self.game_state.get_accumulated_reward("agent_0"), 15.0)
        self.assertEqual(self.game_state.get_last_step_reward("agent_0"), 5.0)
        
        # Set last step reward
        self.game_state.set_last_step_reward("agent_0", 3.0)
        self.assertEqual(self.game_state.get_last_step_reward("agent_0"), 3.0)
        self.assertEqual(self.game_state.get_accumulated_reward("agent_0"), 15.0)  # Unchanged
        
        # Clear last step rewards
        self.game_state.clear_last_step_rewards()
        self.assertEqual(self.game_state.get_last_step_reward("agent_0"), 0.0)
        self.assertEqual(self.game_state.get_accumulated_reward("agent_0"), 15.0)  # Unchanged
    
    def test_reset(self):
        """Test resetting game state."""
        # Modify state
        self.game_state.set_agent_position("agent_0", (2, 2))
        self.game_state.increment_timestep()
        self.game_state.add_reward("agent_0", 10.0)
        self.game_state.mark_recipe_delivered()
        
        # Reset
        self.game_state.reset(self.map_data)
        
        # Verify reset
        self.assertEqual(self.game_state.get_agent_position("agent_0"), (1, 1))
        self.assertEqual(self.game_state.time_steps, 0)
        self.assertEqual(self.game_state.get_accumulated_reward("agent_0"), 0.0)
        self.assertEqual(self.game_state.expected_recipe_index, 0)
        self.assertEqual(len(self.game_state.delivered_recipes), 0)
    
    def test_state_serialization(self):
        """Test getting state as dictionary."""
        self.game_state.set_agent_position("agent_0", (2, 1))
        self.game_state.increment_timestep()
        
        state_dict = self.game_state.get_state_dict()
        
        # Verify all keys present
        expected_keys = [
            "map_static", "map_size", "items_on_ground", "dispensers",
            "agent_positions", "agent_orientations", "recipe_order",
            "expected_recipe_index", "delivered_recipes", "dishes_pending",
            "time_steps", "accumulated_rewards", "last_step_rewards", "agents"
        ]
        for key in expected_keys:
            self.assertIn(key, state_dict)
        
        # Verify some values
        self.assertEqual(state_dict["time_steps"], 1)
        self.assertEqual(state_dict["agent_positions"]["agent_0"], (2, 1))
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.game_state)
        self.assertIn("GameState", repr_str)
        self.assertIn("timestep=0", repr_str)


class TestMapData(unittest.TestCase):
    """Test suite for MapData dataclass."""
    
    def test_creation(self):
        """Test creating MapData."""
        static_map = np.zeros((5, 5), dtype=np.uint8)
        agent_positions = {"agent_0": (2, 2)}
        dispensers = {(1, 1): "tomato"}
        recipes = ["SIMPLE_SALAD"]
        size = (5, 5)
        
        map_data = MapData(
            static_map=static_map,
            agent_positions=agent_positions,
            dispensers=dispensers,
            recipes=recipes,
            size=size
        )
        
        self.assertIsNotNone(map_data)
        self.assertEqual(map_data.size, (5, 5))
        self.assertEqual(len(map_data.recipes), 1)


if __name__ == '__main__':
    unittest.main()
