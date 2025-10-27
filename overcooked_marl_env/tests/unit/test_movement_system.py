"""
Unit tests for MovementSystem.

Tests the movement processing logic in isolation.
"""

import unittest
import numpy as np
from overcooked_marl_env.mechanics import MovementSystem
from overcooked_marl_env.core.items import ItemManager


class TestMovementSystem(unittest.TestCase):
    """Test suite for MovementSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Standard action to direction mapping
        self.action_to_direction = {
            0: (0, -1),  # UP
            1: (0, 1),   # DOWN
            2: (-1, 0),  # LEFT
            3: (1, 0),   # RIGHT
        }
        
        self.movement_system = MovementSystem(self.action_to_direction)
        self.item_manager = ItemManager()
        
        # Create a simple 5x5 map
        # Layout:
        # |||||
        # |   |
        # | # |
        # |   |
        # |||||
        self.map_static = np.array([
            [1, 1, 1, 1, 1],  # Walls (|)
            [1, 0, 0, 0, 1],  # Empty spaces
            [1, 0, 2, 0, 1],  # Counter (#) in middle
            [1, 0, 0, 0, 1],  # Empty spaces
            [1, 1, 1, 1, 1],  # Walls
        ], dtype=np.uint8)
        
        self.map_size = (5, 5)  # height, width
    
    def _create_game_state(self, agent_positions):
        """Helper to create a minimal game state for testing."""
        return {
            "map_static": self.map_static,
            "agent_pos": agent_positions.copy(),
            "agent_orientation": {
                agent_id: (0, -1) for agent_id in agent_positions.keys()
            }
        }
    
    def test_initialization(self):
        """Test MovementSystem initialization."""
        self.assertEqual(len(self.movement_system.action_to_direction), 4)
        self.assertIn(0, self.movement_system.action_to_direction)
        self.assertEqual(self.movement_system.action_to_direction[0], (0, -1))
    
    def test_valid_movement_up(self):
        """Test valid upward movement."""
        agent_pos = {"agent_0": (2, 3)}  # Middle of bottom empty row
        game_state = self._create_game_state(agent_pos)
        
        actions = {"agent_0": 0}  # UP
        
        intended = self.movement_system.process_movement_actions(
            actions, game_state, self.map_size, self.item_manager
        )
        
        # Should move from (2, 3) to (2, 2) - but there's a counter at (2, 2)
        # Actually, let's use (1, 3) -> (1, 2) which should be valid
        agent_pos = {"agent_0": (1, 3)}
        game_state = self._create_game_state(agent_pos)
        
        intended = self.movement_system.process_movement_actions(
            actions, game_state, self.map_size, self.item_manager
        )
        
        self.assertEqual(intended["agent_0"], (1, 2))
        # Orientation should be updated
        self.assertEqual(game_state["agent_orientation"]["agent_0"], (0, -1))
    
    def test_valid_movement_down(self):
        """Test valid downward movement."""
        agent_pos = {"agent_0": (1, 1)}
        game_state = self._create_game_state(agent_pos)
        
        actions = {"agent_0": 1}  # DOWN
        
        intended = self.movement_system.process_movement_actions(
            actions, game_state, self.map_size, self.item_manager
        )
        
        self.assertEqual(intended["agent_0"], (1, 2))
        self.assertEqual(game_state["agent_orientation"]["agent_0"], (0, 1))
    
    def test_valid_movement_left(self):
        """Test valid leftward movement."""
        agent_pos = {"agent_0": (3, 1)}
        game_state = self._create_game_state(agent_pos)
        
        actions = {"agent_0": 2}  # LEFT
        
        intended = self.movement_system.process_movement_actions(
            actions, game_state, self.map_size, self.item_manager
        )
        
        self.assertEqual(intended["agent_0"], (2, 1))
        self.assertEqual(game_state["agent_orientation"]["agent_0"], (-1, 0))
    
    def test_valid_movement_right(self):
        """Test valid rightward movement."""
        agent_pos = {"agent_0": (1, 1)}
        game_state = self._create_game_state(agent_pos)
        
        actions = {"agent_0": 3}  # RIGHT
        
        intended = self.movement_system.process_movement_actions(
            actions, game_state, self.map_size, self.item_manager
        )
        
        self.assertEqual(intended["agent_0"], (2, 1))
        self.assertEqual(game_state["agent_orientation"]["agent_0"], (1, 0))
    
    def test_blocked_by_wall(self):
        """Test movement blocked by wall."""
        agent_pos = {"agent_0": (1, 1)}  # Near top-left wall
        game_state = self._create_game_state(agent_pos)
        
        actions = {"agent_0": 0}  # UP - will hit wall
        
        intended = self.movement_system.process_movement_actions(
            actions, game_state, self.map_size, self.item_manager
        )
        
        # Should stay in place (wall at (1, 0))
        self.assertEqual(intended["agent_0"], (1, 1))
        # But orientation should still update
        self.assertEqual(game_state["agent_orientation"]["agent_0"], (0, -1))
    
    def test_blocked_by_counter(self):
        """Test movement blocked by counter."""
        agent_pos = {"agent_0": (1, 2)}  # Left of counter
        game_state = self._create_game_state(agent_pos)
        
        actions = {"agent_0": 3}  # RIGHT - will hit counter at (2, 2)
        
        intended = self.movement_system.process_movement_actions(
            actions, game_state, self.map_size, self.item_manager
        )
        
        # Should stay in place (counter blocking)
        self.assertEqual(intended["agent_0"], (1, 2))
        # Orientation updates
        self.assertEqual(game_state["agent_orientation"]["agent_0"], (1, 0))
    
    def test_map_boundary_clipping(self):
        """Test that positions are clipped at map boundaries."""
        # Start at bottom-right corner
        agent_pos = {"agent_0": (3, 3)}
        game_state = self._create_game_state(agent_pos)
        
        actions = {"agent_0": 1}  # DOWN - should hit boundary
        
        intended = self.movement_system.process_movement_actions(
            actions, game_state, self.map_size, self.item_manager
        )
        
        # Position (3, 4) is a wall, so should stay at (3, 3)
        self.assertEqual(intended["agent_0"], (3, 3))
    
    def test_non_movement_action(self):
        """Test that non-movement actions (interact) keep agent in place."""
        agent_pos = {"agent_0": (1, 1)}
        game_state = self._create_game_state(agent_pos)
        
        actions = {"agent_0": 4}  # INTERACT (not a movement action)
        
        intended = self.movement_system.process_movement_actions(
            actions, game_state, self.map_size, self.item_manager
        )
        
        # Should stay in same position
        self.assertEqual(intended["agent_0"], (1, 1))
        # Orientation should NOT change
        self.assertEqual(game_state["agent_orientation"]["agent_0"], (0, -1))
    
    def test_multiple_agents_independent_movement(self):
        """Test that multiple agents can move independently."""
        agent_pos = {
            "agent_0": (1, 1),
            "agent_1": (3, 3)
        }
        game_state = self._create_game_state(agent_pos)
        
        actions = {
            "agent_0": 3,  # RIGHT
            "agent_1": 2   # LEFT
        }
        
        intended = self.movement_system.process_movement_actions(
            actions, game_state, self.map_size, self.item_manager
        )
        
        # agent_0: (1, 1) -> (2, 1) BUT (2, 1) might be blocked
        # Let's check: (2, 1) should be empty
        # agent_1: (3, 3) -> (2, 3)
        self.assertEqual(intended["agent_0"], (2, 1))
        self.assertEqual(intended["agent_1"], (2, 3))
    
    def test_orientation_always_updates_on_movement_action(self):
        """Test that orientation updates even when movement is blocked."""
        agent_pos = {"agent_0": (1, 1)}
        game_state = self._create_game_state(agent_pos)
        
        # Try all directions against a wall
        test_cases = [
            (0, (0, -1)),  # UP
            (1, (0, 1)),   # DOWN
            (2, (-1, 0)),  # LEFT
            (3, (1, 0)),   # RIGHT
        ]
        
        for action, expected_orientation in test_cases:
            actions = {"agent_0": action}
            self.movement_system.process_movement_actions(
                actions, game_state, self.map_size, self.item_manager
            )
            
            self.assertEqual(
                game_state["agent_orientation"]["agent_0"],
                expected_orientation,
                f"Orientation should update for action {action}"
            )
    
    def test_get_movement_info_successful(self):
        """Test movement info generation for successful movement."""
        game_state = self._create_game_state({"agent_0": (1, 1)})
        
        info = self.movement_system.get_movement_info(
            "agent_0", 3, (1, 1), (2, 1), game_state, self.item_manager
        )
        
        self.assertIn("moved from", info)
        self.assertIn("(1, 1)", info)
        self.assertIn("(2, 1)", info)
    
    def test_get_movement_info_blocked(self):
        """Test movement info generation for blocked movement."""
        game_state = self._create_game_state({"agent_0": (1, 1)})
        
        info = self.movement_system.get_movement_info(
            "agent_0", 0, (1, 1), (1, 1), game_state, self.item_manager
        )
        
        self.assertIn("blocked", info.lower())
    
    def test_get_movement_info_non_movement(self):
        """Test movement info for non-movement action."""
        game_state = self._create_game_state({"agent_0": (1, 1)})
        
        info = self.movement_system.get_movement_info(
            "agent_0", 4, (1, 1), (1, 1), game_state, self.item_manager
        )
        
        self.assertIn("did not attempt to move", info)
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.movement_system)
        self.assertIn("MovementSystem", repr_str)
        self.assertIn("4", repr_str)  # 4 actions


if __name__ == '__main__':
    unittest.main()
