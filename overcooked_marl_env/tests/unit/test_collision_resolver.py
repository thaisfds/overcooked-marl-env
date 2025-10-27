"""
Unit tests for CollisionResolver.

Tests the collision detection and resolution logic in isolation.
"""

import unittest
from overcooked_marl_env.mechanics import CollisionResolver


class TestCollisionResolver(unittest.TestCase):
    """Test suite for CollisionResolver class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.resolver = CollisionResolver(collision_penalty=0.5)
    
    def test_initialization(self):
        """Test CollisionResolver initialization."""
        self.assertEqual(self.resolver.collision_penalty, 0.5)
        
        # Test custom penalty
        custom_resolver = CollisionResolver(collision_penalty=1.0)
        self.assertEqual(custom_resolver.collision_penalty, 1.0)
    
    def test_no_collision_single_agent(self):
        """Test single agent movement with no collision."""
        intended = {"agent_0": (2, 2)}
        current = {"agent_0": (1, 1)}
        
        final, penalties = self.resolver.resolve_collisions(
            intended, current, verbose=False
        )
        
        self.assertEqual(final["agent_0"], (2, 2))
        self.assertEqual(penalties["agent_0"], 0.0)
    
    def test_no_collision_multiple_agents(self):
        """Test multiple agents moving to different positions."""
        intended = {
            "agent_0": (2, 1),
            "agent_1": (1, 2)
        }
        current = {
            "agent_0": (1, 1),
            "agent_1": (2, 2)
        }
        
        final, penalties = self.resolver.resolve_collisions(
            intended, current, verbose=False
        )
        
        self.assertEqual(final["agent_0"], (2, 1))
        self.assertEqual(final["agent_1"], (1, 2))
        self.assertEqual(penalties["agent_0"], 0.0)
        self.assertEqual(penalties["agent_1"], 0.0)
    
    def test_collision_with_stationary_agent(self):
        """Test agent trying to move to position occupied by another agent."""
        intended = {
            "agent_0": (2, 2),  # Trying to move to agent_1's position
            "agent_1": (2, 2)   # Staying in place
        }
        current = {
            "agent_0": (1, 1),
            "agent_1": (2, 2)
        }
        
        final, penalties = self.resolver.resolve_collisions(
            intended, current, verbose=False
        )
        
        # agent_0 should be blocked
        self.assertEqual(final["agent_0"], (1, 1))
        # agent_1 should stay in place
        self.assertEqual(final["agent_1"], (2, 2))
        # No penalties (agent_0 was just blocked, not in conflict)
        self.assertEqual(penalties["agent_0"], 0.0)
        self.assertEqual(penalties["agent_1"], 0.0)
    
    def test_head_on_collision(self):
        """Test two agents trying to move to same position (conflict)."""
        intended = {
            "agent_0": (2, 2),
            "agent_1": (2, 2)
        }
        current = {
            "agent_0": (1, 2),
            "agent_1": (3, 2)
        }
        
        final, penalties = self.resolver.resolve_collisions(
            intended, current, verbose=False
        )
        
        # Both agents should stay in place
        self.assertEqual(final["agent_0"], (1, 2))
        self.assertEqual(final["agent_1"], (3, 2))
        # Both should receive penalty
        self.assertEqual(penalties["agent_0"], 0.5)
        self.assertEqual(penalties["agent_1"], 0.5)
    
    def test_three_way_conflict(self):
        """Test three agents trying to move to same position."""
        intended = {
            "agent_0": (2, 2),
            "agent_1": (2, 2),
            "agent_2": (2, 2)
        }
        current = {
            "agent_0": (1, 2),
            "agent_1": (3, 2),
            "agent_2": (2, 1)
        }
        
        final, penalties = self.resolver.resolve_collisions(
            intended, current, verbose=False
        )
        
        # All agents should stay in place
        self.assertEqual(final["agent_0"], (1, 2))
        self.assertEqual(final["agent_1"], (3, 2))
        self.assertEqual(final["agent_2"], (2, 1))
        # All should receive penalty
        self.assertEqual(penalties["agent_0"], 0.5)
        self.assertEqual(penalties["agent_1"], 0.5)
        self.assertEqual(penalties["agent_2"], 0.5)
    
    def test_agent_staying_in_place(self):
        """Test that agents staying in place don't cause conflicts."""
        intended = {
            "agent_0": (1, 1),  # Not moving
            "agent_1": (2, 2)   # Moving
        }
        current = {
            "agent_0": (1, 1),
            "agent_1": (1, 2)
        }
        
        final, penalties = self.resolver.resolve_collisions(
            intended, current, verbose=False
        )
        
        self.assertEqual(final["agent_0"], (1, 1))
        self.assertEqual(final["agent_1"], (2, 2))
        self.assertEqual(penalties["agent_0"], 0.0)
        self.assertEqual(penalties["agent_1"], 0.0)
    
    def test_swap_positions(self):
        """Test two agents trying to swap positions."""
        intended = {
            "agent_0": (2, 1),
            "agent_1": (1, 1)
        }
        current = {
            "agent_0": (1, 1),
            "agent_1": (2, 1)
        }
        
        final, penalties = self.resolver.resolve_collisions(
            intended, current, verbose=False
        )
        
        # agent_0 wants to go to (2, 1) where agent_1 currently is
        # agent_1 wants to go to (1, 1) where agent_0 currently is
        # Both should be blocked initially
        self.assertEqual(final["agent_0"], (1, 1))
        self.assertEqual(final["agent_1"], (2, 1))
    
    def test_mixed_scenario(self):
        """Test complex scenario with multiple agents, some moving, some not."""
        intended = {
            "agent_0": (3, 3),  # Moving
            "agent_1": (2, 2),  # Not moving
            "agent_2": (3, 3),  # Conflict with agent_0
            "agent_3": (4, 4)   # Moving independently
        }
        current = {
            "agent_0": (3, 2),
            "agent_1": (2, 2),
            "agent_2": (3, 4),
            "agent_3": (4, 3)
        }
        
        final, penalties = self.resolver.resolve_collisions(
            intended, current, verbose=False
        )
        
        # agent_0 and agent_2 conflict at (3, 3)
        self.assertEqual(final["agent_0"], (3, 2))
        self.assertEqual(final["agent_2"], (3, 4))
        self.assertEqual(penalties["agent_0"], 0.5)
        self.assertEqual(penalties["agent_2"], 0.5)
        
        # agent_1 stays in place
        self.assertEqual(final["agent_1"], (2, 2))
        self.assertEqual(penalties["agent_1"], 0.0)
        
        # agent_3 moves successfully
        self.assertEqual(final["agent_3"], (4, 4))
        self.assertEqual(penalties["agent_3"], 0.0)
    
    def test_is_position_occupied(self):
        """Test internal method for checking if position is occupied."""
        current = {
            "agent_0": (1, 1),
            "agent_1": (2, 2)
        }
        
        # Check if (2, 2) is occupied (should be True for agent_0)
        occupied = self.resolver._is_position_occupied(
            (2, 2), "agent_0", current
        )
        self.assertTrue(occupied)
        
        # Check if (2, 2) is occupied (should be False for agent_1 itself)
        occupied = self.resolver._is_position_occupied(
            (2, 2), "agent_1", current
        )
        self.assertFalse(occupied)
        
        # Check if (3, 3) is occupied (should be False)
        occupied = self.resolver._is_position_occupied(
            (3, 3), "agent_0", current
        )
        self.assertFalse(occupied)
    
    def test_detect_conflicts(self):
        """Test internal method for detecting conflicts."""
        temp_positions = {
            "agent_0": (2, 2),
            "agent_1": (2, 2),
            "agent_2": (3, 3)
        }
        current_positions = {
            "agent_0": (1, 1),
            "agent_1": (1, 2),
            "agent_2": (3, 3)  # Not moving
        }
        
        conflicts = self.resolver._detect_conflicts(
            temp_positions, current_positions
        )
        
        # Should detect conflict at (2, 2)
        self.assertIn((2, 2), conflicts)
        self.assertEqual(len(conflicts[(2, 2)]), 2)
        self.assertIn("agent_0", conflicts[(2, 2)])
        self.assertIn("agent_1", conflicts[(2, 2)])
        
        # Should NOT include (3, 3) since agent_2 is not moving
        self.assertNotIn((3, 3), conflicts)
    
    def test_get_collision_summary(self):
        """Test collision summary generation."""
        intended = {
            "agent_0": (2, 2),
            "agent_1": (3, 3),
            "agent_2": (1, 1)
        }
        final = {
            "agent_0": (2, 2),  # Moved successfully
            "agent_1": (2, 3),  # Blocked (stayed in place but wanted to move)
            "agent_2": (1, 1)   # No movement
        }
        current = {
            "agent_0": (1, 1),
            "agent_1": (2, 3),
            "agent_2": (1, 1)
        }
        
        summary = self.resolver.get_collision_summary(
            intended, final, current
        )
        
        self.assertEqual(summary["agent_0"], "moved_successfully")
        self.assertEqual(summary["agent_1"], "blocked")  # Changed from "no_movement" to "blocked"
        self.assertEqual(summary["agent_2"], "no_movement")
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.resolver)
        self.assertIn("CollisionResolver", repr_str)
        self.assertIn("0.5", repr_str)


if __name__ == '__main__':
    unittest.main()
