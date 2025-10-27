"""Unit tests for ItemInstanceManager."""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from overcooked_marl_env.core.item_instance_manager import ItemInstanceManager


class TestItemInstanceManager(unittest.TestCase):
    """Test suite for ItemInstanceManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.manager = ItemInstanceManager()
    
    def test_initialization(self):
        """Test that manager initializes with empty state."""
        self.assertEqual(self.manager.get_instance_count(), 0)
        self.assertEqual(self.manager.next_item_id, 0)
        self.assertEqual(len(self.manager.item_instances), 0)
    
    def test_create_instance(self):
        """Test creating item instances."""
        item1 = {"type": "ingredient", "name": "tomato"}
        item2 = {"type": "ingredient", "name": "lettuce"}
        
        id1 = self.manager.create_instance(item1)
        id2 = self.manager.create_instance(item2)
        
        self.assertEqual(id1, 0)
        self.assertEqual(id2, 1)
        self.assertEqual(self.manager.get_instance_count(), 2)
    
    def test_get_instance(self):
        """Test retrieving item instances."""
        item = {"type": "plate", "ingredients": []}
        item_id = self.manager.create_instance(item)
        
        retrieved = self.manager.get_instance(item_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["type"], "plate")
        
        # Test getting non-existent instance
        none_item = self.manager.get_instance(999)
        self.assertIsNone(none_item)
        
        # Test getting None ID
        none_item = self.manager.get_instance(None)
        self.assertIsNone(none_item)
    
    def test_update_instance(self):
        """Test updating item instances."""
        item = {"state": "raw"}
        item_id = self.manager.create_instance(item)
        
        success = self.manager.update_instance(item_id, {"state": "chopped"})
        self.assertTrue(success)
        
        updated = self.manager.get_instance(item_id)
        self.assertEqual(updated["state"], "chopped")
        
        # Test updating non-existent instance
        success = self.manager.update_instance(999, {"state": "ready"})
        self.assertFalse(success)
    
    def test_delete_instance(self):
        """Test deleting item instances."""
        item = {"name": "cucumber"}
        item_id = self.manager.create_instance(item)
        
        self.assertEqual(self.manager.get_instance_count(), 1)
        
        success = self.manager.delete_instance(item_id)
        self.assertTrue(success)
        self.assertEqual(self.manager.get_instance_count(), 0)
        
        # Test deleting non-existent instance
        success = self.manager.delete_instance(999)
        self.assertFalse(success)
    
    def test_agent_item_management(self):
        """Test managing items held by agents."""
        item = {"name": "plate"}
        item_id = self.manager.create_instance(item)
        
        # Set agent item
        self.manager.set_agent_item("agent_0", item_id)
        
        # Get agent item
        held_item = self.manager.get_agent_item("agent_0")
        self.assertIsNotNone(held_item)
        self.assertEqual(held_item["name"], "plate")
        
        # Get agent item ID
        held_id = self.manager.get_agent_item_id("agent_0")
        self.assertEqual(held_id, item_id)
        
        # Check if holding
        self.assertTrue(self.manager.is_agent_holding_item("agent_0"))
        
        # Clear agent item
        cleared_id = self.manager.clear_agent_item("agent_0")
        self.assertEqual(cleared_id, item_id)
        self.assertFalse(self.manager.is_agent_holding_item("agent_0"))
    
    def test_ground_item_management(self):
        """Test managing items on the ground."""
        item = {"name": "tomato"}
        item_id = self.manager.create_instance(item)
        
        position = (5, 3)
        
        # Set ground item
        self.manager.set_ground_item(position, item_id)
        
        # Get ground item
        ground_item = self.manager.get_ground_item(position)
        self.assertIsNotNone(ground_item)
        self.assertEqual(ground_item["name"], "tomato")
        
        # Get ground item ID
        ground_id = self.manager.get_ground_item_id(position)
        self.assertEqual(ground_id, item_id)
        
        # Check if position occupied
        self.assertTrue(self.manager.is_position_occupied(position))
        
        # Clear ground item
        cleared_id = self.manager.clear_ground_item(position)
        self.assertEqual(cleared_id, item_id)
        self.assertFalse(self.manager.is_position_occupied(position))
    
    def test_delete_cleans_references(self):
        """Test that deleting an instance cleans up all references."""
        item = {"name": "test_item"}
        item_id = self.manager.create_instance(item)
        
        # Set item for agent and ground
        self.manager.set_agent_item("agent_0", item_id)
        self.manager.set_ground_item((2, 2), item_id)
        
        # Delete the instance
        self.manager.delete_instance(item_id)
        
        # Verify cleanup
        self.assertIsNone(self.manager.get_agent_item("agent_0"))
        self.assertIsNone(self.manager.get_ground_item((2, 2)))
        self.assertEqual(self.manager.get_instance_count(), 0)
    
    def test_reset(self):
        """Test resetting the manager."""
        # Create some items
        self.manager.create_instance({"name": "item1"})
        self.manager.create_instance({"name": "item2"})
        self.manager.set_agent_item("agent_0", 0)
        
        # Reset
        self.manager.reset()
        
        # Verify reset
        self.assertEqual(self.manager.get_instance_count(), 0)
        self.assertEqual(self.manager.next_item_id, 0)
        self.assertEqual(len(self.manager.agent_holding_instance), 0)
        self.assertEqual(len(self.manager.ground_instances), 0)
    
    def test_state_serialization(self):
        """Test getting and loading state."""
        item_id = self.manager.create_instance({"name": "test"})
        self.manager.set_agent_item("agent_0", item_id)
        self.manager.set_ground_item((1, 1), item_id)
        
        # Get state
        state = self.manager.get_state_dict()
        
        # Create new manager and load state
        new_manager = ItemInstanceManager()
        new_manager.load_state_dict(state)
        
        # Verify state was loaded
        self.assertEqual(new_manager.get_instance_count(), 1)
        self.assertEqual(new_manager.next_item_id, self.manager.next_item_id)
        self.assertEqual(new_manager.get_agent_item_id("agent_0"), item_id)
        self.assertEqual(new_manager.get_ground_item_id((1, 1)), item_id)
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.manager)
        self.assertIn("ItemInstanceManager", repr_str)
        self.assertIn("instances=0", repr_str)


if __name__ == '__main__':
    unittest.main()
