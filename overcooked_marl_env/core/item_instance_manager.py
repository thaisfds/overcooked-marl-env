"""
Item Instance Manager for Overcooked MARL Environment.

This module provides a centralized manager for item instances, following the
Single Responsibility Principle. It handles the lifecycle of all item instances
in the game, including creation, retrieval, updates, and deletion.

The ItemInstanceManager separates item instance management from the main
environment logic, making the code more maintainable and testable.
"""

from typing import Dict, Optional, Tuple, Any


class ItemInstanceManager:
    """
    Manages the lifecycle of item instances in the game.
    
    This class is responsible for:
    - Creating unique item instances
    - Storing and retrieving item data
    - Managing item-agent associations
    - Managing item-ground associations
    - Cleaning up deleted items
    
    Attributes:
        item_instances (dict): Maps item_instance_id -> item_dict
        agent_holding_instance (dict): Maps agent_id -> item_instance_id
        ground_instances (dict): Maps (x, y) -> item_instance_id
        next_item_id (int): Counter for generating unique IDs
    """
    
    def __init__(self):
        """Initialize the ItemInstanceManager with empty state."""
        self.item_instances: Dict[int, dict] = {}
        self.agent_holding_instance: Dict[str, Optional[int]] = {}
        self.ground_instances: Dict[Tuple[int, int], int] = {}
        self.next_item_id: int = 0
    
    def reset(self):
        """
        Reset the manager to initial state.
        
        This clears all instances and resets the ID counter.
        """
        self.item_instances.clear()
        self.agent_holding_instance.clear()
        self.ground_instances.clear()
        self.next_item_id = 0
    
    def create_instance(self, item_dict: dict) -> int:
        """
        Create a new item instance with a unique ID.
        
        Args:
            item_dict: Dictionary containing item data (type, state, etc.)
        
        Returns:
            int: Unique ID of the created instance
        
        Example:
            >>> manager = ItemInstanceManager()
            >>> item = {"type": ItemType.INGREDIENT, "name": "tomato"}
            >>> item_id = manager.create_instance(item)
            >>> print(item_id)
            0
        """
        item_id = self.next_item_id
        self.next_item_id += 1
        self.item_instances[item_id] = item_dict.copy()
        return item_id
    
    def get_instance(self, instance_id: Optional[int]) -> Optional[dict]:
        """
        Retrieve an item instance by ID.
        
        Args:
            instance_id: The unique ID of the instance, or None
        
        Returns:
            dict: The item dictionary, or None if not found
        
        Example:
            >>> manager = ItemInstanceManager()
            >>> item_id = manager.create_instance({"name": "lettuce"})
            >>> item = manager.get_instance(item_id)
            >>> print(item["name"])
            lettuce
        """
        if instance_id is None:
            return None
        return self.item_instances.get(instance_id)
    
    def update_instance(self, instance_id: int, new_item_dict: dict) -> bool:
        """
        Update an existing item instance completely.
        
        Args:
            instance_id: The unique ID of the instance
            new_item_dict: New item dictionary to replace the old one
        
        Returns:
            bool: True if update succeeded, False if instance not found
        
        Example:
            >>> manager = ItemInstanceManager()
            >>> item_id = manager.create_instance({"state": "raw"})
            >>> manager.update_instance(item_id, {"state": "chopped"})
            True
        """
        if instance_id not in self.item_instances:
            return False
        self.item_instances[instance_id] = new_item_dict.copy()
        return True
    
    def delete_instance(self, instance_id: int) -> bool:
        """
        Delete an item instance and clean up all references.
        
        This removes the instance from:
        - The main instances dictionary
        - Agent holdings (if any agent is holding it)
        - Ground positions (if it's on the ground)
        
        Args:
            instance_id: The unique ID of the instance to delete
        
        Returns:
            bool: True if deletion succeeded, False if instance not found
        
        Example:
            >>> manager = ItemInstanceManager()
            >>> item_id = manager.create_instance({"trash": True})
            >>> manager.delete_instance(item_id)
            True
        """
        if instance_id not in self.item_instances:
            return False
        
        # Remove from main instances
        del self.item_instances[instance_id]
        
        # Remove from agent holdings
        agents_to_clear = [
            agent_id for agent_id, held_id in self.agent_holding_instance.items()
            if held_id == instance_id
        ]
        for agent_id in agents_to_clear:
            self.agent_holding_instance[agent_id] = None
        
        # Remove from ground
        positions_to_clear = [
            pos for pos, item_id in self.ground_instances.items()
            if item_id == instance_id
        ]
        for pos in positions_to_clear:
            del self.ground_instances[pos]
        
        return True
    
    def get_agent_item(self, agent_id: str) -> Optional[dict]:
        """
        Get the item that an agent is currently holding.
        
        Args:
            agent_id: The ID of the agent
        
        Returns:
            dict: The item dictionary, or None if agent is not holding anything
        
        Example:
            >>> manager = ItemInstanceManager()
            >>> item_id = manager.create_instance({"name": "plate"})
            >>> manager.set_agent_item("agent_0", item_id)
            >>> item = manager.get_agent_item("agent_0")
            >>> print(item["name"])
            plate
        """
        instance_id = self.agent_holding_instance.get(agent_id)
        return self.get_instance(instance_id)
    
    def set_agent_item(self, agent_id: str, instance_id: Optional[int]) -> None:
        """
        Set the item that an agent is holding.
        
        Args:
            agent_id: The ID of the agent
            instance_id: The ID of the item instance, or None to clear
        
        Example:
            >>> manager = ItemInstanceManager()
            >>> item_id = manager.create_instance({"name": "tomato"})
            >>> manager.set_agent_item("agent_0", item_id)
            >>> manager.set_agent_item("agent_0", None)  # Clear
        """
        self.agent_holding_instance[agent_id] = instance_id
    
    def get_agent_item_id(self, agent_id: str) -> Optional[int]:
        """
        Get the instance ID of the item an agent is holding.
        
        Args:
            agent_id: The ID of the agent
        
        Returns:
            int: The instance ID, or None if not holding anything
        """
        return self.agent_holding_instance.get(agent_id)
    
    def get_ground_item(self, position: Tuple[int, int]) -> Optional[dict]:
        """
        Get the item at a ground position.
        
        Args:
            position: Tuple of (x, y) coordinates
        
        Returns:
            dict: The item dictionary, or None if no item at position
        
        Example:
            >>> manager = ItemInstanceManager()
            >>> item_id = manager.create_instance({"name": "cucumber"})
            >>> manager.set_ground_item((5, 3), item_id)
            >>> item = manager.get_ground_item((5, 3))
            >>> print(item["name"])
            cucumber
        """
        instance_id = self.ground_instances.get(position)
        return self.get_instance(instance_id)
    
    def set_ground_item(self, position: Tuple[int, int], instance_id: Optional[int]) -> None:
        """
        Set the item at a ground position.
        
        Args:
            position: Tuple of (x, y) coordinates
            instance_id: The ID of the item instance, or None to clear
        
        Example:
            >>> manager = ItemInstanceManager()
            >>> item_id = manager.create_instance({"name": "onion"})
            >>> manager.set_ground_item((2, 4), item_id)
            >>> manager.set_ground_item((2, 4), None)  # Clear
        """
        if instance_id is None:
            # Clear the position
            if position in self.ground_instances:
                del self.ground_instances[position]
        else:
            self.ground_instances[position] = instance_id
    
    def get_ground_item_id(self, position: Tuple[int, int]) -> Optional[int]:
        """
        Get the instance ID of the item at a ground position.
        
        Args:
            position: Tuple of (x, y) coordinates
        
        Returns:
            int: The instance ID, or None if no item at position
        """
        return self.ground_instances.get(position)
    
    def get_all_instances(self) -> Dict[int, dict]:
        """
        Get all item instances.
        
        Returns:
            dict: Dictionary mapping instance_id -> item_dict
        
        Note:
            Returns a copy to prevent external modification.
        """
        return self.item_instances.copy()
    
    def get_instance_count(self) -> int:
        """
        Get the total number of active instances.
        
        Returns:
            int: Number of instances currently managed
        """
        return len(self.item_instances)
    
    def clear_agent_item(self, agent_id: str) -> Optional[int]:
        """
        Clear an agent's held item and return its ID.
        
        Args:
            agent_id: The ID of the agent
        
        Returns:
            int: The ID of the item that was cleared, or None
        
        Example:
            >>> manager = ItemInstanceManager()
            >>> item_id = manager.create_instance({"name": "plate"})
            >>> manager.set_agent_item("agent_0", item_id)
            >>> cleared_id = manager.clear_agent_item("agent_0")
            >>> print(cleared_id == item_id)
            True
        """
        instance_id = self.agent_holding_instance.get(agent_id)
        self.agent_holding_instance[agent_id] = None
        return instance_id
    
    def clear_ground_item(self, position: Tuple[int, int]) -> Optional[int]:
        """
        Clear an item from a ground position and return its ID.
        
        Args:
            position: Tuple of (x, y) coordinates
        
        Returns:
            int: The ID of the item that was cleared, or None
        
        Example:
            >>> manager = ItemInstanceManager()
            >>> item_id = manager.create_instance({"name": "tomato"})
            >>> manager.set_ground_item((3, 3), item_id)
            >>> cleared_id = manager.clear_ground_item((3, 3))
            >>> print(cleared_id == item_id)
            True
        """
        instance_id = self.ground_instances.get(position)
        if position in self.ground_instances:
            del self.ground_instances[position]
        return instance_id
    
    def is_agent_holding_item(self, agent_id: str) -> bool:
        """
        Check if an agent is holding any item.
        
        Args:
            agent_id: The ID of the agent
        
        Returns:
            bool: True if agent is holding an item, False otherwise
        """
        instance_id = self.agent_holding_instance.get(agent_id)
        return instance_id is not None and instance_id in self.item_instances
    
    def is_position_occupied(self, position: Tuple[int, int]) -> bool:
        """
        Check if a ground position has an item.
        
        Args:
            position: Tuple of (x, y) coordinates
        
        Returns:
            bool: True if position has an item, False otherwise
        """
        instance_id = self.ground_instances.get(position)
        return instance_id is not None and instance_id in self.item_instances
    
    def get_state_dict(self) -> dict:
        """
        Get the complete state as a dictionary for serialization.
        
        Returns:
            dict: Complete state including all instances and associations
        
        This is useful for:
        - Saving/loading game state
        - Debugging
        - State inspection
        """
        return {
            "item_instances": self.item_instances.copy(),
            "agent_holding_instance": self.agent_holding_instance.copy(),
            "ground_instances": self.ground_instances.copy(),
            "next_item_id": self.next_item_id
        }
    
    def load_state_dict(self, state: dict) -> None:
        """
        Load state from a dictionary.
        
        Args:
            state: State dictionary from get_state_dict()
        
        This allows restoring a previously saved state.
        """
        self.item_instances = state.get("item_instances", {}).copy()
        self.agent_holding_instance = state.get("agent_holding_instance", {}).copy()
        self.ground_instances = state.get("ground_instances", {}).copy()
        self.next_item_id = state.get("next_item_id", 0)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"ItemInstanceManager("
                f"instances={len(self.item_instances)}, "
                f"next_id={self.next_item_id})")


__all__ = ['ItemInstanceManager']
