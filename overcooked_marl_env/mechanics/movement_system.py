"""
Movement System - Handles agent movement and orientation updates.

This module implements the Single Responsibility Principle (SRP) by separating
movement logic from the main environment class.

Responsibilities:
- Process movement actions (UP, DOWN, LEFT, RIGHT)
- Update agent orientations based on actions
- Calculate intended positions
- Validate movements against map boundaries
- Validate movements against static obstacles (walls, stations)
"""

from typing import Dict, Tuple, Optional
import numpy as np


class MovementSystem:
    """
    Handles all movement-related logic for agents.
    
    This class is responsible for:
    1. Calculating intended positions based on actions
    2. Updating agent orientations
    3. Validating movements against map boundaries and obstacles
    
    Uses dependency injection to receive game state and item manager.
    """
    
    def __init__(self, action_to_direction: Dict[int, Tuple[int, int]]):
        """
        Initialize the movement system.
        
        Args:
            action_to_direction: Mapping of action codes to direction vectors
                                {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        """
        self.action_to_direction = action_to_direction
    
    def process_movement_actions(
        self,
        actions: Dict[str, int],
        game_state: Dict,
        map_size: Tuple[int, int],
        item_manager
    ) -> Dict[str, Tuple[int, int]]:
        """
        Process all movement actions and calculate intended positions.
        
        This method:
        1. Updates agent orientations based on movement actions
        2. Calculates intended positions
        3. Validates against map boundaries
        4. Validates against static obstacles (walls, stations)
        
        Args:
            actions: Dictionary mapping agent_id -> action code
            game_state: Current game state dictionary
            map_size: Tuple of (height, width)
            item_manager: ItemManager instance for cell codes
        
        Returns:
            Dictionary mapping agent_id -> intended position (x, y)
            Returns current position if movement is invalid
        """
        intended_positions = {}
        space_code = item_manager.get_cell_code(' ')
        
        for agent_id, action in actions.items():
            # Only process movement actions (0-3)
            if action < 4:
                # Get direction vector
                dx, dy = self.action_to_direction.get(action, (0, 0))
                
                # ALWAYS update orientation based on action (even if movement fails)
                game_state["agent_orientation"][agent_id] = (dx, dy)
                
                # Get current position
                current_pos = game_state["agent_pos"][agent_id]
                current_x, current_y = current_pos
                
                # Calculate new position
                new_x = current_x + dx
                new_y = current_y + dy
                
                # Validate against map boundaries
                max_x, max_y = map_size[1] - 1, map_size[0] - 1
                new_x = int(np.clip(new_x, 0, max_x))
                new_y = int(np.clip(new_y, 0, max_y))
                
                # Validate against static obstacles (walls, stations)
                cell_type = game_state["map_static"][new_y, new_x]
                
                if cell_type == space_code:
                    # Valid movement - cell is empty space
                    intended_positions[agent_id] = (new_x, new_y)
                else:
                    # Invalid movement - wall or station blocking
                    intended_positions[agent_id] = current_pos
            else:
                # Non-movement action (e.g., interact)
                # Keep current position
                current_pos = game_state["agent_pos"][agent_id]
                intended_positions[agent_id] = current_pos
        
        return intended_positions
    
    def get_movement_info(
        self,
        agent_id: str,
        action: int,
        current_pos: Tuple[int, int],
        final_pos: Tuple[int, int],
        game_state: Dict,
        item_manager
    ) -> str:
        """
        Generate detailed movement information for debugging.
        
        Args:
            agent_id: Agent identifier
            action: Action code
            current_pos: Agent's position before movement
            final_pos: Agent's position after movement
            game_state: Current game state
            item_manager: ItemManager for cell codes
        
        Returns:
            Human-readable string describing the movement result
        """
        # Non-movement action
        if action >= 4:
            return f"Agent {agent_id} did not attempt to move (action: {action})"
        
        # Get direction
        dx, dy = self.action_to_direction.get(action, (0, 0))
        intended_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        # Convert to int for printing
        current_print = (int(current_pos[0]), int(current_pos[1]))
        final_print = (int(final_pos[0]), int(final_pos[1]))
        intended_print = (int(intended_pos[0]), int(intended_pos[1]))
        
        # Successful movement
        if current_pos != final_pos:
            return f"Agent {agent_id} moved from {current_print} to {final_print}"
        
        # Failed movement
        if intended_pos != current_pos:
            # Check why it failed
            if 0 <= intended_pos[0] < game_state["map_static"].shape[1] and \
               0 <= intended_pos[1] < game_state["map_static"].shape[0]:
                cell_type = game_state["map_static"][intended_pos[1], intended_pos[0]]
                space_code = item_manager.get_cell_code(' ')
                
                if cell_type != space_code:
                    return f"Agent {agent_id} blocked by obstacle at {intended_print}"
                else:
                    return f"Agent {agent_id} blocked by agent collision at {intended_print}"
            else:
                return f"Agent {agent_id} hit map boundary at {intended_print}"
        
        return f"Agent {agent_id} stayed at {current_print}"
    
    def __repr__(self) -> str:
        """String representation of MovementSystem."""
        return f"MovementSystem(actions={len(self.action_to_direction)})"
