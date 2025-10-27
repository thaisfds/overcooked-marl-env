"""
Collision Resolver - Handles agent-to-agent collision detection and resolution.

This module implements the Single Responsibility Principle (SRP) by separating
collision resolution logic from movement processing.

Responsibilities:
- Detect agent-to-agent collisions
- Detect multi-agent conflicts (multiple agents trying to reach same position)
- Resolve conflicts by preventing movement
- Track collision penalties
"""

from typing import Dict, Tuple, List, Set


class CollisionResolver:
    """
    Resolves collisions between agents during movement.
    
    This class is responsible for:
    1. Detecting when an agent tries to move to an occupied position
    2. Detecting when multiple agents try to move to the same position
    3. Resolving conflicts by keeping agents in place
    4. Calculating collision penalties
    
    Collision Rules:
    - If agent A tries to move to position P, but agent B is already there → A stays
    - If agents A and B both try to move to position P → both stay in place
    - Collision penalties are applied to conflicting agents
    """
    
    def __init__(self, collision_penalty: float = 0.5):
        """
        Initialize the collision resolver.
        
        Args:
            collision_penalty: Penalty applied when agents collide (default: 0.5)
        """
        self.collision_penalty = collision_penalty
    
    def resolve_collisions(
        self,
        intended_positions: Dict[str, Tuple[int, int]],
        current_positions: Dict[str, Tuple[int, int]],
        verbose: bool = False
    ) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, float]]:
        """
        Resolve all collisions and conflicts between agents.
        
        Process:
        1. Check if intended positions are occupied by other agents
        2. Detect multi-agent conflicts (same destination)
        3. Resolve conflicts by preventing movement
        4. Calculate penalties for conflicting agents
        
        Args:
            intended_positions: Dict mapping agent_id -> intended position
            current_positions: Dict mapping agent_id -> current position
            verbose: If True, print debug information
        
        Returns:
            Tuple of:
            - final_positions: Dict mapping agent_id -> final position
            - penalties: Dict mapping agent_id -> collision penalty
        """
        final_positions = current_positions.copy()
        penalties = {agent_id: 0.0 for agent_id in current_positions.keys()}
        temp_positions = {}
        
        # Step 1: Filter out positions occupied by other agents
        for agent_id, intended_pos in intended_positions.items():
            # Check if another agent is already at the intended position
            occupied = self._is_position_occupied(
                intended_pos,
                agent_id,
                current_positions
            )
            
            if not occupied:
                temp_positions[agent_id] = intended_pos
            else:
                # Position is occupied - agent cannot move
                temp_positions[agent_id] = current_positions[agent_id]
                if verbose:
                    print(f"Agent {agent_id} cannot move to {intended_pos} "
                          f"because another agent is already there")
        
        # Step 2: Detect and resolve multi-agent conflicts
        conflicts = self._detect_conflicts(temp_positions, current_positions)
        
        # Step 3: Apply conflict resolution
        for position, conflicting_agents in conflicts.items():
            if len(conflicting_agents) == 1:
                # No conflict - single agent moving to this position
                agent_id = conflicting_agents[0]
                final_positions[agent_id] = position
            else:
                # Conflict - multiple agents trying to reach same position
                for agent_id in conflicting_agents:
                    # Keep agents in current positions
                    final_positions[agent_id] = current_positions[agent_id]
                    # Apply penalty
                    penalties[agent_id] = self.collision_penalty
                
                if verbose:
                    print(f"Movement conflict: agents {conflicting_agents} "
                          f"tried to move to {position}, none moved")
        
        return final_positions, penalties
    
    def _is_position_occupied(
        self,
        position: Tuple[int, int],
        agent_id: str,
        current_positions: Dict[str, Tuple[int, int]]
    ) -> bool:
        """
        Check if a position is currently occupied by another agent.
        
        Args:
            position: Position to check (x, y)
            agent_id: ID of agent trying to move (excluded from check)
            current_positions: Current positions of all agents
        
        Returns:
            True if position is occupied by another agent, False otherwise
        """
        for other_id, other_pos in current_positions.items():
            if other_id != agent_id and other_pos == position:
                return True
        return False
    
    def _detect_conflicts(
        self,
        temp_positions: Dict[str, Tuple[int, int]],
        current_positions: Dict[str, Tuple[int, int]]
    ) -> Dict[Tuple[int, int], List[str]]:
        """
        Detect positions where multiple agents are trying to move.
        
        A conflict occurs when multiple agents try to move to the same
        position that is different from their current position.
        
        Args:
            temp_positions: Temporary positions after filtering occupied cells
            current_positions: Current positions of all agents
        
        Returns:
            Dictionary mapping position -> list of agents trying to move there
            Only includes positions that agents are actually moving TO
            (not their current positions)
        """
        position_agents: Dict[Tuple[int, int], List[str]] = {}
        
        for agent_id, pos in temp_positions.items():
            # Only consider conflicts for positions agents are moving TO
            # (not their current position)
            if pos != current_positions[agent_id]:
                if pos in position_agents:
                    position_agents[pos].append(agent_id)
                else:
                    position_agents[pos] = [agent_id]
        
        return position_agents
    
    def get_collision_summary(
        self,
        intended_positions: Dict[str, Tuple[int, int]],
        final_positions: Dict[str, Tuple[int, int]],
        current_positions: Dict[str, Tuple[int, int]]
    ) -> Dict[str, str]:
        """
        Generate a summary of collision results for each agent.
        
        Args:
            intended_positions: Where agents wanted to move
            final_positions: Where agents actually ended up
            current_positions: Where agents started
        
        Returns:
            Dictionary mapping agent_id -> collision status string
        """
        summary = {}
        
        for agent_id in current_positions.keys():
            intended = intended_positions.get(agent_id, current_positions[agent_id])
            final = final_positions.get(agent_id, current_positions[agent_id])
            current = current_positions[agent_id]
            
            if final == intended and final != current:
                summary[agent_id] = "moved_successfully"
            elif final == current and intended != current:
                summary[agent_id] = "blocked"
            else:
                summary[agent_id] = "no_movement"
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of CollisionResolver."""
        return f"CollisionResolver(penalty={self.collision_penalty})"
