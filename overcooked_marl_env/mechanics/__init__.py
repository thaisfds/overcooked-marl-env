"""
Game mechanics module for Overcooked MARL.

This module contains the core game mechanics:
- Movement system (movement_system.py)
- Collision detection and resolution (collision_resolver.py)
- Interaction handling (interaction_handler.py)

Public API Usage:
    >>> from overcooked_marl_env.mechanics import MovementSystem, CollisionResolver, InteractionHandler
    
    >>> movement = MovementSystem()
    >>> collision = CollisionResolver()
    >>> interaction = InteractionHandler()
    
Advanced Usage (direct imports):
    >>> from overcooked_marl_env.mechanics.movement_system import MovementSystem
    >>> from overcooked_marl_env.mechanics.collision_resolver import CollisionResolver
    >>> from overcooked_marl_env.mechanics.interaction_handler import InteractionHandler
"""

from .movement_system import MovementSystem
from .collision_resolver import CollisionResolver
from .interaction_handler import InteractionHandler

__all__ = [
    'MovementSystem',
    'CollisionResolver',
    'InteractionHandler',
]
