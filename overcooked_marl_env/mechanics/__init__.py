"""
Game mechanics module for Overcooked MARL.

This module contains the core game mechanics:
- Movement system
- Collision detection and resolution
- Interaction handling
"""

from .movement_system import MovementSystem
from .collision_resolver import CollisionResolver

__all__ = ['MovementSystem', 'CollisionResolver']
