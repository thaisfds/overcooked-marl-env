"""
Environment module for Overcooked MARL.

This module contains the refactored environment logic following SOLID principles,
including game state management and other environment components.

Note: This folder was originally named 'env/' but was renamed to 'environment/' 
to avoid naming conflicts with the existing env.py file. Once env.py is refactored
and moved into this directory (Phase 8), we can consider renaming back to 'env/'.
"""

from .game_state import GameState, MapData
from .observation_builder import ObservationBuilder

__all__ = ['GameState', 'MapData', 'ObservationBuilder']
