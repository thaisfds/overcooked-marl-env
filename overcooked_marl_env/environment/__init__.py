"""
Environment module for Overcooked MARL.

This module contains the refactored environment logic following SOLID principles:
- Game state management (game_state.py)
- Observation building (observation_builder.py)
- Environment configuration (env_config.py)

Note: This folder was originally named 'env/' but was renamed to 'environment/' 
to avoid naming conflicts with the existing env.py file. Once env.py is refactored
and moved into this directory (Phase 8), we can consider renaming back to 'env/'.

Public API Usage:
    >>> from overcooked_marl_env.environment import GameState, MapData, ObservationBuilder
    >>> from overcooked_marl_env.environment import INITIAL_ORIENTATION, DEFAULT_DISHES_PENDING
    
    >>> game_state = GameState(map_data, agent_ids)
    >>> obs_builder = ObservationBuilder()
    >>> observations = obs_builder.build_observations(state, agents, item_manager)
    
Advanced Usage (direct imports):
    >>> from overcooked_marl_env.environment.game_state import GameState, MapData
    >>> from overcooked_marl_env.environment.observation_builder import ObservationBuilder
    >>> from overcooked_marl_env.environment.env_config import INITIAL_ORIENTATION
"""

from .game_state import GameState, MapData
from .observation_builder import ObservationBuilder
from .featurized_observation_builder import FeaturizedObservationBuilder

# Config exports (re-exported from core.constants for convenience)
from .env_config import (
    INITIAL_ORIENTATION,
    DEFAULT_DISHES_PENDING,
    MAX_EPISODE_STEPS,
    EMPTY_PROCESSING_VECTOR,
    DEFAULT_CUTS_NEEDED,
    CHANNEL_STATIC_MAP,
    CHANNEL_ITEMS_GROUND,
    CHANNEL_AGENT_SELF,
    CHANNEL_AGENTS_OTHER,
    CHANNEL_DELIVERY_POINTS,
    CHANNEL_PROCESSING_START,
    NUM_BASE_CHANNELS,
    MAX_PROCESS_STEPS,
)

__all__ = [
    # Main classes
    'GameState',
    'MapData',
    'ObservationBuilder',
    'FeaturizedObservationBuilder',
    
    # Configuration constants
    'INITIAL_ORIENTATION',
    'DEFAULT_DISHES_PENDING',
    'MAX_EPISODE_STEPS',
    'EMPTY_PROCESSING_VECTOR',
    'DEFAULT_CUTS_NEEDED',
    'CHANNEL_STATIC_MAP',
    'CHANNEL_ITEMS_GROUND',
    'CHANNEL_AGENT_SELF',
    'CHANNEL_AGENTS_OTHER',
    'CHANNEL_DELIVERY_POINTS',
    'CHANNEL_PROCESSING_START',
    'NUM_BASE_CHANNELS',
    'MAX_PROCESS_STEPS',
]
