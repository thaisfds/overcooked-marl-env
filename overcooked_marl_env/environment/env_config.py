"""
Configuration constants for the environment module.

This module re-exports constants from core.constants for backward compatibility
and convenience. All constants are centralized in core/constants.py.
"""

from ..core.constants import (
    # Game state constants
    INITIAL_ORIENTATION,
    DEFAULT_DISHES_PENDING,
    DEFAULT_CUTS_NEEDED,
    MAX_EPISODE_STEPS,
    
    # Observation builder constants
    EMPTY_PROCESSING_VECTOR,
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
    'INITIAL_ORIENTATION',
    'DEFAULT_DISHES_PENDING',
    'MAX_EPISODE_STEPS',
    'DEFAULT_CUTS_NEEDED',
    'EMPTY_PROCESSING_VECTOR',
    'CHANNEL_STATIC_MAP',
    'CHANNEL_ITEMS_GROUND',
    'CHANNEL_AGENT_SELF',
    'CHANNEL_AGENTS_OTHER',
    'CHANNEL_DELIVERY_POINTS',
    'CHANNEL_PROCESSING_START',
    'NUM_BASE_CHANNELS',
    'MAX_PROCESS_STEPS',
]
