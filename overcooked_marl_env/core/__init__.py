"""
Core module for Overcooked MARL.

This module contains core game logic including:
- Item management (items.py)
- Recipe management (recipes.py)
- Item instance management (item_instance_manager.py)
- Game constants (constants.py)

Public API Usage:
    >>> from overcooked_marl_env.core import ItemManager, RecipeManager
    >>> from overcooked_marl_env.core import create_ingredient, create_plate
    >>> from overcooked_marl_env.core import MAX_PROCESS_STEPS, ACTIONS
    
Advanced Usage (direct imports):
    >>> from overcooked_marl_env.core.items import ItemManager
    >>> from overcooked_marl_env.core.recipes import RecipeManager
    >>> from overcooked_marl_env.core.constants import MAX_PROCESS_STEPS
"""

# Items module exports
from .items import (
    ItemState,
    ItemType,
    ItemManager,
    create_ingredient,
    create_plate,
    advance_processing,
    is_ready,
    can_combine,
    can_process,
    add_ingredient_to_plate,
    get_ingredient_display_name,
    get_item_char,
    get_item_color,
    TILES,
)

# Recipes module exports
from .recipes import (
    Recipe,
    RecipeManager,
    RECIPE_DEFINITIONS,
    get_available_recipes,
    get_default_recipe,
)

# Item instance management exports
from .item_instance_manager import ItemInstanceManager

# Constants exports (most commonly used)
from .constants import (
    # Observation constants
    MAX_PROCESS_STEPS,
    TOTAL_CHANNELS,
    NUM_BASE_CHANNELS,
    CHANNEL_STATIC_MAP,
    CHANNEL_ITEMS_GROUND,
    CHANNEL_AGENT_SELF,
    CHANNEL_AGENTS_OTHER,
    CHANNEL_DELIVERY_POINTS,
    CHANNEL_PROCESSING_START,
    EMPTY_PROCESSING_VECTOR,
    
    # Action constants
    ACTIONS,
    ACTION_TO_DIRECTION,
    DIRECTION_NAMES,
    
    # Game constants
    DEFAULT_NUM_AGENTS,
    MAX_EPISODE_STEPS,
    DEFAULT_DISHES_PENDING,
    DEFAULT_CUTS_NEEDED,
    INITIAL_ORIENTATION,
)

__all__ = [
    # Items
    'ItemState',
    'ItemType',
    'ItemManager',
    'create_ingredient',
    'create_plate',
    'advance_processing',
    'is_ready',
    'can_combine',
    'can_process',
    'add_ingredient_to_plate',
    'get_ingredient_display_name',
    'get_item_char',
    'get_item_color',
    'TILES',
    
    # Recipes
    'Recipe',
    'RecipeManager',
    'RECIPE_DEFINITIONS',
    'get_available_recipes',
    'get_default_recipe',
    
    # Item instance management
    'ItemInstanceManager',
    
    # Observation constants
    'MAX_PROCESS_STEPS',
    'TOTAL_CHANNELS',
    'NUM_BASE_CHANNELS',
    'CHANNEL_STATIC_MAP',
    'CHANNEL_ITEMS_GROUND',
    'CHANNEL_AGENT_SELF',
    'CHANNEL_AGENTS_OTHER',
    'CHANNEL_DELIVERY_POINTS',
    'CHANNEL_PROCESSING_START',
    'EMPTY_PROCESSING_VECTOR',
    
    # Action constants
    'ACTIONS',
    'ACTION_TO_DIRECTION',
    'DIRECTION_NAMES',
    
    # Game constants
    'DEFAULT_NUM_AGENTS',
    'MAX_EPISODE_STEPS',
    'DEFAULT_DISHES_PENDING',
    'DEFAULT_CUTS_NEEDED',
    'INITIAL_ORIENTATION',
]
