"""
Core module for Overcooked MARL.

This module contains core game logic including:
- Item management
- Recipe management
- Item instance management
- Game constants
"""

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
    get_ingredient_display_name
)

from .recipes import (
    Recipe,
    RecipeManager,
    RECIPE_DEFINITIONS,
    get_available_recipes,
    get_default_recipe
)

from .item_instance_manager import ItemInstanceManager

from .constants import *

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
    
    # Recipes
    'Recipe',
    'RecipeManager',
    'RECIPE_DEFINITIONS',
    'get_available_recipes',
    'get_default_recipe',
    
    # Item Instance Management
    'ItemInstanceManager',
    
    # Constants (exported from constants module)
    'MAX_PROCESS_STEPS',
    'TOTAL_CHANNELS',
    'NUM_BASE_CHANNELS',
    'DEFAULT_NUM_AGENTS',
    'MAX_EPISODE_STEPS',
    'DEFAULT_MAP_PATH',
    'TUTORIAL_MAP_PATH',
]
