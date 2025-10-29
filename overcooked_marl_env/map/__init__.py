"""
Map loading and parsing module for Overcooked MARL.

This module handles all map-related operations including:
- Loading maps from files (map_loader.py)
- Parsing map data (map_parser.py)
- Validating map structure (map_validator.py)
- Map configuration constants (map_config.py)

Public API Usage:
    >>> from overcooked_marl_env.map import MapLoader, MapParser, MapValidator
    >>> from overcooked_marl_env.map import GRID_DIMENSIONS
    
    >>> loader = MapLoader()
    >>> map_data = loader.load_map("path/to/map.txt")
    
Advanced Usage (direct imports):
    >>> from overcooked_marl_env.map.map_loader import MapLoader
    >>> from overcooked_marl_env.map.map_config import GRID_DIMENSIONS
"""

from .map_parser import MapParser
from .map_validator import MapValidator, MapValidationError
from .map_loader import MapLoader

# Config exports (commonly used constants)
from .map_config import (
    AGENT_ID_PREFIX,
    DISPENSER_INGREDIENT_INDEX,
    DISPENSER_X_INDEX,
    DISPENSER_Y_INDEX,
    MSG_PREFIX_WARNING,
    MSG_PREFIX_ERROR,
    MSG_PREFIX_LOADER,
    MSG_PREFIX_VALIDATION_ERROR,
    GRID_DIMENSIONS,
    POSITION_TUPLE_SIZE,
)

__all__ = [
    # Main classes
    'MapParser',
    'MapValidator',
    'MapValidationError',
    'MapLoader',
    
    # Configuration constants
    'GRID_DIMENSIONS',
    'POSITION_TUPLE_SIZE',
    'AGENT_ID_PREFIX',
    'DISPENSER_INGREDIENT_INDEX',
    'DISPENSER_X_INDEX',
    'DISPENSER_Y_INDEX',
    'MSG_PREFIX_WARNING',
    'MSG_PREFIX_ERROR',
    'MSG_PREFIX_LOADER',
    'MSG_PREFIX_VALIDATION_ERROR',
]
