"""
Map configuration constants for the Overcooked environment.

This module centralizes all map-related constants to make it easy
to adjust map parsing, validation, and display without modifying the map code.
"""

# =================================================================
# CELL CODE CONSTANTS
# =================================================================

# Cell code for empty/walkable space
EMPTY_CELL_CODE = 0

# =================================================================
# MAP PARSING CONSTANTS
# =================================================================

# Agent ID prefix
AGENT_ID_PREFIX = 'agent_'

# Grid dimensions
GRID_DIMENSIONS = 2
POSITION_TUPLE_SIZE = 2

# Dispenser line format: "ingredient x y"
DISPENSER_LINE_PARTS = 3
DISPENSER_INGREDIENT_INDEX = 0
DISPENSER_X_INDEX = 1
DISPENSER_Y_INDEX = 2

# Validation report formatting
REPORT_SEPARATOR_WIDTH = 50

# Section headers in map files
SECTION_HEADER_GRID = '# Grid'
SECTION_HEADER_DISPENSERS = '# Dispensers'
SECTION_HEADER_RECIPES = '# Recipes'

# =================================================================
# MAP VALIDATION CONSTANTS
# =================================================================

# Minimum map dimensions
MIN_MAP_WIDTH = 3
MIN_MAP_HEIGHT = 3

# Maximum map dimensions (reasonable limits)
MAX_MAP_WIDTH = 50
MAX_MAP_HEIGHT = 50

# =================================================================
# MAP PATHS
# =================================================================

# Default map path (relative to package root)
DEFAULT_MAP_PATH = "map/maps/default_map.txt"

# Tutorial map path
TUTORIAL_MAP_PATH = "map/maps/tutorial.txt"

# =================================================================
# VALIDATION MESSAGE PREFIXES
# =================================================================

# Message prefixes for validation output
MSG_PREFIX_WARNING = "[MAP WARNING]"
MSG_PREFIX_ERROR = "[MAP ERROR]"
MSG_PREFIX_LOADER = "[MAP LOADER]"
MSG_PREFIX_VALIDATION_ERROR = "[VALIDATION ERROR]"
MSG_STATUS_OK = "[OK]"
