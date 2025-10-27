"""
Constants for the Overcooked MARL Environment.

This module centralizes all constants used throughout the environment,
following the Single Responsibility Principle.
"""

# =================================================================
# OBSERVATION CONSTANTS
# =================================================================

# Maximum number of processing steps for ingredients
MAX_PROCESS_STEPS = 3

# Number of base observation channels (before processing vector)
NUM_BASE_CHANNELS = 5

# Total channels in observation space
TOTAL_CHANNELS = NUM_BASE_CHANNELS + MAX_PROCESS_STEPS  # 5 + 3 = 8

# =================================================================
# ACTION CONSTANTS
# =================================================================

# Action types
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_INTERACT = 4

# Total number of actions
NUM_ACTIONS = 5

# Action to direction mapping
ACTION_TO_DIRECTION = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0)
}

# Direction names for display
DIRECTION_NAMES = {
    (0, -1): "North",
    (0, 1): "South",
    (1, 0): "East",
    (-1, 0): "West"
}

# =================================================================
# GAME CONSTANTS
# =================================================================

# Default number of agents
DEFAULT_NUM_AGENTS = 2

# Maximum episode steps
MAX_EPISODE_STEPS = 500

# Default number of dishes to complete
DEFAULT_DISHES_PENDING = 3

# =================================================================
# REWARD CONSTANTS
# =================================================================

# Reward for delivering a recipe
REWARD_RECIPE_DELIVERED = 20.0

# Bonus multipliers for sequential delivery
REWARD_SEQUENTIAL_BONUS_MULTIPLIER = 1.5

# Penalty for invalid interaction
PENALTY_INVALID_INTERACTION = -2.0

# Small penalty for time step (optional, currently 0)
PENALTY_TIME_STEP = 0.0

# =================================================================
# CELL TYPE CONSTANTS (for map representation)
# =================================================================

# These are character codes used in map files
CELL_EMPTY = ' '
CELL_WALL = '#'
CELL_COUNTER = '#'
CELL_CHOPPING_BOARD = 'C'
CELL_PLATE_DISPENSER = 'P'
CELL_DELIVERY = 'D'
CELL_INGREDIENT_DISPENSER = 'I'
CELL_TRASH = 'T'
CELL_HORIZONTAL_WALL = '_'
CELL_VERTICAL_WALL = '|'

# Station names for display
STATION_NAMES = {
    'C': "Chopping Board",
    'P': "Plate Dispenser",
    'D': "Delivery",
    'I': "Ingredient Dispenser",
    'T': "Trash",
    '#': "Counter",
    ' ': "Empty"
}

# =================================================================
# RENDERING CONSTANTS
# =================================================================

# Cell size in pixels for pygame rendering
DEFAULT_CELL_SIZE = 64

# Info panel height in pixels
DEFAULT_INFO_PANEL_HEIGHT = 150

# Font sizes
FONT_SIZE_NORMAL = 18
FONT_SIZE_SMALL = 14
FONT_SIZE_LARGE = 24

# Colors (RGB tuples)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (200, 200, 200)
COLOR_DARK_GRAY = (100, 100, 100)
COLOR_LIGHT_GRAY = (240, 240, 240)

# Wall colors
COLOR_WALL = (60, 60, 60)
COLOR_WALL_BORDER = (40, 40, 40)

# Station colors
COLOR_CHOPPING_BOARD = (139, 90, 43)  # Brown
COLOR_PLATE_DISPENSER = (230, 230, 250)  # Lavender
COLOR_DELIVERY = (50, 205, 50)  # Lime green
COLOR_TRASH = (105, 105, 105)  # Dim gray
COLOR_COUNTER = (169, 169, 169)  # Dark gray
COLOR_INGREDIENT_DISPENSER = (255, 215, 0)  # Gold

# Agent colors
COLOR_AGENT_0 = (65, 105, 225)  # Royal blue
COLOR_AGENT_1 = (220, 20, 60)  # Crimson
COLOR_AGENT_2 = (50, 205, 50)  # Lime green
COLOR_AGENT_3 = (255, 140, 0)  # Dark orange

# Item state colors
COLOR_RAW = (200, 50, 50)  # Red
COLOR_PROCESSING = (255, 165, 0)  # Orange
COLOR_READY = (50, 200, 50)  # Green

# Ingredient colors (for visualization)
COLOR_TOMATO = (255, 99, 71)  # Tomato red
COLOR_LETTUCE = (144, 238, 144)  # Light green
COLOR_CUCUMBER = (152, 251, 152)  # Pale green
COLOR_ONION = (245, 222, 179)  # Wheat
COLOR_PLATE = (240, 240, 240)  # White-ish

# =================================================================
# MAP CONSTANTS
# =================================================================

# Default map path (relative to package root)
DEFAULT_MAP_PATH = "map/maps/default_map.txt"

# Tutorial map path
TUTORIAL_MAP_PATH = "map/maps/tutorial.txt"

# Map section markers
MAP_SECTION_GRID = "# Grid Map"
MAP_SECTION_DISPENSERS = "# Dispensers"
MAP_SECTION_RECIPES = "# Recipes"

# =================================================================
# METADATA
# =================================================================

# Environment metadata (for PettingZoo)
ENV_NAME = "overcooked_marl_v0"
RENDER_MODES = ["human", "rgb_array"]

# =================================================================
# VALIDATION CONSTANTS
# =================================================================

# Minimum map dimensions
MIN_MAP_WIDTH = 3
MIN_MAP_HEIGHT = 3

# Maximum map dimensions (reasonable limits)
MAX_MAP_WIDTH = 50
MAX_MAP_HEIGHT = 50

# =================================================================
# UTILITY FUNCTIONS
# =================================================================

def get_agent_color(agent_index: int) -> tuple:
    """
    Get color for an agent by index.
    
    Args:
        agent_index: Zero-based agent index
    
    Returns:
        RGB color tuple
    """
    colors = [COLOR_AGENT_0, COLOR_AGENT_1, COLOR_AGENT_2, COLOR_AGENT_3]
    return colors[agent_index % len(colors)]


def get_station_color(cell_char: str) -> tuple:
    """
    Get color for a station type.
    
    Args:
        cell_char: Character representing the station
    
    Returns:
        RGB color tuple
    """
    station_colors = {
        CELL_CHOPPING_BOARD: COLOR_CHOPPING_BOARD,
        CELL_PLATE_DISPENSER: COLOR_PLATE_DISPENSER,
        CELL_DELIVERY: COLOR_DELIVERY,
        CELL_TRASH: COLOR_TRASH,
        CELL_COUNTER: COLOR_COUNTER,
        CELL_INGREDIENT_DISPENSER: COLOR_INGREDIENT_DISPENSER,
        CELL_WALL: COLOR_WALL,
    }
    return station_colors.get(cell_char, COLOR_GRAY)


def get_direction_name(direction: tuple) -> str:
    """
    Convert direction vector to readable name.
    
    Args:
        direction: Tuple of (dx, dy)
    
    Returns:
        Direction name string
    """
    return DIRECTION_NAMES.get(direction, "Unknown")


def get_station_name(cell_char: str) -> str:
    """
    Convert cell character to station name.
    
    Args:
        cell_char: Character representing the station
    
    Returns:
        Station name string
    """
    return STATION_NAMES.get(cell_char, "Unknown")
