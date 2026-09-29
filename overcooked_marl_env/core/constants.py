"""
Constants for the Overcooked MARL Environment.

This module centralizes all constants used throughout the environment,
following the Single Responsibility Principle.
"""

# =================================================================
# OBSERVATION CONSTANTS
# =================================================================

# Maximum number of processing steps for ingredients (processing vector size)
MAX_PROCESS_STEPS = 3

# Maximum number of recipes in order per episode
MAX_RECIPES_IN_ORDER = 5

# Number of base observation channels (before processing vector)
NUM_BASE_CHANNELS = 6  # Increased from 5 to 6 (added recipe order channel)

# Total channels in observation space
TOTAL_CHANNELS = NUM_BASE_CHANNELS + MAX_PROCESS_STEPS  # 6 + 3 = 9

# Channel indices for observation layers
CHANNEL_STATIC_MAP = 0
CHANNEL_ITEMS_GROUND = 1
CHANNEL_AGENT_SELF = 2
CHANNEL_AGENTS_OTHER = 3
CHANNEL_DELIVERY_POINTS = 4
CHANNEL_RECIPE_ORDER = 5  # Recipe order channel (new)
CHANNEL_PROCESSING_START = 6  # Processing vector starts at channel 6 (was 5)

# Empty processing vector (no item being carried)
EMPTY_PROCESSING_VECTOR = [0] * MAX_PROCESS_STEPS  # Dynamic based on MAX_PROCESS_STEPS

# =================================================================
# ACTION CONSTANTS
# =================================================================

# Action definitions with direction vectors and names
ACTIONS = {
    0: {'name': 'UP', 'direction': (0, -1), 'direction_name': 'North'},
    1: {'name': 'DOWN', 'direction': (0, 1), 'direction_name': 'South'},
    2: {'name': 'LEFT', 'direction': (-1, 0), 'direction_name': 'West'},
    3: {'name': 'RIGHT', 'direction': (1, 0), 'direction_name': 'East'},
    4: {'name': 'INTERACT', 'direction': None, 'direction_name': None},
}

# Total number of actions
NUM_ACTIONS = len(ACTIONS)

# Action to direction mapping (for backward compatibility)
ACTION_TO_DIRECTION = {
    action_id: action_data['direction'] 
    for action_id, action_data in ACTIONS.items() 
    if action_data['direction'] is not None
}

# Direction names for display (for backward compatibility)
DIRECTION_NAMES = {
    action_data['direction']: action_data['direction_name']
    for action_data in ACTIONS.values()
    if action_data['direction'] is not None
}

# =================================================================
# GAME CONSTANTS
# =================================================================

# Default number of agents
DEFAULT_NUM_AGENTS = 2

# Number of steps per second in real-time execution / play
STEPS_PER_SECOND = 8

# Maximum episode steps (3000 steps)
MAX_EPISODE_STEPS = 3000

# Default number of dishes to complete
DEFAULT_DISHES_PENDING = 3

# Default number of cuts needed for ingredients
DEFAULT_CUTS_NEEDED = 3

# Initial orientation for all agents (facing North)
INITIAL_ORIENTATION = (0, -1)

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
# METADATA
# =================================================================

# Environment metadata (for PettingZoo)
ENV_NAME = "overcooked_marl_v0"
RENDER_MODES = ["human", "ascii"]
