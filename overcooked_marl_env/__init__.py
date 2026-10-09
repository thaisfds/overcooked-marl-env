"""
Overcooked MARL Environment Package

A multi-agent reinforcement learning environment based on the Overcooked game,
built with PettingZoo's ParallelEnv API.

This package follows SOLID principles and PettingZoo best practices for
maintainable and extensible code.

Quick Start:
    >>> from overcooked_marl_env import OvercookedMARL
    >>> env = OvercookedMARL(render_mode="human")
    >>> observations, infos = env.reset()
    >>> # ... your training loop ...
    
Using Versioned Wrapper:
    >>> from overcooked_marl_env import env, parallel_env
    >>> environment = env(render_mode="human")
    >>> # PettingZoo-compatible interface
    
Advanced Usage - Import Submodules:
    >>> from overcooked_marl_env.core import ItemManager, RecipeManager
    >>> from overcooked_marl_env.map import MapLoader
    >>> from overcooked_marl_env.rendering import Renderer
    >>> from overcooked_marl_env.environment import GameState, ObservationBuilder
    >>> from overcooked_marl_env.mechanics import MovementSystem, InteractionHandler
    
For detailed documentation, see individual module docstrings.
"""

from gymnasium.envs.registration import register

# Main environment class
from .env import OvercookedMARL

# Versioned wrapper for PettingZoo compatibility
try:
    from .overcooked_marl_v0 import env, parallel_env
except ImportError:
    # Fallback if wrapper has dependency issues
    env = None
    parallel_env = None

from .core.constants import MAX_EPISODE_STEPS

# Registration for Gymnasium compatibility
# Note: For PettingZoo environments use direct instantiation or the env() factory
register(
    id="OvercookedMARL-v0",
    entry_point="overcooked_marl_env.env:OvercookedMARL",
    max_episode_steps=MAX_EPISODE_STEPS,
)

__version__ = "0.1.0"

__all__ = [
    # Main API
    'OvercookedMARL',
    'env',
    'parallel_env',
    
    # Submodules available via:
    # from overcooked_marl_env.core import ...
    # from overcooked_marl_env.map import ...
    # from overcooked_marl_env.rendering import ...
    # from overcooked_marl_env.environment import ...
    # from overcooked_marl_env.mechanics import ...
]