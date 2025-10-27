"""
Overcooked MARL Environment Package

A multi-agent reinforcement learning environment based on the Overcooked game,
built with PettingZoo's ParallelEnv API.

This package follows SOLID principles and PettingZoo best practices for
maintainable and extensible code.

Example:
    >>> from overcooked_marl_env import OvercookedMARL
    >>> env = OvercookedMARL(render_mode="human")
    >>> observations, infos = env.reset()
    
    Or using the versioned wrapper:
    >>> from overcooked_marl_env.overcooked_marl_v0 import env
    >>> environment = env(render_mode="human")
"""

from gymnasium.envs.registration import register

# Import from current env.py (will be moved to env/ folder in Phase 8)
from .env import OvercookedMARL

# Import versioned wrapper
try:
    from .overcooked_marl_v0 import env, parallel_env
except ImportError:
    # Fallback if wrapper has dependency issues
    env = None
    parallel_env = None

# Registration for Gymnasium compatibility
# Note: For PettingZoo environments use direct instantiation or the env() factory
register(
    id="OvercookedMARL-v0",
    entry_point="overcooked_marl_env.env:OvercookedMARL",
    max_episode_steps=500,
)

__version__ = "0.1.0"

__all__ = [
    'OvercookedMARL',
    'env',
    'parallel_env',
]