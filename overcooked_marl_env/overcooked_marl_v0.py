"""
Overcooked MARL Environment - Version 0

This module provides a version-controlled wrapper for the Overcooked MARL environment,
following PettingZoo conventions for environment versioning.

Example:
    >>> from overcooked_marl_env.overcooked_marl_v0 import env
    >>> environment = env(render_mode="human")
    >>> observations, infos = environment.reset()
"""

# Import from current location (env.py in package root)
# This will be updated when env.py is moved to env/overcooked_env.py in Phase 8
from overcooked_marl_env.env import OvercookedMARL


def env(**kwargs):
    """
    Create an instance of the Overcooked MARL environment.
    
    This function provides a convenient factory method for creating environment instances,
    following the PettingZoo pattern for environment creation.
    
    Args:
        **kwargs: Keyword arguments to pass to OvercookedMARL constructor.
            render_mode (str, optional): Rendering mode - "human" (Pygame) or "ascii" (ASCII console)
            map_path (str, optional): Path to the map file
            num_fixed_agents (int, optional): Number of agents (default: 2)
            verbose (bool, optional): Enable verbose logging (default: False)
    
    Returns:
        OvercookedMARL: An instance of the environment
    
    Example:
        >>> env_instance = env(render_mode="human", num_fixed_agents=2)
        >>> obs, info = env_instance.reset()
    """
    return OvercookedMARL(**kwargs)


# For backwards compatibility
parallel_env = env


__all__ = ['env', 'parallel_env']
