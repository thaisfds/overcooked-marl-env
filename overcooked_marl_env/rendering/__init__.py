"""
Rendering module for Overcooked MARL.

This module contains rendering implementations:
- Renderer: Main rendering class supporting ASCII and Pygame modes (renderer.py)
- RENDER_CONFIG: Centralized rendering configuration constants (render_config.py)

Public API Usage:
    >>> from overcooked_marl_env.rendering import Renderer, RENDER_CONFIG
    >>> renderer = Renderer(mode="human")
    >>> renderer.render(game_state, item_manager)
    
    # Access configuration
    >>> cell_size = RENDER_CONFIG['CELL_SIZE']
    >>> colors = RENDER_CONFIG['COLORS']
    
Advanced Usage (direct imports):
    >>> from overcooked_marl_env.rendering.renderer import Renderer
    >>> from overcooked_marl_env.rendering.render_config import RENDER_CONFIG
"""

from .renderer import Renderer
from .render_config import RENDER_CONFIG

__all__ = ['Renderer', 'RENDER_CONFIG']
