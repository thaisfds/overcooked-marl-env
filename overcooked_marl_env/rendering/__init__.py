"""
Rendering module for Overcooked MARL.

This module contains rendering implementations:
- Renderer: Main rendering class supporting ASCII and Pygame modes
- RENDER_CONFIG: Centralized rendering configuration constants
"""

from .renderer import Renderer
from .render_config import RENDER_CONFIG

__all__ = ['Renderer', 'RENDER_CONFIG']
