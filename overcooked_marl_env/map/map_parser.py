"""
Map Parser for Overcooked MARL Environment.

This module provides functionality to parse map files (.txt format)
into structured data that can be used by the game environment.

Map format:
- Grid section: Visual representation using special characters
  * | and _ are walls
  * # are counters/workstations
  * 0, 1, 2, ... are agent starting positions (numeric characters)
  * Letters (C, T, P, D, I) are special interactive stations
    - C: Cutting board
    - T: Trash
    - P: Plate dispenser
    - D: Delivery station
    - I: Ingredient dispenser placeholder
  * Spaces are walkable areas
- Dispensers section: ingredient_name x y
- Recipes section: comma-separated recipe names

Following the Single Responsibility Principle, this class is only
responsible for parsing, not validation or game logic.
"""

import os
import numpy as np
from typing import Dict, List, Tuple
from ..core.items import ItemManager
from .map_config import (
    AGENT_ID_PREFIX, DISPENSER_LINE_PARTS,
    DISPENSER_INGREDIENT_INDEX, DISPENSER_X_INDEX, DISPENSER_Y_INDEX,
    SECTION_HEADER_GRID, SECTION_HEADER_DISPENSERS, SECTION_HEADER_RECIPES
)


class MapParser:
    """
    Parses map files into structured data.
    
    This class is responsible for:
    - Reading map files from disk
    - Parsing grid layout with walls, counters, and stations
    - Extracting agent starting positions from grid (0, 1, 2, ...)
    - Reading dispenser locations and types
    - Parsing recipe lists
    
    The parser separates concerns by only handling the parsing logic,
    leaving validation to MapValidator and game initialization to MapLoader.
    
    Uses ItemManager for consistent cell code mapping with env.py.
    """
    
    def __init__(self):
        """Initialize the map parser with ItemManager for cell codes."""
        self.item_manager = ItemManager()
    
    def parse_file(self, map_path: str) -> dict:
        """
        Parse a complete map file.
        
        Args:
            map_path: Path to the map file
        
        Returns:
            dict: Parsed map data with keys:
                - 'static_map': np.ndarray of cell codes
                - 'agent_positions': dict of agent_id -> (x, y)
                - 'dispensers': dict of (x, y) -> ingredient_name
                - 'recipes': list of recipe names
                - 'size': tuple of (height, width)
        
        Raises:
            FileNotFoundError: If map file doesn't exist
            ValueError: If map format is invalid
        """
        # Resolve absolute path
        abs_path = os.path.abspath(map_path)
        
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Map file not found: {abs_path}")
        
        # Read file content
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into sections
        sections = self._split_sections(content)
        
        # Parse each section
        grid, agent_positions = self._parse_grid_section(sections.get('grid', ''))
        dispensers = self._parse_dispenser_section(sections.get('dispensers', ''))
        recipes = self._parse_recipe_section(sections.get('recipes', ''))
        
        # Calculate map size
        height, width = grid.shape
        
        return {
            'static_map': grid,
            'agent_positions': agent_positions,
            'dispensers': dispensers,
            'recipes': recipes,
            'size': (height, width)
        }
    
    def _split_sections(self, content: str) -> Dict[str, str]:
        """
        Split file content into sections.
        
        Args:
            content: Full file content
        
        Returns:
            dict: Section name -> section content
        """
        sections = {}
        current_section = None
        section_lines = []
        
        for line in content.split('\n'):
            line = line.rstrip()
            
            # Check for section headers
            if line.startswith(SECTION_HEADER_GRID):
                if current_section and section_lines:
                    sections[current_section] = '\n'.join(section_lines)
                current_section = 'grid'
                section_lines = []
            elif line.startswith(SECTION_HEADER_DISPENSERS):
                if current_section and section_lines:
                    sections[current_section] = '\n'.join(section_lines)
                current_section = 'dispensers'
                section_lines = []
            elif line.startswith(SECTION_HEADER_RECIPES):
                if current_section and section_lines:
                    sections[current_section] = '\n'.join(section_lines)
                current_section = 'recipes'
                section_lines = []
            elif current_section:
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    section_lines.append(line)
        
        # Add last section
        if current_section and section_lines:
            sections[current_section] = '\n'.join(section_lines)
        
        return sections
    
    def _parse_grid_section(self, grid_text: str) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
        """
        Parse the grid section of the map.
        
        The grid is a visual representation where:
        - | and _ are walls
        - # are counters
        - Letters (C, T, P, D, I) are special stations
        - Digits (0, 1, 2, ...) are agent starting positions
        - Spaces are walkable
        
        Args:
            grid_text: Grid section as string
        
        Returns:
            tuple: (grid_array, agent_positions)
                - grid_array: np.ndarray of cell codes
                - agent_positions: dict of agent_id -> (x, y)
        
        Raises:
            ValueError: If grid format is invalid
        """
        if not grid_text.strip():
            raise ValueError("Grid section is empty")
        
        lines = [line for line in grid_text.split('\n') if line]
        
        if not lines:
            raise ValueError("No valid grid lines found")
        
        # Don't skip any lines - include all of them (even border lines)
        grid_lines = lines
        
        if not grid_lines:
            raise ValueError("No grid content found")
        
        # Determine grid dimensions
        max_width = max(len(line) for line in grid_lines)
        height = len(grid_lines)
        
        # Initialize grid and agent positions
        grid = np.zeros((height, max_width), dtype=np.uint8)
        agent_positions = {}
        
        # Parse each cell
        for y, line in enumerate(grid_lines):
            for x, char in enumerate(line):
                # Check if it's an agent position (digit)
                if char.isdigit():
                    agent_id = f"{AGENT_ID_PREFIX}{char}"
                    agent_positions[agent_id] = (x, y)
                    grid[y, x] = self.item_manager.get_cell_code(' ')  # Agent position is walkable
                # Check if it's a known character
                elif char in self.item_manager.CHAR_TO_NUM:
                    grid[y, x] = self.item_manager.get_cell_code(char)
                else:
                    # Unknown character - treat as empty for now
                    grid[y, x] = self.item_manager.get_cell_code(' ')
        
        return grid, agent_positions
    
    def _parse_dispenser_section(self, dispenser_text: str) -> Dict[Tuple[int, int], str]:
        """
        Parse the dispenser section.
        
        Format: ingredient_name x y
        Example: tomato 2 5
        
        Args:
            dispenser_text: Dispenser section as string
        
        Returns:
            dict: (x, y) -> ingredient_name
        
        Raises:
            ValueError: If dispenser format is invalid
        """
        dispensers = {}
        
        if not dispenser_text.strip():
            return dispensers
        
        for line in dispenser_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != DISPENSER_LINE_PARTS:
                raise ValueError(f"Invalid dispenser format: {line}. Expected: ingredient x y")
            
            ingredient_name = parts[DISPENSER_INGREDIENT_INDEX]
            try:
                x = int(parts[DISPENSER_X_INDEX])
                y = int(parts[DISPENSER_Y_INDEX])
            except ValueError:
                raise ValueError(f"Invalid dispenser coordinates: {line}")
            
            dispensers[(x, y)] = ingredient_name
        
        return dispensers
    
    def _parse_recipe_section(self, recipe_text: str) -> List[str]:
        """
        Parse the recipe section.
        
        Format: Comma-separated recipe names
        Example: SIMPLE_SALAD, GARDEN_SALAD, FRESH_SALAD
        
        Args:
            recipe_text: Recipe section as string
        
        Returns:
            list: Recipe names
        """
        recipes = []
        
        if not recipe_text.strip():
            return recipes
        
        # Split by commas and clean up
        for recipe in recipe_text.split(','):
            recipe = recipe.strip()
            if recipe:
                recipes.append(recipe)
        
        return recipes
    
    def parse_grid_string(self, grid_string: str) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
        """
        Parse a grid string directly (useful for testing).
        
        Args:
            grid_string: Grid as a multi-line string
        
        Returns:
            tuple: (grid_array, agent_positions)
        """
        return self._parse_grid_section(grid_string)


__all__ = ['MapParser']
