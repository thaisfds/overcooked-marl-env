"""
Map Validator for Overcooked MARL Environment.

This module provides validation functionality for parsed map data,
ensuring maps meet all requirements before being used in the game.

Following the Single Responsibility Principle, this class is only
responsible for validation, not parsing or loading.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from .map_config import (
    GRID_DIMENSIONS, POSITION_TUPLE_SIZE,
    REPORT_SEPARATOR_WIDTH, MIN_MAP_WIDTH, MIN_MAP_HEIGHT,
    MAX_MAP_WIDTH, MAX_MAP_HEIGHT,
    MSG_PREFIX_WARNING, MSG_PREFIX_ERROR, MSG_STATUS_OK
)
from ..core.items import TILES
from ..core.recipes import get_available_recipes


class MapValidationError(Exception):
    """Exception raised when map validation fails."""
    pass


class MapValidator:
    """
    Validates parsed map data.
    
    This class is responsible for:
    - Validating map dimensions
    - Checking agent positions are valid
    - Validating dispenser configurations
    - Verifying recipe definitions
    - Ensuring map has required elements
    
    Separation of validation from parsing makes the code more
    maintainable and testable.
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize the map validator.
        
        Args:
            strict: If True, raises exceptions on validation errors.
                   If False, logs warnings but continues.
        """
        self.strict = strict
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
    
    def _is_valid_position_tuple(self, pos: any) -> bool:
        """
        Check if position is a valid (x, y) tuple.
        
        Args:
            pos: Position to check
            
        Returns:
            bool: True if valid tuple of 2 integers
        """
        return isinstance(pos, tuple) and len(pos) == POSITION_TUPLE_SIZE
    
    def _is_position_in_bounds(self, pos: Tuple[int, int], width: int, height: int) -> bool:
        """
        Check if position is within map bounds.
        
        Args:
            pos: (x, y) position
            width: Map width
            height: Map height
            
        Returns:
            bool: True if position is within bounds
        """
        x, y = pos
        return 0 <= x < width and 0 <= y < height
    
    def _has_errors(self) -> bool:
        """Check if any validation errors exist."""
        return bool(self.validation_errors)
    
    def validate_map(self, map_data: Dict) -> bool:
        """
        Validate complete map data.
        
        Args:
            map_data: Dictionary with keys:
                - 'grid': np.ndarray
                - 'agent_positions': dict
                - 'dispensers': dict
                - 'recipes': list
                - 'size': tuple
        
        Returns:
            bool: True if validation passes
        
        Raises:
            MapValidationError: If strict=True and validation fails
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        # Required keys
        required_keys = ['static_map', 'agent_positions', 'dispensers', 'recipes', 'size']
        for key in required_keys:
            if key not in map_data:
                self._add_error(f"Missing required key: {key}")
        
        if self.validation_errors:
            return self._handle_validation_result()
        
        # Validate grid first - critical for other validations
        grid_valid = self.validate_grid(map_data['static_map'])
        if not grid_valid:
            # Cannot continue validation if grid is invalid
            return self._handle_validation_result()
        
        # Validate each component
        self.validate_size(map_data['size'])
        self.validate_agent_positions(
            map_data['agent_positions'],
            map_data['static_map']
        )
        self.validate_dispensers(
            map_data['dispensers'],
            map_data['static_map']
        )
        self.validate_recipes(map_data['recipes'])
        
        return self._handle_validation_result()
    
    def validate_grid(self, grid: np.ndarray) -> bool:
        """
        Validate grid array.
        
        Args:
            grid: Numpy array representing the map
        
        Returns:
            bool: True if valid
        """
        if not isinstance(grid, np.ndarray):
            self._add_error(f"Grid must be numpy array, got {type(grid)}")
            return False
        
        if grid.ndim != GRID_DIMENSIONS:
            self._add_error(f"Grid must be {GRID_DIMENSIONS}D, got {grid.ndim}D")
            return False
        
        if grid.size == 0:
            self._add_error("Grid is empty")
            return False
        
        if grid.dtype != np.uint8:
            self._add_warning(f"Grid dtype should be uint8, got {grid.dtype}")
        
        return not self._has_errors()
    
    def validate_size(self, size: Tuple[int, int]) -> bool:
        """
        Validate map size.
        
        Args:
            size: Tuple of (height, width)
        
        Returns:
            bool: True if valid
        """
        if not isinstance(size, tuple) or len(size) != POSITION_TUPLE_SIZE:
            self._add_error(f"Size must be tuple of {POSITION_TUPLE_SIZE} integers, got {size}")
            return False
        
        height, width = size
        
        # Check minimum dimensions
        if height < MIN_MAP_HEIGHT or width < MIN_MAP_WIDTH:
            self._add_error(
                f"Map too small: {size}. "
                f"Minimum: ({MIN_MAP_HEIGHT}, {MIN_MAP_WIDTH})"
            )
        
        # Check maximum dimensions
        if height > MAX_MAP_HEIGHT or width > MAX_MAP_WIDTH:
            self._add_error(
                f"Map too large: {size}. "
                f"Maximum: ({MAX_MAP_HEIGHT}, {MAX_MAP_WIDTH})"
            )
        
        return not self._has_errors()
    
    def validate_agent_positions(
        self,
        agent_positions: Dict[str, Tuple[int, int]],
        grid: np.ndarray
    ) -> bool:
        """
        Validate agent starting positions.
        
        Args:
            agent_positions: Dict mapping agent_id -> (x, y)
            grid: Map grid for boundary checking
        
        Returns:
            bool: True if valid
        """
        if not agent_positions:
            self._add_error("No agent positions defined")
            return False
        
        height, width = grid.shape
        occupied_positions: Set[Tuple[int, int]] = set()
        
        for agent_id, pos in agent_positions.items():
            # Check position format
            if not self._is_valid_position_tuple(pos):
                self._add_error(
                    f"Agent {agent_id} position must be (x, y) tuple, got {pos}"
                )
                continue
            
            x, y = pos
            
            # Check bounds
            if not self._is_position_in_bounds(pos, width, height):
                self._add_error(
                    f"Agent {agent_id} position {pos} out of bounds. "
                    f"Map size: ({height}, {width})"
                )
                continue
            
            # Check for duplicate positions
            if pos in occupied_positions:
                self._add_error(
                    f"Agent {agent_id} position {pos} overlaps with another agent"
                )
            else:
                occupied_positions.add(pos)
            
            # Check if position is walkable (not a wall)
            cell_value = grid[y, x]
            if cell_value != TILES[' ']['code']:  # Not empty space
                self._add_warning(
                    f"Agent {agent_id} starts on non-empty cell at {pos} "
                    f"(cell code: {cell_value})"
                )
        
        return not self._has_errors()
    
    def validate_dispensers(
        self,
        dispensers: Dict[Tuple[int, int], str],
        grid: np.ndarray
    ) -> bool:
        """
        Validate dispenser positions and ingredients.
        
        Args:
            dispensers: Dict mapping (x, y) -> ingredient_name
            grid: Map grid for boundary checking
        
        Returns:
            bool: True if valid
        """
        if not dispensers:
            self._add_warning("No dispensers defined in map")
            # Not a critical error - game can still work without custom dispensers
            return True
        
        height, width = grid.shape
        
        for pos, ingredient in dispensers.items():
            # Check position format
            if not self._is_valid_position_tuple(pos):
                self._add_error(
                    f"Dispenser position must be (x, y) tuple, got {pos}"
                )
                continue
            
            x, y = pos
            
            # Check bounds
            if not self._is_position_in_bounds(pos, width, height):
                self._add_error(
                    f"Dispenser at {pos} out of bounds. "
                    f"Map size: ({height}, {width})"
                )
                continue
            
            # Check ingredient name is valid string
            if not isinstance(ingredient, str) or not ingredient:
                self._add_error(
                    f"Dispenser at {pos} has invalid ingredient: {ingredient}"
                )
        
        return not self._has_errors()
    
    def validate_recipes(self, recipes: List[str]) -> bool:
        """
        Validate recipe list.
        
        Args:
            recipes: List of recipe names
        
        Returns:
            bool: True if valid
        """
        if not recipes:
            self._add_warning("No recipes defined - using default")
            # Not a critical error - game will use defaults
            return True
        
        available_recipes = get_available_recipes()
        
        for recipe_name in recipes:
            if recipe_name not in available_recipes:
                self._add_error(
                    f"Unknown recipe: {recipe_name}. "
                    f"Available: {', '.join(available_recipes)}"
                )
        
        return not self._has_errors()
    
    def _add_error(self, message: str) -> None:
        """Add a validation error."""
        self.validation_errors.append(message)
    
    def _add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.validation_warnings.append(message)
    
    def _handle_validation_result(self) -> bool:
        """
        Handle validation result based on strict mode.
        
        Returns:
            bool: True if validation passes
        
        Raises:
            MapValidationError: If strict=True and errors exist
        """
        # Print warnings
        for warning in self.validation_warnings:
            print(f"{MSG_PREFIX_WARNING} {warning}")
        
        # Handle errors
        if self.validation_errors:
            error_message = "\n".join(
                f"  - {error}" for error in self.validation_errors
            )
            
            if self.strict:
                raise MapValidationError(
                    f"Map validation failed with {len(self.validation_errors)} error(s):\n"
                    f"{error_message}"
                )
            else:
                print(f"{MSG_PREFIX_ERROR} Validation failed:\n{error_message}")
                return False
        
        return True
    
    def get_validation_report(self) -> str:
        """
        Get a detailed validation report.
        
        Returns:
            str: Formatted validation report
        """
        separator = "=" * REPORT_SEPARATOR_WIDTH
        report = ["Map Validation Report", separator]
        
        if not self.validation_errors and not self.validation_warnings:
            report.append(f"{MSG_STATUS_OK} All validations passed")
        else:
            if self.validation_errors:
                report.append(f"\nErrors ({len(self.validation_errors)}):")
                for error in self.validation_errors:
                    report.append(f"  {MSG_PREFIX_ERROR} {error}")
            
            if self.validation_warnings:
                report.append(f"\nWarnings ({len(self.validation_warnings)}):")
                for warning in self.validation_warnings:
                    report.append(f"  {MSG_PREFIX_WARNING} {warning}")
        
        report.append(separator)
        return "\n".join(report)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"MapValidator(strict={self.strict})"


__all__ = ['MapValidator', 'MapValidationError']
