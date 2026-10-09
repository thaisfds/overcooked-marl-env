"""
Map Loader for Overcooked MARL Environment.

This module provides a high-level interface for loading and validating maps.
It coordinates MapParser and MapValidator to provide clean, validated map data.

Following the Facade pattern, this class simplifies the map loading process
and provides a single entry point for all map-related operations.
"""

import os
import numpy as np
from typing import Optional, Any
from .map_parser import MapParser
from .map_validator import MapValidator, MapValidationError
from .map_config import MSG_PREFIX_LOADER, MSG_PREFIX_VALIDATION_ERROR
from ..environment.game_state import MapData


class MapLoader:
    """
    High-level interface for loading maps.
    
    This class coordinates:
    - Map file parsing (via MapParser)
    - Map data validation (via MapValidator)
    - Default map fallback
    - Error handling
    
    This follows the Facade pattern, providing a simple interface
    to a complex subsystem.
    """
    
    def __init__(
        self,
        strict_validation: bool = True,
        recipe_manager: Optional[Any] = None,
        item_manager: Optional[Any] = None
    ):
        """
        Initialize the map loader.
        
        Args:
            strict_validation: If True, raises exceptions on validation errors.
                             If False, logs warnings and tries to continue.
            recipe_manager: Optional RecipeManager instance for custom recipe validation.
            item_manager: Optional ItemManager instance for custom cell codes/tiles.
        """
        self.recipe_manager = recipe_manager
        self.item_manager = item_manager
        self.parser = MapParser(item_manager=item_manager)
        self.validator = MapValidator(strict=strict_validation, recipe_manager=recipe_manager)
        self._default_map_path: Optional[str] = None
    
    def load_map(
        self,
        map_path: Optional[str] = None,
        use_default_on_error: bool = True
    ) -> MapData:
        """
        Load and validate a map file.
        
        Args:
            map_path: Path to map file. If None, uses default map.
            use_default_on_error: If True and map_path fails,
                                 falls back to default map.
        
        Returns:
            MapData: Validated map data ready for use
        
        Raises:
            FileNotFoundError: If map file not found and no default
            MapValidationError: If validation fails in strict mode
        """
        # Determine which map to load
        if map_path is None:
            map_path = self._get_default_map_path()
            # print(f"{MSG_PREFIX_LOADER} Loading default map: {map_path}")
        else:
            # print(f"{MSG_PREFIX_LOADER} Loading map: {map_path}")
            pass
        # Try to load the requested map
        try:
            map_data = self._load_and_validate(map_path)
            # print(f"{MSG_PREFIX_LOADER} Map loaded successfully")
            return map_data
            
        except (FileNotFoundError, MapValidationError, Exception) as e:
            print(f"{MSG_PREFIX_LOADER} Error loading map: {e}")
            
            # Try default map as fallback
            if use_default_on_error and map_path != self._get_default_map_path():
                print(f"{MSG_PREFIX_LOADER} Attempting to load default map as fallback...")
                try:
                    default_path = self._get_default_map_path()
                    map_data = self._load_and_validate(default_path)
                    print(f"{MSG_PREFIX_LOADER} Default map loaded successfully")
                    return map_data
                except Exception as fallback_error:
                    print(f"{MSG_PREFIX_LOADER} Default map also failed: {fallback_error}")
            
            # No fallback or fallback failed
            raise
    
    def _auto_load_adjacent_recipes(self, map_path: Optional[str]) -> None:
        """Auto-loads recipes.json if located in the same directory as map_path."""
        if not map_path or not os.path.exists(map_path):
            return
        adjacent_recipes = os.path.join(os.path.dirname(os.path.abspath(map_path)), "recipes.json")
        if os.path.exists(adjacent_recipes):
            if self.recipe_manager is None:
                from ..core.recipes import RecipeManager
                self.recipe_manager = RecipeManager()
                self.validator.recipe_manager = self.recipe_manager
            self.recipe_manager.load_recipes_from_json(adjacent_recipes)
            if self.item_manager:
                self.item_manager.rebuild_ingredient_creators()

    def _load_and_validate(self, map_path: str) -> MapData:
        """
        Load and validate a map file.
        
        Args:
            map_path: Path to map file
        
        Returns:
            MapData: Validated map data
        
        Raises:
            FileNotFoundError: If map file not found
            MapValidationError: If validation fails
        """
        # Auto-load adjacent recipes if present
        self._auto_load_adjacent_recipes(map_path)
        
        # Parse the map file
        parsed_data = self.parser.parse_file(map_path)
        
        # Validate the parsed data
        is_valid = self.validator.validate_map(parsed_data)
        
        if not is_valid:
            # Get validation report for debugging
            report = self.validator.get_validation_report()
            print(report)
            raise MapValidationError("Map validation failed - see report above")
        
        # Convert to MapData
        return self._convert_to_map_data(parsed_data)
    
    def _convert_to_map_data(self, parsed_data: dict) -> MapData:
        """
        Convert parsed dictionary to MapData dataclass.
        
        Args:
            parsed_data: Dictionary from MapParser
        
        Returns:
            MapData: Structured map data
        """
        return MapData(
            static_map=parsed_data['static_map'],
            agent_positions=parsed_data['agent_positions'],
            dispensers=parsed_data['dispensers'],
            recipes=parsed_data['recipes'],
            size=parsed_data['size']
        )
    
    def _get_default_map_path(self) -> str:
        """
        Get path to the default map file.
        
        Returns:
            str: Absolute path to default map
        
        Raises:
            FileNotFoundError: If default map not found
        """
        if self._default_map_path is None:
            # Try multiple possible locations
            possible_paths = [
                # Relative to this file
                os.path.join(os.path.dirname(__file__), 'maps', 'default_map.txt'),
                # Relative to package root
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'map', 'maps', 'default_map.txt'
                ),
                # Absolute path from workspace root
                os.path.join('maps', 'default_map.txt'),
            ]
            
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    self._default_map_path = abs_path
                    break
            
            if self._default_map_path is None:
                raise FileNotFoundError(
                    f"Default map not found. Tried:\n" +
                    "\n".join(f"  - {p}" for p in possible_paths)
                )
        
        return self._default_map_path
    
    def set_default_map_path(self, path: str) -> None:
        """
        Set custom default map path.
        
        Args:
            path: Path to default map file
        
        Raises:
            FileNotFoundError: If path doesn't exist
        """
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Map file not found: {abs_path}")
        
        self._default_map_path = abs_path
        print(f"{MSG_PREFIX_LOADER} Default map path set to: {abs_path}")
    
    def validate_map_file(self, map_path: str, verbose: bool = True) -> bool:
        """
        Validate a map file without loading it into the game.
        
        Useful for testing map files before using them.
        
        Args:
            map_path: Path to map file
            verbose: If True, prints validation report
        
        Returns:
            bool: True if map is valid
        """
        try:
            self._auto_load_adjacent_recipes(map_path)
            parsed_data = self.parser.parse_file(map_path)
            is_valid = self.validator.validate_map(parsed_data)
            
            if verbose:
                print(self.validator.get_validation_report())
            
            return is_valid
            
        except Exception as e:
            if verbose:
                print(f"{MSG_PREFIX_VALIDATION_ERROR} {e}")
            return False
    
    def get_map_info(self, map_path: str) -> dict:
        """
        Get information about a map without full validation.
        
        Args:
            map_path: Path to map file
        
        Returns:
            dict: Map information (size, agents, dispensers, recipes)
        """
        try:
            self._auto_load_adjacent_recipes(map_path)
            parsed_data = self.parser.parse_file(map_path)
            
            return {
                'path': map_path,
                'size': parsed_data['size'],
                'num_agents': len(parsed_data['agent_positions']),
                'agent_ids': list(parsed_data['agent_positions'].keys()),
                'num_dispensers': len(parsed_data['dispensers']),
                'dispenser_items': list(set(parsed_data['dispensers'].values())),
                'recipes': parsed_data['recipes'],
                'num_recipes': len(parsed_data['recipes'])
            }
            
        except Exception as e:
            return {
                'path': map_path,
                'error': str(e)
            }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MapLoader("
            f"strict={self.validator.strict}, "
            f"default_map={self._default_map_path})"
        )


__all__ = ['MapLoader']


if __name__ == "__main__":
    import sys
    import argparse
    from overcooked_marl_env.core.recipes import RecipeManager
    
    parser = argparse.ArgumentParser(description="Validate an Overcooked MARL environment map file.")
    parser.add_argument("map_file", help="Path to the map text file to validate")
    parser.add_argument("--recipes", help="Optional path to a custom recipes.json config file to load before validation")
    
    args = parser.parse_args()
    
    loader = MapLoader(strict_validation=False)
    
    # Resolve and load custom recipes if provided or if adjacent to the map file
    recipes_path = args.recipes
    if not recipes_path:
        # Check adjacent directory for recipes.json
        adjacent_path = os.path.join(os.path.dirname(os.path.abspath(args.map_file)), "recipes.json")
        if os.path.exists(adjacent_path):
            recipes_path = adjacent_path
            print(f"[MAP VALIDATOR CLI] Found adjacent recipes file at: {recipes_path}")
            
    if recipes_path:
        if os.path.exists(recipes_path):
            manager = RecipeManager()
            manager.load_recipes_from_json(recipes_path)
            print(f"[MAP VALIDATOR CLI] Loaded custom recipes/ingredients from: {recipes_path}")
        else:
            print(f"[MAP VALIDATOR CLI] Error: Custom recipes file not found: {recipes_path}")
            sys.exit(1)
            
    print(f"[MAP VALIDATOR CLI] Validating map file: {args.map_file}...")
    valid = loader.validate_map_file(args.map_file, verbose=True)
    
    if valid:
        print("[MAP VALIDATOR CLI] Map validation PASSED! The map is ready to use.")
        sys.exit(0)
    else:
        print("[MAP VALIDATOR CLI] Map validation FAILED! Please fix the errors reported above.")
        sys.exit(1)
