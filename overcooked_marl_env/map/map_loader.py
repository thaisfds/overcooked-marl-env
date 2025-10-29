"""
Map Loader for Overcooked MARL Environment.

This module provides a high-level interface for loading and validating maps.
It coordinates MapParser and MapValidator to provide clean, validated map data.

Following the Facade pattern, this class simplifies the map loading process
and provides a single entry point for all map-related operations.
"""

import os
import numpy as np
from typing import Optional
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
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize the map loader.
        
        Args:
            strict_validation: If True, raises exceptions on validation errors.
                             If False, logs warnings and tries to continue.
        """
        self.parser = MapParser()
        self.validator = MapValidator(strict=strict_validation)
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
            print(f"{MSG_PREFIX_LOADER} Loading default map: {map_path}")
        else:
            print(f"{MSG_PREFIX_LOADER} Loading map: {map_path}")
        
        # Try to load the requested map
        try:
            map_data = self._load_and_validate(map_path)
            print(f"{MSG_PREFIX_LOADER} Map loaded successfully")
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
