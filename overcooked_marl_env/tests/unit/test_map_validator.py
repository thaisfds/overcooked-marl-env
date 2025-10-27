"""
Unit tests for MapValidator.

Tests the map validation functionality, ensuring maps meet all
requirements before being used in the game.
"""

import unittest
import numpy as np
from overcooked_marl_env.map.map_validator import MapValidator, MapValidationError
from overcooked_marl_env.core.constants import (
    MIN_MAP_WIDTH, MIN_MAP_HEIGHT,
    MAX_MAP_WIDTH, MAX_MAP_HEIGHT
)


class TestMapValidator(unittest.TestCase):
    """Test cases for MapValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator_strict = MapValidator(strict=True)
        self.validator_lenient = MapValidator(strict=False)
    
    def _create_valid_map_data(self) -> dict:
        """Create a simple valid map data dict for testing."""
        return {
            'static_map': np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8),
            'agent_positions': {'agent_0': (0, 0), 'agent_1': (2, 1)},
            'dispensers': {(1, 0): 'tomato'},
            'recipes': ['SIMPLE_SALAD'],
            'size': (3, 3)
        }
    
    def test_validate_valid_map(self):
        """Test validation passes for a valid map."""
        map_data = self._create_valid_map_data()
        
        result = self.validator_strict.validate_map(map_data)
        
        self.assertTrue(result)
        self.assertEqual(len(self.validator_strict.validation_errors), 0)
    
    def test_missing_required_keys(self):
        """Test validation fails when required keys are missing."""
        incomplete_map = {
            'static_map': np.array([[0, 0]]),
            # Missing other required keys
        }
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(incomplete_map)
    
    def test_validate_grid_not_numpy_array(self):
        """Test validation fails if grid is not a numpy array."""
        map_data = self._create_valid_map_data()
        map_data['static_map'] = [[0, 0, 0], [0, 0, 0]]  # Python list, not numpy
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(map_data)
    
    def test_validate_grid_wrong_dimensions(self):
        """Test validation fails for non-2D grid."""
        map_data = self._create_valid_map_data()
        map_data['static_map'] = np.array([0, 1, 2], dtype=np.uint8)  # 1D
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(map_data)
    
    def test_validate_grid_empty(self):
        """Test validation fails for empty grid."""
        map_data = self._create_valid_map_data()
        map_data['static_map'] = np.array([], dtype=np.uint8)
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(map_data)
    
    def test_validate_size_too_small(self):
        """Test validation fails for map smaller than minimum."""
        map_data = self._create_valid_map_data()
        # Create a map smaller than minimum
        small_size = (MIN_MAP_HEIGHT - 1, MIN_MAP_WIDTH - 1)
        map_data['size'] = small_size
        map_data['static_map'] = np.zeros(small_size, dtype=np.uint8)
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(map_data)
    
    def test_validate_size_too_large(self):
        """Test validation fails for map larger than maximum."""
        map_data = self._create_valid_map_data()
        # Create a map larger than maximum
        large_size = (MAX_MAP_HEIGHT + 1, MAX_MAP_WIDTH + 1)
        map_data['size'] = large_size
        map_data['static_map'] = np.zeros(large_size, dtype=np.uint8)
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(map_data)
    
    def test_validate_no_agents(self):
        """Test validation fails when no agents are defined."""
        map_data = self._create_valid_map_data()
        map_data['agent_positions'] = {}
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(map_data)
    
    def test_validate_agent_out_of_bounds(self):
        """Test validation fails when agent position is out of bounds."""
        map_data = self._create_valid_map_data()
        # Grid is 2x3, so (5, 5) is out of bounds
        map_data['agent_positions'] = {'A1': (5, 5)}
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(map_data)
    
    def test_validate_agent_negative_position(self):
        """Test validation fails for negative agent position."""
        map_data = self._create_valid_map_data()
        map_data['agent_positions'] = {'A1': (-1, 0)}
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(map_data)
    
    def test_validate_overlapping_agents(self):
        """Test validation fails when agents share the same position."""
        map_data = self._create_valid_map_data()
        # Both agents at same position
        map_data['agent_positions'] = {'agent_0': (0, 0), 'agent_1': (0, 0)}
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(map_data)
    
    def test_validate_dispenser_out_of_bounds(self):
        """Test validation fails when dispenser is out of bounds."""
        map_data = self._create_valid_map_data()
        map_data['dispensers'] = {(10, 10): 'tomato'}
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(map_data)
    
    def test_validate_dispenser_invalid_ingredient(self):
        """Test validation fails for invalid dispenser ingredient."""
        map_data = self._create_valid_map_data()
        map_data['dispensers'] = {(1, 0): ''}  # Empty ingredient name
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(map_data)
    
    def test_validate_empty_dispensers_warning(self):
        """Test that empty dispensers generates a warning, not error."""
        map_data = self._create_valid_map_data()
        map_data['dispensers'] = {}
        
        # Should still pass validation
        result = self.validator_strict.validate_map(map_data)
        self.assertTrue(result)
        
        # But should have a warning
        self.assertGreater(len(self.validator_strict.validation_warnings), 0)
    
    def test_validate_empty_recipes_warning(self):
        """Test that empty recipes generates a warning, not error."""
        map_data = self._create_valid_map_data()
        map_data['recipes'] = []
        
        # Should still pass validation
        result = self.validator_strict.validate_map(map_data)
        self.assertTrue(result)
        
        # But should have a warning
        self.assertGreater(len(self.validator_strict.validation_warnings), 0)
    
    def test_validate_unknown_recipe(self):
        """Test validation fails for unknown recipe names."""
        map_data = self._create_valid_map_data()
        map_data['recipes'] = ['UNKNOWN_RECIPE']
        
        with self.assertRaises(MapValidationError):
            self.validator_strict.validate_map(map_data)
    
    def test_lenient_mode_no_exception(self):
        """Test lenient mode doesn't raise exceptions."""
        map_data = self._create_valid_map_data()
        map_data['agent_positions'] = {}  # Invalid: no agents
        
        # Should not raise exception in lenient mode
        result = self.validator_lenient.validate_map(map_data)
        
        self.assertFalse(result)
        self.assertGreater(len(self.validator_lenient.validation_errors), 0)
    
    def test_validation_report_all_passed(self):
        """Test validation report for successful validation."""
        map_data = self._create_valid_map_data()
        self.validator_strict.validate_map(map_data)
        
        report = self.validator_strict.get_validation_report()
        
        self.assertIn('✓', report)
        self.assertIn('passed', report.lower())
    
    def test_validation_report_with_errors(self):
        """Test validation report shows errors."""
        map_data = self._create_valid_map_data()
        map_data['agent_positions'] = {}  # Invalid
        
        try:
            self.validator_strict.validate_map(map_data)
        except MapValidationError:
            pass
        
        report = self.validator_strict.get_validation_report()
        
        self.assertIn('Error', report)
        self.assertIn('✗', report)
    
    def test_validation_report_with_warnings(self):
        """Test validation report shows warnings."""
        map_data = self._create_valid_map_data()
        map_data['dispensers'] = {}  # Generates warning
        
        self.validator_strict.validate_map(map_data)
        report = self.validator_strict.get_validation_report()
        
        self.assertIn('Warning', report)
        self.assertIn('⚠', report)
    
    def test_multiple_validation_runs(self):
        """Test that validator can be reused for multiple maps."""
        map1 = self._create_valid_map_data()
        map2 = self._create_valid_map_data()
        map2['recipes'] = ['GARDEN_SALAD']
        
        # First validation
        result1 = self.validator_strict.validate_map(map1)
        self.assertTrue(result1)
        
        # Second validation should clear previous errors
        result2 = self.validator_strict.validate_map(map2)
        self.assertTrue(result2)
    
    def test_agent_on_non_empty_cell_warning(self):
        """Test warning when agent starts on non-empty cell."""
        map_data = self._create_valid_map_data()
        # Place agent on cell with value 1 (non-empty)
        map_data['static_map'][0, 0] = 1
        map_data['agent_positions'] = {'agent_0': (0, 0)}
        
        result = self.validator_strict.validate_map(map_data)
        
        # Should pass but with warning
        self.assertTrue(result)
        self.assertGreater(len(self.validator_strict.validation_warnings), 0)


if __name__ == '__main__':
    unittest.main()
