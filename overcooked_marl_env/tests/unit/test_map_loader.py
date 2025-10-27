"""
Unit tests for MapLoader.

Tests the high-level map loading interface, including coordination
of parsing and validation, error handling, and fallback mechanisms.
"""

import unittest
import os
import tempfile
import shutil
from overcooked_marl_env.map.map_loader import MapLoader
from overcooked_marl_env.map.map_validator import MapValidationError
from overcooked_marl_env.environment.game_state import MapData


class TestMapLoader(unittest.TestCase):
    """Test cases for MapLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = MapLoader(strict_validation=True)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_temp_map(self, content: str, filename: str = 'test_map.txt') -> str:
        """
        Create a temporary map file for testing.
        
        Args:
            content: Map file content
            filename: Name for the temporary file
        
        Returns:
            str: Path to temporary file
        """
        temp_file = os.path.join(self.temp_dir, filename)
        with open(temp_file, 'w') as f:
            f.write(content)
        return temp_file
    
    def _create_valid_map_content(self) -> str:
        """Create valid map file content for testing."""
        return """# Grid Map
___________
|#I######I|
|#   0   #|
|#       #|
|#   1   #|
|#########|
___________

# Dispensers
tomato 2 1
lettuce 8 1

# Recipes
SIMPLE_SALAD
"""
    
    def test_load_valid_map(self):
        """Test loading a valid map file."""
        map_content = self._create_valid_map_content()
        map_path = self._create_temp_map(map_content)
        
        map_data = self.loader.load_map(map_path, use_default_on_error=False)
        
        # Check that MapData is returned
        self.assertIsInstance(map_data, MapData)
        
        # Check data integrity
        self.assertGreater(map_data.size[0], 0)
        self.assertGreater(map_data.size[1], 0)
        self.assertEqual(len(map_data.agent_positions), 2)
        self.assertEqual(len(map_data.dispensers), 2)
        self.assertIn('SIMPLE_SALAD', map_data.recipes)
    
    def test_load_nonexistent_file_no_fallback(self):
        """Test that loading nonexistent file raises error without fallback."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_map(
                'nonexistent_map.txt',
                use_default_on_error=False
            )
    
    def test_load_invalid_map_no_fallback(self):
        """Test that loading invalid map raises validation error."""
        # Create an invalid map (no agents)
        invalid_content = """# Grid Map
_______
|     |
|     |
_______

# Dispensers

# Recipes
"""
        map_path = self._create_temp_map(invalid_content)
        
        with self.assertRaises(MapValidationError):
            self.loader.load_map(map_path, use_default_on_error=False)
    
    def test_validate_map_file_valid(self):
        """Test validation of a valid map file."""
        map_content = self._create_valid_map_content()
        map_path = self._create_temp_map(map_content)
        
        is_valid = self.loader.validate_map_file(map_path, verbose=False)
        
        self.assertTrue(is_valid)
    
    def test_validate_map_file_invalid(self):
        """Test validation of an invalid map file."""
        invalid_content = """# Grid
0 0 0

# Dispensers

# Recipes
"""
        map_path = self._create_temp_map(invalid_content)
        
        is_valid = self.loader.validate_map_file(map_path, verbose=False)
        
        self.assertFalse(is_valid)
    
    def test_get_map_info_valid(self):
        """Test getting information about a valid map."""
        map_content = self._create_valid_map_content()
        map_path = self._create_temp_map(map_content)
        
        info = self.loader.get_map_info(map_path)
        
        self.assertGreater(info['size'][0], 0)
        self.assertGreater(info['size'][1], 0)
        self.assertEqual(info['num_agents'], 2)
        self.assertIn('agent_0', info['agent_ids'])
        self.assertIn('agent_1', info['agent_ids'])
        self.assertEqual(info['num_dispensers'], 2)
        self.assertIn('tomato', info['dispenser_items'])
        self.assertIn('SIMPLE_SALAD', info['recipes'])
    
    def test_get_map_info_nonexistent(self):
        """Test getting info for nonexistent map returns error in dict."""
        info = self.loader.get_map_info('nonexistent.txt')
        
        self.assertIn('error', info)
    
    def test_set_custom_default_map(self):
        """Test setting a custom default map path."""
        map_content = self._create_valid_map_content()
        custom_default = self._create_temp_map(map_content, 'custom_default.txt')
        
        self.loader.set_default_map_path(custom_default)
        
        # Load without specifying path (should use custom default)
        map_data = self.loader.load_map()
        
        self.assertIsInstance(map_data, MapData)
    
    def test_set_invalid_default_map_path(self):
        """Test that setting nonexistent default path raises error."""
        with self.assertRaises(FileNotFoundError):
            self.loader.set_default_map_path('nonexistent.txt')
    
    def test_load_map_with_multiple_agents(self):
        """Test loading map with multiple agents."""
        map_content = """# Grid Map
_______________
|0           1|
|             |
|             |
|2           3|
_______________

# Dispensers

# Recipes
SIMPLE_SALAD
"""
        map_path = self._create_temp_map(map_content)
        map_data = self.loader.load_map(map_path, use_default_on_error=False)
        
        self.assertEqual(len(map_data.agent_positions), 4)
        self.assertIn('agent_0', map_data.agent_positions)
        self.assertIn('agent_3', map_data.agent_positions)
    
    def test_load_map_with_complex_dispensers(self):
        """Test loading map with various dispensers."""
        map_content = """# Grid Map
_______________
|IIII        |
|      0     |
|            |
_______________

# Dispensers
tomato 1 1
lettuce 2 1
onion 3 1
cucumber 4 1

# Recipes
SIMPLE_SALAD
"""
        map_path = self._create_temp_map(map_content)
        map_data = self.loader.load_map(map_path, use_default_on_error=False)
        
        self.assertEqual(len(map_data.dispensers), 4)
        self.assertEqual(map_data.dispensers[(1, 1)], 'tomato')
        self.assertEqual(map_data.dispensers[(4, 1)], 'cucumber')
    
    def test_load_map_with_multiple_recipes(self):
        """Test loading map with multiple recipes."""
        map_content = """# Grid Map
_______
|  0  |
_______

# Dispensers

# Recipes
SIMPLE_SALAD, GARDEN_SALAD, FRESH_SALAD
"""
        map_path = self._create_temp_map(map_content)
        map_data = self.loader.load_map(map_path, use_default_on_error=False)
        
        self.assertEqual(len(map_data.recipes), 3)
        self.assertIn('SIMPLE_SALAD', map_data.recipes)
        self.assertIn('GARDEN_SALAD', map_data.recipes)
        self.assertIn('FRESH_SALAD', map_data.recipes)
    
    def test_lenient_loader_continues_on_warnings(self):
        """Test that lenient loader continues despite warnings."""
        lenient_loader = MapLoader(strict_validation=False)
        
        # Map with warnings (empty dispensers)
        map_content = """# Grid Map
___________
|         |
|    0    |
|         |
___________

# Dispensers

# Recipes
SIMPLE_SALAD
"""
        map_path = self._create_temp_map(map_content)
        
        # Should not raise exception
        map_data = lenient_loader.load_map(map_path, use_default_on_error=False)
        
        self.assertIsInstance(map_data, MapData)
    
    def test_map_data_structure(self):
        """Test that loaded map returns proper MapData structure."""
        map_content = self._create_valid_map_content()
        map_path = self._create_temp_map(map_content)
        
        map_data = self.loader.load_map(map_path, use_default_on_error=False)
        
        # Check all attributes exist
        self.assertTrue(hasattr(map_data, 'static_map'))
        self.assertTrue(hasattr(map_data, 'agent_positions'))
        self.assertTrue(hasattr(map_data, 'dispensers'))
        self.assertTrue(hasattr(map_data, 'recipes'))
        self.assertTrue(hasattr(map_data, 'size'))
    
    def test_loader_repr(self):
        """Test string representation of loader."""
        repr_str = repr(self.loader)
        
        self.assertIn('MapLoader', repr_str)
        self.assertIn('strict', repr_str)


if __name__ == '__main__':
    unittest.main()
