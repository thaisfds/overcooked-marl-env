"""
Unit tests for MapParser.

Tests the map file parsing functionality with the correct format:
- | and _ are walls
- # are counters
- Digits (0, 1, 2...) are agent positions
- Letters (C, T, P, D, I) are special stations
"""

import unittest
import os
import tempfile
import numpy as np
from overcooked_marl_env.map.map_parser import MapParser


class TestMapParser(unittest.TestCase):
    """Test cases for MapParser class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = MapParser()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_temp_map(self, content: str) -> str:
        """
        Create a temporary map file for testing.
        
        Args:
            content: Map file content
        
        Returns:
            str: Path to temporary file
        """
        temp_file = os.path.join(self.temp_dir, 'test_map.txt')
        with open(temp_file, 'w') as f:
            f.write(content)
        return temp_file
    
    def test_parse_simple_map(self):
        """Test parsing a simple valid map."""
        map_content = """# Grid Map
_______
|I 0  |
|  1  |
_______

# Dispensers
tomato 1 1

# Recipes
SIMPLE_SALAD
"""
        map_path = self._create_temp_map(map_content)
        result = self.parser.parse_file(map_path)
        
        # Check all expected keys present
        self.assertIn('static_map', result)
        self.assertIn('agent_positions', result)
        self.assertIn('dispensers', result)
        self.assertIn('recipes', result)
        self.assertIn('size', result)
        
        # Check agent positions (0 and 1 are agents)
        self.assertIn('agent_0', result['agent_positions'])
        self.assertIn('agent_1', result['agent_positions'])
        
        # Check dispenser
        self.assertEqual(result['dispensers'][(1, 1)], 'tomato')
        
        # Check recipe
        self.assertIn('SIMPLE_SALAD', result['recipes'])
    
    def test_parse_grid_with_walls(self):
        """Test grid parsing with walls."""
        map_content = """# Grid Map
___________
|         |
|    0    |
|         |
___________

# Dispensers

# Recipes
"""
        map_path = self._create_temp_map(map_content)
        result = self.parser.parse_file(map_path)
        
        grid = result['static_map']
        
        # Walls: | = 8, _ = 9
        self.assertEqual(grid[0, 0], 9)  # _ (top border)
        self.assertEqual(grid[1, 0], 8)  # | (left border)
    
    def test_parse_grid_with_counters(self):
        """Test grid parsing with counters (#)."""
        map_content = """# Grid Map
_______
|#####|
|# 0 #|
|#####|
_______

# Dispensers

# Recipes
"""
        map_path = self._create_temp_map(map_content)
        result = self.parser.parse_file(map_path)
        
        grid = result['static_map']
        
        # Counters: # = 1
        self.assertEqual(grid[1, 1], 1)  # # (counter)
    
    def test_parse_grid_with_special_stations(self):
        """Test grid parsing with special stations (C, T, P, D, I)."""
        map_content = """# Grid Map
___________
|C  T  P |
|   0    |
|D     I |
___________

# Dispensers

# Recipes
"""
        map_path = self._create_temp_map(map_content)
        result = self.parser.parse_file(map_path)
        
        grid = result['static_map']
        
        # Check special station codes
        # C = 3, T = 7, P = 5, D = 6, I = 2
        self.assertEqual(grid[1, 1], 3)  # C (cutting board)
        self.assertEqual(grid[1, 4], 7)  # T (trash)
        self.assertEqual(grid[1, 7], 5)  # P (plate)
        self.assertEqual(grid[3, 1], 6)  # D (delivery)
        self.assertEqual(grid[3, 7], 2)  # I (ingredient dispenser)
    
    def test_parse_multiple_agents(self):
        """Test parsing map with multiple agents (0, 1, 2, 3)."""
        map_content = """# Grid Map
___________
|0       1|
|         |
|2       3|
___________

# Dispensers

# Recipes
"""
        map_path = self._create_temp_map(map_content)
        result = self.parser.parse_file(map_path)
        
        agents = result['agent_positions']
        
        self.assertEqual(len(agents), 4)
        self.assertIn('agent_0', agents)
        self.assertIn('agent_1', agents)
        self.assertIn('agent_2', agents)
        self.assertIn('agent_3', agents)
        
        # Check positions (accounting for border line at top)
        self.assertEqual(agents['agent_0'], (1, 1))
        self.assertEqual(agents['agent_1'], (9, 1))
        self.assertEqual(agents['agent_2'], (1, 3))
        self.assertEqual(agents['agent_3'], (9, 3))
    
    def test_parse_multiple_dispensers(self):
        """Test parsing map with multiple dispensers."""
        map_content = """# Grid Map
_________
|III 0  |
_________

# Dispensers
tomato 1 1
lettuce 2 1
onion 3 1

# Recipes
"""
        map_path = self._create_temp_map(map_content)
        result = self.parser.parse_file(map_path)
        
        dispensers = result['dispensers']
        
        self.assertEqual(len(dispensers), 3)
        self.assertEqual(dispensers[(1, 1)], 'tomato')
        self.assertEqual(dispensers[(2, 1)], 'lettuce')
        self.assertEqual(dispensers[(3, 1)], 'onion')
    
    def test_parse_multiple_recipes(self):
        """Test parsing map with multiple comma-separated recipes."""
        map_content = """# Grid Map
_______
|  0  |
_______

# Dispensers

# Recipes
SIMPLE_SALAD, GARDEN_SALAD, FRESH_SALAD
"""
        map_path = self._create_temp_map(map_content)
        result = self.parser.parse_file(map_path)
        
        recipes = result['recipes']
        
        self.assertEqual(len(recipes), 3)
        self.assertIn('SIMPLE_SALAD', recipes)
        self.assertIn('GARDEN_SALAD', recipes)
        self.assertIn('FRESH_SALAD', recipes)
    
    def test_parse_with_empty_sections(self):
        """Test parsing map with empty optional sections."""
        map_content = """# Grid Map
_______
|  0  |
_______

# Dispensers

# Recipes
"""
        map_path = self._create_temp_map(map_content)
        result = self.parser.parse_file(map_path)
        
        # Empty sections should result in empty collections
        self.assertEqual(len(result['dispensers']), 0)
        self.assertEqual(len(result['recipes']), 0)
    
    def test_parse_complex_grid(self):
        """Test parsing a complex grid with mixed elements."""
        map_content = """# Grid Map
_______________
|#           #|
|C     0     C|
|#           #|
|C           C|
|#T###########|
|             |
|      1      |
|#II#P#DD##I##|
_______________

# Dispensers
lettuce 2 9
tomato 3 9

# Recipes
SIMPLE_SALAD, GARDEN_SALAD
"""
        map_path = self._create_temp_map(map_content)
        result = self.parser.parse_file(map_path)
        
        # Check agents
        self.assertEqual(len(result['agent_positions']), 2)
        self.assertIn('agent_0', result['agent_positions'])
        self.assertIn('agent_1', result['agent_positions'])
        
        # Check dispensers
        self.assertEqual(len(result['dispensers']), 2)
        
        # Check recipes
        self.assertEqual(len(result['recipes']), 2)
        
        # Check grid has correct dimensions
        grid = result['static_map']
        self.assertGreater(grid.shape[0], 0)
        self.assertGreater(grid.shape[1], 0)
    
    def test_map_size_calculation(self):
        """Test that map size is correctly calculated."""
        map_content = """# Grid Map
_______________
|             |
|      0      |
|             |
_______________

# Dispensers

# Recipes
"""
        map_path = self._create_temp_map(map_content)
        result = self.parser.parse_file(map_path)
        
        # Size should match grid shape
        height, width = result['size']
        self.assertEqual((height, width), result['static_map'].shape)
    
    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            self.parser.parse_file('nonexistent_map.txt')
    
    def test_missing_grid_section(self):
        """Test error handling for missing grid section."""
        map_content = """# Dispensers
tomato 1 1

# Recipes
SIMPLE_SALAD
"""
        map_path = self._create_temp_map(map_content)
        
        with self.assertRaises(ValueError):
            self.parser.parse_file(map_path)
    
    def test_invalid_dispenser_format(self):
        """Test error handling for invalid dispenser format."""
        map_content = """# Grid Map
_______
|  0  |
_______

# Dispensers
invalid dispenser line

# Recipes
"""
        map_path = self._create_temp_map(map_content)
        
        with self.assertRaises(ValueError):
            self.parser.parse_file(map_path)
    
    def test_parse_grid_string_directly(self):
        """Test parsing grid from string without file."""
        grid_string = """___________
|#  0    #|
|         |
|#  1    #|
___________"""
        
        grid, agents = self.parser.parse_grid_string(grid_string)
        
        # Check agents
        self.assertEqual(len(agents), 2)
        self.assertIn('agent_0', agents)
        self.assertIn('agent_1', agents)
        
        # Check grid shape
        self.assertGreater(grid.shape[0], 0)
        self.assertGreater(grid.shape[1], 0)
    
    def test_walkable_spaces(self):
        """Test that spaces are parsed as walkable (code 0)."""
        map_content = """# Grid Map
_______
|     |
|  0  |
|     |
_______

# Dispensers

# Recipes
"""
        map_path = self._create_temp_map(map_content)
        result = self.parser.parse_file(map_path)
        
        grid = result['static_map']
        
        # Spaces should be code 0
        # Check some positions that should be spaces
        agent_pos = result['agent_positions']['agent_0']
        # Agent position itself should be walkable
        self.assertEqual(grid[agent_pos[1], agent_pos[0]], 0)


if __name__ == '__main__':
    unittest.main()
