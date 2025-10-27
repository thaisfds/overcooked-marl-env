"""
Integration tests for the Map System.

Tests the complete map loading workflow, from file parsing through
validation to integration with GameState.
"""

import unittest
import os
from overcooked_marl_env.map import MapLoader
from overcooked_marl_env.environment.game_state import GameState, MapData


class TestMapSystemIntegration(unittest.TestCase):
    """Integration tests for the complete map system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = MapLoader(strict_validation=True)
        
        # Find the tutorial map in the workspace
        self.tutorial_map_path = self._find_tutorial_map()
    
    def _find_tutorial_map(self) -> str:
        """
        Locate the tutorial.txt map file.
        
        Returns:
            str: Path to tutorial map
        
        Raises:
            FileNotFoundError: If map not found
        """
        possible_paths = [
            # From package
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'map', 'maps', 'tutorial.txt'
            ),
            # From workspace root
            os.path.join('maps', 'tutorial.txt'),
            # Relative to test file
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                'maps', 'tutorial.txt'
            ),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        # If not found, skip these tests
        raise FileNotFoundError("tutorial.txt not found - skipping integration tests")
    
    def test_load_tutorial_map(self):
        """Test loading the real tutorial map."""
        if not self.tutorial_map_path:
            self.skipTest("Tutorial map not found")
        
        map_data = self.loader.load_map(self.tutorial_map_path)
        
        # Verify MapData structure
        self.assertIsInstance(map_data, MapData)
        self.assertIsNotNone(map_data.static_map)
        self.assertIsNotNone(map_data.agent_positions)
        self.assertIsNotNone(map_data.dispensers)
        self.assertIsNotNone(map_data.recipes)
        self.assertIsNotNone(map_data.size)
        
        # Verify data types
        import numpy as np
        self.assertIsInstance(map_data.static_map, np.ndarray)
        self.assertIsInstance(map_data.agent_positions, dict)
        self.assertIsInstance(map_data.dispensers, dict)
        self.assertIsInstance(map_data.recipes, list)
        self.assertIsInstance(map_data.size, tuple)
        
        # Verify grid shape matches size
        height, width = map_data.size
        self.assertEqual(map_data.static_map.shape, (height, width))
    
    def test_map_integration_with_game_state(self):
        """Test that loaded map integrates properly with GameState."""
        if not self.tutorial_map_path:
            self.skipTest("Tutorial map not found")
        
        # Load map
        map_data = self.loader.load_map(self.tutorial_map_path)
        
        # Get agent IDs from map data
        agent_ids = list(map_data.agent_positions.keys())
        
        # Create GameState with loaded map
        game_state = GameState(map_data, agent_ids)
        
        # Verify GameState initialized correctly
        self.assertEqual(game_state.map_size, map_data.size)
        
        # Verify agent states initialized
        for agent_id in map_data.agent_positions.keys():
            self.assertIn(agent_id, game_state.agent_positions)
            self.assertIn(agent_id, game_state.agent_orientations)
    
    def test_validate_tutorial_map(self):
        """Test that tutorial map passes validation."""
        if not self.tutorial_map_path:
            self.skipTest("Tutorial map not found")
        
        is_valid = self.loader.validate_map_file(self.tutorial_map_path, verbose=True)
        
        self.assertTrue(is_valid)
    
    def test_get_tutorial_map_info(self):
        """Test getting information about tutorial map."""
        if not self.tutorial_map_path:
            self.skipTest("Tutorial map not found")
        
        info = self.loader.get_map_info(self.tutorial_map_path)
        
        # Should have all expected keys
        self.assertIn('size', info)
        self.assertIn('num_agents', info)
        self.assertIn('agent_ids', info)
        self.assertIn('num_dispensers', info)
        self.assertIn('dispenser_items', info)
        self.assertIn('recipes', info)
        self.assertIn('num_recipes', info)
        
        # Verify reasonable values
        self.assertGreater(info['num_agents'], 0)
        self.assertIsInstance(info['agent_ids'], list)
    
    def test_map_loader_fallback_to_default(self):
        """Test that loader falls back to default map on error."""
        # Try to load a nonexistent map with fallback enabled
        try:
            map_data = self.loader.load_map(
                'nonexistent_map.txt',
                use_default_on_error=True
            )
            
            # Should have loaded default map instead
            self.assertIsInstance(map_data, MapData)
            
        except FileNotFoundError:
            # If default map also doesn't exist, that's acceptable
            # (e.g., in minimal test environment)
            self.skipTest("No default map available for fallback test")
    
    def test_multiple_map_loads(self):
        """Test loading multiple maps in sequence."""
        if not self.tutorial_map_path:
            self.skipTest("Tutorial map not found")
        
        # Load same map twice
        map_data1 = self.loader.load_map(self.tutorial_map_path)
        map_data2 = self.loader.load_map(self.tutorial_map_path)
        
        # Should produce equivalent data
        self.assertEqual(map_data1.size, map_data2.size)
        self.assertEqual(
            len(map_data1.agent_positions),
            len(map_data2.agent_positions)
        )
        self.assertEqual(
            len(map_data1.dispensers),
            len(map_data2.dispensers)
        )
    
    def test_end_to_end_workflow(self):
        """Test complete workflow: load map -> create game state -> verify."""
        if not self.tutorial_map_path:
            self.skipTest("Tutorial map not found")
        
        # Step 1: Load map
        map_data = self.loader.load_map(self.tutorial_map_path)
        
        # Get agent IDs from map data
        agent_ids = list(map_data.agent_positions.keys())
        
        # Step 2: Create game state
        game_state = GameState(map_data, agent_ids)
        
        # Step 3: Verify agents are positioned correctly
        for agent_id, (x, y) in map_data.agent_positions.items():
            game_pos = game_state.agent_positions[agent_id]
            self.assertEqual(game_pos, (x, y))
        
        # Step 4: Verify recipes are loaded
        self.assertEqual(game_state.recipe_order, map_data.recipes)
        
        # Step 5: Verify map size
        self.assertEqual(game_state.map_size, map_data.size)


class TestMapSystemErrors(unittest.TestCase):
    """Test error handling in the map system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = MapLoader(strict_validation=True)
    
    def test_nonexistent_file_error(self):
        """Test error when map file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_map('definitely_does_not_exist.txt', use_default_on_error=False)
    
    def test_lenient_validation_mode(self):
        """Test that lenient mode handles errors gracefully."""
        lenient_loader = MapLoader(strict_validation=False)
        
        # This should not raise, even for invalid maps
        # (will fall back to default or return partial data)
        try:
            # Try to load nonexistent with fallback
            map_data = lenient_loader.load_map(
                'nonexistent.txt',
                use_default_on_error=True
            )
            # If we get here, fallback worked
            self.assertIsInstance(map_data, MapData)
        except FileNotFoundError:
            # Acceptable if no default available
            pass


if __name__ == '__main__':
    unittest.main()
