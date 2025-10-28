"""
Unit tests for Renderer (Phase 7).

Tests the rendering system for both ASCII and RGB modes.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from overcooked_marl_env.rendering import Renderer
from overcooked_marl_env.core.items import ItemType, ItemState, ItemManager


class TestRendererInitialization(unittest.TestCase):
    """Test Renderer initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.item_manager = ItemManager()
    
    def test_init_human_mode(self):
        """Test initialization with human render mode."""
        renderer = Renderer(self.item_manager, render_mode="human")
        
        self.assertEqual(renderer.render_mode, "human")
        self.assertEqual(renderer.item_manager, self.item_manager)
        self.assertIsNone(renderer.window)
        self.assertIsNone(renderer.clock)
    
    def test_init_rgb_array_mode(self):
        """Test initialization with rgb_array render mode."""
        renderer = Renderer(self.item_manager, render_mode="rgb_array")
        
        self.assertEqual(renderer.render_mode, "rgb_array")
        self.assertEqual(renderer.item_manager, self.item_manager)
        self.assertIsNone(renderer.window)
        self.assertIsNone(renderer.clock)
    
    def test_init_none_mode(self):
        """Test initialization with None render mode."""
        renderer = Renderer(self.item_manager, render_mode=None)
        
        self.assertIsNone(renderer.render_mode)
    
    def test_char_map_created(self):
        """Test that character map is properly created."""
        renderer = Renderer(self.item_manager, render_mode="human")
        
        # Check that char_map exists and has expected entries
        self.assertIsInstance(renderer.char_map, dict)
        self.assertIn(self.item_manager.get_cell_code(' '), renderer.char_map)
        self.assertIn(self.item_manager.get_cell_code('#'), renderer.char_map)
        self.assertIn(self.item_manager.get_cell_code('C'), renderer.char_map)
        
        # Check mappings
        self.assertEqual(renderer.char_map[self.item_manager.get_cell_code(' ')], ' ')
        self.assertEqual(renderer.char_map[self.item_manager.get_cell_code('#')], '#')
        self.assertEqual(renderer.char_map[self.item_manager.get_cell_code('C')], 'C')


class TestRendererHelpers(unittest.TestCase):
    """Test Renderer helper methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.item_manager = ItemManager()
        self.renderer = Renderer(self.item_manager, render_mode="human")
    
    def test_get_orientation_name_north(self):
        """Test converting north direction to name."""
        result = self.renderer._get_orientation_name((0, -1))
        self.assertEqual(result, "North")
    
    def test_get_orientation_name_south(self):
        """Test converting south direction to name."""
        result = self.renderer._get_orientation_name((0, 1))
        self.assertEqual(result, "South")
    
    def test_get_orientation_name_east(self):
        """Test converting east direction to name."""
        result = self.renderer._get_orientation_name((1, 0))
        self.assertEqual(result, "East")
    
    def test_get_orientation_name_west(self):
        """Test converting west direction to name."""
        result = self.renderer._get_orientation_name((-1, 0))
        self.assertEqual(result, "West")
    
    def test_get_orientation_name_unknown(self):
        """Test converting unknown direction to name."""
        result = self.renderer._get_orientation_name((1, 1))
        self.assertEqual(result, "Unknown")
    
    def test_get_station_name_counter(self):
        """Test converting counter cell code to name."""
        cell_code = self.item_manager.get_cell_code('#')
        result = self.renderer._get_station_name(cell_code)
        self.assertEqual(result, "Counter")
    
    def test_get_station_name_chopping(self):
        """Test converting chopping board cell code to name."""
        cell_code = self.item_manager.get_cell_code('C')
        result = self.renderer._get_station_name(cell_code)
        self.assertEqual(result, "Chopping Board")
    
    def test_get_station_name_delivery(self):
        """Test converting delivery cell code to name."""
        cell_code = self.item_manager.get_cell_code('D')
        result = self.renderer._get_station_name(cell_code)
        self.assertEqual(result, "Delivery")
    
    def test_get_station_name_unknown(self):
        """Test converting unknown cell code to name."""
        result = self.renderer._get_station_name(999)
        self.assertEqual(result, "Unknown")


class TestRendererItemChar(unittest.TestCase):
    """Test item character representation for ASCII rendering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.item_manager = ItemManager()
        self.renderer = Renderer(self.item_manager, render_mode="human")
    
    def test_get_item_char_none(self):
        """Test getting character for None item."""
        result = self.renderer._get_item_char(None)
        self.assertEqual(result, ' ')
    
    def test_get_item_char_tomato_raw(self):
        """Test getting character for raw tomato."""
        item = {
            "type": ItemType.INGREDIENT,
            "name": "tomato",
            "state": ItemState.RAW
        }
        result = self.renderer._get_item_char(item)
        self.assertEqual(result, 't')  # lowercase for raw
    
    def test_get_item_char_tomato_ready(self):
        """Test getting character for ready tomato."""
        item = {
            "type": ItemType.INGREDIENT,
            "name": "tomato",
            "state": ItemState.READY
        }
        result = self.renderer._get_item_char(item)
        self.assertEqual(result, 'T')  # uppercase for ready
    
    def test_get_item_char_lettuce_chopping(self):
        """Test getting character for lettuce being chopped."""
        item = {
            "type": ItemType.INGREDIENT,
            "name": "lettuce",
            "state": ItemState.CHOPPING_1
        }
        result = self.renderer._get_item_char(item)
        self.assertEqual(result, '1')  # number for chopping state
    
    def test_get_item_char_cucumber_chopping_2(self):
        """Test getting character for cucumber chopping step 2."""
        item = {
            "type": ItemType.INGREDIENT,
            "name": "cucumber",
            "state": ItemState.CHOPPING_2
        }
        result = self.renderer._get_item_char(item)
        self.assertEqual(result, '2')
    
    def test_get_item_char_empty_plate(self):
        """Test getting character for empty plate."""
        item = {
            "type": ItemType.DISH,
            "ingredients": []
        }
        result = self.renderer._get_item_char(item)
        self.assertEqual(result, 'p')  # lowercase p for empty plate
    
    def test_get_item_char_plate_with_ingredient(self):
        """Test getting character for plate with tomato."""
        item = {
            "type": ItemType.DISH,
            "ingredients": ["tomato"]
        }
        result = self.renderer._get_item_char(item)
        self.assertEqual(result, 't')  # first letter of first ingredient


class TestRendererItemColor(unittest.TestCase):
    """Test item color for RGB rendering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.item_manager = ItemManager()
        self.renderer = Renderer(self.item_manager, render_mode="rgb_array")
    
    def test_get_item_color_none(self):
        """Test getting color for None item."""
        result = self.renderer._get_item_color(None)
        self.assertEqual(result, (255, 255, 255))
    
    def test_get_item_color_tomato_raw(self):
        """Test getting color for raw tomato (darker)."""
        item = {
            "type": ItemType.INGREDIENT,
            "name": "tomato",
            "state": ItemState.RAW
        }
        result = self.renderer._get_item_color(item)
        # Should be darker than base color
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        # Raw tomato should be darker red
        self.assertLess(result[0], 255)
    
    def test_get_item_color_tomato_ready(self):
        """Test getting color for ready tomato (full color)."""
        item = {
            "type": ItemType.INGREDIENT,
            "name": "tomato",
            "state": ItemState.READY
        }
        result = self.renderer._get_item_color(item)
        # Should be full tomato red color
        self.assertEqual(result, (255, 99, 71))
    
    def test_get_item_color_lettuce_ready(self):
        """Test getting color for ready lettuce."""
        item = {
            "type": ItemType.INGREDIENT,
            "name": "lettuce",
            "state": ItemState.READY
        }
        result = self.renderer._get_item_color(item)
        # Should be green
        self.assertEqual(result, (144, 238, 144))
    
    def test_get_item_color_empty_plate(self):
        """Test getting color for empty plate."""
        item = {
            "type": ItemType.DISH,
            "ingredients": []
        }
        result = self.renderer._get_item_color(item)
        # Should be light gray/white
        self.assertEqual(result, (240, 240, 240))


class TestRendererMainRender(unittest.TestCase):
    """Test main render method routing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.item_manager = ItemManager()
        
        # Create minimal game state
        self.game_state = {
            "map_static": np.array([[0, 1], [1, 0]]),
            "ground_instances": {},
            "item_instances": {},
            "agent_pos": {"agent_0": (0, 0)},
            "agent_orientation": {"agent_0": (1, 0)},
            "agent_holding_instance": {},
            "time_steps": 0,
            "dishes_pending": 3,
            "delivered_recipes": [],
            "accumulated_rewards": {"agent_0": 0.0},
            "last_step_rewards": {"agent_0": 0.0}
        }
        
        self.map_size = (2, 2)
        self.agents = ["agent_0"]
        self.recipe_order = ["salad"]
    
    def test_render_none_mode(self):
        """Test that render returns None when mode is None."""
        renderer = Renderer(self.item_manager, render_mode=None)
        
        result = renderer.render(
            self.game_state,
            self.map_size,
            self.agents,
            self.recipe_order
        )
        
        self.assertIsNone(result)
    
    @patch('builtins.print')
    def test_render_human_mode(self, mock_print):
        """Test that render routes to text rendering for human mode."""
        renderer = Renderer(self.item_manager, render_mode="human")
        
        result = renderer.render(
            self.game_state,
            self.map_size,
            self.agents,
            self.recipe_order
        )
        
        # Should return None for human mode
        self.assertIsNone(result)
        
        # Should have printed something
        self.assertTrue(mock_print.called)
    
    @patch('overcooked_marl_env.rendering.renderer.pygame')
    @patch('overcooked_marl_env.rendering.renderer.gfxdraw')
    def test_render_rgb_array_mode(self, mock_gfxdraw, mock_pygame):
        """Test that render routes to RGB rendering for rgb_array mode."""
        # Create a proper mock surface
        mock_surface = MagicMock()
        mock_surface.fill = MagicMock()
        
        # Mock pygame initialization
        mock_pygame.init = MagicMock()
        mock_pygame.display.init = MagicMock()
        mock_pygame.display.set_mode = MagicMock(return_value=mock_surface)
        mock_pygame.time.Clock = MagicMock(return_value=MagicMock())
        mock_pygame.event.pump = MagicMock()
        mock_pygame.display.flip = MagicMock()
        mock_pygame.event.get = MagicMock(return_value=[])
        mock_pygame.Rect = MagicMock(return_value=MagicMock())
        mock_pygame.draw.rect = MagicMock()
        mock_pygame.draw.circle = MagicMock()
        
        # Mock gfxdraw
        mock_gfxdraw.filled_circle = MagicMock()
        mock_gfxdraw.aacircle = MagicMock()
        
        # Mock surfarray to return a valid array
        mock_pygame.surfarray.pixels3d = MagicMock(return_value=np.zeros((120, 120, 3)))
        
        renderer = Renderer(self.item_manager, render_mode="rgb_array")
        
        result = renderer.render(
            self.game_state,
            self.map_size,
            self.agents,
            self.recipe_order
        )
        
        # Should return RGB array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result.shape), 3)
        self.assertEqual(result.shape[2], 3)  # RGB channels


class TestRendererClose(unittest.TestCase):
    """Test renderer cleanup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.item_manager = ItemManager()
    
    def test_close_without_window(self):
        """Test closing renderer without initialized window."""
        renderer = Renderer(self.item_manager, render_mode="human")
        
        # Should not raise an error
        renderer.close()
        
        self.assertIsNone(renderer.window)
        self.assertIsNone(renderer.clock)
    
    @patch('overcooked_marl_env.rendering.renderer.pygame')
    def test_close_with_window(self, mock_pygame):
        """Test closing renderer with initialized window."""
        renderer = Renderer(self.item_manager, render_mode="rgb_array")
        
        # Simulate initialized window
        renderer.window = MagicMock()
        renderer.clock = MagicMock()
        
        renderer.close()
        
        # Should have called pygame cleanup
        mock_pygame.display.quit.assert_called_once()
        mock_pygame.quit.assert_called_once()
        
        # Should have cleared references
        self.assertIsNone(renderer.window)
        self.assertIsNone(renderer.clock)


class TestRendererAgentColor(unittest.TestCase):
    """Test agent color assignment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.item_manager = ItemManager()
        self.renderer = Renderer(self.item_manager, render_mode="rgb_array")
    
    def test_get_agent_color_0(self):
        """Test color for agent 0."""
        result = self.renderer._get_agent_color(0)
        self.assertEqual(result, (255, 100, 100))  # Red
    
    def test_get_agent_color_1(self):
        """Test color for agent 1."""
        result = self.renderer._get_agent_color(1)
        self.assertEqual(result, (100, 100, 255))  # Blue
    
    def test_get_agent_color_2(self):
        """Test color for agent 2."""
        result = self.renderer._get_agent_color(2)
        self.assertEqual(result, (100, 255, 100))  # Green
    
    def test_get_agent_color_3(self):
        """Test color for agent 3."""
        result = self.renderer._get_agent_color(3)
        self.assertEqual(result, (255, 255, 100))  # Yellow
    
    def test_get_agent_color_wraps(self):
        """Test that agent colors wrap around for agent 4."""
        result = self.renderer._get_agent_color(4)
        # Should be same as agent 0 (wraps around)
        self.assertEqual(result, (255, 100, 100))


class TestRendererCellColor(unittest.TestCase):
    """Test cell color mapping."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.item_manager = ItemManager()
        self.renderer = Renderer(self.item_manager, render_mode="rgb_array")
    
    def test_get_cell_color_empty(self):
        """Test color for empty cell."""
        cell_code = self.item_manager.get_cell_code(' ')
        result = self.renderer._get_cell_color(cell_code)
        self.assertEqual(result, (255, 255, 255))  # White
    
    def test_get_cell_color_counter(self):
        """Test color for counter cell."""
        cell_code = self.item_manager.get_cell_code('#')
        result = self.renderer._get_cell_color(cell_code)
        self.assertEqual(result, (180, 180, 180))  # Gray
    
    def test_get_cell_color_chopping(self):
        """Test color for chopping board cell."""
        cell_code = self.item_manager.get_cell_code('C')
        result = self.renderer._get_cell_color(cell_code)
        self.assertEqual(result, (139, 69, 19))  # Brown
    
    def test_get_cell_color_delivery(self):
        """Test color for delivery cell."""
        cell_code = self.item_manager.get_cell_code('D')
        result = self.renderer._get_cell_color(cell_code)
        self.assertEqual(result, (50, 205, 50))  # Green
    
    def test_get_cell_color_unknown(self):
        """Test color for unknown cell code."""
        result = self.renderer._get_cell_color(999)
        self.assertEqual(result, (200, 200, 200))  # Default gray


class TestRendererRepr(unittest.TestCase):
    """Test string representation."""
    
    def test_repr_human_mode(self):
        """Test __repr__ for human mode."""
        item_manager = ItemManager()
        renderer = Renderer(item_manager, render_mode="human")
        
        result = repr(renderer)
        self.assertEqual(result, "Renderer(mode=human)")
    
    def test_repr_rgb_array_mode(self):
        """Test __repr__ for rgb_array mode."""
        item_manager = ItemManager()
        renderer = Renderer(item_manager, render_mode="rgb_array")
        
        result = repr(renderer)
        self.assertEqual(result, "Renderer(mode=rgb_array)")
    
    def test_repr_none_mode(self):
        """Test __repr__ for None mode."""
        item_manager = ItemManager()
        renderer = Renderer(item_manager, render_mode=None)
        
        result = repr(renderer)
        self.assertEqual(result, "Renderer(mode=None)")


if __name__ == '__main__':
    unittest.main()
