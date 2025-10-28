"""
Tests for the Renderer class (Phase 7).

This module tests the rendering functionality including:
- Renderer initialization
- Color mappings
- Item rendering
- Agent rendering
- Text rendering
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from overcooked_marl_env.rendering.renderer import Renderer
from overcooked_marl_env.core.items import ItemManager, ItemType, ItemState


class TestRendererInitialization(unittest.TestCase):
    """Test Renderer initialization."""
    
    def setUp(self):
        self.item_manager = ItemManager()
    
    def test_renderer_init_human_mode(self):
        """Test renderer initialization in human mode."""
        renderer = Renderer(self.item_manager, render_mode="human")
        self.assertEqual(renderer.render_mode, "human")
        self.assertIsNone(renderer.window)
        self.assertEqual(renderer.cell_size, 60)
    
    def test_renderer_init_rgb_mode(self):
        """Test renderer initialization in rgb_array mode."""
        renderer = Renderer(self.item_manager, render_mode="rgb_array")
        self.assertEqual(renderer.render_mode, "rgb_array")
        self.assertIsNone(renderer.window)
    
    def test_renderer_init_none_mode(self):
        """Test renderer initialization with no render mode."""
        renderer = Renderer(self.item_manager, render_mode=None)
        self.assertIsNone(renderer.render_mode)
    
    def test_char_map_initialization(self):
        """Test that character map is properly initialized."""
        renderer = Renderer(self.item_manager, render_mode="human")
        self.assertIn(self.item_manager.get_cell_code(' '), renderer.char_map)
        self.assertIn(self.item_manager.get_cell_code('#'), renderer.char_map)
        self.assertIn(self.item_manager.get_cell_code('C'), renderer.char_map)


class TestRendererColorMethods(unittest.TestCase):
    """Test color-related methods."""
    
    def setUp(self):
        self.item_manager = ItemManager()
        self.renderer = Renderer(self.item_manager, render_mode="rgb_array")
    
    def test_get_ingredient_base_color_tomato(self):
        """Test base color for tomato."""
        color = self.renderer._get_ingredient_base_color('tomato')
        self.assertEqual(color, (255, 99, 71))
    
    def test_get_ingredient_base_color_lettuce(self):
        """Test base color for lettuce."""
        color = self.renderer._get_ingredient_base_color('lettuce')
        self.assertEqual(color, (144, 238, 144))
    
    def test_get_ingredient_base_color_cucumber(self):
        """Test base color for cucumber."""
        color = self.renderer._get_ingredient_base_color('cucumber')
        self.assertEqual(color, (60, 179, 113))
    
    def test_get_ingredient_base_color_unknown(self):
        """Test base color for unknown ingredient."""
        color = self.renderer._get_ingredient_base_color('unknown')
        self.assertEqual(color, (200, 200, 200))
    
    def test_get_cell_color_empty(self):
        """Test cell color for empty space."""
        cell_code = self.item_manager.get_cell_code(' ')
        color = self.renderer._get_cell_color(cell_code)
        self.assertEqual(color, (255, 255, 255))
    
    def test_get_cell_color_counter(self):
        """Test cell color for counter."""
        cell_code = self.item_manager.get_cell_code('#')
        color = self.renderer._get_cell_color(cell_code)
        self.assertEqual(color, (180, 180, 180))
    
    def test_get_cell_color_chopping(self):
        """Test cell color for chopping board."""
        cell_code = self.item_manager.get_cell_code('C')
        color = self.renderer._get_cell_color(cell_code)
        self.assertEqual(color, (210, 180, 140))  # Light brown
    
    def test_get_cell_color_ingredient_dispenser(self):
        """Test cell color for ingredient dispenser."""
        cell_code = self.item_manager.get_cell_code('I')
        color = self.renderer._get_cell_color(cell_code)
        self.assertEqual(color, (210, 180, 140))  # Light brown
    
    def test_get_item_color_raw_ingredient(self):
        """Test color for RAW ingredient."""
        item_dict = {
            "type": ItemType.INGREDIENT,
            "name": "tomato",
            "state": ItemState.RAW
        }
        color = self.renderer._get_item_color(item_dict)
        # RAW is 60% of base color
        base = (255, 99, 71)
        expected = tuple(int(c * 0.6) for c in base)
        self.assertEqual(color, expected)
    
    def test_get_item_color_ready_ingredient(self):
        """Test color for READY ingredient."""
        item_dict = {
            "type": ItemType.INGREDIENT,
            "name": "lettuce",
            "state": ItemState.READY
        }
        color = self.renderer._get_item_color(item_dict)
        # READY uses full base color
        self.assertEqual(color, (144, 238, 144))
    
    def test_get_item_color_chopping_ingredient(self):
        """Test color for CHOPPING ingredient."""
        item_dict = {
            "type": ItemType.INGREDIENT,
            "name": "cucumber",
            "state": ItemState.CHOPPING_1
        }
        color = self.renderer._get_item_color(item_dict)
        # CHOPPING is 80% of base color
        base = (60, 179, 113)
        expected = tuple(int(c * 0.8) for c in base)
        self.assertEqual(color, expected)
    
    def test_get_item_color_dish(self):
        """Test color for dish/plate."""
        item_dict = {
            "type": ItemType.DISH,
            "ingredients": []
        }
        color = self.renderer._get_item_color(item_dict)
        self.assertEqual(color, (240, 240, 240))
    
    def test_get_item_color_none(self):
        """Test color for None item."""
        color = self.renderer._get_item_color(None)
        self.assertEqual(color, (255, 255, 255))


class TestRendererHelperMethods(unittest.TestCase):
    """Test helper methods."""
    
    def setUp(self):
        self.item_manager = ItemManager()
        self.renderer = Renderer(self.item_manager, render_mode="human")
    
    def test_get_orientation_name_north(self):
        """Test orientation name for north."""
        name = self.renderer._get_orientation_name((0, -1))
        self.assertEqual(name, "North")
    
    def test_get_orientation_name_south(self):
        """Test orientation name for south."""
        name = self.renderer._get_orientation_name((0, 1))
        self.assertEqual(name, "South")
    
    def test_get_orientation_name_east(self):
        """Test orientation name for east."""
        name = self.renderer._get_orientation_name((1, 0))
        self.assertEqual(name, "East")
    
    def test_get_orientation_name_west(self):
        """Test orientation name for west."""
        name = self.renderer._get_orientation_name((-1, 0))
        self.assertEqual(name, "West")
    
    def test_get_orientation_name_unknown(self):
        """Test orientation name for unknown direction."""
        name = self.renderer._get_orientation_name((1, 1))
        self.assertEqual(name, "Unknown")
    
    def test_get_station_name_chopping(self):
        """Test station name for chopping board."""
        cell_code = self.item_manager.get_cell_code('C')
        name = self.renderer._get_station_name(cell_code)
        self.assertEqual(name, "Chopping Board")
    
    def test_get_station_name_delivery(self):
        """Test station name for delivery."""
        cell_code = self.item_manager.get_cell_code('D')
        name = self.renderer._get_station_name(cell_code)
        self.assertEqual(name, "Delivery")
    
    def test_get_station_name_trash(self):
        """Test station name for trash."""
        cell_code = self.item_manager.get_cell_code('T')
        name = self.renderer._get_station_name(cell_code)
        self.assertEqual(name, "Trash")
    
    def test_get_item_char_raw_tomato(self):
        """Test character for raw tomato."""
        item_dict = {
            "type": ItemType.INGREDIENT,
            "name": "tomato",
            "state": ItemState.RAW
        }
        char = self.renderer._get_item_char(item_dict)
        self.assertEqual(char, 't')
    
    def test_get_item_char_ready_lettuce(self):
        """Test character for ready lettuce."""
        item_dict = {
            "type": ItemType.INGREDIENT,
            "name": "lettuce",
            "state": ItemState.READY
        }
        char = self.renderer._get_item_char(item_dict)
        self.assertEqual(char, 'L')
    
    def test_get_item_char_chopping(self):
        """Test character for chopping ingredient."""
        item_dict = {
            "type": ItemType.INGREDIENT,
            "name": "cucumber",
            "state": ItemState.CHOPPING_2
        }
        char = self.renderer._get_item_char(item_dict)
        self.assertEqual(char, '2')
    
    def test_get_item_char_empty_plate(self):
        """Test character for empty plate."""
        item_dict = {
            "type": ItemType.DISH,
            "ingredients": []
        }
        char = self.renderer._get_item_char(item_dict)
        self.assertEqual(char, 'p')
    
    def test_get_item_char_plate_with_ingredient(self):
        """Test character for plate with ingredient."""
        item_dict = {
            "type": ItemType.DISH,
            "ingredients": ["tomato"]
        }
        char = self.renderer._get_item_char(item_dict)
        self.assertEqual(char, 't')


class TestRendererTextMode(unittest.TestCase):
    """Test text rendering mode."""
    
    def setUp(self):
        self.item_manager = ItemManager()
        self.renderer = Renderer(self.item_manager, render_mode="human")
    
    @patch('builtins.print')
    def test_render_text_basic(self, mock_print):
        """Test basic text rendering."""
        game_state = {
            "map_static": np.array([[0, 1], [1, 0]]),
            "ground_instances": {},
            "item_instances": {},
            "agent_pos": {"agent_0": (0, 0)},
            "agent_orientation": {"agent_0": (0, -1)},
            "agent_holding_instance": {},
            "time_steps": 0,
            "dishes_pending": 3,
            "delivered_recipes": [],
            "accumulated_rewards": {"agent_0": 0.0},
            "last_step_rewards": {"agent_0": 0.0}
        }
        
        result = self.renderer.render_text(game_state, (2, 2), ["agent_0"], [])
        
        # Should call print multiple times
        self.assertGreater(mock_print.call_count, 0)
        self.assertIsNone(result)
    
    def test_render_with_none_mode(self):
        """Test render with None mode returns None."""
        renderer = Renderer(self.item_manager, render_mode=None)
        result = renderer.render({}, (5, 5), [], [])
        self.assertIsNone(result)


class TestRendererRGBMode(unittest.TestCase):
    """Test RGB rendering mode."""
    
    def setUp(self):
        self.item_manager = ItemManager()
        self.renderer = Renderer(self.item_manager, render_mode="rgb_array")
    
    @patch('pygame.display.init')
    @patch('pygame.init')
    @patch('pygame.display.set_mode')
    @patch('pygame.time.Clock')
    @patch('pygame.surfarray.pixels3d')
    @patch('pygame.draw.rect')
    @patch('pygame.event.pump')
    @patch('pygame.display.flip')
    def test_render_rgb_array_initialization(self, mock_flip, mock_pump, mock_rect,
                                            mock_pixels, mock_clock, 
                                            mock_set_mode, mock_init, mock_display_init):
        """Test RGB array rendering initializes pygame."""
        mock_surface = MagicMock()
        mock_surface.fill = MagicMock()
        mock_set_mode.return_value = mock_surface
        mock_pixels.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        game_state = {
            "map_static": np.array([[0, 1], [1, 0]]),
            "ground_instances": {},
            "item_instances": {},
            "agent_pos": {},
            "dispensers": {}
        }
        
        result = self.renderer.render_rgb_array(game_state, (2, 2), [])
        
        # Should initialize pygame
        mock_init.assert_called_once()
        mock_display_init.assert_called_once()
        mock_set_mode.assert_called_once()
        
        # Should return numpy array
        self.assertIsInstance(result, np.ndarray)


class TestRendererClose(unittest.TestCase):
    """Test renderer cleanup."""
    
    def setUp(self):
        self.item_manager = ItemManager()
    
    @patch('pygame.display.quit')
    @patch('pygame.quit')
    def test_close_with_window(self, mock_pygame_quit, mock_display_quit):
        """Test closing renderer with active window."""
        renderer = Renderer(self.item_manager, render_mode="rgb_array")
        renderer.window = MagicMock()
        
        renderer.close()
        
        mock_display_quit.assert_called_once()
        mock_pygame_quit.assert_called_once()
        self.assertIsNone(renderer.window)
    
    def test_close_without_window(self):
        """Test closing renderer without window doesn't crash."""
        renderer = Renderer(self.item_manager, render_mode="rgb_array")
        renderer.close()  # Should not raise exception


class TestRendererRepr(unittest.TestCase):
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__ method."""
        item_manager = ItemManager()
        renderer = Renderer(item_manager, render_mode="human")
        
        repr_str = repr(renderer)
        self.assertEqual(repr_str, "Renderer(mode=human)")


if __name__ == '__main__':
    unittest.main()
