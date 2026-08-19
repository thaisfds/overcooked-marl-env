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
        """Test renderer initialization in human mode (Pygame)."""
        renderer = Renderer(self.item_manager, render_mode="human")
        self.assertEqual(renderer.render_mode, "human")
        self.assertIsNone(renderer.window)
        self.assertEqual(renderer.cell_size, 60)
    
    def test_renderer_init_ascii_mode(self):
        """Test renderer initialization in ascii mode."""
        renderer = Renderer(self.item_manager, render_mode="ascii")
        self.assertEqual(renderer.render_mode, "ascii")
        self.assertIsNone(renderer.window)
    
    def test_renderer_init_none_mode(self):
        """Test renderer initialization with no render mode."""
        renderer = Renderer(self.item_manager, render_mode=None)
        self.assertIsNone(renderer.render_mode)
    



class TestRendererTextMode(unittest.TestCase):
    """Test text rendering mode."""
    
    def setUp(self):
        self.item_manager = ItemManager()
        self.renderer = Renderer(self.item_manager, render_mode="ascii")
    
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
        self.renderer = Renderer(self.item_manager, render_mode="human")
    
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
        
        result = self.renderer.render_rgb_array(game_state, (2, 2), [], [])
        
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
        renderer = Renderer(self.item_manager, render_mode="human")
        renderer.window = MagicMock()
        
        renderer.close()
        
        mock_display_quit.assert_called_once()
        mock_pygame_quit.assert_called_once()
        self.assertIsNone(renderer.window)
    
    def test_close_without_window(self):
        """Test closing renderer without window doesn't crash."""
        renderer = Renderer(self.item_manager, render_mode="human")
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
