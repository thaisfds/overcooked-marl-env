"""
Renderer - Handles all rendering logic for the Overcooked environment.

This module implements the Single Responsibility Principle (SRP) by separating
rendering logic from the main environment class.

Supports two rendering modes:
- "human": ASCII text rendering to console
- "rgb_array": Pygame graphical rendering
"""

import pygame
from pygame import gfxdraw
import numpy as np
from typing import Dict, Tuple, Optional
from ..core.items import (
    ItemType, 
    ItemState, 
    get_ingredient_display_name, 
    get_item_color,
    get_item_char,
    INGREDIENTS
)
from ..core.constants import DIRECTION_NAMES
from .render_config import RENDER_CONFIG


class Renderer:
    """
    Handles rendering of the Overcooked environment.
    
    This class is responsible for:
    1. ASCII text rendering to console
    2. Pygame graphical rendering
    3. Visual representation of game state
    4. Agent status display
    
    Separating rendering from game logic makes the code more maintainable
    and follows the Single Responsibility Principle.
    """
    
    def __init__(self, item_manager, render_mode: str = "human"):
        """
        Initialize the renderer.
        
        Args:
            item_manager: ItemManager instance for cell code lookups
            render_mode: "human" for ASCII or "rgb_array" for Pygame
        """
        self.item_manager = item_manager
        self.render_mode = render_mode
        
        # Pygame-specific attributes
        self.window = None
        self.clock = None
        self.cell_size = RENDER_CONFIG['cell_size']
        self.should_close = False
    
    def render(
        self,
        game_state: Dict,
        map_size: Tuple[int, int],
        agents: list,
        recipe_order: list
    ) -> Optional[np.ndarray]:
        """
        Main render method that routes to appropriate renderer.
        
        Args:
            game_state: Current game state dictionary
            map_size: Tuple of (height, width)
            agents: List of agent IDs
            recipe_order: List of recipe names in order
        
        Returns:
            None for "human" mode, RGB array for "rgb_array" mode
        """
        if self.render_mode is None:
            return None
        
        if self.render_mode == "human":
            return self.render_text(game_state, map_size, agents, recipe_order)
        elif self.render_mode == "rgb_array":
            return self.render_rgb_array(game_state, map_size, agents, recipe_order)
        
        return None
    
    def render_text(
        self,
        game_state: Dict,
        map_size: Tuple[int, int],
        agents: list,
        recipe_order: list
    ) -> None:
        """
        Renders the environment as ASCII art to the console.
        
        Args:
            game_state: Current game state dictionary
            map_size: Tuple of (height, width)
            agents: List of agent IDs
            recipe_order: List of recipe names in order
        """
        H, W = map_size
        grid = [[' ' for _ in range(W)] for _ in range(H)]
        
        # Draw static map elements
        for y in range(H):
            for x in range(W):
                cell_code = game_state["map_static"][y, x]
                grid[y][x] = self.item_manager.get_cell_char(cell_code)
                
                # Draw items on ground
                ground_instance_id = game_state["ground_instances"].get((x, y), None)
                if ground_instance_id is not None:
                    ground_item = game_state["item_instances"].get(ground_instance_id)
                    grid[y][x] = get_item_char(ground_item)
        
        # Draw agents
        for agent_id, pos in game_state["agent_pos"].items():
            x, y = pos
            agent_num = self._get_agent_number(agent_id)
            
            # Show agent number
            grid[y][x] = str(agent_num)
            
            # Check if agent is carrying something
            carrying_instance_id = game_state["agent_holding_instance"].get(agent_id, None)
            if carrying_instance_id is not None:
                carrying_item = game_state["item_instances"].get(carrying_instance_id)
                # Print what the agent is carrying next to the grid
                print(f"Agent {agent_num} is carrying: {get_ingredient_display_name(carrying_item)}")
        
        # Print the grid to console
        padding = RENDER_CONFIG['ascii_grid_padding']
        print("\n" + "-" * (W + padding))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("-" * (W + padding))
        
        # Print game state information
        self._print_game_info(game_state, recipe_order)
        
        # Print agent status
        self._print_agent_status(game_state, agents, map_size)
        
        return None
    
    def render_rgb_array(
        self,
        game_state: Dict,
        map_size: Tuple[int, int],
        agents: list,
        recipe_order: list
    ) -> np.ndarray:
        """
        Renders the environment as an RGB array using Pygame.
        
        Args:
            game_state: Current game state dictionary
            map_size: Tuple of (height, width)
            agents: List of agent IDs
            recipe_order: List of recipe names in order
        
        Returns:
            numpy array of shape (H, W, 3) with RGB values
        """
        if self.window is None:
            self._initialize_pygame(map_size)
        
        H, W = map_size
        
        # Fill background
        self.window.fill(RENDER_CONFIG['background_color'])
        
        # Draw grid cells
        for y in range(H):
            for x in range(W):
                self._draw_cell(x, y, game_state)
        
        # Draw agents
        for agent_id, pos in game_state["agent_pos"].items():
            self._draw_agent(agent_id, pos, game_state)
        
        # Draw HUD (Heads-Up Display) with game information
        self._draw_hud(game_state, agents, recipe_order)
        
        # Update display
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(RENDER_CONFIG['fps'])
        
        # Return RGB array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )
    
    def _draw_hud(self, game_state: Dict, agents: list, recipe_order: list):
        """
        Draw Heads-Up Display with game information below the grid.
        
        Args:
            game_state: Current game state dictionary
            agents: List of agent IDs
            recipe_order: List of recipe names in order
        """
        # Initialize font if not already done
        if not hasattr(self, 'font'):
            pygame.font.init()
            self.font = pygame.font.Font(None, RENDER_CONFIG['font_size'])
            self.font_small = pygame.font.Font(None, RENDER_CONFIG['font_size_small'])
        
        # HUD position - below the game grid
        map_height = (game_state["map_static"].shape[0] * self.cell_size)
        hud_y_start = self.grid_offset_y + map_height + self.hud_padding
        hud_x = self.hud_padding
        hud_y = hud_y_start
        line_height = RENDER_CONFIG['hud_line_height']
        
        # Current step
        current_step = game_state.get("time_steps", 0)
        step_text = self.font.render(f"Step: {current_step}", True, RENDER_CONFIG['text_black'])
        self.window.blit(step_text, (hud_x, hud_y))
        hud_y += line_height + RENDER_CONFIG['hud_extra_spacing']
        
        # Agent information - one line per agent with well-separated fields
        for agent_id in sorted(agents):
            # Get agent data
            carrying_instance_id = game_state.get("agent_holding_instance", {}).get(agent_id, None)
            
            # Get item display
            if carrying_instance_id is not None and carrying_instance_id in game_state.get("item_instances", {}):
                carrying_item = game_state["item_instances"][carrying_instance_id]
                inventory = get_ingredient_display_name(carrying_item)
            else:
                inventory = "Empty"
            
            # Get rewards
            step_reward = game_state.get("last_step_rewards", {}).get(agent_id, 0.0)
            total_reward = game_state.get("accumulated_rewards", {}).get(agent_id, 0.0)
            
            # Format with well-defined spacing for each field
            agent_num = self._get_agent_number(agent_id)
            
            # Agent label
            agent_label = f"Agent {agent_num}:"
            label_surface = self.font_small.render(agent_label, True, RENDER_CONFIG['text_black'])
            self.window.blit(label_surface, (hud_x, hud_y))
            
            # Step reward field
            step_text = f"Step: {step_reward:+6.2f}"
            step_surface = self.font_small.render(step_text, True, RENDER_CONFIG['text_step_reward'])
            self.window.blit(step_surface, (hud_x + RENDER_CONFIG['hud_step_reward_x'], hud_y))
            
            # Total reward field
            total_text = f"Total: {total_reward:+7.2f}"
            total_surface = self.font_small.render(total_text, True, RENDER_CONFIG['text_total_reward'])
            self.window.blit(total_surface, (hud_x + RENDER_CONFIG['hud_total_reward_x'], hud_y))
            
            # Inventory field
            inv_text = f"Inv: {inventory}"
            inv_surface = self.font_small.render(inv_text, True, RENDER_CONFIG['text_inventory'])
            self.window.blit(inv_surface, (hud_x + RENDER_CONFIG['hud_inventory_x'], hud_y))
            
            hud_y += line_height
        
        hud_y += RENDER_CONFIG['hud_extra_spacing']
        
        # Orders - show only pending orders (not yet delivered)
        delivered = game_state.get("delivered_recipes", [])
        pending_orders = self._calculate_pending_orders(recipe_order, delivered)
        
        # Display pending orders
        if pending_orders:
            orders_str = ", ".join(pending_orders)
            orders_text = self.font_small.render(f"Pending Orders: {orders_str}", True, RENDER_CONFIG['text_black'])
        else:
            orders_text = self.font_small.render("Pending Orders: All completed!", True, RENDER_CONFIG['text_success'])
        
        self.window.blit(orders_text, (hud_x, hud_y))
        hud_y += line_height
        
        # Delivered - single line
        delivered = game_state.get("delivered_recipes", [])
        if delivered:
            delivered_str = ", ".join(delivered)
            delivered_text = self.font_small.render(f"Delivered: {delivered_str}", True, RENDER_CONFIG['text_success'])
        else:
            delivered_text = self.font_small.render("Delivered: None", True, RENDER_CONFIG['text_disabled'])
        self.window.blit(delivered_text, (hud_x, hud_y))
    
    def close(self):
        """Clean up Pygame resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.should_close = False
    
    def check_events(self) -> bool:
        """Check pygame events and return False if window should close."""
        if self.render_mode != "rgb_array" or self.window is None:
            return True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_close = True
                return False
        return True
    
    # ===================================================================
    # Helper Methods - ASCII Rendering
    # ===================================================================
    
    def _get_agent_number(self, agent_id: str) -> int:
        """Extract agent number from agent_id (e.g., 'agent_0' -> 0)."""
        return int(agent_id.split('_')[1])
    
    def _calculate_pending_orders(self, recipe_order: list, delivered_recipes: list) -> list:
        """
        Calculate pending orders by removing delivered recipes from the order list.
        
        Args:
            recipe_order: Original list of recipes to prepare
            delivered_recipes: List of recipes that have been delivered
            
        Returns:
            List of recipes still pending
        """
        pending_orders = list(recipe_order)  # Make a copy
        
        # Count how many of each recipe was delivered
        delivered_count = {}
        for delivered_recipe in delivered_recipes:
            delivered_count[delivered_recipe] = delivered_count.get(delivered_recipe, 0) + 1
        
        # Remove delivered recipes from pending (respecting count)
        for recipe_name, count in delivered_count.items():
            for _ in range(count):
                if recipe_name in pending_orders:
                    pending_orders.remove(recipe_name)  # Remove first occurrence
        
        return pending_orders
    
    def _print_game_info(self, game_state: Dict, recipe_order: list):
        """Print game state information."""
        print(f"Step: {game_state['time_steps']}")
        print(f"Remaining Dishes: {game_state['dishes_pending']}")
        
        # Show orders and delivered
        if len(recipe_order) > 0:
            # Debug: Show original recipe order
            print(f"\n[DEBUG] Recipe Order (original): {recipe_order}")
            print(f"[DEBUG] Delivered Recipes: {game_state['delivered_recipes']}")
            
            # Calculate pending orders using utility method
            pending_orders = self._calculate_pending_orders(recipe_order, game_state['delivered_recipes'])
            
            print(f"[DEBUG] Pending Orders (calculated): {pending_orders}")
            
            if len(pending_orders) > 0:
                orders_str = ", ".join(pending_orders)
                print(f"\nOrders (Pending): {orders_str}")
            else:
                print(f"\nOrders (Pending): (none - all completed!)")
            
            if len(game_state['delivered_recipes']) > 0:
                delivered_str = ", ".join(game_state['delivered_recipes'])
                print(f"Delivered: {delivered_str}")
            else:
                print(f"Delivered: (none)")
        else:
            print(f"Orders: (none)")
    
    def _print_agent_status(self, game_state: Dict, agents: list, map_size: Tuple[int, int]):
        """Print status information for all agents."""
        print("\n--- Agent Status ---")
        for agent_id in agents:
            agent_pos = game_state["agent_pos"][agent_id]
            x, y = int(agent_pos[0]), int(agent_pos[1])
            
            # Get orientation and what's in front
            facing_info = self._get_agent_facing_info(agent_id, game_state, map_size)
            
            # Get inventory
            carrying_instance_id = game_state["agent_holding_instance"].get(agent_id)
            if carrying_instance_id is not None:
                carrying_item = game_state["item_instances"].get(carrying_instance_id)
                inventory_str = get_ingredient_display_name(carrying_item) if carrying_item else "Empty"
            else:
                inventory_str = "Empty"
            
            # Get rewards
            acc_reward = game_state["accumulated_rewards"].get(agent_id, 0.0)
            step_reward = game_state["last_step_rewards"].get(agent_id, 0.0)
            
            print(f"  {agent_id} at ({x},{y} {facing_info}) - Accumulated: {acc_reward:.2f} - Step: {step_reward:.2f} - Inventory: {inventory_str}")
    
    def _get_agent_facing_info(self, agent_id: str, game_state: Dict, map_size: Tuple[int, int]) -> str:
        """
        Returns information about what the agent is facing.
        
        Args:
            agent_id: Agent identifier
            game_state: Current game state
            map_size: Tuple of (height, width)
        
        Returns:
            String describing orientation and facing station
        """
        orientation = game_state["agent_orientation"][agent_id]
        orientation_name = self._get_orientation_name(orientation)
        
        # Get cell in front
        x, y = game_state["agent_pos"][agent_id]
        dx, dy = orientation
        front_x, front_y = x + dx, y + dy
        
        # Check if within bounds
        if 0 <= front_x < map_size[1] and 0 <= front_y < map_size[0]:
            cell_code = game_state["map_static"][front_y, front_x]
            # Get character and then tile name
            char = self.item_manager.get_cell_char(cell_code)
            station_name = self.item_manager.get_tile_name(char)
            return f"{orientation_name} -> {station_name}"
        else:
            return f"{orientation_name} -> Wall"
    
    def _get_orientation_name(self, direction: Tuple[int, int]) -> str:
        """Converts direction vector to readable name."""
        return DIRECTION_NAMES.get(direction, "Unknown")
    
    # ===================================================================
    # Helper Methods - Pygame Rendering
    # ===================================================================
    
    def _draw_circle(self, x: int, y: int, radius: int, fill_color: Tuple[int, int, int], border_color: Tuple[int, int, int] = None):
        """
        Draw a filled circle with anti-aliased border.
        
        Args:
            x: Center X coordinate
            y: Center Y coordinate
            radius: Circle radius
            fill_color: RGB color for the filled circle
            border_color: RGB color for the border (defaults to black if not provided)
        """
        if border_color is None:
            border_color = RENDER_CONFIG['text_black']
        
        gfxdraw.filled_circle(self.window, x, y, radius, fill_color)
        gfxdraw.aacircle(self.window, x, y, radius, border_color)
    
    def _get_cell_center(self, x: int, y: int, offset_y: int = 0) -> Tuple[int, int]:
        """
        Calculate the pixel coordinates of the center of a grid cell.
        
        Args:
            x: Grid X coordinate
            y: Grid Y coordinate
            offset_y: Optional Y offset in pixels
            
        Returns:
            Tuple of (center_x, center_y) in pixel coordinates
        """
        center_x = self.grid_offset_x + x * self.cell_size + self.cell_size // 2
        center_y = self.grid_offset_y + y * self.cell_size + self.cell_size // 2 - offset_y
        return center_x, center_y
    
    def _initialize_pygame(self, map_size: Tuple[int, int]):
        """Initialize Pygame window with space for HUD below the grid."""
        pygame.init()
        pygame.display.init()
        
        H, W = map_size
        # Add space for HUD below the grid and padding
        self.hud_height = RENDER_CONFIG['hud_height']
        self.hud_padding = RENDER_CONFIG['hud_padding']
        self.grid_offset_x = self.hud_padding
        self.grid_offset_y = self.hud_padding
        
        # Calculate window dimensions - ensure minimum width for HUD text
        grid_width = W * self.cell_size
        grid_height = H * self.cell_size
        
        min_width = RENDER_CONFIG['hud_min_width']
        window_width = max(grid_width + 2 * self.hud_padding, min_width)
        window_height = self.grid_offset_y + grid_height + self.hud_padding + self.hud_height
        
        # Center the grid horizontally if window is wider than grid
        if window_width > grid_width + 2 * self.hud_padding:
            self.grid_offset_x = (window_width - grid_width) // 2
        
        self.window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Overcooked MARL")
        self.clock = pygame.time.Clock()
    
    def _draw_cell(self, x: int, y: int, game_state: Dict):
        """
        Draw a single cell of the grid.
        
        Args:
            x: X coordinate
            y: Y coordinate
            game_state: Current game state
        """
        cell_code = game_state["map_static"][y, x]
        rect = pygame.Rect(
            self.grid_offset_x + x * self.cell_size,
            self.grid_offset_y + y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        
        # Get cell color based on type
        color = self._get_cell_color(cell_code)
        pygame.draw.rect(self.window, color, rect)
        pygame.draw.rect(self.window, RENDER_CONFIG['border_color'], rect, RENDER_CONFIG['grid_border_width'])  # Border
        
        # Draw visual indicators for stations
        center_x, center_y = self._get_cell_center(x, y)
        
        # Chopping board - draw knife line
        if cell_code == self.item_manager.get_cell_code('C'):
            knife_padding = RENDER_CONFIG['knife_padding']
            knife_y = self.grid_offset_y + y * self.cell_size + RENDER_CONFIG['knife_y_position']
            pygame.draw.line(
                self.window, 
                RENDER_CONFIG['border_color'], 
                (self.grid_offset_x + x * self.cell_size + knife_padding, knife_y),
                (self.grid_offset_x + x * self.cell_size + self.cell_size - knife_padding, knife_y), 
                RENDER_CONFIG['knife_line_width']
            )
        
        # Trash - draw X
        elif cell_code == self.item_manager.get_cell_code('T'):
            padding = self.cell_size // RENDER_CONFIG['trash_x_padding_ratio']
            line_width = RENDER_CONFIG['trash_line_width']
            # Diagonal 1
            pygame.draw.line(
                self.window, 
                RENDER_CONFIG['text_black'],
                (self.grid_offset_x + x * self.cell_size + padding, self.grid_offset_y + y * self.cell_size + padding),
                (self.grid_offset_x + x * self.cell_size + self.cell_size - padding, self.grid_offset_y + y * self.cell_size + self.cell_size - padding), 
                line_width
            )
            # Diagonal 2
            pygame.draw.line(
                self.window, 
                RENDER_CONFIG['text_black'],
                (self.grid_offset_x + x * self.cell_size + self.cell_size - padding, self.grid_offset_y + y * self.cell_size + padding),
                (self.grid_offset_x + x * self.cell_size + padding, self.grid_offset_y + y * self.cell_size + self.cell_size - padding), 
                line_width
            )
        
        # Delivery - draw arrow pointing right
        elif cell_code == self.item_manager.get_cell_code('D'):
            arrow_size = self.cell_size // RENDER_CONFIG['arrow_size_ratio']
            arrow_points = [
                (center_x - arrow_size, center_y - arrow_size),  # Top left
                (center_x - arrow_size, center_y + arrow_size),  # Bottom left
                (center_x + arrow_size, center_y),               # Right tip
            ]
            pygame.draw.polygon(self.window, RENDER_CONFIG['text_white'], arrow_points)
            pygame.draw.polygon(self.window, RENDER_CONFIG['border_color'], arrow_points, RENDER_CONFIG['arrow_border_width'])
        
        # Plate Dispenser - draw empty plate
        elif cell_code == self.item_manager.get_cell_code('P'):
            # Draw plate as circle
            plate_radius = self.cell_size // RENDER_CONFIG['plate_dispenser_ratio']
            self._draw_circle(center_x, center_y, plate_radius, RENDER_CONFIG['plate_color'], RENDER_CONFIG['plate_border_color'])
        
        # Ingredient Dispenser - draw colored circle for the ingredient type
        elif cell_code == self.item_manager.get_cell_code('I'):
            # Check if there's dispenser data for this position
            if (x, y) in game_state.get("dispensers", {}):
                dispenser_data = game_state["dispensers"][(x, y)]
                
                # Get ingredient name (should be a string)
                ingredient_name = dispenser_data if isinstance(dispenser_data, str) else "tomato"
                
                # Use centralized color - RAW ingredient color (60% brightness)
                base_color = INGREDIENTS[ingredient_name]['color']
                ingredient_color = tuple(int(c * 0.6) for c in base_color)  # Same as RAW
                
                indicator_radius = self.cell_size // RENDER_CONFIG['station_indicator_ratio']
                
                # Draw colored circle indicating ingredient type
                self._draw_circle(center_x, center_y, indicator_radius, ingredient_color)
        
        # Draw items on ground
        ground_instance_id = game_state["ground_instances"].get((x, y), None)
        if ground_instance_id is not None:
            ground_item = game_state["item_instances"].get(ground_instance_id)
            self._draw_item(x, y, ground_item)
    
    def _draw_agent(self, agent_id: str, pos: Tuple[int, int], game_state: Dict):
        """
        Draw an agent.
        
        Args:
            agent_id: Agent identifier
            pos: Agent position (x, y)
            game_state: Current game state
        """
        x, y = pos
        agent_num = self._get_agent_number(agent_id)
        
        # Draw agent as circle
        center_x, center_y = self._get_cell_center(x, y)
        radius = self.cell_size // RENDER_CONFIG['agent_radius_ratio']
        
        self._draw_circle(center_x, center_y, radius, RENDER_CONFIG['agent_color'], RENDER_CONFIG['agent_border_color'])
        
        # Draw orientation indicator (white circle showing direction)
        orientation = game_state["agent_orientation"][agent_id]
        dx, dy = orientation
        orientation_distance = RENDER_CONFIG['agent_orientation_distance']
        indicator_x = center_x + int(dx * radius * orientation_distance)
        indicator_y = center_y + int(dy * radius * orientation_distance)
        
        indicator_radius = radius // RENDER_CONFIG['agent_orientation_ratio']
        self._draw_circle(indicator_x, indicator_y, indicator_radius, RENDER_CONFIG['orientation_color'])
        
        # Draw agent number in white
        font = pygame.font.SysFont(None, RENDER_CONFIG['font_size_agent'])
        text = font.render(str(agent_num), True, RENDER_CONFIG['agent_number_color'])
        text_rect = text.get_rect(center=(center_x, center_y))
        self.window.blit(text, text_rect)
        
        # Draw item being carried (positioned on top of orientation indicator)
        carrying_instance_id = game_state["agent_holding_instance"].get(agent_id, None)
        if carrying_instance_id is not None:
            carrying_item = game_state["item_instances"].get(carrying_instance_id)
            # Position item slightly in front of agent (not a full cell)
            orientation = game_state["agent_orientation"][agent_id]
            dx, dy = orientation
            # Calculate position: move only partially in the direction
            offset_pixels = int(self.cell_size * RENDER_CONFIG['carried_item_offset'])
            item_center_x = center_x + (dx * offset_pixels)
            item_center_y = center_y + (dy * offset_pixels)
            # Draw item at calculated position using pixel coordinates directly
            self._draw_item_at_pixel(item_center_x, item_center_y, carrying_item)
    
    def _get_cell_color(self, cell_code: int) -> Tuple[int, int, int]:
        """Get RGB color for a cell type using ItemManager's tile colors."""
        char = self.item_manager.get_cell_char(cell_code)
        return self.item_manager.get_tile_color(char)
    
    def _draw_plate_ingredients(self, center_x: int, center_y: int, plate_radius: int, ingredients: list):
        """
        Helper method to draw ingredients on a plate.
        
        Args:
            center_x: Center X coordinate of the plate
            center_y: Center Y coordinate of the plate
            plate_radius: Radius of the plate
            ingredients: List of ingredient names
        """
        if len(ingredients) == 0:
            # Empty plate - just the white circle (already drawn)
            pass
        elif len(ingredients) == 1:
            # Single ingredient - centered on plate
            ing_color = INGREDIENTS[ingredients[0]]['color']
            ing_radius = plate_radius // RENDER_CONFIG['plate_ingredient_ratio']
            self._draw_circle(center_x, center_y, ing_radius, ing_color)
        elif len(ingredients) == 2:
            # Two ingredients - side by side
            ing_radius = plate_radius // RENDER_CONFIG['plate_multi_ingredient_ratio']
            offset_x = plate_radius // RENDER_CONFIG['plate_multi_ingredient_ratio']
            for i, ingredient_name in enumerate(ingredients):
                ing_color = INGREDIENTS[ingredient_name]['color']
                ing_x = center_x - offset_x + (i * offset_x * 2)
                self._draw_circle(ing_x, center_y, ing_radius, ing_color)
        else:
            # Three or more ingredients - arranged in triangle
            ing_radius = plate_radius // RENDER_CONFIG['plate_multi_ingredient_ratio']
            for i, ingredient_name in enumerate(ingredients[:3]):
                ing_color = INGREDIENTS[ingredient_name]['color']
                
                # Position ingredients in a triangle pattern
                if i == 0:  # Top
                    ing_x, ing_y = center_x, center_y - plate_radius // RENDER_CONFIG['plate_multi_ingredient_ratio']
                elif i == 1:  # Bottom left
                    ing_x, ing_y = center_x - plate_radius // RENDER_CONFIG['plate_multi_ingredient_ratio'], center_y + plate_radius // 4
                else:  # Bottom right
                    ing_x, ing_y = center_x + plate_radius // RENDER_CONFIG['plate_multi_ingredient_ratio'], center_y + plate_radius // 4
                
                self._draw_circle(ing_x, ing_y, ing_radius, ing_color)
    
    def _draw_item_at_pixel(self, pixel_x: int, pixel_y: int, item_dict: Optional[Dict]):
        """
        Draw an item at specific pixel coordinates (for carried items).
        
        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            item_dict: Item dictionary
        """
        if not item_dict:
            return
        
        color = get_item_color(item_dict)
        item_type = item_dict.get("type")
        
        if item_type == ItemType.INGREDIENT:
            # Draw as small circle
            radius = self.cell_size // RENDER_CONFIG['item_radius_ratio']
            self._draw_circle(pixel_x, pixel_y, radius, color)
        
        elif item_type == ItemType.DISH:
            # Draw plate as white circle
            plate_radius = self.cell_size // RENDER_CONFIG['plate_radius_ratio']
            self._draw_circle(pixel_x, pixel_y, plate_radius, RENDER_CONFIG['plate_color'], RENDER_CONFIG['plate_border_color'])
            
            # Draw ingredients on plate
            ingredients = item_dict.get("ingredients", [])
            self._draw_plate_ingredients(pixel_x, pixel_y, plate_radius, ingredients)
    
    def _draw_item(self, x: int, y: int, item_dict: Optional[Dict], offset: int = 0):
        """
        Draw an item on a cell.
        
        Args:
            x: X coordinate
            y: Y coordinate
            item_dict: Item dictionary
            offset: Pixel offset from center (for carried items)
        """
        if not item_dict:
            return
        
        color = get_item_color(item_dict)
        center_x, center_y = self._get_cell_center(x, y, offset)
        item_type = item_dict.get("type")
        
        if item_type == ItemType.INGREDIENT:
            # Draw as circle
            radius = self.cell_size // RENDER_CONFIG['item_radius_ratio']
            self._draw_circle(center_x, center_y, radius, color)
        
        elif item_type == ItemType.DISH:
            # Draw plate as white circle
            plate_radius = self.cell_size // RENDER_CONFIG['plate_radius_ratio']
            self._draw_circle(center_x, center_y, plate_radius, RENDER_CONFIG['plate_color'], RENDER_CONFIG['plate_border_color'])
            
            # Draw ingredients on plate
            ingredients = item_dict.get("ingredients", [])
            self._draw_plate_ingredients(center_x, center_y, plate_radius, ingredients)
    
    def __repr__(self) -> str:
        """String representation of Renderer."""
        return f"Renderer(mode={self.render_mode})"


__all__ = ['Renderer']
