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
from ..core.items import ItemType, ItemState, get_ingredient_display_name


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
        self.cell_size = 60  # Size of each cell in pixels
        
        # Character map for ASCII rendering
        self.char_map = {
            self.item_manager.get_cell_code(' '): ' ',
            self.item_manager.get_cell_code('#'): '#',
            self.item_manager.get_cell_code('C'): 'C',
            self.item_manager.get_cell_code('P'): 'P',
            self.item_manager.get_cell_code('D'): 'D',
            self.item_manager.get_cell_code('I'): 'I',
            self.item_manager.get_cell_code('T'): 'T',
            self.item_manager.get_cell_code('|'): '|',
            self.item_manager.get_cell_code('_'): '_',
        }
    
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
            return self.render_rgb_array(game_state, map_size, agents)
        
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
                grid[y][x] = self.char_map.get(cell_code, '?')
                
                # Draw items on ground
                ground_instance_id = game_state["ground_instances"].get((x, y), None)
                if ground_instance_id is not None:
                    ground_item = game_state["item_instances"].get(ground_instance_id)
                    grid[y][x] = self._get_item_char(ground_item)
        
        # Draw agents
        for agent_id, pos in game_state["agent_pos"].items():
            x, y = pos
            agent_num = int(agent_id.split('_')[1])
            
            # Show agent number
            grid[y][x] = str(agent_num)
            
            # Check if agent is carrying something
            carrying_instance_id = game_state["agent_holding_instance"].get(agent_id, None)
            if carrying_instance_id is not None:
                carrying_item = game_state["item_instances"].get(carrying_instance_id)
                # Print what the agent is carrying next to the grid
                print(f"Agent {agent_num} is carrying: {get_ingredient_display_name(carrying_item)}")
        
        # Print the grid to console
        print("\n" + "-" * (W + 2))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("-" * (W + 2))
        
        # Print game state information
        self._print_game_info(game_state, recipe_order)
        
        # Print agent status
        self._print_agent_status(game_state, agents, map_size)
        
        return None
    
    def render_rgb_array(
        self,
        game_state: Dict,
        map_size: Tuple[int, int],
        agents: list
    ) -> np.ndarray:
        """
        Renders the environment as an RGB array using Pygame.
        
        Args:
            game_state: Current game state dictionary
            map_size: Tuple of (height, width)
            agents: List of agent IDs
        
        Returns:
            numpy array of shape (H, W, 3) with RGB values
        """
        if self.window is None:
            self._initialize_pygame(map_size)
        
        H, W = map_size
        
        # Fill background
        self.window.fill((200, 200, 200))  # Light gray background
        
        # Draw grid cells
        for y in range(H):
            for x in range(W):
                self._draw_cell(x, y, game_state)
        
        # Draw agents
        for agent_id, pos in game_state["agent_pos"].items():
            self._draw_agent(agent_id, pos, game_state)
        
        # Update display
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS
        
        # Return RGB array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )
    
    def close(self):
        """Clean up Pygame resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
    
    # ===================================================================
    # Helper Methods - ASCII Rendering
    # ===================================================================
    
    def _get_item_char(self, item_dict: Optional[Dict]) -> str:
        """
        Get character representation for an item.
        
        Args:
            item_dict: Item dictionary
        
        Returns:
            Single character representing the item
        """
        if not item_dict:
            return ' '
        
        item_type = item_dict.get("type")
        
        if item_type == ItemType.INGREDIENT:
            name = item_dict["name"]
            state = item_dict.get("state", ItemState.RAW)
            
            # Map ingredients to letters
            name_map = {'tomato': 't', 'lettuce': 'l', 'cucumber': 'c'}
            char = name_map.get(name, '?')
            
            # Uppercase if READY, lowercase if RAW, number if CHOPPING
            if state == ItemState.READY:
                return char.upper()
            elif state == ItemState.RAW:
                return char.lower()
            elif state == ItemState.CHOPPING_1:
                return '1'
            elif state == ItemState.CHOPPING_2:
                return '2'
            elif state == ItemState.CHOPPING_3:
                return '3'
            return char
            
        elif item_type == ItemType.DISH:
            ingredients = item_dict.get("ingredients", [])
            if not ingredients:
                return 'p'  # empty plate
            else:
                # Show first letter of first ingredient
                return ingredients[0][0].lower()
        
        return '?'
    
    def _print_game_info(self, game_state: Dict, recipe_order: list):
        """Print game state information."""
        print(f"Step: {game_state['time_steps']}")
        print(f"Remaining Dishes: {game_state['dishes_pending']}")
        
        # Show orders and delivered
        if len(recipe_order) > 0:
            # Calculate pending orders: original order minus delivered
            pending_orders = list(recipe_order)
            for delivered_recipe in game_state['delivered_recipes']:
                if delivered_recipe in pending_orders:
                    pending_orders.remove(delivered_recipe)  # Remove first occurrence
            
            if len(pending_orders) > 0:
                orders_str = ", ".join(pending_orders)
                print(f"Orders: {orders_str}")
            else:
                print(f"Orders: (none)")
            
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
            station_name = self._get_station_name(cell_code)
            return f"{orientation_name} -> {station_name}"
        else:
            return f"{orientation_name} -> Wall"
    
    def _get_orientation_name(self, direction: Tuple[int, int]) -> str:
        """Converts direction vector to readable name."""
        direction_map = {
            (0, -1): "North",
            (0, 1): "South",
            (1, 0): "East",
            (-1, 0): "West"
        }
        return direction_map.get(direction, "Unknown")
    
    def _get_station_name(self, cell_code: int) -> str:
        """Converts cell code to readable name."""
        station_map = {
            self.item_manager.get_cell_code('C'): "Chopping Board",
            self.item_manager.get_cell_code('P'): "Plate Dispenser",
            self.item_manager.get_cell_code('D'): "Delivery",
            self.item_manager.get_cell_code('I'): "Ingredient Dispenser",
            self.item_manager.get_cell_code('T'): "Trash",
            self.item_manager.get_cell_code('#'): "Counter",
            self.item_manager.get_cell_code(' '): "Empty"
        }
        return station_map.get(cell_code, "Unknown")
    
    # ===================================================================
    # Helper Methods - Pygame Rendering
    # ===================================================================
    
    def _initialize_pygame(self, map_size: Tuple[int, int]):
        """Initialize Pygame window."""
        pygame.init()
        pygame.display.init()
        
        H, W = map_size
        window_width = W * self.cell_size
        window_height = H * self.cell_size
        
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
            x * self.cell_size,
            y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        
        # Get cell color based on type
        color = self._get_cell_color(cell_code)
        pygame.draw.rect(self.window, color, rect)
        pygame.draw.rect(self.window, (100, 100, 100), rect, 1)  # Border
        
        # Draw visual indicators for stations
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        
        # Chopping board - draw knife line
        if cell_code == self.item_manager.get_cell_code('C'):
            pygame.draw.line(
                self.window, 
                (100, 100, 100), 
                (x * self.cell_size + 15, y * self.cell_size + 30),
                (x * self.cell_size + self.cell_size - 15, y * self.cell_size + 30), 
                3
            )
        
        # Trash - draw X
        elif cell_code == self.item_manager.get_cell_code('T'):
            padding = self.cell_size // 4
            # Diagonal 1
            pygame.draw.line(
                self.window, 
                (0, 0, 0),
                (x * self.cell_size + padding, y * self.cell_size + padding),
                (x * self.cell_size + self.cell_size - padding, y * self.cell_size + self.cell_size - padding), 
                4
            )
            # Diagonal 2
            pygame.draw.line(
                self.window, 
                (0, 0, 0),
                (x * self.cell_size + self.cell_size - padding, y * self.cell_size + padding),
                (x * self.cell_size + padding, y * self.cell_size + self.cell_size - padding), 
                4
            )
        
        # Delivery - draw arrow pointing right
        elif cell_code == self.item_manager.get_cell_code('D'):
            arrow_size = self.cell_size // 4
            arrow_points = [
                (center_x - arrow_size, center_y - arrow_size),  # Top left
                (center_x - arrow_size, center_y + arrow_size),  # Bottom left
                (center_x + arrow_size, center_y),               # Right tip
            ]
            pygame.draw.polygon(self.window, (255, 255, 255), arrow_points)
            pygame.draw.polygon(self.window, (100, 100, 100), arrow_points, 2)
        
        # Plate Dispenser - draw empty plate
        elif cell_code == self.item_manager.get_cell_code('P'):
            # Draw plate as circle (larger size)
            plate_radius = self.cell_size // 3
            
            # Draw filled white circle
            gfxdraw.filled_circle(
                self.window, 
                center_x, 
                center_y, 
                plate_radius, 
                (250, 250, 250)  # White plate
            )
            # Draw border
            gfxdraw.aacircle(
                self.window, 
                center_x, 
                center_y, 
                plate_radius, 
                (100, 100, 100)  # Gray border
            )
        
        # Ingredient Dispenser - draw colored circle for the ingredient type
        elif cell_code == self.item_manager.get_cell_code('I'):
            # Check if there's dispenser data for this position
            if (x, y) in game_state.get("dispensers", {}):
                dispenser_data = game_state["dispensers"][(x, y)]
                
                # Get ingredient name
                if isinstance(dispenser_data, str):
                    ingredient_name = dispenser_data
                else:
                    # Legacy mapping if needed
                    ingredient_map = {5: "tomato", 6: "lettuce", 7: "cucumber"}
                    ingredient_name = ingredient_map.get(dispenser_data, "tomato")
                
                # Use centralized color method - RAW ingredient color (60% brightness)
                base_color = self._get_ingredient_base_color(ingredient_name)
                ingredient_color = tuple(int(c * 0.6) for c in base_color)  # Same as RAW
                
                indicator_radius = self.cell_size // 4
                
                # Draw colored circle indicating ingredient type
                gfxdraw.filled_circle(self.window, center_x, center_y, indicator_radius, ingredient_color)
                gfxdraw.aacircle(self.window, center_x, center_y, indicator_radius, (0, 0, 0))
        
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
        agent_num = int(agent_id.split('_')[1])
        
        # All agents are black
        agent_color = (0, 0, 0)
        
        # Draw agent as circle
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3
        
        gfxdraw.filled_circle(self.window, center_x, center_y, radius, agent_color)
        gfxdraw.aacircle(self.window, center_x, center_y, radius, (50, 50, 50))
        
        # Draw orientation indicator (white circle showing direction)
        orientation = game_state["agent_orientation"][agent_id]
        dx, dy = orientation
        indicator_x = center_x + int(dx * radius * 0.7)
        indicator_y = center_y + int(dy * radius * 0.7)
        
        gfxdraw.filled_circle(self.window, indicator_x, indicator_y, radius // 4, (255, 255, 255))
        
        # Draw agent number in white
        font = pygame.font.SysFont(None, 24)
        text = font.render(str(agent_num), True, (255, 255, 255))
        text_rect = text.get_rect(center=(center_x, center_y))
        self.window.blit(text, text_rect)
        
        # Draw item being carried (positioned on top of orientation indicator)
        carrying_instance_id = game_state["agent_holding_instance"].get(agent_id, None)
        if carrying_instance_id is not None:
            carrying_item = game_state["item_instances"].get(carrying_instance_id)
            # Position item slightly in front of agent (not a full cell)
            orientation = game_state["agent_orientation"][agent_id]
            dx, dy = orientation
            # Calculate position: move only partially in the direction (30% of cell size)
            offset_pixels = int(self.cell_size * 0.3)
            item_center_x = center_x + (dx * offset_pixels)
            item_center_y = center_y + (dy * offset_pixels)
            # Draw item at calculated position using pixel coordinates directly
            self._draw_item_at_pixel(item_center_x, item_center_y, carrying_item)
    
    def _get_cell_color(self, cell_code: int) -> Tuple[int, int, int]:
        """Get RGB color for a cell type."""
        color_map = {
            self.item_manager.get_cell_code(' '): (255, 255, 255),  # White - empty
            self.item_manager.get_cell_code('#'): (180, 180, 180),  # Gray - counter
            self.item_manager.get_cell_code('C'): (210, 180, 140),  # Light brown - chopping (tan)
            self.item_manager.get_cell_code('P'): (100, 149, 237),  # Blue - plate
            self.item_manager.get_cell_code('D'): (50, 205, 50),    # Green - delivery
            self.item_manager.get_cell_code('I'): (210, 180, 140),  # Light brown - ingredient dispenser (tan)
            self.item_manager.get_cell_code('T'): (105, 105, 105),  # Dark gray - trash
            self.item_manager.get_cell_code('|'): (70, 70, 70),     # Dark - wall
            self.item_manager.get_cell_code('_'): (70, 70, 70),     # Dark - wall
        }
        return color_map.get(cell_code, (200, 200, 200))
    
    def _get_ingredient_base_color(self, ingredient_name: str) -> Tuple[int, int, int]:
        """
        Get the base RGB color for an ingredient (unified color definition).
        This is used for both dispensers and items.
        
        Args:
            ingredient_name: Name of the ingredient ('tomato', 'lettuce', 'cucumber')
        
        Returns:
            RGB color tuple
        """
        ingredient_colors = {
            'tomato': (255, 99, 71),      # Tomato red
            'lettuce': (144, 238, 144),   # Light green
            'cucumber': (60, 179, 113),   # Medium sea green
        }
        return ingredient_colors.get(ingredient_name, (200, 200, 200))
    
    def _get_item_color(self, item_dict: Optional[Dict]) -> Tuple[int, int, int]:
        """Get RGB color for an item."""
        if not item_dict:
            return (255, 255, 255)
        
        item_type = item_dict.get("type")
        
        if item_type == ItemType.INGREDIENT:
            name = item_dict["name"]
            state = item_dict.get("state", ItemState.RAW)
            
            # Get base color from centralized method
            base_color = self._get_ingredient_base_color(name)
            
            # RAW ingredients are darker (60% brightness)
            if state == ItemState.RAW:
                return tuple(int(c * 0.6) for c in base_color)
            # READY ingredients use full base color
            elif state == ItemState.READY:
                return base_color
            # CHOPPING ingredients are slightly darker (80% brightness)
            else:  # CHOPPING
                return tuple(int(c * 0.8) for c in base_color)
        
        elif item_type == ItemType.DISH:
            return (240, 240, 240)  # Light gray for plate
        
        return (200, 200, 200)
    
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
        
        color = self._get_item_color(item_dict)
        item_type = item_dict.get("type")
        
        if item_type == ItemType.INGREDIENT:
            # Draw as small circle (slightly larger now)
            radius = self.cell_size // 6
            gfxdraw.filled_circle(self.window, pixel_x, pixel_y, radius, color)
            gfxdraw.aacircle(self.window, pixel_x, pixel_y, radius, (0, 0, 0))
        
        elif item_type == ItemType.DISH:
            # Draw plate as white circle (slightly larger now)
            plate_radius = self.cell_size // 5
            
            # Draw filled white circle (plate base)
            gfxdraw.filled_circle(self.window, pixel_x, pixel_y, plate_radius, (250, 250, 250))
            gfxdraw.aacircle(self.window, pixel_x, pixel_y, plate_radius, (100, 100, 100))
            
            # Draw ingredients on plate as small circles on top
            ingredients = item_dict.get("ingredients", [])
            
            if len(ingredients) == 0:
                # Empty plate - just the white circle
                pass
            elif len(ingredients) == 1:
                # Single ingredient - centered on plate
                ing_color = self._get_ingredient_base_color(ingredients[0])
                ing_radius = plate_radius // 2
                gfxdraw.filled_circle(self.window, pixel_x, pixel_y, ing_radius, ing_color)
                gfxdraw.aacircle(self.window, pixel_x, pixel_y, ing_radius, (0, 0, 0))
            elif len(ingredients) == 2:
                # Two ingredients - side by side
                ing_radius = plate_radius // 3
                offset_x = plate_radius // 3
                for i, ingredient_name in enumerate(ingredients):
                    ing_color = self._get_ingredient_base_color(ingredient_name)
                    ing_x = pixel_x - offset_x + (i * offset_x * 2)
                    gfxdraw.filled_circle(self.window, ing_x, pixel_y, ing_radius, ing_color)
                    gfxdraw.aacircle(self.window, ing_x, pixel_y, ing_radius, (0, 0, 0))
            else:
                # Three or more ingredients - arranged in triangle
                ing_radius = plate_radius // 3
                for i, ingredient_name in enumerate(ingredients[:3]):
                    ing_color = self._get_ingredient_base_color(ingredient_name)
                    
                    # Position ingredients in a triangle pattern
                    if i == 0:  # Top
                        ing_x, ing_y = pixel_x, pixel_y - plate_radius // 3
                    elif i == 1:  # Bottom left
                        ing_x, ing_y = pixel_x - plate_radius // 3, pixel_y + plate_radius // 4
                    else:  # Bottom right
                        ing_x, ing_y = pixel_x + plate_radius // 3, pixel_y + plate_radius // 4
                    
                    gfxdraw.filled_circle(self.window, ing_x, ing_y, ing_radius, ing_color)
                    gfxdraw.aacircle(self.window, ing_x, ing_y, ing_radius, (0, 0, 0))
    
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
        
        color = self._get_item_color(item_dict)
        
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2 - offset
        
        item_type = item_dict.get("type")
        
        if item_type == ItemType.INGREDIENT:
            # Draw as circle (same size as carried items)
            radius = self.cell_size // 6
            gfxdraw.filled_circle(self.window, center_x, center_y, radius, color)
            gfxdraw.aacircle(self.window, center_x, center_y, radius, (0, 0, 0))
        
        elif item_type == ItemType.DISH:
            # Draw plate as white circle (same size as carried items)
            plate_radius = self.cell_size // 5
            
            # Draw filled white circle (plate base)
            gfxdraw.filled_circle(self.window, center_x, center_y, plate_radius, (250, 250, 250))
            gfxdraw.aacircle(self.window, center_x, center_y, plate_radius, (100, 100, 100))
            
            # Draw ingredients on plate as small circles on top
            ingredients = item_dict.get("ingredients", [])
            
            if len(ingredients) == 0:
                # Empty plate - just the white circle
                pass
            elif len(ingredients) == 1:
                # Single ingredient - centered on plate
                ing_color = self._get_ingredient_base_color(ingredients[0])
                ing_radius = plate_radius // 2
                gfxdraw.filled_circle(self.window, center_x, center_y, ing_radius, ing_color)
                gfxdraw.aacircle(self.window, center_x, center_y, ing_radius, (0, 0, 0))
            elif len(ingredients) == 2:
                # Two ingredients - side by side
                ing_radius = plate_radius // 3
                offset_x = plate_radius // 3
                for i, ingredient_name in enumerate(ingredients):
                    ing_color = self._get_ingredient_base_color(ingredient_name)
                    ing_x = center_x - offset_x + (i * offset_x * 2)
                    gfxdraw.filled_circle(self.window, ing_x, center_y, ing_radius, ing_color)
                    gfxdraw.aacircle(self.window, ing_x, center_y, ing_radius, (0, 0, 0))
            else:
                # Three or more ingredients - arranged in triangle
                ing_radius = plate_radius // 3
                for i, ingredient_name in enumerate(ingredients[:3]):
                    ing_color = self._get_ingredient_base_color(ingredient_name)
                    
                    # Position ingredients in a triangle pattern
                    if i == 0:  # Top
                        ing_x, ing_y = center_x, center_y - plate_radius // 3
                    elif i == 1:  # Bottom left
                        ing_x, ing_y = center_x - plate_radius // 3, center_y + plate_radius // 4
                    else:  # Bottom right
                        ing_x, ing_y = center_x + plate_radius // 3, center_y + plate_radius // 4
                    
                    gfxdraw.filled_circle(self.window, ing_x, ing_y, ing_radius, ing_color)
                    gfxdraw.aacircle(self.window, ing_x, ing_y, ing_radius, (0, 0, 0))
    
    def __repr__(self) -> str:
        """String representation of Renderer."""
        return f"Renderer(mode={self.render_mode})"


__all__ = ['Renderer']
