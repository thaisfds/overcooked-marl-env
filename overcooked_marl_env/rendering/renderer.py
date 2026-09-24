"""
Renderer - Handles all rendering logic for the Overcooked environment.

This module implements the Single Responsibility Principle (SRP) by separating
rendering logic from the main environment class.

Supports two rendering modes:
- "human": Pygame graphical rendering (visual for humans)
- "ascii": ASCII text rendering to console
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
from ..core.constants import DIRECTION_NAMES, MAX_EPISODE_STEPS, STEPS_PER_SECOND
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
            render_mode: "human" for Pygame or "ascii" for ASCII
        """
        self.item_manager = item_manager
        self.render_mode = render_mode
        
        # Pygame-specific attributes
        self.window = None
        self.clock = None
        self.cell_size = RENDER_CONFIG['cell_size']
        self.should_close = False
        self.handle_window_events = True
    
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
            RGB array for "human" mode (Pygame), None for "ascii" mode (ASCII console)
        """
        if self.render_mode is None:
            return None
        
        if self.render_mode == "human":
            return self.render_rgb_array(game_state, map_size, agents, recipe_order)
        elif self.render_mode == "ascii":
            return self.render_text(game_state, map_size, agents, recipe_order)
        
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
        if self.window is None and self.render_mode == "human":
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
            self._draw_agent_vision(agent_id, pos, game_state)
        
        
        # Draw HUD (Heads-Up Display) with game information
        self._draw_hud(game_state, agents, recipe_order)
        
        # Scale and blit virtual_screen (self.window) to actual display window (self.display_window)
        if self.render_mode == "human" and hasattr(self, 'display_window') and self.display_window is not None:
            # Fill the display background with dark coffee color
            self.display_window.fill((75, 54, 42))
            
            # Calculate aspect ratio scaling
            ratio_w = self.window_width / self.virtual_width
            ratio_h = self.window_height / self.virtual_height
            scale = min(ratio_w, ratio_h)
            
            new_width = int(self.virtual_width * scale)
            new_height = int(self.virtual_height * scale)
            
            # Scale the virtual screen surface
            scaled_surf = pygame.transform.smoothscale(self.window, (new_width, new_height))
            
            # Center the scaled surface in the actual display window
            offset_x = (self.window_width - new_width) // 2
            offset_y = (self.window_height - new_height) // 2
            
            self.display_window.blit(scaled_surf, (offset_x, offset_y))
            
            # Update display
            if getattr(self, 'handle_window_events', True) and pygame.display.get_init():
                for event in pygame.event.get():
                    if event.type == pygame.VIDEORESIZE:
                        self.handle_resize(event.w, event.h, map_size)
                    elif event.type == pygame.QUIT:
                        self.should_close = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_F11:
                            pygame.display.toggle_fullscreen()
                            infoObject = pygame.display.Info()
                            self.handle_resize(infoObject.current_w, infoObject.current_h, map_size)
                        elif event.key == pygame.K_ESCAPE:
                            self.should_close = True
            else:
                pygame.event.pump()
                
            pygame.display.flip()
            self.clock.tick(RENDER_CONFIG['fps'])
        
        # Return RGB array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )
    
    def _draw_hud(self, game_state: Dict, agents: list, recipe_order: list):
        """
        Draw Heads-Up Display with game information.
        - Pending orders at the top as cards with ingredient requirements.
        - Team Score at the top right.
        - Timer and Delivered recipes in the bottom left.
        - Chef details in the bottom right side-by-side.
        """
        # Initialize fonts if not already done
        if not hasattr(self, 'font'):
            pygame.font.init()
            self.font = pygame.font.Font(None, RENDER_CONFIG['font_size'])
            self.font_small = pygame.font.Font(None, RENDER_CONFIG['font_size_small'])
            self.font_large = pygame.font.Font(None, 36)
            self.font_small_bold = pygame.font.Font(None, 18)
        
        # Recipes mapping to their respective ingredients
        RECIPE_INGREDIENTS = {
            "SIMPLE_SALAD": ["tomato", "lettuce"],
            "GARDEN_SALAD": ["tomato", "lettuce", "onion"],
            "FRESH_SALAD": ["tomato", "lettuce", "cucumber"],
            "ONION_SOUP": ["onion", "onion", "onion"],
            "ONION_SALAD": ["onion", "lettuce"]
        }
        
        # Colors for the ingredient visual badges
        ING_COLORS = {
            "tomato": ((220, 70, 60), "T"),
            "lettuce": ((90, 180, 90), "L"),
            "cucumber": ((40, 120, 80), "C"),
            "onion": ((190, 150, 100), "O")
        }
        
        # Get pending orders from game state
        delivered = game_state.get("delivered_recipes", [])
        pending_orders = self._calculate_pending_orders(recipe_order, delivered)
        
        # ===================================================================
        # 1. TOP HUD - Active Recipes and Team Score
        # ===================================================================
        
        # 1.1 Draw Recipe Cards on the left
        card_x_start = self.hud_padding
        card_y = 12
        card_width = 110
        card_height = 50
        card_spacing = 8
        
        for idx, recipe in enumerate(pending_orders[:5]):
            card_x = card_x_start + idx * (card_width + card_spacing)
            
            # Card background (Warm espresso card)
            pygame.draw.rect(self.window, (95, 72, 58), (card_x, card_y, card_width, card_height), border_radius=6)
            pygame.draw.rect(self.window, (186, 158, 138), (card_x, card_y, card_width, card_height), width=1, border_radius=6)
            
            # Recipe Display Name
            display_name = recipe.replace("_", " ").title()
            if len(display_name) > 13:
                display_name = display_name[:11] + ".."
            name_surf = self.font_small_bold.render(display_name, True, (255, 250, 240))
            self.window.blit(name_surf, (card_x + 6, card_y + 4))
            
            # Recipe Ingredients (small circular badges)
            ingredients = RECIPE_INGREDIENTS.get(recipe, [])
            for ing_idx, ing in enumerate(ingredients):
                color, char = ING_COLORS.get(ing, ((150, 150, 150), "?"))
                badge_x = card_x + 8 + ing_idx * 22
                badge_y = card_y + 24
                pygame.draw.circle(self.window, color, (badge_x + 7, badge_y + 8), 8)
                char_surf = self.font_small_bold.render(char, True, (255, 255, 255))
                self.window.blit(char_surf, (badge_x + 4, badge_y + 2))
                
        # 1.2 Draw Team Score on the right
        score_card_width = 110
        score_card_height = 50
        score_card_x = self.virtual_width - self.hud_padding - score_card_width
        
        # Draw Score Card background (Warm Caramel)
        pygame.draw.rect(self.window, (115, 86, 68), (score_card_x, card_y, score_card_width, score_card_height), border_radius=6)
        pygame.draw.rect(self.window, (186, 158, 138), (score_card_x, card_y, score_card_width, score_card_height), width=1, border_radius=6)
        
        total_score = sum(game_state.get("accumulated_rewards", {}).values())
        score_val_surf = self.font_large.render(f"{total_score:g}", True, (255, 215, 0)) # Shiny Gold
        score_lbl_surf = self.font_small_bold.render("SCORE", True, (240, 220, 205)) # Creamy
        
        # Center text horizontally inside the card
        lbl_x = score_card_x + (score_card_width - score_lbl_surf.get_width()) // 2
        val_x = score_card_x + (score_card_width - score_val_surf.get_width()) // 2
        
        self.window.blit(score_lbl_surf, (lbl_x, card_y + 4))
        self.window.blit(score_val_surf, (val_x, card_y + 20))
        
        # ===================================================================
        # 2. BOTTOM HUD - Timer, Delivered, and Chef details
        # ===================================================================
        
        map_height = (game_state["map_static"].shape[0] * self.cell_size)
        hud_y_start = self.grid_offset_y + map_height + self.hud_padding
        hud_x = self.hud_padding
        hud_width = self.virtual_width - 2 * self.hud_padding
        hud_height = self.bottom_hud_height
        
        # Draw Background Panel (Sleek Cappuccino Theme Card)
        pygame.draw.rect(self.window, (75, 54, 42), (hud_x, hud_y_start, hud_width, hud_height), border_radius=12)
        pygame.draw.rect(self.window, (139, 115, 98), (hud_x, hud_y_start, hud_width, hud_height), width=2, border_radius=12)
        
        # 2.1 Timer and Delivered recipes (Bottom Left)
        left_column_x = hud_x + 20
        
        # Time remaining calculation
        current_step = game_state.get("time_steps", 0)
        remaining_seconds = max(0, int((MAX_EPISODE_STEPS - current_step) / STEPS_PER_SECOND))
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # Timer Display
        clock_color = (255, 100, 100) if remaining_seconds <= 30 else (255, 245, 230)
        timer_lbl_surf = self.font_small.render("TIMER", True, (220, 200, 185))
        timer_surf = self.font_large.render(time_str, True, clock_color)
        self.window.blit(timer_lbl_surf, (left_column_x, hud_y_start + 12))
        self.window.blit(timer_surf, (left_column_x, hud_y_start + 28))
        
        # Delivered line (continuous line below the timer)
        delivered_lbl_surf = self.font_small.render("DELIVERED:", True, (220, 200, 185))
        self.window.blit(delivered_lbl_surf, (left_column_x, hud_y_start + 70))
        
        delivered_badges_y = hud_y_start + 68
        if delivered:
            badge_x = left_column_x + 95
            for recipe in delivered[-5:]:  # Limit to last 5 to fit layout width
                recipe_display = recipe.replace("_", " ").title()
                recipe_txt = self.font_small_bold.render(recipe_display, True, (215, 245, 220))
                recipe_width = recipe_txt.get_width() + 14
                pygame.draw.rect(self.window, (86, 125, 100), (badge_x, delivered_badges_y, recipe_width, 18), border_radius=4)
                self.window.blit(recipe_txt, (badge_x + 7, delivered_badges_y + 1))
                badge_x += recipe_width + 8
        else:
            none_txt = self.font_small.render("None", True, (160, 140, 130))
            self.window.blit(none_txt, (left_column_x + 95, delivered_badges_y + 2))
            
        # 2.2 Chef Status Cards (Bottom Right - side-by-side)
        chef_card_width = 180
        chef_card_height = 95
        chef_card_spacing = 15
        
        right_column_start = self.virtual_width - self.hud_padding - (chef_card_width * 2 + chef_card_spacing)
        
        for idx, agent_id in enumerate(sorted(agents)):
            card_x = right_column_start + idx * (chef_card_width + chef_card_spacing)
            card_y = hud_y_start + 12
            
            # Card background panel (Warm light brown)
            pygame.draw.rect(self.window, (95, 72, 58), (card_x, card_y, chef_card_width, chef_card_height), border_radius=8)
            pygame.draw.rect(self.window, (166, 138, 118), (card_x, card_y, chef_card_width, chef_card_height), width=1, border_radius=8)
            
            # Agent identity badge (colored circle)
            agent_num = self._get_agent_number(agent_id)
            badge_color = (100, 175, 185) if agent_num == 0 else (210, 130, 80) # Cozy Teal or Caramel
            pygame.draw.circle(self.window, badge_color, (card_x + 15, card_y + 15), 6)
            
            # Title
            title_txt = self.font.render(f"Chef {agent_num}", True, (255, 250, 240))
            self.window.blit(title_txt, (card_x + 28, card_y + 6))
            
            # Rewards
            step_reward = game_state.get("last_step_rewards", {}).get(agent_id, 0.0)
            total_reward = game_state.get("accumulated_rewards", {}).get(agent_id, 0.0)
            
            # Inventory item name
            carrying_instance_id = game_state.get("agent_holding_instance", {}).get(agent_id, None)
            if carrying_instance_id is not None and carrying_instance_id in game_state.get("item_instances", {}):
                carrying_item = game_state["item_instances"][carrying_instance_id]
                inventory = get_ingredient_display_name(carrying_item)
                if len(inventory) > 15:
                    inventory = inventory[:13] + ".."
            else:
                inventory = "Empty"
                
            score_txt = self.font_small_bold.render(f"Score: {total_reward:+.1f}", True, (255, 235, 200) if total_reward >= 0 else (255, 150, 150))
            step_txt = self.font_small_bold.render(f"Step: {step_reward:+.1f}", True, (255, 245, 230))
            inv_color = (200, 235, 190) if inventory != "Empty" else (220, 200, 185)
            inv_txt = self.font_small_bold.render(f"Holding: {inventory}", True, inv_color)
            
            # Find nearest object for the text HUD
            ax, ay = game_state["agent_pos"][agent_id]
            closest_obj = "None"
            min_dist = float('inf')
            H, W = game_state["map_static"].shape
            for y in range(H):
                for x in range(W):
                    char = self.item_manager.get_cell_char(game_state["map_static"][y, x])
                    dist = abs(ax - x) + abs(ay - y)
                    if char in ['C', 'P', 'D'] and dist < min_dist:
                        min_dist = dist
                        name_map = {'C': 'Board', 'P': 'Plates', 'D': 'Delivery'}
                        closest_obj = f"{name_map[char]} ({dist} steps)"
            for (x, y), ing in game_state.get("dispensers", {}).items():
                dist = abs(ax - x) + abs(ay - y)
                if dist < min_dist:
                    min_dist = dist
                    closest_obj = f"{ing.title()} ({dist} steps)"
                    
            vis_txt = self.font_small_bold.render(f"Focus: {closest_obj}", True, (200, 220, 255))
            
            # Optional log analysis to file (default: disabled)
            try:
                import os
                from pathlib import Path
                log_filename = os.environ.get("OVERCOOKED_VISION_LOG", None)
                if log_filename and log_filename.strip() and log_filename.strip().lower() not in ("0", "false", "off", "none"):
                    log_dir = Path(__file__).resolve().parent.parent.parent / "experiments" / "logs" / "vision_analysis"
                    os.makedirs(log_dir, exist_ok=True)
                    log_str = log_filename.strip()
                    if not log_str.endswith(".log"):
                        log_str += ".log"
                    log_filepath = log_dir / log_str if not Path(log_str).is_absolute() else Path(log_str)
                    with open(log_filepath, "a") as f:
                        time_step = game_state.get("time_steps", 0)
                        f.write(f"[Step {time_step:04d}] {agent_id} | Pos: {ax:02d},{ay:02d} | Hold: {inventory:10} | Focus: {closest_obj:20} | Score: {total_reward:+.1f} | Step Rwd: {step_reward:+.1f}\n")
            except Exception:
                pass
            
            self.window.blit(score_txt, (card_x + 12, card_y + 30))
            self.window.blit(step_txt, (card_x + 100, card_y + 30))
            self.window.blit(inv_txt, (card_x + 12, card_y + 50))
            self.window.blit(vis_txt, (card_x + 12, card_y + 70))
    
    def _draw_agent_vision(self, agent_id: str, pos: Tuple[int, int], game_state: Dict):
        """Draw lines to represent what the agent's neural network is focusing on."""
        H, W = game_state["map_static"].shape
        ax, ay = pos
        center_x, center_y = self._get_cell_center(ax, ay)
        
        closest_c = None
        closest_p = None
        closest_d = None
        min_dist_c = float('inf')
        min_dist_p = float('inf')
        min_dist_d = float('inf')
        
        for y in range(H):
            for x in range(W):
                cell_code = game_state["map_static"][y, x]
                char = self.item_manager.get_cell_char(cell_code)
                dist = abs(ax - x) + abs(ay - y)
                if char == 'C' and dist < min_dist_c:
                    min_dist_c = dist
                    closest_c = (x, y)
                elif char == 'P' and dist < min_dist_p:
                    min_dist_p = dist
                    closest_p = (x, y)
                elif char == 'D' and dist < min_dist_d:
                    min_dist_d = dist
                    closest_d = (x, y)
                    
        recipe_queue = list(game_state.get("recipe_order", []))
        for delivered in game_state.get("delivered_recipes", []):
            if delivered in recipe_queue:
                recipe_queue.remove(delivered)
        
        needed_ingredients = []
        if len(recipe_queue) > 0:
            rec = recipe_queue[0]
            if "SALAD" in rec:
                needed_ingredients = ["tomato", "lettuce", "cucumber"]
            elif "SOUP" in rec:
                needed_ingredients = ["onion"]
                
        closest_ing = None
        min_dist_i = float('inf')
        for (x, y), ing in game_state.get("dispensers", {}).items():
            if ing.lower() in needed_ingredients:
                dist = abs(ax - x) + abs(ay - y)
                if dist < min_dist_i:
                    min_dist_i = dist
                    closest_ing = (x, y)
                    
        line_surf = pygame.Surface((self.virtual_width, self.virtual_height), pygame.SRCALPHA)
        
        def draw_dashed_line(surf, color, start_pos, end_pos, width=2, dash_length=8):
            x1, y1 = start_pos
            x2, y2 = end_pos
            dl = ((x2-x1)**2 + (y2-y1)**2)**0.5
            if dl == 0: return
            dashes = int(dl / dash_length)
            for i in range(dashes):
                if i % 2 == 0:
                    sx = x1 + (x2-x1) * (i/dashes)
                    sy = y1 + (y2-y1) * (i/dashes)
                    ex = x1 + (x2-x1) * ((i+1)/dashes)
                    ey = y1 + (y2-y1) * ((i+1)/dashes)
                    pygame.draw.line(surf, color, (sx, sy), (ex, ey), width)
                    
        # Draw translucent dashed lines pointing to key focuses
        if closest_c:
            tx, ty = self._get_cell_center(*closest_c)
            draw_dashed_line(line_surf, (200, 200, 200, 100), (center_x, center_y), (tx, ty), 2)
        if closest_p:
            tx, ty = self._get_cell_center(*closest_p)
            draw_dashed_line(line_surf, (100, 100, 255, 100), (center_x, center_y), (tx, ty), 2)
        if closest_d:
            tx, ty = self._get_cell_center(*closest_d)
            draw_dashed_line(line_surf, (255, 215, 0, 100), (center_x, center_y), (tx, ty), 2)
        if closest_ing:
            tx, ty = self._get_cell_center(*closest_ing)
            draw_dashed_line(line_surf, (100, 255, 100, 100), (center_x, center_y), (tx, ty), 2)
            
        self.window.blit(line_surf, (0, 0))
    
    def close(self):
        """Clean up Pygame resources."""
        if (hasattr(self, 'display_window') and self.display_window is not None) or (self.window is not None):
            pygame.display.quit()
            pygame.quit()
            if hasattr(self, 'display_window'):
                self.display_window = None
        self.window = None
        self.clock = None
        self.should_close = False
    
    def check_events(self) -> bool:
        """Check pygame events and return False if window should close."""
        if self.render_mode != "human" or self.window is None:
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
        # Show remaining time formatted as MM:SS based on STEPS_PER_SECOND
        current_step = game_state.get("time_steps", 0)
        remaining_seconds = max(0, int((MAX_EPISODE_STEPS - current_step) / STEPS_PER_SECOND))
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        time_str = f"{minutes:02d}:{seconds:02d}"
        print(f"Time Remaining: {time_str} (Step: {current_step}/{MAX_EPISODE_STEPS})")
        print(f"Remaining Dishes: {game_state['dishes_pending']}")
        
        # Show orders and delivered
        if len(recipe_order) > 0:
            
            # Calculate pending orders using utility method
            pending_orders = self._calculate_pending_orders(recipe_order, game_state['delivered_recipes'])
            
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
        self.hud_padding = RENDER_CONFIG['hud_padding']
        self.top_hud_height = 70
        self.bottom_hud_height = 110
        self.hud_height = self.bottom_hud_height
        
        # Fixed cell size for the internal virtual surface
        self.cell_size = 60
        self.grid_offset_x = self.hud_padding
        self.grid_offset_y = self.hud_padding + self.top_hud_height
        
        # Calculate window dimensions - ensure minimum width for HUD text
        grid_width = W * self.cell_size
        grid_height = H * self.cell_size
        
        min_width = RENDER_CONFIG['hud_min_width']
        window_width = max(grid_width + 2 * self.hud_padding, min_width)
        window_height = self.grid_offset_y + grid_height + self.hud_padding + self.bottom_hud_height
        
        # Center the grid horizontally if window is wider than grid
        if window_width > grid_width + 2 * self.hud_padding:
            self.grid_offset_x = (window_width - grid_width) // 2
            
        self.virtual_width = window_width
        self.virtual_height = window_height
        
        # Virtual screen surface where all rendering takes place at fixed 60px cell size
        self.window = pygame.Surface((self.virtual_width, self.virtual_height))
        
        # Actual display window that will be shown to the user and can be resized
        self.window_width = window_width
        self.window_height = window_height
        self.display_window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Overcooked MARL")
        self.clock = pygame.time.Clock()
        
    def handle_resize(self, width: int, height: int, map_size: Tuple[int, int]):
        """Handle window resize event by updating the actual display window size."""
        self.window_width = width
        self.window_height = height
        self.display_window = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    
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
