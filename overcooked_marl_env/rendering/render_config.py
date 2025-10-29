"""
Rendering configuration constants for the Overcooked environment.

This module centralizes all visual rendering parameters to make it easy
to adjust the appearance of the game without modifying the renderer code.
"""

# Rendering constants
RENDER_CONFIG = {
    # Grid and cell dimensions
    'cell_size': 60,              # Size of each grid cell in pixels
    
    # HUD (Heads-Up Display) layout
    'hud_height': 150,            # Height reserved for HUD below grid
    'hud_padding': 20,            # Padding around HUD and grid
    'hud_line_height': 25,        # Height of each line in HUD
    'hud_min_width': 800,         # Minimum window width for HUD text
    'hud_extra_spacing': 5,       # Extra spacing between sections
    
    # HUD field positions (X coordinates)
    'hud_step_reward_x': 100,     # X position for step reward field
    'hud_total_reward_x': 250,    # X position for total reward field
    'hud_inventory_x': 420,       # X position for inventory field
    
    # Font sizes
    'font_size': 24,              # Main font size
    'font_size_small': 20,        # Small font size for details
    'font_size_agent': 24,        # Agent number font size
    
    # Agent rendering
    'agent_radius_ratio': 3,      # cell_size // 3 for agent circle
    'agent_orientation_ratio': 4, # radius // 4 for orientation indicator
    'agent_orientation_distance': 0.7,  # Distance of orientation indicator (ratio of radius)
    
    # Item rendering
    'item_radius_ratio': 6,       # cell_size // 6 for ingredient circle
    'plate_radius_ratio': 5,      # cell_size // 5 for plate circle
    'plate_ingredient_ratio': 2,  # plate_radius // 2 for single ingredient
    'plate_multi_ingredient_ratio': 3,  # plate_radius // 3 for multiple ingredients
    
    # Station visual indicators
    'station_indicator_ratio': 4, # cell_size // 4 for indicator size
    'knife_padding': 15,          # Padding for knife line on chopping board
    'knife_y_position': 30,       # Y position for knife line
    'knife_line_width': 3,        # Width of knife line
    'trash_x_padding_ratio': 4,   # cell_size // 4 for trash X padding
    'trash_line_width': 4,        # Width of trash X lines
    'arrow_size_ratio': 4,        # cell_size // 4 for delivery arrow
    'arrow_border_width': 2,      # Border width for delivery arrow
    'plate_dispenser_ratio': 3,   # cell_size // 3 for plate on dispenser
    
    # Item positioning
    'carried_item_offset': 0.3,   # Offset ratio for carried items (30% of cell)
    
    # Rendering settings
    'fps': 60,                    # Frames per second for pygame
    'grid_border_width': 1,       # Width of cell borders
    'ascii_grid_padding': 2,      # Padding around ASCII grid (characters)
    
    # Colors
    'background_color': (200, 200, 200),    # Light gray background
    'border_color': (100, 100, 100),        # Grid cell borders
    'agent_color': (0, 0, 0),               # Black agents
    'agent_border_color': (50, 50, 50),     # Agent border
    'agent_number_color': (255, 255, 255),  # White agent numbers
    'orientation_color': (255, 255, 255),   # White orientation indicator
    'plate_color': (250, 250, 250),         # White plate
    'plate_border_color': (100, 100, 100),  # Gray plate border
    
    # Text colors
    'text_black': (0, 0, 0),                # Black text
    'text_white': (255, 255, 255),          # White text
    'text_step_reward': (80, 0, 80),        # Purple for step reward
    'text_total_reward': (0, 0, 150),       # Dark blue for total reward
    'text_inventory': (0, 80, 0),           # Dark green for inventory
    'text_success': (0, 150, 0),            # Green for success messages
    'text_disabled': (120, 120, 120),       # Gray for disabled text
}
