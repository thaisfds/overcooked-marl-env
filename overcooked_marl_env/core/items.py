"""
Item system using dictionaries with metadata for flexible recipe management.
"""

from .constants import DEFAULT_CUTS_NEEDED

# =================================================================
# ITEM PROCESSING CONSTANTS
# =================================================================

# Number of cuts required for each chopping state
CHOPPING_STATE_1_CUTS = 1
CHOPPING_STATE_2_CUTS = 2
CHOPPING_STATE_3_CUTS = 3

# Brightness multipliers for ingredient colors based on state
BRIGHTNESS_RAW = 0.6          # 60% brightness for raw ingredients
BRIGHTNESS_CHOPPING = 0.8     # 80% brightness for chopping states
BRIGHTNESS_READY = 1.0        # 100% brightness for ready ingredients

# Default fallback colors
DEFAULT_COLOR_UNKNOWN = (200, 200, 200)  # Gray for unknown items
DEFAULT_COLOR_EMPTY = (255, 255, 255)     # White for empty/None
DEFAULT_COLOR_PLATE = (240, 240, 240)     # Light gray for plates

# First ingredient index in a dish
FIRST_INGREDIENT_INDEX = 0

# =================================================================
# INGREDIENT DEFINITIONS
# =================================================================

# Ingredient definitions: name -> {color, cuts_needed, char, state_chars}
INGREDIENTS = {
    'tomato': {
        'color': (255, 0, 0), 
        'cuts_needed': DEFAULT_CUTS_NEEDED, 
        'char': 't',
        'state_chars': {
            'RAW': 't',
            'CHOPPING_1': '1',
            'CHOPPING_2': '2',
            'CHOPPING_3': '3',
            'READY': 'T'
        }
    },
    'lettuce': {
        'color': (107,142,35), 
        'cuts_needed': DEFAULT_CUTS_NEEDED, 
        'char': 'l',
        'state_chars': {
            'RAW': 'l',
            'CHOPPING_1': '1',
            'CHOPPING_2': '2',
            'CHOPPING_3': '3',
            'READY': 'L'
        }
    },
    'cucumber': {
        'color': (0,100,0), 
        'cuts_needed': DEFAULT_CUTS_NEEDED, 
        'char': 'c',
        'state_chars': {
            'RAW': 'c',
            'CHOPPING_1': '1',
            'CHOPPING_2': '2',
            'CHOPPING_3': '3',
            'READY': 'C'
        }
    },
    'onion': {
        'color': (238, 130, 238), 
        'cuts_needed': DEFAULT_CUTS_NEEDED, 
        'char': 'o',
        'state_chars': {
            'RAW': 'o',
            'CHOPPING_1': '1',
            'CHOPPING_2': '2',
            'CHOPPING_3': '3',
            'READY': 'O'
        }
    },
}

def register_custom_ingredient(name, color, cuts_needed=3, char=None, state_chars=None):
    """
    Register a custom ingredient dynamically.
    
    Args:
        name: Name of the ingredient
        color: RGB tuple (e.g., (255, 0, 0))
        cuts_needed: Number of cuts required (default: 3)
        char: Default character representing the ingredient (default: first letter of name)
        state_chars: Dictionary mapping ItemState to character representation
    """
    name_lower = name.lower()
    if char is None:
        char = name_lower[0] if name_lower else 'x'
    
    if state_chars is None:
        state_chars = {
            'RAW': char,
            'CHOPPING_1': '1',
            'CHOPPING_2': '2',
            'CHOPPING_3': '3',
            'READY': char.upper()
        }
    
    INGREDIENTS[name_lower] = {
        'color': tuple(color),
        'cuts_needed': int(cuts_needed),
        'char': char,
        'state_chars': state_chars
    }

# Dish character representation
DISH_CHAR = 'p'  # Empty plate character

# Tile definitions: char -> {code, name, color}
TILES = {
    ' ': {'code': 0, 'name': "Empty", 'color': (245, 235, 224)},
    '#': {'code': 1, 'name': "Counter", 'color': (115, 87, 72)},
    'I': {'code': 2, 'name': "Ingredient Dispenser", 'color': (143, 111, 95)},
    'C': {'code': 3, 'name': "Chopping Board", 'color': (196, 160, 137)},
    'F': {'code': 4, 'name': "Fryer", 'color': (130, 110, 95)},  # Not used yet
    'P': {'code': 5, 'name': "Plate Dispenser", 'color': (166, 133, 114)},
    'D': {'code': 6, 'name': "Delivery", 'color': (140, 165, 140)},
    'T': {'code': 7, 'name': "Trash", 'color': (69, 53, 44)},
    '|': {'code': 8, 'name': "Wall", 'color': (69, 53, 44)},
    '_': {'code': 9, 'name': "Wall", 'color': (69, 53, 44)},
}

class ItemState:
    """Constants for item processing states."""
    RAW = "RAW"
    CHOPPING_1 = "CHOPPING_1"
    CHOPPING_2 = "CHOPPING_2"
    CHOPPING_3 = "CHOPPING_3"
    READY = "READY"
    EMPTY = "EMPTY"
    BUILDING = "BUILDING"
    COMPLETE = "COMPLETE"


class ItemType:
    """Constants for item types."""
    INGREDIENT = "ingredient"
    DISH = "dish"
    RECIPE = "recipe"


def create_ingredient(name, cuts_needed=DEFAULT_CUTS_NEEDED):
    """
    Creates a basic ingredient that needs processing.
    
    Args:
        name: Name of the ingredient (e.g., "tomato", "lettuce")
        cuts_needed: Number of cuts required to process (default: DEFAULT_CUTS_NEEDED)
    
    Returns:
        Dictionary representing the ingredient
    """
    # Get ingredient data
    ingredient_data = INGREDIENTS.get(name.lower())
    if ingredient_data:
        base_color = ingredient_data['color']
        cuts_needed = ingredient_data['cuts_needed']
    else:
        base_color = DEFAULT_COLOR_UNKNOWN
    
    return {
        "name": name,
        "type": ItemType.INGREDIENT,
        "state": ItemState.RAW,
        "cuts_needed": cuts_needed,
        "cuts_done": 0,
        "combinable": False,
        "base_color": base_color  # Store base color in item
    }


def create_plate():
    """
    Creates an empty plate.
    
    Returns:
        Dictionary representing an empty plate
    """
    return {
        "name": "plate",
        "type": ItemType.DISH,
        "state": ItemState.EMPTY,
        "ingredients": [],
        "color": (240, 240, 240)  # Light gray for plate
    }


def advance_processing(item):
    """
    Advances the processing state of an item by one step.
    
    Args:
        item: Item dictionary to process
    
    Returns:
        Updated item dictionary
    """
    if item["type"] != ItemType.INGREDIENT:
        return item
    
    item["cuts_done"] += 1
    cuts_done = item["cuts_done"]
    cuts_needed = item["cuts_needed"]
    
    # Update state based on progress
    if cuts_done >= cuts_needed:
        item["state"] = ItemState.READY
        item["combinable"] = True
    elif cuts_done == CHOPPING_STATE_1_CUTS:
        item["state"] = ItemState.CHOPPING_1
    elif cuts_done == CHOPPING_STATE_2_CUTS:
        item["state"] = ItemState.CHOPPING_2
    elif cuts_done == CHOPPING_STATE_3_CUTS:
        item["state"] = ItemState.CHOPPING_3
    
    return item


def is_ready(item):
    """
    Checks if item is ready to be combined.
    
    Args:
        item: Item dictionary to check
    
    Returns:
        True if item is in READY state
    """
    return item.get("state") == ItemState.READY


def can_combine(item):
    """
    Checks if item can be combined into a plate.
    
    Args:
        item: Item dictionary to check
    
    Returns:
        True if item can be combined
    """
    return item.get("combinable", False)


def is_processing(item):
    """
    Checks if item is currently being processed.
    
    Args:
        item: Item dictionary to check
    
    Returns:
        True if item is in a CHOPPING state
    """
    state = item.get("state", "")
    return state in [ItemState.CHOPPING_1, ItemState.CHOPPING_2, ItemState.CHOPPING_3]


def get_item_char(item_dict):
    """
    Get ASCII character representation for an item based on its type and state.
    
    Args:
        item_dict: Item dictionary
    
    Returns:
        Single character representing the item
    """
    if not item_dict:
        return ' '
    
    item_type = item_dict.get("type")
    
    if item_type == ItemType.INGREDIENT:
        name = item_dict["name"].lower()
        state = item_dict.get("state", ItemState.RAW)
        
        # Get character from state_chars mapping with safe fallback
        if name in INGREDIENTS:
            state_chars = INGREDIENTS[name]['state_chars']
            return state_chars.get(state, INGREDIENTS[name]['char'])
        else:
            return name[0] if name else 'x'
        
    elif item_type == ItemType.DISH:
        ingredients = item_dict.get("ingredients", [])
        if not ingredients:
            return DISH_CHAR  # empty plate
        else:
            # Show character of first ingredient
            first_ingredient = ingredients[FIRST_INGREDIENT_INDEX].lower()
            if first_ingredient in INGREDIENTS:
                return INGREDIENTS[first_ingredient]['char']
            else:
                return first_ingredient[0] if first_ingredient else 'x'
    
    return '?'


def can_process(item):
    """
    Checks if item can be processed further.
    
    Args:
        item: Item dictionary to check
    
    Returns:
        True if item is RAW or in a CHOPPING state
    """
    if item["type"] != ItemType.INGREDIENT:
        return False
    
    state = item.get("state", "")
    return state == ItemState.RAW or is_processing(item)


def add_ingredient_to_plate(plate, ingredient_name):
    """
    Adds an ingredient to a plate.
    
    Args:
        plate: Plate dictionary
        ingredient_name: Name of the ingredient to add
    
    Returns:
        Updated plate dictionary
    """
    if plate["type"] != ItemType.DISH:
        return plate
    
    if "ingredients" not in plate:
        plate["ingredients"] = []
    
    plate["ingredients"].append(ingredient_name)
    
    # Update state
    if plate["state"] == ItemState.EMPTY:
        plate["state"] = ItemState.BUILDING
    
    return plate


def get_ingredient_display_name(item):
    """
    Gets a display-friendly name for an item including its state.
    
    Args:
        item: Item dictionary
    
    Returns:
        String representation for display
    """
    if item["type"] == ItemType.INGREDIENT:
        name = item["name"].capitalize()
        state = item["state"]
        
        if state == ItemState.RAW:
            return f"{name} (raw)"
        elif state == ItemState.READY:
            return f"{name} (ready)"
        elif is_processing(item):
            cuts = item["cuts_done"]
            total = item["cuts_needed"]
            return f"{name} ({cuts}/{total})"
    
    elif item["type"] == ItemType.DISH:
        if item["state"] == ItemState.EMPTY:
            return "Empty Plate"
        else:
            ingredients = ", ".join(item["ingredients"])
            return f"Plate ({ingredients})"
    
    return str(item)


def get_item_color(item):
    """
    Gets the RGB color for an item based on its type and state.
    
    Args:
        item: Item dictionary
    
    Returns:
        RGB color tuple (r, g, b)
    """
    if not item:
        return DEFAULT_COLOR_EMPTY
    
    item_type = item.get("type")
    
    if item_type == ItemType.INGREDIENT:
        state = item.get("state", ItemState.RAW)
        base_color = item.get("base_color", DEFAULT_COLOR_UNKNOWN)
        
        # RAW ingredients are darker
        if state == ItemState.RAW:
            return tuple(int(c * BRIGHTNESS_RAW) for c in base_color)
        # READY ingredients use full base color
        elif state == ItemState.READY:
            return tuple(int(c * BRIGHTNESS_READY) for c in base_color)
        # CHOPPING states are intermediate brightness
        else:  # CHOPPING_1, CHOPPING_2, CHOPPING_3
            return tuple(int(c * BRIGHTNESS_CHOPPING) for c in base_color)
    
    elif item_type == ItemType.DISH:
        return item.get("color", DEFAULT_COLOR_PLATE)
    
    return DEFAULT_COLOR_UNKNOWN



# Legacy support - map old item codes to new system
class ItemManager:
    """Manager for handling item codes and cell codes (for map compatibility)."""
    
    def __init__(self):
        # Build CHAR_TO_NUM from TILES
        self.CHAR_TO_NUM = {char: tile['code'] for char, tile in TILES.items()}
        
        # Build reverse mapping (NUM_TO_CHAR) for ASCII rendering
        self.NUM_TO_CHAR = {tile['code']: char for char, tile in TILES.items()}
        
        # Store tiles reference
        self.TILES = TILES
        
        # Build ingredient creators from INGREDIENTS dictionary
        self.INGREDIENT_CREATORS = {
            name: lambda n=name: create_ingredient(n)
            for name in INGREDIENTS.keys()
        }
        
        # Store ingredients reference
        self.INGREDIENTS = INGREDIENTS
        
    def rebuild_ingredient_creators(self):
        """Rebuild the ingredient creators dictionary from global INGREDIENTS."""
        self.INGREDIENT_CREATORS = {
            name: lambda n=name: create_ingredient(n)
            for name in INGREDIENTS.keys()
        }
    
    def get_cell_code(self, char: str) -> int:
        """Get numeric code for a map cell character."""
        return self.CHAR_TO_NUM.get(char, TILES[' ']['code'])
    
    def get_cell_char(self, code: int) -> str:
        """Get character for a numeric cell code."""
        return self.NUM_TO_CHAR.get(code, '?')
    
    def get_tile_color(self, char: str) -> tuple:
        """Get RGB color for a tile character."""
        tile = TILES.get(char)
        return tile['color'] if tile else DEFAULT_COLOR_UNKNOWN
    
    def get_tile_name(self, char: str) -> str:
        """Get readable name for a tile character."""
        tile = TILES.get(char)
        return tile['name'] if tile else "Unknown"
    
    def create_ingredient_by_name(self, name: str):
        """Create an ingredient by name."""
        creator = self.INGREDIENT_CREATORS.get(name.lower())
        if creator:
            return creator()
        return create_ingredient(name, cuts_needed=DEFAULT_CUTS_NEEDED)
