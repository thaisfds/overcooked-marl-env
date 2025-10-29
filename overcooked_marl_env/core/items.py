"""
Item system using dictionaries with metadata for flexible recipe management.
"""

# Ingredient definitions: name -> {color, cuts_needed}
INGREDIENTS = {
    'tomato': {'color': (255, 0, 0), 'cuts_needed': 3},      # Tomato red
    'lettuce': {'color': (107,142,35), 'cuts_needed': 3},   # Light green
    'cucumber': {'color': (0,100,0), 'cuts_needed': 3},   # Medium sea green
}

# Tile definitions: char -> {code, name, color}
TILES = {
    ' ': {'code': 0, 'name': "Empty", 'color': (255, 255, 255)},
    '#': {'code': 1, 'name': "Counter", 'color': (180, 180, 180)},
    'I': {'code': 2, 'name': "Ingredient Dispenser", 'color': (210, 180, 140)},
    'C': {'code': 3, 'name': "Chopping Board", 'color': (210, 180, 140)},
    'F': {'code': 4, 'name': "Fryer", 'color': (200, 200, 200)},  # Not used yet
    'P': {'code': 5, 'name': "Plate Dispenser", 'color': (100, 149, 237)},
    'D': {'code': 6, 'name': "Delivery", 'color': (50, 205, 50)},
    'T': {'code': 7, 'name': "Trash", 'color': (105, 105, 105)},
    '|': {'code': 8, 'name': "Wall", 'color': (70, 70, 70)},
    '_': {'code': 9, 'name': "Wall", 'color': (70, 70, 70)},
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


def create_ingredient(name, cuts_needed=3):
    """
    Creates a basic ingredient that needs processing.
    
    Args:
        name: Name of the ingredient (e.g., "tomato", "lettuce")
        cuts_needed: Number of cuts required to process (default: 3)
    
    Returns:
        Dictionary representing the ingredient
    """
    # Get ingredient data
    ingredient_data = INGREDIENTS.get(name.lower())
    if ingredient_data:
        base_color = ingredient_data['color']
        cuts_needed = ingredient_data['cuts_needed']
    else:
        base_color = (200, 200, 200)  # Default gray for unknown ingredients
    
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
    elif cuts_done == 1:
        item["state"] = ItemState.CHOPPING_1
    elif cuts_done == 2:
        item["state"] = ItemState.CHOPPING_2
    elif cuts_done == 3:
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
        return (255, 255, 255)
    
    item_type = item.get("type")
    
    if item_type == ItemType.INGREDIENT:
        state = item.get("state", ItemState.RAW)
        base_color = item.get("base_color", (200, 200, 200))
        
        # RAW ingredients are darker (60% brightness)
        if state == ItemState.RAW:
            return tuple(int(c * 0.6) for c in base_color)
        # READY ingredients use full base color
        elif state == ItemState.READY:
            return base_color
        # CHOPPING states are intermediate brightness (80%)
        else:  # CHOPPING_1, CHOPPING_2, CHOPPING_3
            return tuple(int(c * 0.8) for c in base_color)
    
    elif item_type == ItemType.DISH:
        return item.get("color", (240, 240, 240))
    
    return (200, 200, 200)



# Legacy support - map old item codes to new system
class ItemManager:
    """Manager for handling item codes and cell codes (for map compatibility)."""
    
    def __init__(self):
        # Build CHAR_TO_NUM from TILES
        self.CHAR_TO_NUM = {char: tile['code'] for char, tile in TILES.items()}
        
        # Store tiles reference
        self.TILES = TILES
        
        # Build ingredient creators from INGREDIENTS dictionary
        self.INGREDIENT_CREATORS = {
            name: lambda n=name: create_ingredient(n)
            for name in INGREDIENTS.keys()
        }
        
        # Store ingredients reference
        self.INGREDIENTS = INGREDIENTS
    
    def get_cell_code(self, char: str) -> int:
        """Get numeric code for a map cell character."""
        return self.CHAR_TO_NUM.get(char, 0)
    
    def get_tile_color(self, char: str) -> tuple:
        """Get RGB color for a tile character."""
        tile = TILES.get(char)
        return tile['color'] if tile else (200, 200, 200)
    
    def get_tile_name(self, char: str) -> str:
        """Get readable name for a tile character."""
        tile = TILES.get(char)
        return tile['name'] if tile else "Unknown"
    
    def create_ingredient_by_name(self, name: str):
        """Create an ingredient by name."""
        creator = self.INGREDIENT_CREATORS.get(name.lower())
        if creator:
            return creator()
        return create_ingredient(name, cuts_needed=3)
