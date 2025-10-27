"""
Item system using dictionaries with metadata for flexible recipe management.
"""

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
    return {
        "name": name,
        "type": ItemType.INGREDIENT,
        "state": ItemState.RAW,
        "cuts_needed": cuts_needed,
        "cuts_done": 0,
        "combinable": False
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
        "ingredients": []
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


# Legacy support - map old item codes to new system
class ItemManager:
    """Manager for handling item codes and cell codes (for map compatibility)."""
    
    def __init__(self):
        self.CHAR_TO_NUM = {
            ' ': 0,
            '#': 1,
            'I': 2,
            'C': 3,
            'F': 4,
            'P': 5,
            'D': 6,
            'T': 7,
            '|': 8,  # Wall (vertical borders)
            '_': 9,  # Wall (horizontal borders)
        }
        
        # Map ingredient names to create functions
        self.INGREDIENT_CREATORS = {
            "tomato": lambda: create_ingredient("tomato", cuts_needed=3),
            "lettuce": lambda: create_ingredient("lettuce", cuts_needed=3),
            "cucumber": lambda: create_ingredient("cucumber", cuts_needed=3),
        }
    
    def get_cell_code(self, char: str) -> int:
        """Get numeric code for a map cell character."""
        return self.CHAR_TO_NUM.get(char, 0)
    
    def create_ingredient_by_name(self, name: str):
        """Create an ingredient by name."""
        creator = self.INGREDIENT_CREATORS.get(name.lower())
        if creator:
            return creator()
        return create_ingredient(name, cuts_needed=3)
