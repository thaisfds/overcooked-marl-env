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
import os
import json
from typing import Dict, Optional, Any, List, Tuple

# =================================================================
# INGREDIENT REGISTRY & DEFINITIONS
# =================================================================

class IngredientRegistry:
    """
    Registry managing ingredient definitions and states.
    Loads defaults from default_ingredients.json and supports dynamic JSON extension.
    """

    def __init__(self):
        self._ingredients: Dict[str, Dict] = {}
        self.load_defaults()

    @property
    def ingredients(self) -> Dict[str, Dict]:
        return self._ingredients

    def load_defaults(self):
        """Loads default base ingredients strictly from default_ingredients.json."""
        json_path = os.path.join(os.path.dirname(__file__), "default_ingredients.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        self.load_from_json(json_path)

    def load_from_json(self, file_path: str):
        """Loads and registers ingredients from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("ingredients", data) if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "name" in item:
                        self.register(
                            name=item["name"],
                            color=item.get("color"),
                            cuts_needed=item.get("cuts_needed", DEFAULT_CUTS_NEEDED),
                            char=item.get("char"),
                            state_chars=item.get("state_chars"),
                            stages=item.get("stages")
                        )
        except Exception:
            pass

    def register(
        self,
        name: str,
        color=None,
        cuts_needed=DEFAULT_CUTS_NEEDED,
        char=None,
        state_chars=None,
        stages=None
    ) -> Dict:
        """Register or update an ingredient in the registry."""
        name_lower = name.lower()
        if color is None:
            color = DEFAULT_COLOR_UNKNOWN

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

        # Normalize stages
        if stages is None:
            if cuts_needed > 0:
                stages = [{"station": "C", "action": "chop", "steps": int(cuts_needed)}]
            else:
                stages = []
        else:
            total_steps = sum(s.get("steps", 1) for s in stages)
            if cuts_needed == DEFAULT_CUTS_NEEDED and total_steps > 0:
                cuts_needed = total_steps

        entry = {
            'color': tuple(color),
            'cuts_needed': int(cuts_needed),
            'char': char,
            'state_chars': state_chars,
            'stages': [s.copy() for s in stages]
        }
        self._ingredients[name_lower] = entry
        return entry

    def get(self, name: str, default=None) -> Optional[Dict]:
        return self._ingredients.get(name.lower(), default)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._ingredients

    def __getitem__(self, name: str) -> Dict:
        return self._ingredients[name.lower()]

    def __setitem__(self, name: str, value: Dict):
        self._ingredients[name.lower()] = value


# Singleton instance and backward-compatible global dictionary alias
ingredient_registry = IngredientRegistry()
INGREDIENTS = ingredient_registry.ingredients


def register_custom_ingredient(name, color=None, cuts_needed=3, char=None, state_chars=None, stages=None):
    """
    Register a custom ingredient dynamically.
    Delegates to the centralized ingredient_registry.
    """
    return ingredient_registry.register(
        name=name, color=color, cuts_needed=cuts_needed,
        char=char, state_chars=state_chars, stages=stages
    )

# Dish character representation
DISH_CHAR = 'p'  # Empty plate character

# Tile definitions: char -> {code, name, color}
TILES = {
    ' ': {'code': 0, 'name': "Empty", 'color': (245, 235, 224)},
    '#': {'code': 1, 'name': "Counter", 'color': (115, 87, 72)},
    'I': {'code': 2, 'name': "Ingredient Dispenser", 'color': (143, 111, 95)},
    'C': {'code': 3, 'name': "Chopping Board", 'color': (196, 160, 137)},
    'F': {'code': 4, 'name': "Fryer", 'color': (130, 110, 95)},
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


def create_ingredient(name, cuts_needed=DEFAULT_CUTS_NEEDED, stages=None):
    """
    Creates an ingredient that needs processing (or instant if cuts_needed=0).
    
    Args:
        name: Name of the ingredient (e.g., "tomato", "lettuce", "meat", "bun")
        cuts_needed: Number of cuts required to process (default: DEFAULT_CUTS_NEEDED)
        stages: Optional list of multi-stage processing steps, e.g.
                [{"station": "C", "action": "chop", "steps": 3}, {"station": "S", "action": "cook", "steps": 20}]
    
    Returns:
        Dictionary representing the ingredient
    """
    name_lower = name.lower()
    ingredient_data = INGREDIENTS.get(name_lower)
    if ingredient_data:
        base_color = ingredient_data.get('color', DEFAULT_COLOR_UNKNOWN)
        cuts_needed = ingredient_data.get('cuts_needed', cuts_needed)
        if stages is None and "stages" in ingredient_data:
            stages = [s.copy() for s in ingredient_data["stages"]]
    else:
        base_color = DEFAULT_COLOR_UNKNOWN
    
    if stages is None:
        if cuts_needed > 0:
            stages = [{"station": "C", "action": "chop", "steps": int(cuts_needed)}]
        else:
            stages = []

    is_instant = (cuts_needed == 0 and len(stages) == 0)
    initial_state = ItemState.READY if is_instant else ItemState.RAW

    return {
        "name": name,
        "type": ItemType.INGREDIENT,
        "state": initial_state,
        "cuts_needed": cuts_needed,
        "cuts_done": 0,
        "cuts_completed": 0,
        "stages": [s.copy() for s in stages],
        "current_stage_idx": 0,
        "combinable": is_instant,
        "base_color": base_color,
        "shaping_reward_accumulated": 0.0,
        "contributors": [],
        "last_placed_by": None
    }


def create_plate(ingredients=None):
    """
    Creates an empty plate (or a plate with initial ingredients).
    
    Args:
        ingredients: Optional list of ingredient names to initialize on the plate
        
    Returns:
        Dictionary representing a plate
    """
    ing_list = list(ingredients) if ingredients else []
    return {
        "name": "plate",
        "type": ItemType.DISH,
        "state": ItemState.EMPTY if not ing_list else ItemState.BUILDING,
        "ingredients": ing_list,
        "color": (240, 240, 240),  # Light gray for plate
        "shaping_reward_accumulated": 0.0,
        "contributors": [],
        "last_placed_by": None
    }


def advance_processing(item, station="C"):
    """
    Advances the processing state of an item by one step.
    
    Args:
        item: Item dictionary to process
        station: Station code/char (default: "C")
    
    Returns:
        Updated item dictionary
    """
    if not item or item.get("type") != ItemType.INGREDIENT:
        return item
    
    item["cuts_done"] = item.get("cuts_done", 0) + 1
    item["cuts_completed"] = item["cuts_done"]
    cuts_done = item["cuts_done"]
    cuts_needed = item.get("cuts_needed", DEFAULT_CUTS_NEEDED)

    stages = item.get("stages")
    if not stages:
        stages = [{"station": station, "action": "chop", "steps": cuts_needed}]
        item["stages"] = stages
        item["current_stage_idx"] = 0

    stage_idx = item.get("current_stage_idx", 0)
    if stage_idx < len(stages):
        current_stage = stages[stage_idx]
        current_stage["progress"] = current_stage.get("progress", 0) + 1
        if current_stage["progress"] >= current_stage.get("steps", 1):
            # Completed current stage
            item["current_stage_idx"] = stage_idx + 1
            if item["current_stage_idx"] >= len(stages):
                item["state"] = ItemState.READY
                item["combinable"] = True
            else:
                next_stage = stages[item["current_stage_idx"]]
                item["state"] = f"STAGE_{item['current_stage_idx']}_{next_stage.get('action', 'proc').upper()}"
        else:
            if station == "C":
                if cuts_done == CHOPPING_STATE_1_CUTS:
                    item["state"] = ItemState.CHOPPING_1
                elif cuts_done == CHOPPING_STATE_2_CUTS:
                    item["state"] = ItemState.CHOPPING_2
                elif cuts_done == CHOPPING_STATE_3_CUTS:
                    item["state"] = ItemState.CHOPPING_3
    elif cuts_done >= cuts_needed:
        item["state"] = ItemState.READY
        item["combinable"] = True

    return item


def is_ready(item):
    """
    Checks if item is ready to be combined.
    
    Args:
        item: Item dictionary to check
    
    Returns:
        True if item is in READY state
    """
    if not item or not isinstance(item, dict):
        return False
    return item.get("state") == ItemState.READY


def is_plate(item):
    """
    Checks if item is a plate/dish.
    
    Args:
        item: Item dictionary to check
        
    Returns:
        True if item is of type DISH or represents a plate.
    """
    if not item or not isinstance(item, dict):
        return False
    return item.get("type") == ItemType.DISH or "plate" in item.get("name", "").lower()


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


def can_process(item, station="C"):
    """
    Checks if item can be processed further at a given station.
    
    Args:
        item: Item dictionary to check
        station: Station code/char (default: "C")
    
    Returns:
        True if item can be processed at this station
    """
    if not item or item.get("type") != ItemType.INGREDIENT:
        return False
    
    if is_ready(item):
        return False

    stages = item.get("stages")
    if not stages:
        stages = [{"station": "C", "action": "chop", "steps": item.get("cuts_needed", DEFAULT_CUTS_NEEDED)}]
        item["stages"] = stages
        item["current_stage_idx"] = 0

    stage_idx = item.get("current_stage_idx", 0)
    if stage_idx < len(stages):
        target_station = stages[stage_idx].get("station", "C")
        return target_station == station
    return False


def add_ingredient_to_plate(plate, ingredient_name, ingredient_item=None):
    """
    Adds an ingredient to a plate.
    
    Args:
        plate: Plate dictionary
        ingredient_name: Name of the ingredient to add
        ingredient_item: Optional ingredient dictionary to transfer accumulated shaping reward
    
    Returns:
        Updated plate dictionary
    """
    if plate["type"] != ItemType.DISH:
        return plate
    
    if "ingredients" not in plate:
        plate["ingredients"] = []
    
    # Do not allow duplicate ingredients on the same plate
    if ingredient_name in plate["ingredients"]:
        return plate

    plate["ingredients"].append(ingredient_name)

    
    if "shaping_reward_accumulated" not in plate:
        plate["shaping_reward_accumulated"] = 0.0

    if "contributors" not in plate:
        plate["contributors"] = []
        
    if ingredient_item and isinstance(ingredient_item, dict):
        plate["shaping_reward_accumulated"] += ingredient_item.get("shaping_reward_accumulated", 0.0)
        for c in ingredient_item.get("contributors", []):
            if c not in plate["contributors"]:
                plate["contributors"].append(c)
    
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



# Central Item and Tile Manager
class ItemManager:
    """Manager for handling item codes, dynamic tiles, and cell codes."""
    
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
        if char in self.CHAR_TO_NUM:
            return self.CHAR_TO_NUM[char]
        if char in TILES:
            return TILES[char]['code']
        return TILES[' ']['code']
    
    def get_cell_char(self, code: int) -> str:
        """Get character for a numeric cell code."""
        if code in self.NUM_TO_CHAR:
            return self.NUM_TO_CHAR[code]
        for char, tile in TILES.items():
            if tile.get('code') == code:
                return char
        return '?'
    
    def get_tile_color(self, char: str) -> tuple:
        """Get RGB color for a tile character."""
        tile = TILES.get(char)
        return tile['color'] if tile else DEFAULT_COLOR_UNKNOWN
    
    def get_tile_name(self, char: str) -> str:
        """Get readable name for a tile character."""
        tile = TILES.get(char)
        return tile['name'] if tile else "Unknown"
    
    def register_tile(self, char: str, code: int, name: str, color: tuple):
        """Registers a new tile dynamically and updates cell code/char mappings."""
        register_tile(char, code, name, color)
        self.CHAR_TO_NUM[char] = int(code)
        self.NUM_TO_CHAR[int(code)] = char

    def create_ingredient_by_name(self, name: str):
        """Create an ingredient by name."""
        creator = self.INGREDIENT_CREATORS.get(name.lower())
        if creator:
            return creator()
        return create_ingredient(name, cuts_needed=DEFAULT_CUTS_NEEDED)


def register_tile(char: str, code: int, name: str, color: tuple):
    """
    Registers a new tile type dynamically in TILES.
    
    Args:
        char: Single character on the map grid (e.g. 'S', 'W')
        code: Unique numeric code for observation maps
        name: Human readable station name (e.g. "Stove")
        color: RGB tuple for Pygame rendering (e.g. (255, 100, 50))
    """
    TILES[char] = {'code': int(code), 'name': str(name), 'color': tuple(color)}

