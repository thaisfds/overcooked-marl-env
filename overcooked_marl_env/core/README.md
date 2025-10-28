# Core System - Items and Recipes

The `core` module is the foundation of the game system, defining all items, recipes, constants, and item instance lifecycle management.

## Files

```
core/
├── __init__.py                 # Module exports
├── constants.py                # Game constants
├── items.py                    # Item types and utilities
├── recipes.py                  # Available recipes
└── item_instance_manager.py    # Item instance management
```

## Responsibilities

- Define all item types (ingredients, plate, processed items)
- Manage item states (RAW, CHOPPED_1, CHOPPED_2, CHOPPED_3)
- Define available recipes and their requirements
- Create, track, and destroy item instances
- Validate recipe completion

## Detailed Files

### `constants.py`

Defines all game constants.

#### ItemType (Enum)

```python
class ItemType(IntEnum):
    EMPTY = 0
    LETTUCE = 1
    TOMATO = 2
    CUCUMBER = 3
    PLATE = 4
    CHOPPED_LETTUCE = 5
    CHOPPED_TOMATO = 6
    CHOPPED_CUCUMBER = 7
    SIMPLE_SALAD = 8
    GARDEN_SALAD = 9
    FRESH_SALAD = 10
```

**Usage**: Identify each type of item in the game.

#### ItemState (Enum)

```python
class ItemState(IntEnum):
    RAW = 0          # Not processed
    CHOPPED_1 = 1    # 1 cut
    CHOPPED_2 = 2    # 2 cuts
    CHOPPED_3 = 3    # 3 cuts (fully processed)
```

**Usage**: Track item processing progress.

#### TileType (Enum)

```python
class TileType(IntEnum):
    EMPTY = 0
    WALL = 1
    COUNTER = 2
    CUTTING_BOARD = 3
    INGREDIENT_DISPENSER = 4
    PLATE_DISPENSER = 5
    DELIVERY_STATION = 6
```

**Usage**: Identify map tiles.

### `items.py`

Manages item information and utilities.

#### ItemInfo (NamedTuple)

```python
@dataclass
class ItemInfo:
    name: str          # Display name
    symbol: str        # ASCII symbol
    color: Tuple[int, int, int]  # RGB color
```

**Usage**: Store visual and display information for each item type.

#### ITEM_INFO Dictionary

Maps each `ItemType` to its `ItemInfo`:

```python
ITEM_INFO = {
    ItemType.LETTUCE: ItemInfo("Lettuce", "L", (0, 255, 0)),
    ItemType.TOMATO: ItemInfo("Tomato", "T", (255, 0, 0)),
    ItemType.CUCUMBER: ItemInfo("Cucumber", "C", (0, 128, 0)),
    ItemType.PLATE: ItemInfo("Plate", "P", (200, 200, 200)),
    # ...
}
```

**Access**:
```python
info = ITEM_INFO[ItemType.LETTUCE]
print(info.name)  # "Lettuce"
print(info.symbol)  # "L"
```

#### Utility Functions

**`get_item_name(item_type: ItemType) -> str`**
```python
name = get_item_name(ItemType.LETTUCE)  # "Lettuce"
```

**`get_item_symbol(item_type: ItemType) -> str`**
```python
symbol = get_item_symbol(ItemType.TOMATO)  # "T"
```

**`get_item_color(item_type: ItemType) -> Tuple[int, int, int]`**
```python
color = get_item_color(ItemType.CUCUMBER)  # (0, 128, 0)
```

**`get_chopped_version(item_type: ItemType) -> ItemType`**
```python
chopped = get_chopped_version(ItemType.LETTUCE)  # ItemType.CHOPPED_LETTUCE
```

**`is_ingredient(item_type: ItemType) -> bool`**
```python
is_ingredient(ItemType.LETTUCE)  # True
is_ingredient(ItemType.PLATE)    # False
```

**`is_chopped(item_type: ItemType) -> bool`**
```python
is_chopped(ItemType.CHOPPED_LETTUCE)  # True
is_chopped(ItemType.LETTUCE)          # False
```

### `recipes.py`

Defines all available recipes.

#### Recipe Class

```python
@dataclass
class Recipe:
    name: str
    ingredients: Set[ItemType]
    reward: float
```

**Attributes**:
- `name`: Recipe identifier
- `ingredients`: Set of required ItemTypes
- `reward`: Points awarded on delivery

#### Available Recipes

```python
SIMPLE_SALAD = Recipe(
    name="SIMPLE_SALAD",
    ingredients={ItemType.CHOPPED_LETTUCE},
    reward=10.0
)

GARDEN_SALAD = Recipe(
    name="GARDEN_SALAD",
    ingredients={ItemType.CHOPPED_LETTUCE, ItemType.CHOPPED_TOMATO},
    reward=15.0
)

FRESH_SALAD = Recipe(
    name="FRESH_SALAD",
    ingredients={
        ItemType.CHOPPED_LETTUCE,
        ItemType.CHOPPED_TOMATO,
        ItemType.CHOPPED_CUCUMBER
    },
    reward=20.0
)
```

#### Utility Functions

**`get_recipe_for_ingredients(ingredients: Set[ItemType]) -> Optional[Recipe]`**

Checks if ingredients match a recipe:

```python
ingredients = {ItemType.CHOPPED_LETTUCE, ItemType.CHOPPED_TOMATO}
recipe = get_recipe_for_ingredients(ingredients)
# Returns GARDEN_SALAD (15.0 reward)

invalid_ingredients = {ItemType.CHOPPED_LETTUCE, ItemType.CUCUMBER}
recipe = get_recipe_for_ingredients(invalid_ingredients)
# Returns None (no matching recipe)
```

**`is_valid_recipe(ingredients: Set[ItemType]) -> bool`**

Validates if ingredients form a valid recipe:

```python
is_valid_recipe({ItemType.CHOPPED_LETTUCE})  # True (SIMPLE_SALAD)
is_valid_recipe({ItemType.LETTUCE})           # False (not chopped)
```

### `item_instance_manager.py`

Manages item instance lifecycle.

#### ItemInstanceManager Class

```python
class ItemInstanceManager:
    def __init__(self):
        self.items: Dict[int, ItemInstance] = {}
        self.next_id: int = 1
```

**Responsibilities**:
- Assign unique IDs to items
- Track all active items
- Store item properties (type, state, contents)
- Clean up destroyed items

#### ItemInstance

```python
@dataclass
class ItemInstance:
    id: int
    item_type: ItemType
    state: ItemState = ItemState.RAW
    contents: List[ItemType] = field(default_factory=list)
```

**Attributes**:
- `id`: Unique identifier
- `item_type`: What the item is
- `state`: Processing progress
- `contents`: Items inside (for plates)

#### Methods

**`create_item(item_type: ItemType) -> int`**

Creates new item and returns its ID:

```python
manager = ItemInstanceManager()
item_id = manager.create_item(ItemType.LETTUCE)
# Returns 1 (first item)
```

**`get_item(item_id: int) -> Optional[ItemInstance]`**

Retrieves item by ID:

```python
item = manager.get_item(item_id)
print(item.item_type)  # ItemType.LETTUCE
print(item.state)      # ItemState.RAW
```

**`update_item_state(item_id: int, new_state: ItemState)`**

Updates item processing state:

```python
manager.update_item_state(item_id, ItemState.CHOPPED_1)
# Item now has 1 cut
```

**`add_to_plate(plate_id: int, ingredient_type: ItemType)`**

Adds ingredient to plate:

```python
plate_id = manager.create_item(ItemType.PLATE)
manager.add_to_plate(plate_id, ItemType.CHOPPED_LETTUCE)
manager.add_to_plate(plate_id, ItemType.CHOPPED_TOMATO)

plate = manager.get_item(plate_id)
print(plate.contents)  
# [ItemType.CHOPPED_LETTUCE, ItemType.CHOPPED_TOMATO]
```

**`get_plate_contents(plate_id: int) -> List[ItemType]`**

Gets plate contents:

```python
contents = manager.get_plate_contents(plate_id)
# [ItemType.CHOPPED_LETTUCE, ItemType.CHOPPED_TOMATO]
```

**`destroy_item(item_id: int)`**

Removes item from tracking:

```python
manager.destroy_item(item_id)
# Item no longer exists
```

## Complete Workflow

### Process Ingredient

```python
# 1. Create raw lettuce
item_id = manager.create_item(ItemType.LETTUCE)

# 2. Place on cutting board (position-based in GameState)

# 3. Cut 3 times
manager.update_item_state(item_id, ItemState.CHOPPED_1)  # 1st cut
manager.update_item_state(item_id, ItemState.CHOPPED_2)  # 2nd cut
manager.update_item_state(item_id, ItemState.CHOPPED_3)  # 3rd cut - done

# 4. Transform to chopped version
chopped_type = get_chopped_version(ItemType.LETTUCE)
# Returns ItemType.CHOPPED_LETTUCE
```

### Prepare Recipe

```python
# 1. Create plate
plate_id = manager.create_item(ItemType.PLATE)

# 2. Add processed ingredients
manager.add_to_plate(plate_id, ItemType.CHOPPED_LETTUCE)
manager.add_to_plate(plate_id, ItemType.CHOPPED_TOMATO)

# 3. Validate recipe
contents = manager.get_plate_contents(plate_id)
ingredients_set = set(contents)

recipe = get_recipe_for_ingredients(ingredients_set)
if recipe:
    print(f"Recipe complete: {recipe.name}")
    print(f"Reward: {recipe.reward}")
    # GARDEN_SALAD, reward 15.0

# 4. Deliver at delivery station (handled by InteractionHandler)
```

## Tests

```bash
# Test item system
python -m pytest overcooked_marl_env/tests/unit/test_item_instance_manager.py

# Check all core utilities
python -m pytest overcooked_marl_env/tests/unit/ -k "item or recipe"
```

## See Also

- [Environment Documentation](../environment/README.md) - How items are used in game state
- [Mechanics Documentation](../mechanics/README.md) - How items are manipulated
- [Main Package Documentation](../README.md) - Complete architecture
