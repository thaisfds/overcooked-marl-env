# Core System - Items, Recipes, and Constants

The `core` module forms the domain model of the Overcooked MARL environment, defining item states, item types, recipe matches, game constants, and the lifecycle of cooking assets.

## Sub-package Contents

```
core/
├── __init__.py                 # Module exports
├── constants.py                # Action, observation, and reward constants
├── items.py                    # Item creation, registration, and properties
├── recipes.py                  # Recipes verification and config loaders
└── item_instance_manager.py    # Item unique ID tracking and instance operations
```

---

## Detailed Components

### 1. Constants (`constants.py`)
Centralizes key environment parameters:
* **Observation Channels:** Defines `TOTAL_CHANNELS = 9` (6 base channels + 3 processing state channels).
* **Actions Mapping:** Defines action directions (`0: Up`, `1: Down`, `2: Left`, `3: Right`, `4: Interact`).
* **Game Parameters:** Defines episode limits (`MAX_EPISODE_STEPS = 800`) and cuts required (`DEFAULT_CUTS_NEEDED = 3`).
* **Rewards:** Delivery rewards (`+20.0`), sequential delivery multiplier (`1.5x`), and invalid interaction penalty (`-2.0`).

### 2. Items & Ingredients (`items.py`)
Rather than rigid subclassing, items are represented by dynamic metadata dictionaries containing properties like type, state, and list of contents.

#### Item Types and States
* **ItemType:** `ItemType.INGREDIENT` ("ingredient"), `ItemType.DISH` ("dish"), and `ItemType.RECIPE` ("recipe").
* **ItemState:** `RAW`, `CHOPPING_1`, `CHOPPING_2`, `CHOPPING_3`, `READY`, `EMPTY`, `BUILDING`, `COMPLETE`.

#### Centralized Ingredient Registry (`INGREDIENTS`)
The global registry tracks ingredient visualization (colors, characters) and processing:
```python
INGREDIENTS = {
    'tomato': {'color': (255, 0, 0), 'cuts_needed': 3, 'char': 't', 'state_chars': {...}},
    'lettuce': {'color': (107, 142, 35), 'cuts_needed': 3, 'char': 'l', 'state_chars': {...}},
    'cucumber': {'color': (0, 100, 0), 'cuts_needed': 3, 'char': 'c', 'state_chars': {...}}
}
```

#### Dynamic Extensions
Register new ingredients programmatically:
```python
from overcooked_marl_env.core.items import register_custom_ingredient

register_custom_ingredient(
    name="onion",
    color=(235, 150, 235),
    cuts_needed=3,
    char="o"
)
```

#### ItemManager
Mapeia caracteres do grid do mapa para códigos numéricos:
* ` ` (Empty Floor) → Code `0`
* `#` (Counter) → Code `1`
* `I` (Ingredient Dispenser) → Code `2`
* `C` (Chopping Board) → Code `3`
* `P` (Plate Dispenser) → Code `5`
* `D` (Delivery Station) → Code `6`
* `T` (Trash) → Code `7`
* `|` / `_` (Walls) → Codes `8` / `9`

### 3. Recipes & Dynamic Loading (`recipes.py`)
Determines if combinations of processed ingredients on a plate form a valid dish order.

#### Recipe Dataclass
* `name`: Recipe name (e.g. `"GARDEN_SALAD"`).
* `ingredients`: List of required items (e.g. `["lettuce", "tomato"]`).
* `reward`: Delivery reward score.

#### Loading Custom Recipes & Ingredients from JSON
Load a complete JSON configuration of recipes and ingredients dynamically:
```python
recipe_manager = RecipeManager()
recipe_manager.load_recipes_from_json("example/recipes.json")
```

---

## Verification & Testing

Verify items and recipes logic:
```bash
python -m unittest overcooked_marl_env/tests/unit/test_item_instance_manager.py
```
