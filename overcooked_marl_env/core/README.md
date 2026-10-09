# Core System - Items, Recipes, and Constants

The `core` module forms the domain model of the Overcooked MARL environment, defining item states, multi-stage processing pipelines, item types, recipe matches, order queue management, and game constants.

## Sub-package Contents

```
core/
├── __init__.py                 # Module exports
├── constants.py                # Action, observation, physical and reward constants
├── items.py                    # IngredientRegistry, item creation, multi-stage processing, tiles
├── recipes.py                  # Recipe definitions, OrderQueue (FIFO), RecipeManager
├── default_ingredients.json    # Declarative configuration for base ingredients
└── default_recipes.json        # Declarative configuration for default recipes
```

---

## Detailed Components

### 1. Constants (`constants.py`)
Centralizes key environment parameters following the Single Responsibility Principle:
* **Observation Channels (Spatial Tensor):** Defines `TOTAL_CHANNELS = 9` (6 base channels + 3 processing state channels).
* **Actions Mapping:** Defines action directions (`0: Up`, `1: Down`, `2: Left`, `3: Right`, `4: Interact`).
* **Game Parameters:** Defines episode limits (`MAX_EPISODE_STEPS = 3000`), real-time execution (`STEPS_PER_SECOND = 8`), and default cuts required (`DEFAULT_CUTS_NEEDED = 3`).
* **Milestone Rewards:** Managed dynamically in `RewardValues` ([`reward_shaper.py`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/mechanics/reward_shaper.py)) and recipe-specific values in [`default_recipes.json`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/core/default_recipes.json).

---

### 2. Items & Ingredients (`items.py`)
Items are represented as dynamic metadata dictionaries containing type, processing state, stage progression, and contents.

#### Item Types and States
* **`ItemType`:** `ItemType.INGREDIENT` ("ingredient"), `ItemType.DISH` ("dish"), and `ItemType.RECIPE` ("recipe").
* **`ItemState`:** `RAW`, `CHOPPING_1`, `CHOPPING_2`, `CHOPPING_3`, `READY`, `EMPTY`, `BUILDING`, `COMPLETE`.

#### Centralized Declarative Ingredient Registry (`IngredientRegistry`)
Base ingredients (`tomato`, `lettuce`, `cucumber`, `onion`) are loaded automatically from [`default_ingredients.json`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/core/default_ingredients.json):
```python
from overcooked_marl_env.core import ingredient_registry, INGREDIENTS

# Query ingredient
tomato_data = ingredient_registry.get("tomato")
```

#### Multi-Stage Pipelines & Instant Ingredients
* **Multi-Stage Processing (`stages`):**
  Ingredients can define a pipeline of transformation stages across different stations:
  ```python
  from overcooked_marl_env.core import register_custom_ingredient

  register_custom_ingredient(
      name="patty_meat",
      color=(139, 69, 19),
      stages=[
          {"station": "C", "action": "chop", "steps": 3},
          {"station": "S", "action": "cook", "steps": 20}
      ]
  )
  ```
* **Instant Ingredients (`cuts_needed=0`):**
  Ingredients that need no processing (e.g. bread bun, cheese) start immediately in `ItemState.READY`.
* **Character-less Ingredients:**
  Ingredients identify by `name`, eliminating ASCII letter collision bottlenecks.

#### Dynamic Tile Registration
New interactive stations or tiles can be registered dynamically in one line:
```python
from overcooked_marl_env.core import register_tile

register_tile(char='S', code=10, name="Stove", color=(255, 100, 50))
```

---

### 3. Recipes & Order Queue (`recipes.py`)

#### Centralized Order Queue (`OrderQueue`)
Eliminates duplicated queue calculation across observation builders, HUD renderers, and reward shapers:
```python
from overcooked_marl_env.core import OrderQueue

pending = OrderQueue.get_pending(recipe_order, delivered_recipes)
next_order = OrderQueue.get_next_pending(recipe_order, delivered_recipes)
```

#### RecipeManager & Declarative Loading
Default recipes (`SIMPLE_SALAD`, `GARDEN_SALAD`, `FRESH_SALAD`) are loaded from [`default_recipes.json`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/core/default_recipes.json):
```python
from overcooked_marl_env.core import RecipeManager

recipe_manager = RecipeManager()
recipe_manager.load_recipes_from_json("custom_map_recipes.json")
```

---

## Verification & Testing

Verify items, recipes, order queue, and registry logic:
```bash
./venv/bin/python -m unittest discover -s overcooked_marl_env/tests
```
