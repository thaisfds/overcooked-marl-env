# Core System — Items, Recipes, and Constants

Domain model of the Overcooked MARL environment: item states, multi-stage processing pipelines, declarative ingredient/recipe registries, FIFO order queue, and constants.

---

## Sub-package Contents

```text
core/
├── __init__.py                 # Module exports
├── constants.py                # Action, observation, physical, and reward constants
├── items.py                    # IngredientRegistry, item creation, multi-stage processing, tiles
├── recipes.py                  # Recipe definitions, OrderQueue (FIFO), RecipeManager
├── default_ingredients.json    # Declarative base ingredients config
└── default_recipes.json        # Declarative default recipes config
```

## Key Components

* **Constants (`constants.py`):** Centralizes `TOTAL_CHANNELS = 9`, actions (`0: Up`, `1: Down`, `2: Left`, `3: Right`, `4: Interact`), `MAX_EPISODE_STEPS = 3000`, `STEPS_PER_SECOND = 8`, and `DEFAULT_CUTS_NEEDED = 3`. Milestone rewards live in [`reward_shaper.py`](../mechanics/reward_shaper.py) and recipe rewards in [`default_recipes.json`](./default_recipes.json).
* **Items & Ingredients (`items.py`):**
  * `ItemType`: `INGREDIENT`, `DISH`, `RECIPE` | `ItemState`: `RAW`, `CHOPPING_1..3`, `READY`, `EMPTY`, `BUILDING`, `COMPLETE`.
  * `IngredientRegistry` loads base ingredients (`tomato`, `lettuce`, `cucumber`, `onion`) from [`default_ingredients.json`](./default_ingredients.json).
  * Supports **multi-stage pipelines** (`stages`), **instant ingredients** (`cuts_needed=0`), **character-less IDs**, and **dynamic tile registration** (`register_tile`).
* **Recipes & Order Queue (`recipes.py`):**
  * `OrderQueue` provides centralized FIFO order tracking (`get_pending`, `get_next_pending`).
  * `RecipeManager` loads default recipes (`SIMPLE_SALAD`, `GARDEN_SALAD`, `FRESH_SALAD`) from [`default_recipes.json`](./default_recipes.json) or custom JSON files.

<details>
<summary><b>Usage & Extension Examples</b></summary>

```python
from overcooked_marl_env.core import (
    ingredient_registry, register_custom_ingredient, register_tile,
    OrderQueue, RecipeManager,
)

# Query base ingredient
tomato_data = ingredient_registry.get("tomato")

# Register a multi-stage custom ingredient
register_custom_ingredient(
    name="patty_meat",
    color=(139, 69, 19),
    stages=[
        {"station": "C", "action": "chop", "steps": 3},
        {"station": "S", "action": "cook", "steps": 20},
    ],
)

# Register a custom station tile
register_tile(char="S", code=10, name="Stove", color=(255, 100, 50))

# Query FIFO pending orders & load custom recipes
pending = OrderQueue.get_pending(recipe_order, delivered_recipes)
next_order = OrderQueue.get_next_pending(recipe_order, delivered_recipes)

recipe_manager = RecipeManager()
recipe_manager.load_recipes_from_json("custom_map_recipes.json")
```
</details>

## Testing

```bash
./venv/bin/python -m unittest discover -s overcooked_marl_env/tests
```

---

⬅**[Back to Package Documentation](../README.md)**
