# Mechanics System — Physics, Interactions & Reward Shaping

Implements movement, multi-agent collision resolution, Potential-Based Reward Shaping (PBRS), and modular workstation interactions via the **Station Strategy Pattern**.

---

## Sub-package Contents

```text
mechanics/
├── __init__.py              # Module exports
├── movement_system.py       # Directional walking (0:U, 1:D, 2:L, 3:R) & orientation
├── collision_resolver.py    # Multi-agent position conflict resolution
├── reward_shaper.py         # Potential-based shaping, milestones & delivery multipliers
├── interaction_handler.py   # High-level interaction coordinator
└── interactions/            # Station Strategy Pattern handlers
    ├── base_station.py      # Abstract StationHandler
    ├── station_registry.py  # Dynamic station lookup
    ├── chopping_handler.py  # Universal processing appliance ('C', 'S', etc.)
    ├── dispenser_handler.py # Plate ('P') & Ingredient ('I') dispensers
    ├── counter_handler.py   # General counters/tables ('#')
    ├── trash_handler.py     # Waste & shaping reward clawback ('T')
    └── delivery_handler.py  # OrderQueue matching, multipliers & victory ('D')
```

## Core Systems

* **Movement & Collisions:** `process_movement_actions` updates orientation and computes candidate moves (`0` Up, `1` Down, `2` Left, `3` Right). `resolve_collisions` blocks overlapping moves (configurable penalty, default `0.0`).
* **Station Strategy Pattern (`interactions/`):** Action `4` (Interact) dispatches dynamically through `StationRegistry` to isolated `StationHandler` classes (`C`, `P`, `I`, `#`, `T`, `D`), supporting multi-stage pipelines (e.g., chop on `C` → cook on `S` → plate).
* **Reward Shaper (`reward_shaper.py`):**
  * Living cost: `-0.005/step`
  * Milestones: Pick up needed (`+5.0`), Place on station (`+5.0`), Processing step (`+1.0`), Complete stage (`+8.0`), Combine on plate (`+10.0` + bonus), Complete plate (`+15.0`).
  * Dynamic Delivery Multiplier: `1.0 + 0.5 × (k - 1)` for the *k*-th delivery.
  * Episode Victory Bonus: `+5,000.0 pts` when all pending orders are delivered.

<details>
<summary><b>Extending with Custom Stations (Code Examples)</b></summary>

**Option A — Reuse Universal Processing Appliance (`ChoppingBoardHandler`):**
```python
from overcooked_marl_env.mechanics.interactions.chopping_handler import ChoppingBoardHandler

env.item_manager.register_tile("S", code=20, name="Stove", color=(255, 120, 50))
env.interaction_handler.register_station_handler("S", ChoppingBoardHandler(env.interaction_handler, station_char="S"))
```

**Option B — Custom `StationHandler` Subclass:**
```python
from overcooked_marl_env.mechanics.interactions.base_station import StationHandler

class CustomMachineHandler(StationHandler):
    def handle(self, agent_id, tile_pos, carrying_item, carrying_instance_id,
               ground_item, ground_instance_id, game_state, verbose=False):
        return step_reward, global_bonus

env.interaction_handler.register_station_handler("M", CustomMachineHandler(env.interaction_handler))
```
</details>

## Testing

```bash
./venv/bin/python -m unittest discover -s overcooked_marl_env/tests
```

---

⬅**[Back to Package Documentation](../README.md)**
