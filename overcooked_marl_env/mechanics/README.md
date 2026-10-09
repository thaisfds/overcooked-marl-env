# Mechanics System - Movement, Collision, and Modular Interactions

The `mechanics` module implements the environment physics, agent collisions, reward shaping, and workstation interactions.

## Sub-package Contents

```
mechanics/
├── __init__.py              # Module exports
├── movement_system.py       # Agent walking and orientation updates
├── collision_resolver.py    # Multi-agent collision resolution
├── reward_shaper.py         # Potential-based shaping and intermediate bonuses
├── interaction_handler.py   # Interaction coordinator and backward compatibility adapter
└── interactions/            # Station Strategy Pattern implementations
    ├── __init__.py          # Strategy exports
    ├── base_station.py      # Abstract StationHandler base class
    ├── station_registry.py  # StationRegistry dynamic strategy lookup
    ├── chopping_handler.py  # Universal processing station ('C', 'S', etc.)
    ├── dispenser_handler.py # Plate ('P') & Ingredient ('I') dispensers
    ├── counter_handler.py   # General counters/tables ('#')
    ├── trash_handler.py     # Waste and reward clawback ('T')
    └── delivery_handler.py  # Recipe validation, HUD matching, and victory ('D')
```

---

## Detailed Components

### 1. Movement System (`movement_system.py`)
Computes new coordinates and updates orientation based on action directions:
* **Directions Mapping:**
  - `0`: Move UP (North, `(0, -1)`)
  - `1`: Move DOWN (South, `(0, 1)`)
  - `2`: Move LEFT (West, `(-1, 0)`)
  - `3`: Move RIGHT (East, `(1, 0)`)
* **Processing:** `process_movement_actions(actions: dict, game_state: dict, map_size: tuple, item_manager)` computes candidate moves. The agent's orientation always reflects the intended movement direction, ensuring consistent facing-tile interactions.

### 2. Collision Resolver (`collision_resolver.py`)
Resolves multi-agent position conflicts at each step:
* `resolve_collisions(intended_positions: dict, current_positions: dict, verbose: bool)`
  - Blocks movement if the candidate square is occupied by a stationary agent.
  - Blocks both agents if they attempt to enter the same coordinate simultaneously.
* **Penalty:** Configurable collision penalty (default: `0.0`).

### 3. Interaction Handler & Station Strategy Pattern (`interaction_handler.py` & `interactions/`)
Coordinates all interact actions (Action 4) when agents face adjacent workstations:
* **Station Strategy Pattern:** Instead of a monolithic `if/elif` block, each station type is an isolated `StationHandler` subclass managed by `StationRegistry`.
* **Handlers:**
  * `ChoppingBoardHandler ('C')`: Placing raw items, progressive chopping, picking ready items, and direct plate assembly. Can also be parameterized for any appliance char (e.g. `'S'`).
  * `PlateDispenserHandler ('P')`: Dispensing clean plates and instant combination with held ready items.
  * `IngredientDispenserHandler ('I')`: Taking fresh ingredients from dispensers.
  * `CounterHandler ('#')`: Placing items, picking up items, direct combinations on counter, and anti-clutter / useless cycle penalty checks.
  * `TrashHandler ('T')`: Discarding items with strict shaping reward clawback.
  * `DeliveryHandler ('D')`: Validating plate contents against `OrderQueue`, calculating sequence order bonuses, dynamic delivery multipliers, and episode victory rewards.

### 4. Multi-Stage Ingredient Processing Across Stations
Ingredients can have multi-stage pipelines (e.g. Patty: chop on `'C'` $\to$ cook on `'S'` $\to$ combine onto plate):
1. **Station Matching:** `can_process(item, station)` verifies if the current stage of the item targets this specific station.
2. **Intermediate Pickups:** When an item completes its steps for station `'C'`, it transitions to the next stage (e.g., `'S'`). Because it can no longer be processed at `'C'`, the agent can pick it up to transport it to `'S'`.
3. **Completion Milestone:** When a processing stage finishes, agents receive the `chop_completed` milestone reward (+8.0 pts). When the final stage completes, the item transitions to `ItemState.READY`.

### 5. Extending with Custom Stations
To add a new station (e.g. `'S'` for Stove or `'F'` for Fryer):

**Option A: Universal Processing Appliance (Reuse `ChoppingBoardHandler`)**
If the new station processes multi-stage items through progress steps:
```python
from overcooked_marl_env.mechanics.interactions.chopping_handler import ChoppingBoardHandler

env.item_manager.register_tile('S', code=20, name="Stove", color=(255, 120, 50))
env.interaction_handler.register_station_handler('S', ChoppingBoardHandler(env.interaction_handler, station_char='S'))
```

**Option B: Custom Strategy Subclass**
For stations with entirely custom logic:
```python
from overcooked_marl_env.mechanics.interactions.base_station import StationHandler

class CustomMachineHandler(StationHandler):
    def handle(self, agent_id, tile_pos, carrying_item, carrying_instance_id,
               ground_item, ground_instance_id, game_state, verbose=False):
        # Custom logic here
        return step_reward, global_bonus

env.interaction_handler.register_station_handler('M', CustomMachineHandler(env.interaction_handler))
```

### 6. Reward Shaper (`reward_shaper.py`)
Computes potential-based milestone rewards and delivery multipliers:
* **Living Cost:** -0.005 per step.
* **Milestone Progression:**
  - Pick up needed ingredient: +5.0 pts
  - Place ingredient on station: +5.0 pts
  - Station processing cut/step: +1.0 pt
  - Complete station stage: +8.0 pts milestone
  - Combine ready ingredient with plate: +10.0 pts base + plate composition bonus
  - Complete plate assembly: +15.0 pts milestone
* **Dynamic Delivery Multipliers:**
  $$\text{Multiplier} = 1.0 + 0.5 \times (k - 1) \quad \text{for the } k\text{-th delivery}$$
* **Episode Victory Bonus:** +5000.0 pts when all pending orders are delivered.

---

## Verification & Testing

Run unit tests to verify physics and interaction mechanics:
```bash
./venv/bin/python -m unittest discover -s overcooked_marl_env/tests
```