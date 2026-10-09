# Overcooked MARL Environment - Package Documentation

This is the main package containing the multi-agent reinforcement learning environment implementation for Overcooked MARL. This document describes the modular architecture, core subsystems, observation representations, and extension mechanisms.

## Package Structure

```
overcooked_marl_env/
├── __init__.py              # Package exports (OvercookedMARL, FeaturizedObservationBuilder, OrderQueue)
├── env.py                   # Main environment wrapper class (OvercookedMARL)
├── overcooked_marl_v0.py   # PettingZoo parallel environment interface
├── core/                    # Items, recipes, order queue, and constants system
│   ├── items.py             # Item types, multi-stage pipelines, helpers (is_plate, is_ready)
│   ├── recipes.py           # Recipes, RecipeManager, OrderQueue (FIFO calculation)
│   └── constants.py         # Physics, timing, rewards, and dimensions constants
├── environment/             # State management and observation builders
│   ├── game_state.py        # Complete game state representation
│   ├── observation_builder.py # Multi-channel spatial grid observations (9 channels)
│   └── featurized_observation_builder.py # Dense 150D scale-invariant vector observations
├── map/                     # Map loading, parsing, and structure validation
├── mechanics/               # Movement, collisions, reward shaping, and interactions
│   ├── movement_system.py   # Directional movement & orientation tracking
│   ├── collision_resolver.py# Multi-agent collision resolution
│   ├── reward_shaper.py     # Potential-based reward shaping & intermediate bonuses
│   ├── interaction_handler.py # High-level interaction coordinator
│   └── interactions/        # Station Strategy Pattern (modular station handlers)
│       ├── base_station.py  # Abstract StationHandler base class
│       ├── station_registry.py # Dynamic strategy registry
│       ├── chopping_handler.py # Chopping boards ('C')
│       ├── dispenser_handler.py # Plate ('P') & Ingredient ('I') dispensers
│       ├── counter_handler.py # Workstation counters ('#')
│       ├── trash_handler.py # Trash can clawback ('T')
│       └── delivery_handler.py # Delivery verification and rewards ('D')
├── rendering/               # Pygame (human) and ASCII console visualizers
└── tests/                   # Complete unit and integration test suite
```

---

## Subsystems

### 1. Core System (`core/`)
Manages items, recipes, order queues, and game constants:
* **`items.py`**:
  * Multi-stage processing pipelines via `stages: [{"station": "C", "action": "chop", "steps": 3}, ...]`
  * Support for instant ready ingredients (`cuts_needed: 0`) and character-less ingredients without ASCII alphabet exhaustion.
  * Centralized item query helpers (`is_plate(item)`, `is_ready(item)`, `can_process(item, station)`).
* **`recipes.py`**:
  * `OrderQueue`: Centralized deterministic FIFO pending order calculations (`get_pending`, `get_next_pending`).
  * `RecipeManager`: Declarative JSON recipe/ingredient loader (`load_recipes_from_json`).

### 2. Environment & Observation Builders (`environment/`)
* **Featurized Observation Builder (150 Dimensions)**:
  * Scale-invariant, dense vector representation invariant to map size and number of agents.
  * Layout:
    * `[0:24]` Self Agent State (ego, hand inventory, goal alignment)
    * `[24:32]` Facing Tile Vision (station, contents, progress)
    * `[32:80]` Partner Agents (up to 3 partners * 16 dims = 48 dims)
    * `[80:100]` HUD Recipe Orders (goal conditioning, 2 recipes * 10 dims = 20 dims)
    * `[100:108]` Attentive Dispensers (missing ingredients directional vectors)
    * `[108:132]` Fixed Stations & Appliance Beacons
    * `[132:150]` Counters & Multi-Agent Transfer Midpoints
* **Observation Builder (Spatial Grid)**:
  * 9-channel tensor representation for convolutional policies.

### 3. Mechanics & Station Strategy Pattern (`mechanics/`)
* **Station Strategy Pattern**: Every kitchen station is an isolated, testable strategy implementing `StationHandler`.
* **Registration**: Station behaviors are dynamically dispatched by `StationRegistry` based on the facing tile's station code, avoiding bloated monolithic code.
* **Reward Shaping**: `RewardShaper` handles potential-based shaping, preventing reward hacking and maintaining strict game-theoretic convergence.

---

## Testing & Verification

Run the complete test suite to ensure mathematical and operational integrity:
```bash
./venv/bin/python -m unittest discover -s overcooked_marl_env/tests
```
