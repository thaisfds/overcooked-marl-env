# Environment System - Game State and Observations

The `environment` module manages the internal representation of the kitchen state and formats egocentric observations for the agents.

## Sub-package Contents

```
environment/
├── __init__.py              # Module exports
├── game_state.py            # MapData and game state container keys
└── observation_builder.py   # Multi-channel egocentric tensor generation
```

---

## Game State Representation

Instead of holding references in a complex object, the game state is represented as a structured Python dictionary, facilitating serialization and compatibility with RL algorithms.

### 1. MapData Dataclass
Stores parsed static structure of a loaded map:
* `static_map`: `np.ndarray` of static tile numeric codes.
* `agent_positions`: `Dict[str, Tuple[int, int]]` mapping agent IDs to starting coordinates.
* `dispensers`: `Dict[Tuple[int, int], str]` mapping coordinates to ingredient dispenser names.
* `recipes`: `List[str]` of required recipes.
* `size`: Map shape `(height, width)`.

### 2. Game State Dictionary Structure
The state dictionary initialized by `_initialize_game_state` contains:
* `map_static`: Static tile grid.
* `agent_pos`: Coordinates `(x, y)` for active agents.
* `carrying`: Backward compatible inventory tracking.
* `items_on_ground`: Integer grid for floor items.
* `dishes_pending`: Quantity of dishes remaining to terminate the episode.
* `time_steps`: Step counter.
* `dispensers`: Mappings of dispensers.
* `recipe_order`: List of recipe orders expected.
* `expected_recipe_index`: Target delivery index.
* `delivered_recipes`: Delivery history.
* `item_instances` & `agent_holding_instance` & `ground_instances`: Unique ID tracking of item objects on plates, boards, or hands.

---

## 📊 The 9 Observation Channels

`ObservationBuilder` builds a tensor of shape `(height, width, 9)` for each active agent:

| Channel | Layer | Description |
| :--- | :--- | :--- |
| **0** | **Static Map** | Static walls, counters, and stations encoded by numeric codes (0-9). |
| **1** | **Floor Items** | Items placed on counters or floor, encoded as: `10 + ingredient_index + (state_offset * 10)`. |
| **2** | **Self Agent** | Binary mask containing `1` only at the observing agent's position. |
| **3** | **Other Agents** | Binary mask containing `1` at the coordinates of teammates. |
| **4** | **Delivery Points** | Binary mask showing delivery station locations. |
| **5** | **Recipe Orders** | Sequence layer representing the list of pending recipe IDs to complete. |
| **6-8** | **Processing States** | Layer indicating remaining cuts to complete chopping progress (cuts 1/3, 2/3, 3/3). |

---

## Verification & Testing

Verify game state and observation tensor shapes:
```bash
python -m unittest overcooked_marl_env/tests/unit/test_game_state.py
python -m unittest overcooked_marl_env/tests/unit/test_observation_builder.py
```
