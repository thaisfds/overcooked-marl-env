# Environment System - Game State and Observations

The `environment` module manages the internal representation of the kitchen state and formats observations for multi-agent reinforcement learning.

## Sub-package Contents

```
environment/
├── __init__.py                      # Module exports
├── env_config.py                    # Environment constants re-exports
├── game_state.py                    # MapData dataclass and GameState container
├── observation_builder.py           # Multi-channel egocentric spatial tensor (9 channels)
└── featurized_observation_builder.py# Dense scale-invariant vector observations (150 dimensions)
```

---

## 1. Game State Representation (`game_state.py`)

The game state is represented as a structured Python dictionary, facilitating serialization and compatibility with PettingZoo and RLlib.

### MapData Dataclass
Stores parsed static structure of a loaded map:
* `static_map`: `np.ndarray` of static tile numeric codes.
* `agent_positions`: `Dict[str, Tuple[int, int]]` mapping agent IDs to starting coordinates.
* `dispensers`: `Dict[Tuple[int, int], str]` mapping coordinates to ingredient dispenser names.
* `recipes`: `List[str]` of required recipes.
* `size`: Map shape `(height, width)`.

### GameState Class
* Encapsulates agent positions, orientations, inventory, and ground items.
* Delegates pending recipe orders and current target verification to [`OrderQueue`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/core/recipes.py).

---

## 2. Observation Builders

The environment supports two complementary observation representations:

### A. Featurized Observation Vector (150 Dimensions) — Default for PPO
`FeaturizedObservationBuilder` produces a dense, normalized 1D vector of shape `(150,)` invariant to map dimensions, number of counters, and grid scale:

| Index Range | Dimension Count | Feature Block | Description |
| :--- | :--- | :--- | :--- |
| `[0:24]` | 24 | **Self Agent State** | Normalized `(x, y)`, orientation, inventory category one-hot, 8-dim multihot ingredients, processing progress (`0.0` to `1.0`, supports instant ingredients and multi-stage items), dish target alignment, time ratio, deliveries ratio. |
| `[24:32]` | 8 | **Facing Tile Vision** | Station type norm (`0.2` Counter, `0.4` any Universal Processing Appliance, `0.6` Dispenser, `0.8` Delivery, `1.0` Plate/Trash), item presence, dish status, recipe matching, cutting progress `_get_item_cuts_progress`. |
| `[32:80]` | 48 | **Partner Agents** | Up to 3 partner agents * 16 dims each (relative offset `dx, dy`, distance, orientation, carried item, target alignment, facing station code, cutting progress). Zero-padded if fewer agents. |
| `[80:100]` | 20 | **HUD Recipe Orders** | Goal conditioning via [`OrderQueue`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/core/recipes.py): top 2 pending orders * 10 dims each (recipe index norm, 8-dim multihot required ingredients, is_next flag). |
| `[100:108]` | 8 | **Attentive Dispensers** | Normalized directional vectors `(dx, dy)` to the 4 dispensers for ingredients currently missing on the held plate. |
| `[108:132]` | 24 | **Fixed Stations & Beacons** | 2 nearest Delivery stations (4 dims), 2 nearest Plate dispensers (4 dims), 1 nearest Trash can (2 dims), 2 nearest Universal appliances (12 dims), Goal-directed appliance beacon (2 dims). |
| `[132:150]` | 18 | **Counters & Transfers** | 2 nearest empty counters (4 dims), 2 nearest counters with plates (4 dims), 2 nearest with ready food (4 dims), 2 nearest transfer counters (4 dims), transfer occupancy flags (2 dims). |

---

### B. Spatial Grid Observation (9 Channels) — Convolutional
`ObservationBuilder` builds a tensor of shape `(height, width, 9)` for spatial policies:

| Channel | Layer | Description |
| :--- | :--- | :--- |
| **0** | **Static Map** | Static walls, counters, and stations encoded by numeric codes (0-9). |
| **1** | **Floor Items** | Items placed on counters or floor: `10 + ingredient_index + (state_offset * 10)` (state offset 4 for instant/ready). |
| **2** | **Self Agent** | Binary mask containing `1` only at the observing agent's position. |
| **3** | **Other Agents** | Binary mask containing `1` at teammate positions. |
| **4** | **Delivery Points** | Binary mask showing delivery station locations. |
| **5** | **Recipe Orders** | Sequence layer representing the list of recipe IDs in order, dynamically highlighting the true next pending slot via [`OrderQueue`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/core/recipes.py) (status 100) and delivered slots (status 200). |
| **6-8** | **Processing States** | Layer indicating remaining cuts to complete chopping progress (cuts 1/3, 2/3, 3/3, or instant readiness). |

---

## Verification & Testing

Verify state management and observation shapes:
```bash
./venv/bin/python -m unittest discover -s overcooked_marl_env/tests
```
