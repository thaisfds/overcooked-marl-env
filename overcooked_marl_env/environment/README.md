# Environment System — Game State & Observations

Manages the internal kitchen state (`MapData`, `GameState`) and builds multi-agent RL observations in either dense 150D vector or 9-channel spatial grid format.

---

## Sub-package Contents

```text
environment/
├── __init__.py                        # Module exports
├── env_config.py                      # Environment constants re-exports
├── game_state.py                      # MapData dataclass and GameState container
├── observation_builder.py             # 9-channel egocentric spatial tensor
└── featurized_observation_builder.py  # 150D scale-invariant vector (default for RL)
```

## Game State (`game_state.py`)

* **`MapData`:** Stores parsed static map layout (`static_map`, `agent_positions`, `dispensers`, `recipes`, `size`).
* **`GameState`:** Tracks agent positions, orientations, inventories, and ground items, delegating order queue queries to [`OrderQueue`](../core/recipes.py).

## Observation Builders

### A. Featurized Observation Vector (`150D`) — Default
`FeaturizedObservationBuilder` produces a normalized 1D vector `(150,)` invariant to map size and grid scale:

| Slice | Dims | Feature Block | Summary |
| :--- | :---: | :--- | :--- |
| `[0:24]` | 24 | **Self Agent State** | Normalized `(x, y)`, orientation, inventory one-hot, 8D ingredient multihot, processing progress, goal alignment, time & delivery ratios. |
| `[24:32]` | 8 | **Facing Tile Vision** | Station type norm (`0.2` Counter, `0.4` Appliance, `0.6` Dispenser, `0.8` Delivery, `1.0` Plate/Trash), item presence, dish status, recipe match, cut progress. |
| `[32:80]` | 48 | **Partner Agents** | Up to 3 partners × 16 dims (`dx, dy`, distance, orientation, held item, alignment, facing station, progress); zero-padded if fewer. |
| `[80:100]` | 20 | **HUD Recipe Orders** | Top 2 pending [`OrderQueue`](../core/recipes.py) orders × 10 dims (recipe index, 8D required ingredients, `is_next` flag). |
| `[100:108]` | 8 | **Attentive Dispensers** | Normalized `(dx, dy)` vectors to up to 4 dispensers for ingredients missing on the held plate. |
| `[108:132]` | 24 | **Fixed Stations & Beacons** | Nearest 2 Deliveries (4), 2 Plate dispensers (4), 1 Trash (2), 2 Universal appliances (12), Goal appliance beacon (2). |
| `[132:150]` | 18 | **Counters & Transfers** | Nearest 2 empty counters (4), 2 with plates (4), 2 with ready food (4), 2 transfer counters (4), transfer occupancy flags (2). |

<details>
<summary><b>B. Spatial Grid Observation (9 Channels) — Convolutional</b></summary>

`ObservationBuilder` produces a `(height, width, 9)` tensor:

| Channel | Layer | Description |
| :---: | :--- | :--- |
| **0** | **Static Map** | Walls, counters, and stations encoded by numeric codes (`0–9`). |
| **1** | **Floor Items** | Ground/counter items: `10 + ingredient_index + (state_offset * 10)`. |
| **2** | **Self Agent** | Binary mask at the observing agent's position. |
| **3** | **Other Agents** | Binary mask at teammate positions. |
| **4** | **Delivery Points** | Binary mask for delivery stations. |
| **5** | **Recipe Orders** | Order queue sequence highlighting next pending (`100`) and delivered (`200`) slots via [`OrderQueue`](../core/recipes.py). |
| **6–8** | **Processing States** | Remaining cuts (`1/3`, `2/3`, `3/3`, or instant readiness). |
</details>

## Testing

```bash
./venv/bin/python -m unittest discover -s overcooked_marl_env/tests
```

---

⬅**[Back to Package Documentation](../README.md)**
