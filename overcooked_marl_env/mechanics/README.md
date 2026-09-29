# Mechanics System - Movement, Collision, and Interaction

The `mechanics` module implements the environment physics, agent collisions, and item interactions.

## Sub-package Contents

```
mechanics/
├── __init__.py              # Module exports
├── movement_system.py       # Agent walking and orientation updates
├── collision_resolver.py    # Collision detection and resolutions
└── interaction_handler.py   # Kitchen interactions (pick, place, chop, deliver)
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
* **Processing:** `process_movement_actions(actions: dict, game_state: dict, map_size: tuple, item_manager)` computes candidate moves. Even if blocked by obstacles, the agent's orientation will always face the direction of the movement action.

### 2. Collision Resolver (`collision_resolver.py`)
Resolves multi-agent position conflicts at each step:
* `resolve_collisions(intended_positions: dict, current_positions: dict, verbose: bool)`
  Resolves conflicts by:
  - Blocking an agent if its candidate square is occupied by a stationary agent.
  - Blocking both agents if they attempt to enter the same coordinate simultaneously.
* **Penalty:** A collision penalty can be optionally configured (default: 0.0).

### 3. Interaction Handler (`interaction_handler.py`)
Executes all interactions (action 4) when agents stand adjacent to workstation counters or dispensers:
* `handle_interaction(agent_id: str, game_state: dict, verbose: bool) -> Tuple[float, float]`
  Checks the tile directly in front of the agent and routes to specific interaction routines, returning `(local_reward, global_reward_bonus)`.

---

## Interaction Types (Action 4)

1. **Pick Up Ingredient:** Take a raw item from adjacent dispensers or counters. (Local Reward: `+1.0`).
2. **Pick Up Plate:** Take a clean empty plate from a plate dispenser (`P`). (Local Reward: `+1.0`).
3. **Place on Counter:** Lay down carried item on an empty counter (`#`). (Local Reward: `+1.0` if sorting/cleaning up).
4. **Chop on Board:** Chop raw ingredients at cutting boards (`C`) to progress their state (`RAW` → `CHOPPING_1` → `CHOPPING_2` → `CHOPPING_3` / `READY`).
   * Cuts 1 & 2: Local Reward `+1.0`
   * Cut 3 (Ready): Local Reward `+5.0`
5. **Combine on Plate:** Add processed ingredients to a plate. (Local Reward: `+100.0` for combining).
6. **Deliver Recipe:** Deliver a plate containing the correct sequence of ingredients to the Delivery station (`D`).
   * Earns recipe reward (e.g. `SIMPLE_SALAD = 10.0`, `GARDEN_SALAD = 15.0`, `FRESH_SALAD = 20.0`).
   * Earns sequence delivery multiplier (`1.5x` bonus if delivered in order).
   * Distributes global reward to all agents.

---

## Integration in `step()`

```python
def step(actions):
    # 1. Process candidate movement steps
    intended_positions = movement_system.process_movement_actions(...)
    
    # 2. Resolve collisions
    final_positions, penalties = collision_resolver.resolve_collisions(...)
    
    # 3. Update agent positions
    for agent_id, pos in final_positions.items():
        state["agent_pos"][agent_id] = pos
        
    # 4. Handle interact actions
    for agent_id, action in actions.items():
        if action == 4:
            handle_interaction(agent_id, state)
```

---

## Verification & Testing

Verify physics and interaction mechanics:
```bash
python -m unittest overcooked_marl_env/tests/unit/test_movement_system.py
python -m unittest overcooked_marl_env/tests/unit/test_collision_resolver.py
python -m unittest overcooked_marl_env/tests/unit/test_interaction_handler.py
```


### Treinar modelos
```bash
venv/bin/python experiments/training/train_ppo.py --num-agents 1 --map default_map.txt --episodes 150 --workers 4
```