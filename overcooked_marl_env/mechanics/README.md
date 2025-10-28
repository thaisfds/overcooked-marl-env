# Mechanics System - Movement, Collision, and Interaction

The `mechanics` module implements game physics and all player interactions.

## Files

```
mechanics/
├── __init__.py              # Module exports
├── movement_system.py       # Agent movement
├── collision_resolver.py    # Collision detection
└── interaction_handler.py   # Item interactions
```

## Responsibilities

- Calculate agent movement (up, down, left, right)
- Detect and prevent collisions (walls, agents)
- Execute interactions (pick, place, process, combine, deliver)
- Calculate rewards for actions
- Validate interaction legality

## Detailed Files

### `movement_system.py`

Handles agent movement.

#### MovementSystem

```python
class MovementSystem:
    DIRECTIONS = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
    }
```

**Methods**:

**`calculate_new_position(current_pos: Tuple[int, int], action: int) -> Tuple[int, int]`**

Calculates new position after movement:

```python
movement_system = MovementSystem()
current_pos = (3, 4)
action = 0  # Move up

new_pos = movement_system.calculate_new_position(current_pos, action)
# Returns (2, 4) - moved up one row
```

**Direction Mapping**:
- Action 0 (UP): `row - 1`
- Action 1 (DOWN): `row + 1`
- Action 2 (LEFT): `col - 1`
- Action 3 (RIGHT): `col + 1`

### `collision_resolver.py`

Detects collisions.

#### CollisionResolver

```python
class CollisionResolver:
    def __init__(self, map_data: MapData):
        self.map_data = map_data
```

**Methods**:

**`has_collision(pos: Tuple[int, int], game_state: GameState) -> bool`**

Checks if position is blocked:

```python
collision_resolver = CollisionResolver(map_data)
pos = (3, 4)

if collision_resolver.has_collision(pos, game_state):
    print("Cannot move here!")
else:
    print("Position is free")
```

**Collision Types**:

1. **Out of bounds**: Position outside map
```python
has_collision((-1, 5), game_state)  # True - outside map
```

2. **Wall**: Position is a wall tile
```python
has_collision((0, 0), game_state)  # True - corner wall
```

3. **Other agent**: Another agent at position
```python
# If agent_1 is at (3, 4)
has_collision((3, 4), game_state)  # True - occupied
```

**Free positions**: EMPTY, COUNTER, CUTTING_BOARD, DISPENSER, DELIVERY_STATION

### `interaction_handler.py`

Executes all interactions.

#### InteractionHandler

```python
class InteractionHandler:
    def __init__(self, item_manager: ItemInstanceManager):
        self.item_manager = item_manager
```

**Main Method**:

**`handle_interaction(agent: str, game_state: GameState) -> float`**

Executes interaction and returns reward.

```python
interaction_handler = InteractionHandler(item_manager)
reward = interaction_handler.handle_interaction("agent_0", game_state)
# Returns reward value (0.0 if no valid interaction)
```

## The 6 Interaction Types

### 1. Pick Up Item from Floor/Counter

**Condition**: Agent has empty hands + Item at adjacent position

**Process**:
1. Check adjacent tiles (up, down, left, right)
2. Find first item on floor/counter
3. Remove item from position
4. Place in agent's inventory

**Reward**: +0.1

**Example**:
```
Before:     After:
[L][ ]      [ ][ ]    Agent picks lettuce
[ ][A]  →   [ ][A]    Inventory: L
```

### 2. Pick Up from Dispenser

**Condition**: Agent has empty hands + Adjacent to dispenser

**Dispensers**:
- **INGREDIENT_DISPENSER**: Spawns random ingredient (LETTUCE, TOMATO, or CUCUMBER)
- **PLATE_DISPENSER**: Spawns PLATE

**Process**:
1. Check adjacent dispenser
2. Create new item
3. Place in agent's inventory

**Reward**: +0.1

**Example**:
```
Before:     After:
[I][ ]      [I][ ]    Agent takes from dispenser
[ ][A]  →   [ ][A]    Inventory: LETTUCE (random)
```

### 3. Place Item on Floor/Counter

**Condition**: Agent holding item + Adjacent empty floor/counter

**Process**:
1. Check adjacent empty positions
2. Remove item from agent inventory
3. Place item at position

**Reward**: +0.0 (neutral)

**Example**:
```
Before:     After:
[ ][ ]      [L][ ]    Agent places lettuce
[ ][A]  →   [ ][A]    Inventory: empty
(holding L)
```

### 4. Process on Cutting Board

**Condition**: Agent empty hands + Adjacent cutting board with raw ingredient

**Process**:
1. Check cutting board position
2. Check item is raw ingredient (LETTUCE, TOMATO, CUCUMBER)
3. Advance cutting state:
   - RAW → CHOPPED_1 (+0.1 reward)
   - CHOPPED_1 → CHOPPED_2 (+0.2 reward)
   - CHOPPED_2 → CHOPPED_3 (+0.5 reward, transform to chopped type)

**Example**:
```
Cut 1:  LETTUCE [RAW] → LETTUCE [CHOPPED_1]       +0.1
Cut 2:  LETTUCE [CHOPPED_1] → LETTUCE [CHOPPED_2] +0.2
Cut 3:  LETTUCE [CHOPPED_2] → CHOPPED_LETTUCE     +0.5
```

**Total reward for complete processing**: 0.8

### 5. Combine on Plate

**Condition**: Agent holding processed ingredient + Adjacent plate

**Process**:
1. Check agent holds chopped ingredient
2. Check adjacent position has PLATE
3. Add ingredient to plate's contents
4. Remove from agent inventory

**Reward**: +0.3

**Example**:
```
Step 1: Add chopped lettuce
[P][ ]      [P][ ]    Plate contents: [CHOPPED_LETTUCE]
[ ][A]  →   [ ][A]    Reward: +0.3
(holding CHOPPED_LETTUCE)

Step 2: Add chopped tomato
[P][ ]      [P][ ]    Plate contents: [CHOPPED_LETTUCE, CHOPPED_TOMATO]
[ ][A]  →   [ ][A]    Reward: +0.3
(holding CHOPPED_TOMATO)
```

### 6. Deliver Recipe

**Condition**: Agent holding plate with valid recipe + At delivery station

**Process**:
1. Check agent holds PLATE
2. Get plate contents
3. Validate recipe (check recipes.py)
4. If valid:
   - Award recipe reward
   - Remove plate
   - Increment delivered count
   - Remove from pending orders

**Rewards**:
- SIMPLE_SALAD: +10.0
- GARDEN_SALAD: +15.0
- FRESH_SALAD: +20.0

**Example**:
```
Plate contents: [CHOPPED_LETTUCE, CHOPPED_TOMATO]
Recipe match: GARDEN_SALAD
Reward: +15.0
```

## Integration in step()

```python
def step(actions):
    rewards = {agent: 0.0 for agent in agents}
    
    for agent, action in actions.items():
        if action < 4:  # Movement (0-3)
            new_pos = movement_system.calculate_new_position(
                game_state.get_agent_position(agent),
                action
            )
            
            if not collision_resolver.has_collision(new_pos, game_state):
                game_state.move_agent(agent, new_pos)
        
        elif action == 4:  # Interact
            reward = interaction_handler.handle_interaction(agent, game_state)
            rewards[agent] = reward
    
    return observations, rewards, dones, truncs, infos
```

## Reward Table

| Action | Reward | Notes |
|--------|---------|-------|
| Pick up item | +0.1 | From floor/counter/dispenser |
| Place item | +0.0 | Neutral |
| Cut 1/3 | +0.1 | First cut |
| Cut 2/3 | +0.2 | Second cut |
| Cut 3/3 | +0.5 | Complete processing |
| Add to plate | +0.3 | Combine ingredient |
| Deliver SIMPLE_SALAD | +10.0 | 1 ingredient |
| Deliver GARDEN_SALAD | +15.0 | 2 ingredients |
| Deliver FRESH_SALAD | +20.0 | 3 ingredients |
| Invalid interaction | +0.0 | No effect |

**Total for FRESH_SALAD** (3 ingredients):
- Pick 3 ingredients: 3 × 0.1 = 0.3
- Pick 1 plate: 0.1
- Process 3 ingredients: 3 × (0.1 + 0.2 + 0.5) = 2.4
- Combine 3 times: 3 × 0.3 = 0.9
- Deliver: 20.0
- **TOTAL**: 23.7 points

## Design Patterns

### Single Responsibility

Each system has one job:
- **MovementSystem**: Only calculates positions
- **CollisionResolver**: Only checks validity
- **InteractionHandler**: Only executes interactions

### State Queries

Systems query GameState, not modify directly:
```python
# CollisionResolver queries
pos = game_state.get_agent_position(agent)
item = game_state.get_item_at_position(pos)

# InteractionHandler modifies through GameState methods
game_state.set_agent_inventory(agent, item_id)
game_state.place_item_at_position(pos, item_id)
```

### Reward Immediate Feedback

Rewards returned instantly, not accumulated:
```python
reward = handle_interaction(agent, game_state)
# Agent immediately knows result
```

## Tests

```bash
# Test movement
python -m pytest overcooked_marl_env/tests/unit/test_movement_system.py

# Test collision
python -m pytest overcooked_marl_env/tests/unit/test_collision_resolver.py

# Test interactions
python -m pytest overcooked_marl_env/tests/unit/test_interaction_handler.py

# All mechanics tests
python -m pytest overcooked_marl_env/tests/unit/ -k "movement or collision or interaction"
```

## See Also

- [Core Documentation](../core/README.md) - Item types and recipes
- [Environment Documentation](../environment/README.md) - Game state structure
- [Map Documentation](../map/README.md) - Tile types
- [Main Package Documentation](../README.md) - Complete architecture
