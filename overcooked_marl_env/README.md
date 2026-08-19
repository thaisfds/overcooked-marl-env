# Overcooked MARL Environment - Package Documentation

This is the main package containing all environment implementation. This document describes the internal architecture and how the different subsystems work together.

## Package Structure

```
overcooked_marl_env/
├── __init__.py              # Package initialization and exports
├── env.py                   # Main environment class (OvercookedMARL)
├── overcooked_marl_v0.py   # PettingZoo parallel environment
├── core/                    # Items, recipes, and constants system
├── environment/             # Game state and observation building
├── map/                     # Map loading and validation
├── mechanics/               # Movement, collision, and interaction
├── rendering/               # ASCII and Pygame rendering
└── tests/                   # Unit and integration tests
```

## Architecture

### Execution Flow

```
User Code
    ↓
OvercookedMARL (env.py)
    ↓
OvercookedMARLParallelEnv (overcooked_marl_v0.py)
    ↓
┌────────────────┬───────────────────┬──────────────────┐
│   GameState    │  MovementSystem   │  ObservationBuilder │
│  (game state)  │   (mechanics)     │  (observations)    │
└────────────────┴───────────────────┴──────────────────┘
         ↓                ↓                    ↓
    ItemManager    CollisionResolver    Renderer
```

## Subsystems

### 1. Core System (`core/`)
Manages items, recipes, and game constants.

**Main Components**:
- `constants.py` - Game constants and item types
- `items.py` - Item types, states, and utilities
- `recipes.py` - Available recipes and validation
- `item_instance_manager.py` - Item lifecycle management

**Responsibilities**:
- Define item types (LETTUCE, TOMATO, CUCUMBER, PLATE, etc.)
- Manage item states (RAW, CHOPPED_1, CHOPPED_2, CHOPPED_3)
- Validate and check recipe completion
- Create and destroy item instances

### 2. Environment System (`environment/`)
Manages game state and generates observations.

**Main Components**:
- `game_state.py` - Complete game state
- `observation_builder.py` - Observation tensor generation

**Responsibilities**:
- Maintain agent positions and inventory
- Track items on map and their states
- Generate multi-channel observations (9 channels)
- Manage pending orders and score

### 3. Map System (`map/`)
Loads, validates, and parses maps.

**Main Components**:
- `map_parser.py` - Text to grid conversion
- `map_validator.py` - Structure validation
- `map_loader.py` - Complete map loading

**Responsibilities**:
- Read `.txt` map files
- Validate tile types and agent positions
- Convert to internal grid representation
- Provide default maps (default_map, tutorial)

### 4. Mechanics System (`mechanics/`)
Implements game physics and interactions.

**Main Components**:
- `movement_system.py` - Agent movement
- `collision_resolver.py` - Collision handling
- `interaction_handler.py` - Item interactions

**Responsibilities**:
- Move agents (up, down, left, right)
- Resolve collisions (walls, other agents)
- Execute interactions:
  - Pick up items from floor/counter/dispenser
  - Place items on floor/counter/cutting board
  - Process items (cutting)
  - Combine ingredients on plate
  - Deliver completed recipes

### 5. Rendering System (`rendering/`)
Visualizes the game state.

**Main Components**:
- `renderer.py` - ASCII and Pygame rendering

**Rendering Modes**:
- **ASCII** (`render_mode="human"`): Fast console rendering
- **Pygame** (`render_mode="rgb_array"`): Graphical window with HUD

## Complete Step Execution

```python
def step(actions):
    # 1. Process movements
    for agent, action in actions:
        if action < 4:  # Movement
            new_pos = movement_system.calculate_new_position(agent, action)
            if not collision_resolver.has_collision(new_pos):
                game_state.move_agent(agent, new_pos)
        
        elif action == 4:  # Interaction
            interaction_handler.execute_interaction(agent, game_state)
    
    # 2. Update game state
    game_state.update()
    
    # 3. Check completed recipes
    rewards = interaction_handler.check_deliveries(game_state)
    
    # 4. Generate observations
    observations = observation_builder.build(game_state)
    
    # 5. Check termination
    dones = game_state.check_termination()
    
    return observations, rewards, dones, truncations, infos
```

## Design Patterns

### Single Responsibility Principle
Each module has a single, well-defined responsibility:
- `GameState`: Only stores state
- `MovementSystem`: Only calculates positions
- `CollisionResolver`: Only validates positions
- `InteractionHandler`: Only executes interactions

### Dependency Injection
Systems receive their dependencies explicitly:
```python
observation_builder = ObservationBuilder(height, width)
interaction_handler = InteractionHandler(item_manager)
```

### Immutability (when possible)
Constants and configurations are immutable:
```python
CELL_SIZE = 40  # Never changes
MAP_SYMBOLS = {...}  # Frozen
```

## Testability

All systems are thoroughly tested:
- **Unit tests**: Each component in isolation
- **Integration tests**: Subsystem interaction
- **API tests**: PettingZoo compliance

```bash
# Run all tests
python -m pytest overcooked_marl_env/tests/

# Unit tests only
python -m pytest overcooked_marl_env/tests/unit/

# Integration tests
python -m pytest overcooked_marl_env/tests/integration/
```

## Spaces (Gymnasium)

### Observation Space
```python
Box(
    low=0, 
    high=255, 
    shape=(height, width, 9),
    dtype=np.uint8
)
```

**9 Channels**:
0. Static map (walls, counters, stations)
1. Floor items (coded numeric representation)
2. Current agent position (binary mask)
3. Other agents (binary mask)
4. Delivery stations
5. Active Recipe Orders queue (recipe order channel)
6. Items with 1/3 cuts
7. Items with 2/3 cuts
8. Items with 3/3 cuts (ready)

### Action Space
```python
Discrete(5)
```

0-3: Movement (up, down, left, right)
4: Interact

## Configuration

### Environment Parameters

```python
env = OvercookedMARL(
    render_mode="human",           # "human" or "rgb_array"
    map_path="path/to/map.txt",    # Custom map
    max_steps=1000,                # Episode max steps
    observation_mode="grid"        # Observation type
)
```

### Reward System

| Event | Reward |
|-------|---------|
| Pick up ingredient | +0.1 |
| Start cutting | +0.1 |
| Complete cut (1/3, 2/3) | +0.2 |
| Finish cutting (3/3) | +0.5 |
| Place on plate | +0.3 |
| Deliver SIMPLE_SALAD | +10.0 |
| Deliver GARDEN_SALAD | +15.0 |
| Deliver FRESH_SALAD | +20.0 |

## Extensibility

### Add New Recipe

1. Edit `core/recipes.py`:
```python
SUPER_SALAD = Recipe(
    name="SUPER_SALAD",
    ingredients={ItemType.LETTUCE, ItemType.TOMATO, 
                 ItemType.CUCUMBER, ItemType.CARROT},
    reward=25.0
)
```

2. Add to `RECIPES` list

### Add New Item Type

1. Edit `core/constants.py`:
```python
class ItemType(IntEnum):
    # ... existing types ...
    CARROT = 10
```

2. Edit `core/items.py`:
```python
ITEM_INFO = {
    # ...
    ItemType.CARROT: ItemInfo("Carrot", "🥕", (255, 165, 0))
}
```

### Create Custom Map

1. Create `.txt` file in `map/maps/`
2. Use valid symbols: `_|-#CIPD012`
3. Optionally add an adjacent `recipes.json` to configure new ingredients and custom recipes.
4. Load: `env = OvercookedMARL(map_path="...")`

## Performance

- **Observation building**: O(height × width × channels)
- **Collision check**: O(1) - grid lookup
- **Interaction**: O(1) - position-based
- **Recipe validation**: O(ingredients) - small constant

The environment is optimized for fast steps (~1000 FPS without rendering).

## See Also

- [Core Documentation](core/README.md) - Items and recipes system
- [Environment Documentation](environment/README.md) - Game state and observations
- [Map Documentation](map/README.md) - Map system
- [Mechanics Documentation](mechanics/README.md) - Physics and interactions
- [Rendering Documentation](rendering/README.md) - Visualization
