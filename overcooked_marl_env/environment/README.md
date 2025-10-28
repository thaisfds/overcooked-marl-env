# Environment System - Game State and Observations

The `environment` module manages the complete game state and generates observations for agents.

## Files

```
environment/
├── __init__.py              # Module exports
├── game_state.py            # Complete game state
└── observation_builder.py   # Observation tensor generation
```

## Responsibilities

- Maintain agent positions and inventory
- Track items on map (floor, counters, cutting boards)
- Generate multi-channel observations (8 channels)
- Manage pending orders (recipes to deliver)
- Track score and delivered recipes

## Detailed Files

### `game_state.py`

Contains complete game state.

#### MapData

```python
@dataclass
class MapData:
    grid: np.ndarray                    # Static map (walls, stations)
    height: int
    width: int
    agent_positions: Dict[str, Tuple[int, int]]
    item_positions: Dict[Tuple[int, int], int]  # pos -> item_id
    cutting_board_positions: Set[Tuple[int, int]]
    delivery_positions: Set[Tuple[int, int]]
```

**Usage**: Stores static map information.

#### GameState

```python
class GameState:
    def __init__(self, map_data: MapData):
        self.map_data = map_data
        self.agent_inventory: Dict[str, Optional[int]] = {}
        self.item_manager = ItemInstanceManager()
        self.pending_orders: List[Recipe] = []
        self.score: float = 0.0
        self.delivered_count: int = 0
        self.step_count: int = 0
```

**Main Attributes**:
- `map_data`: Map structure and static positions
- `agent_inventory`: What each agent is holding (item_id or None)
- `item_manager`: Manages all item instances
- `pending_orders`: Active recipe orders
- `score`: Accumulated points
- `delivered_count`: Number of completed recipes
- `step_count`: Episode steps

**Key Methods**:

**`get_agent_position(agent: str) -> Tuple[int, int]`**
```python
pos = game_state.get_agent_position("agent_0")
# Returns (3, 4)
```

**`move_agent(agent: str, new_pos: Tuple[int, int])`**
```python
game_state.move_agent("agent_0", (3, 5))
# Moves agent to position (3, 5)
```

**`get_agent_inventory(agent: str) -> Optional[int]`**
```python
item_id = game_state.get_agent_inventory("agent_0")
# Returns item_id if holding something, None otherwise
```

**`set_agent_inventory(agent: str, item_id: Optional[int])`**
```python
game_state.set_agent_inventory("agent_0", item_id)
# Agent now holds this item
```

**`get_item_at_position(pos: Tuple[int, int]) -> Optional[int]`**
```python
item_id = game_state.get_item_at_position((5, 6))
# Returns item_id if there's an item at position, None otherwise
```

**`place_item_at_position(pos: Tuple[int, int], item_id: int)`**
```python
game_state.place_item_at_position((5, 6), item_id)
# Places item at this position
```

**`remove_item_at_position(pos: Tuple[int, int]) -> Optional[int]`**
```python
item_id = game_state.remove_item_at_position((5, 6))
# Removes and returns item from this position
```

**`add_score(points: float)`**
```python
game_state.add_score(15.0)
# Adds points to score
```

**`add_pending_order(recipe: Recipe)`**
```python
game_state.add_pending_order(GARDEN_SALAD)
# Adds new order to be fulfilled
```

**`remove_pending_order(recipe: Recipe)`**
```python
game_state.remove_pending_order(GARDEN_SALAD)
# Removes order when delivered
```

### `observation_builder.py`

Generates observation tensors.

#### ObservationBuilder

```python
class ObservationBuilder:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
```

**Main Method**:

**`build(game_state: GameState, agent: str) -> np.ndarray`**

Generates 8-channel observation for an agent.

```python
obs_builder = ObservationBuilder(7, 7)
observation = obs_builder.build(game_state, "agent_0")
# Returns: np.ndarray with shape (7, 7, 8)
```

## The 8 Observation Channels

### Channel 0: Static Map

Encodes the map structure:
- `TileType.EMPTY` (0): Empty floor
- `TileType.WALL` (1): Wall
- `TileType.COUNTER` (2): Counter
- `TileType.CUTTING_BOARD` (3): Cutting board
- `TileType.INGREDIENT_DISPENSER` (4): Ingredient dispenser
- `TileType.PLATE_DISPENSER` (5): Plate dispenser
- `TileType.DELIVERY_STATION` (6): Delivery station

**Purpose**: Agent knows where walls and stations are.

### Channel 1: Floor Items

Shows items on the ground or counters:
- `ItemType.EMPTY` (0): No item
- `ItemType.LETTUCE` (1): Lettuce
- `ItemType.CHOPPED_LETTUCE` (5): Chopped lettuce
- etc.

**Purpose**: Agent sees available items to pick up.

### Channel 2: Current Agent Position

Binary channel (0 or 1):
- `1` at the current agent's position
- `0` everywhere else

**Purpose**: Agent knows where it is.

### Channel 3: Other Agents

Binary channel (0 or 1):
- `1` at positions of other agents
- `0` everywhere else

**Purpose**: Agent sees teammates and avoids collisions.

### Channel 4: Delivery Stations

Binary channel (0 or 1):
- `1` at delivery station positions
- `0` everywhere else

**Purpose**: Agent knows where to deliver recipes.

### Channel 5: Items with 1/3 Cuts

Binary channel (0 or 1):
- `1` where items have `ItemState.CHOPPED_1`
- `0` everywhere else

**Purpose**: Agent sees processing progress.

### Channel 6: Items with 2/3 Cuts

Binary channel (0 or 1):
- `1` where items have `ItemState.CHOPPED_2`
- `0` everywhere else

**Purpose**: Agent sees almost-ready items.

### Channel 7: Items with 3/3 Cuts (Ready)

Binary channel (0 or 1):
- `1` where items have `ItemState.CHOPPED_3`
- `0` everywhere else

**Purpose**: Agent identifies fully processed ingredients.

## Observation Building Flow

```
1. Create empty observation tensor (H × W × 8)
   ↓
2. Fill Channel 0 with static map
   ↓
3. Fill Channel 1 with floor items
   ↓
4. Mark current agent position in Channel 2
   ↓
5. Mark other agents in Channel 3
   ↓
6. Mark delivery stations in Channel 4
   ↓
7. Mark items by processing state in Channels 5-7
   ↓
8. Return complete observation
```

## Example Observation

For a 7×7 map with agent at (3, 3):

```python
observation[3, 3, 2] = 1  # Current agent here
observation[5, 5, 3] = 1  # Other agent here
observation[2, 4, 1] = ItemType.LETTUCE  # Lettuce on floor
observation[4, 2, 0] = TileType.CUTTING_BOARD  # Cutting board
observation[2, 4, 5] = 1  # Item with 1 cut at (2, 4)
```

## Design Considerations

### Why 8 channels?

- **Channel 0**: Static environment structure
- **Channels 1-4**: Dynamic element positions
- **Channels 5-7**: Processing state information

This separation allows neural networks to easily learn:
- Where things are (spatial)
- What things are (categorical)
- What state things are in (progression)

### Why egocentric observation?

Each agent receives their own observation with Channel 2 marking their position. This makes learning easier as agents don't need to identify themselves.

### Observation space size

For a 7×7 map:
- Shape: `(7, 7, 8)`
- Total elements: `7 × 7 × 8 = 392`
- Memory: ~392 bytes (uint8)

Very compact and efficient!

## Tests

```bash
# Test game state
python -m pytest overcooked_marl_env/tests/unit/test_game_state.py

# Test observation building
python -m pytest overcooked_marl_env/tests/unit/test_observation_builder.py
```

## See Also

- [Core Documentation](../core/README.md) - Item types and states
- [Map Documentation](../map/README.md) - Map structure
- [Mechanics Documentation](../mechanics/README.md) - How state changes
- [Main Package Documentation](../README.md) - Complete architecture
