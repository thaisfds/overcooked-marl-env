# Map System - Loading and Validation

The `map` module handles loading, parsing, and validating maps from text files.

## Files

```
map/
├── __init__.py          # Module exports
├── map_parser.py        # Text to grid conversion
├── map_validator.py     # Structure validation
├── map_loader.py        # Complete map loading
└── maps/                # Map files
    ├── default_map.txt  # Small 7×7 map
    └── tutorial.txt     # Large 15×11 map
```

## Responsibilities

- Read `.txt` map files
- Parse text to grid representation
- Validate map structure and rules
- Extract agent starting positions
- Provide default maps

## Detailed Files

### `map_parser.py`

Converts text files to grid arrays.

#### MapParser

```python
class MapParser:
    def parse(self, map_text: str) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
        # Returns (grid, agent_positions)
```

**Input**: Text map
```
_______
|C  # |
|0   1|
|#IPD#|
-------
```

**Output**: 
- `grid`: numpy array with TileType values
- `agent_positions`: `{"agent_0": (2, 1), "agent_1": (2, 5)}`

#### Symbol Mapping

```python
SYMBOL_TO_TILE = {
    ' ': TileType.EMPTY,
    '_': TileType.WALL,
    '-': TileType.WALL,
    '|': TileType.WALL,
    '#': TileType.COUNTER,
    'C': TileType.CUTTING_BOARD,
    'I': TileType.INGREDIENT_DISPENSER,
    'P': TileType.PLATE_DISPENSER,
    'D': TileType.DELIVERY_STATION,
    '0': TileType.EMPTY,  # Agent 0 start
    '1': TileType.EMPTY,  # Agent 1 start
}
```

**Processing**:
1. Read line by line
2. Convert each symbol to TileType
3. Extract agent positions ('0', '1')
4. Build grid and position dict

### `map_validator.py`

Validates map structure.

#### MapValidator

```python
class MapValidator:
    def validate(self, grid: np.ndarray, agent_positions: Dict) -> bool:
        # Raises ValueError if invalid
```

**Validation Rules**:

1. **Exactly 2 agents**: Must have '0' and '1' positions
2. **At least 1 cutting board**: Must have 'C'
3. **At least 1 ingredient dispenser**: Must have 'I'
4. **At least 1 plate dispenser**: Must have 'P'
5. **At least 1 delivery station**: Must have 'D'
6. **Surrounded by walls**: Perimeter must be walls
7. **Valid agent positions**: Agents start on EMPTY tiles

**Example errors**:
```python
# Missing agent
ValueError: "Map must have exactly 2 agent starting positions"

# Missing station
ValueError: "Map must have at least one cutting board"

# Agent on wall
ValueError: "Agent starting position must be on empty tile"
```

### `map_loader.py`

Main interface for loading maps.

#### MapLoader

```python
class MapLoader:
    def __init__(self):
        self.parser = MapParser()
        self.validator = MapValidator()
    
    def load(self, map_path: str) -> MapData:
        # Returns complete MapData
```

**Usage**:
```python
loader = MapLoader()
map_data = loader.load("overcooked_marl_env/map/maps/default_map.txt")

print(map_data.height)  # 7
print(map_data.width)   # 7
print(map_data.agent_positions)  # {"agent_0": (2, 1), "agent_1": (2, 5)}
```

**Process**:
1. Read file
2. Parse to grid + positions
3. Validate structure
4. Extract station positions
5. Return MapData

**Default Maps**:
```python
# Load default small map
map_data = loader.load_default()  # Uses default_map.txt

# Load tutorial map
map_data = loader.load("overcooked_marl_env/map/maps/tutorial.txt")
```

## Included Maps

### default_map.txt (7×7)

```
_______
|C  # |
|0   1|
|#IPD#|
-------
```

**Features**:
- Small and simple
- 1 cutting board (top-left)
- 1 counter (top-right)
- Dispensers at bottom
- Agent starts: (2, 1) and (2, 5)

**Good for**: Quick tests, basic training

### tutorial.txt (15×11)

```
_______________
|C    #    #  |
|#    #    C  |
|             |
|0           1|
|             |
|#    #    #  |
|C    #    C  |
|#           #|
|# I  P  D  #|
---------------
```

**Features**:
- Large open space
- 4 cutting boards
- Multiple counters
- More complex navigation

**Good for**: Advanced scenarios, multi-step cooperation

## Creating Custom Maps

### Basic Structure

```
_______________  ← Top wall
|             |  ← Side walls
|  C   #   P  |  ← Stations
|  0      1   |  ← Agents
|  #  I  D  # |  ← More stations
---------------  ← Bottom wall
```

### Required Elements

✓ **Exactly 2 agents** ('0' and '1')
✓ **At least 1 cutting board** ('C')
✓ **At least 1 ingredient dispenser** ('I')
✓ **At least 1 plate dispenser** ('P')
✓ **At least 1 delivery station** ('D')
✓ **Complete wall perimeter** ('_', '-', '|')

### Optional Elements

- **Counters** ('#'): Place/store items temporarily
- **Empty space** (' '): Movement area
- **Multiple stations**: Can have many of each type

### Example: Minimal Map

```
_____
|CIP|
|0 1|
|# D|
-----
```

**Valid**: Has all required elements!

### Example: Complex Map

```
_________
|C # # C|
|I      |
|  0 1  |
|    P  |
|# D # D|
---------
```

**Valid**: Multiple stations, open layout!

### Common Mistakes

❌ **Missing agent**
```
_____
|C 1|  ← Only agent 1, need 0 too!
|IPD|
-----
```

❌ **No walls**
```
C # P
0   1  ← No perimeter walls!
I D #
```

❌ **Agent on wall**
```
_____
|CIP|
|0-1|  ← Agent 1 on wall!
|# D|
-----
```

❌ **Missing station**
```
_____
|C P|  ← No ingredient dispenser!
|0 1|
|# D|
-----
```

## Map Loading in Code

### Load default map

```python
from overcooked_marl_env import OvercookedMARL

env = OvercookedMARL()  # Uses default_map.txt
```

### Load custom map

```python
env = OvercookedMARL(
    map_path="overcooked_marl_env/map/maps/tutorial.txt"
)
```

### Load your own map

```python
# 1. Create mymap.txt in maps/
# 2. Load it
env = OvercookedMARL(
    map_path="overcooked_marl_env/map/maps/mymap.txt"
)
```

## Error Handling

```python
try:
    map_data = loader.load("invalid_map.txt")
except FileNotFoundError:
    print("Map file not found")
except ValueError as e:
    print(f"Invalid map: {e}")
```

## Tests

```bash
# Test map parsing
python -m pytest overcooked_marl_env/tests/unit/test_map_parser.py

# Test map validation
python -m pytest overcooked_marl_env/tests/unit/test_map_validator.py

# Test map loading
python -m pytest overcooked_marl_env/tests/unit/test_map_loader.py

# Integration test
python -m pytest overcooked_marl_env/tests/integration/test_map_system.py
```

## See Also

- [Environment Documentation](../environment/README.md) - How maps are used
- [Core Documentation](../core/README.md) - Tile types
- [Main Package Documentation](../README.md) - Complete architecture
