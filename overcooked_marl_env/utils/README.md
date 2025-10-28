# Utils Module - Utility Functions

The `utils` module contains utility functions and classes that can be used across different components of the environment.

## Files

```
utils/
└── __init__.py     # Module initialization
```

## Status

Currently, this module is empty and reserved for future utilities that can be shared between different environment components.

## Possible Future Additions

### Validation Functions
```python
def validate_action(action: int) -> bool:
    """Validate if action is within valid range (0-4)."""
    return 0 <= action <= 4

def validate_position(pos: Tuple[int, int], height: int, width: int) -> bool:
    """Validate if position is within map boundaries."""
    row, col = pos
    return 0 <= row < height and 0 <= col < width
```

### Conversion Functions
```python
def position_to_index(pos: Tuple[int, int], width: int) -> int:
    """Convert 2D position to 1D index."""
    row, col = pos
    return row * width + col

def index_to_position(index: int, width: int) -> Tuple[int, int]:
    """Convert 1D index to 2D position."""
    return (index // width, index % width)
```

### Distance Functions
```python
def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two positions."""
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
```

### Observation Processing
```python
def normalize_observation(obs: np.ndarray) -> np.ndarray:
    """Normalize observation values to [0, 1] range."""
    return obs / obs.max()

def flatten_observation(obs: np.ndarray) -> np.ndarray:
    """Flatten multi-channel observation to 1D vector."""
    return obs.flatten()
```

### Logging and Debugging
```python
def log_episode_stats(episode: int, score: float, steps: int):
    """Log episode statistics."""
    print(f"Episode {episode}: Score={score}, Steps={steps}")

def print_game_state_summary(game_state: GameState):
    """Print summary of current game state."""
    print(f"Agents: {game_state.agent_positions}")
    print(f"Score: {game_state.score}")
    print(f"Items: {len(game_state.item_manager.items)}")
```

### Configuration Helpers
```python
def load_config(config_path: str) -> Dict:
    """Load configuration from JSON/YAML file."""
    # Implementation here
    pass

def validate_config(config: Dict) -> bool:
    """Validate configuration parameters."""
    # Implementation here
    pass
```

## How to Contribute

If you implement utility functions that could be useful for other components:

1. Add the function to this module
2. Add tests in `tests/unit/test_utils.py`
3. Document here in this README
4. Follow coding standards and type hints

## Example Usage

Once utilities are added, they can be imported:

```python
from overcooked_marl_env.utils import validate_action, manhattan_distance

# Validate action
if validate_action(action):
    env.step({agent: action})

# Calculate distance
dist = manhattan_distance(agent_pos, target_pos)
if dist <= 1:
    print("Target is adjacent!")
```

## Design Guidelines

Utilities should be:
- **Pure functions** when possible (no side effects)
- **Well-documented** with docstrings
- **Type-hinted** for clarity
- **Tested** with unit tests
- **General-purpose** (not too specific to one component)

## See Also

- [Core Documentation](../core/README.md) - Items and constants system
- [Environment Documentation](../environment/README.md) - Game state
- [Main Package Documentation](../README.md) - Complete overview
