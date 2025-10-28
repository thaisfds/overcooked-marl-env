# Overcooked MARL Environment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.24.0+-green.svg)](https://pettingzoo.farama.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1+-orange.svg)](https://gymnasium.farama.org/)

A cooperative Multi-Agent Reinforcement Learning (MARL) environment based on the Overcooked game, where agents work together in a kitchen to prepare and deliver recipes.

## What is this environment?

This is a PettingZoo/Gymnasium environment where **2 agents** must cooperate to:

1. **Collect ingredients** from dispensers
2. **Process ingredients** on cutting boards (3 cuts required)
3. **Combine ingredients** on plates
4. **Deliver complete recipes** at delivery stations

Agents receive **shared rewards** when delivering recipes, encouraging cooperation.

### Why use this environment?

- Fully PettingZoo compatible - works with popular MARL frameworks
- Structured observations - multi-channel tensor with complete information
- Dense reward system - feedback on every important action
- Customizable maps - create your own layouts in text format
- Visual rendering - fast ASCII or Pygame with complete HUD

## Installation

```bash
# Clone the repository
git clone https://github.com/thaisfds/NewOvercookedEnv.git
cd NewOvercookedEnv

# Install in development mode
pip install -e .
```

**Dependencies**: gymnasium, pettingzoo, numpy, pygame, matplotlib

## How to use

### Basic example

```python
from overcooked_marl_env import OvercookedMARL

# Create the environment
env = OvercookedMARL(render_mode="human")

# Reset
observations, infos = env.reset()

# Episode loop
for step in range(100):
    # Choose actions (random here)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    
    # Execute step
    obs, rewards, dones, truncs, infos = env.step(actions)
    
    # Render
    env.render()
    
    if any(dones.values()):
        break

env.close()
```

### Run examples

```bash
# Small map (7x7) with scripted actions
python example/default_map_test.py

# Large map (15x11) with complete demonstration
python example/tutorial_map_test.py
```

## Actions and Observations

### Action Space

`Discrete(5)` - 5 possible actions:
- **0**: Move up
- **1**: Move down  
- **2**: Move left
- **3**: Move right
- **4**: Interact (pick/place/process/combine/deliver)

### Observation Space

`Box(height, width, 8)` - Multi-channel tensor:
- **Channel 0**: Static map (walls, stations)
- **Channel 1**: Items on floor
- **Channel 2**: Current agent position
- **Channel 3**: Other agents positions
- **Channel 4**: Delivery stations
- **Channels 5-7**: Processing state (1/3, 2/3, 3/3 cuts)

## Available Recipes

| Recipe | Ingredients | Reward |
|---------|--------------|------------|
| **SIMPLE_SALAD** | Lettuce (processed) | 10.0 |
| **GARDEN_SALAD** | Lettuce + Tomato (processed) | 15.0 |
| **FRESH_SALAD** | Lettuce + Tomato + Cucumber (processed) | 20.0 |

## Create Custom Maps

Create a `.txt` file in `overcooked_marl_env/map/maps/`:

```
_______
|C  # |
|0   1|
|#IPD#|
-------
```

**Legend**:
- `_|-`: Walls
- `#`: Counter (place/pick items)
- `C`: Cutting board
- `I`: Ingredient dispenser
- `P`: Plate dispenser
- `D`: Delivery station
- `0,1`: Agent starting positions

Use in code:

```python
env = OvercookedMARL(map_path="overcooked_marl_env/map/maps/your_map.txt")
```

## Rendering

### ASCII Mode (`render_mode="human"`)
Fast console rendering with symbols.

### Pygame Mode (`render_mode="rgb_array"`)
Graphical window with:
- Colored grid
- HUD with agent information
- List of pending orders
- Delivered recipes counter

## Tests

```bash
# All tests (205+ tests)
python -m pytest overcooked_marl_env/tests/

# Specific test
python -m pytest overcooked_marl_env/tests/test_pettingzoo_api.py
```

## Project Structure

```
NewOvercookedEnv/
├── overcooked_marl_env/     # Main package
│   ├── core/                # Items and recipes system
│   ├── environment/         # Game state and observations
│   ├── map/                 # Map system
│   ├── mechanics/           # Movement, collision, interaction
│   ├── rendering/           # ASCII and Pygame rendering
│   └── tests/               # Unit and integration tests
├── example/                 # Example scripts
└── docs/                    # Additional documentation
```

Detailed documentation: See READMEs in each package subfolder.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a branch for your feature
3. Add tests
4. Submit a Pull Request

## License

MIT License - see LICENSE file for details.

## Contact

- **Author**: thaisfds
- **Repository**: [github.com/thaisfds/NewOvercookedEnv](https://github.com/thaisfds/NewOvercookedEnv)
- **Issues**: [GitHub Issues](https://github.com/thaisfds/NewOvercookedEnv/issues)

---

Happy Cooking!