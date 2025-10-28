# Rendering System - ASCII and Pygame Visualization

The `rendering` module provides two rendering modes: fast ASCII console output and graphical Pygame window with HUD.

## Files

```
rendering/
├── __init__.py      # Module exports
└── renderer.py      # ASCII and Pygame rendering
```

## Responsibilities

- Render game state visually
- Support ASCII mode (console)
- Support Pygame mode (graphical window)
- Display HUD with agent information
- Show pending orders and score
- Clean up resources properly

## Detailed File

### `renderer.py`

Complete rendering system.

#### Renderer Class

```python
class Renderer:
    def __init__(self, 
                 height: int, 
                 width: int, 
                 render_mode: str = "human"):
        self.height = height
        self.width = width
        self.render_mode = render_mode
        self.screen = None  # For Pygame
```

**Modes**:
- `"human"`: ASCII console rendering
- `"rgb_array"`: Pygame graphical window

#### Main Methods

**`render(game_state: GameState) -> Optional[np.ndarray]`**

Renders current game state:

```python
renderer = Renderer(7, 7, render_mode="human")
renderer.render(game_state)
# Prints ASCII to console

renderer_rgb = Renderer(7, 7, render_mode="rgb_array")
rgb_array = renderer_rgb.render(game_state)
# Returns np.ndarray with RGB image
```

**`close()`**

Cleans up resources:

```python
renderer.close()
# Closes Pygame window if open
```

## ASCII Mode

### How it works

1. Creates text representation of grid
2. Uses symbols for different elements
3. Prints to console with colors (if terminal supports)

### Symbol Mapping

| Element | Symbol | Description |
|---------|--------|-------------|
| Empty | `.` | Floor space |
| Wall | `█` | Wall tile |
| Counter | `#` | Counter |
| Cutting Board | `C` | Cutting board |
| Ingredient Disp. | `I` | Ingredient dispenser |
| Plate Dispenser | `P` | Plate dispenser |
| Delivery Station | `D` | Delivery station |
| Agent 0 | `0` | First agent |
| Agent 1 | `1` | Second agent |
| Lettuce | `L` | Lettuce item |
| Tomato | `T` | Tomato item |
| Cucumber | `C` | Cucumber item |
| Plate | `p` | Plate |
| Chopped items | `l/t/c` | Lowercase = chopped |

### Example ASCII Output

```
═══════════════════════════════════
Episode Step: 42 | Score: 15.0
Delivered: 1 | Pending Orders: 2
═══════════════════════════════════
█████████
█C  #  █
█       █
█0     1█
█ #IPD #█
█████████

Agent 0 (Blue): Holding Lettuce [Raw]
Agent 1 (Red): Empty hands

Pending Orders:
1. SIMPLE_SALAD (Lettuce) - Reward: 10.0
2. GARDEN_SALAD (Lettuce + Tomato) - Reward: 15.0
═══════════════════════════════════
```

### Performance

- **Very fast**: ~10,000+ FPS
- **No dependencies**: Works anywhere Python runs
- **Good for**: Training, debugging, headless servers

## Pygame Mode

### How it works

1. Creates Pygame window
2. Draws colored grid
3. Renders items as colored rectangles
4. Displays HUD below grid
5. Returns RGB array

### Visual Elements

#### Grid Rendering

Each tile is a 40×40 pixel colored rectangle:

**Colors**:
- Empty: White (255, 255, 255)
- Wall: Dark gray (64, 64, 64)
- Counter: Light gray (192, 192, 192)
- Cutting Board: Brown (139, 69, 19)
- Ingredient Dispenser: Light green (144, 238, 144)
- Plate Dispenser: Light blue (173, 216, 230)
- Delivery Station: Gold (255, 215, 0)

**Items on grid**:
- Lettuce: Green (0, 255, 0)
- Tomato: Red (255, 0, 0)
- Cucumber: Dark green (0, 128, 0)
- Plate: Light gray (200, 200, 200)
- Chopped items: Darker variants

**Agents**:
- Agent 0: Blue circle
- Agent 1: Red circle

#### HUD (Heads-Up Display)

Located below the grid, displays 5 lines:

**Line 1**: Episode info
```
Step: 42 | Score: 15.0 | Delivered: 1
```

**Line 2**: Agent 0 status
```
Agent 0 (Blue): Holding LETTUCE [CHOPPED_2]
```

**Line 3**: Agent 1 status
```
Agent 1 (Red): Empty hands
```

**Line 4**: Pending orders header
```
Pending Orders (2):
```

**Line 5+**: Order list
```
- SIMPLE_SALAD (reward: 10.0)
- GARDEN_SALAD (reward: 15.0)
```

### Window Configuration

```python
CELL_SIZE = 40  # pixels per grid cell
HUD_HEIGHT = 200  # pixels for HUD area
FONT_SIZE = 16  # text size

# Total window size
width_pixels = width * CELL_SIZE
height_pixels = height * CELL_SIZE + HUD_HEIGHT
```

For 7×7 map:
- Grid: 280×280 pixels
- HUD: 280×200 pixels
- Total: 280×480 pixels

### Performance

- **Moderate speed**: ~60 FPS (limited by display)
- **Dependencies**: Requires pygame
- **Good for**: Visualization, debugging, demos

## Resource Cleanup

Always close renderer when done:

```python
renderer = Renderer(7, 7, render_mode="rgb_array")

try:
    for step in range(100):
        renderer.render(game_state)
finally:
    renderer.close()  # Ensures Pygame window closes
```

Or use environment's close:

```python
env = OvercookedMARL(render_mode="rgb_array")
try:
    # ... use environment ...
finally:
    env.close()  # Calls renderer.close() internally
```

## Usage in Environment

The environment handles rendering automatically:

```python
# ASCII mode
env = OvercookedMARL(render_mode="human")
env.reset()
env.render()  # Prints to console

# Pygame mode
env = OvercookedMARL(render_mode="rgb_array")
env.reset()
rgb = env.render()  # Returns numpy array

# No rendering
env = OvercookedMARL(render_mode=None)
# render() does nothing
```

## Comparison of Modes

| Feature | ASCII | Pygame |
|---------|-------|--------|
| **Speed** | Very fast (~10k FPS) | Moderate (~60 FPS) |
| **Dependencies** | None | pygame |
| **Visual Quality** | Low (text) | High (graphics) |
| **HUD** | Yes (text) | Yes (rendered) |
| **Headless Support** | Yes | Limited |
| **Good For** | Training, servers | Demos, debugging |
| **Output** | None (prints) | RGB array |

## Tests

```bash
# Test renderer
python -m pytest overcooked_marl_env/tests/test_renderer.py

# Test with actual environment
python example/default_map_test.py  # ASCII mode
```

## Customization

### Change Colors

Edit the `COLORS` dictionary in `renderer.py`:

```python
COLORS = {
    TileType.EMPTY: (255, 255, 255),      # White floor
    TileType.WALL: (64, 64, 64),          # Dark walls
    ItemType.LETTUCE: (0, 255, 0),        # Green lettuce
    # ... customize as needed
}
```

### Change Cell Size

```python
CELL_SIZE = 60  # Larger cells
```

### Change HUD Height

```python
HUD_HEIGHT = 300  # More space for info
```

### Add Custom Rendering

Extend the `Renderer` class:

```python
class CustomRenderer(Renderer):
    def render(self, game_state):
        # Call parent
        rgb = super().render(game_state)
        
        # Add custom drawing
        # ... your code ...
        
        return rgb
```

## Troubleshooting

### Problem: Pygame doesn't initialize
**Solution**: Check installation with `pip install pygame>=2.5.0`

### Problem: ASCII colors not showing
**Solution**: Terminal may not support colors. Try different terminal or disable colors.

### Problem: Window doesn't close
**Solution**: Always call `env.close()` or `renderer.close()`

### Problem: Slow rendering
**Solution**: Use ASCII mode for training, Pygame only for visualization

## See Also

- [Core Documentation](../core/README.md) - Item symbols and colors
- [Environment Documentation](../environment/README.md) - Game state
- [Main Package Documentation](../README.md) - Complete integration
