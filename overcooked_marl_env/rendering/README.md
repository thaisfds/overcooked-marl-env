# Rendering System - ASCII and Pygame Visualization

The `rendering` module provides dual visualization: fast console-based ASCII output and a rich Pygame graphical window.

## Sub-package Contents

```
rendering/
├── __init__.py      # Module exports
├── render_config.py # Pygame color themes, FPS, and grid layout settings
└── renderer.py      # Main renderer class
```

---

## Detailed Components

### `renderer.py`
Provides high-level rendering capabilities.

#### Renderer Class
* `__init__(item_manager, render_mode: Optional[str] = "human")`
  Instantiates the renderer.
  - `item_manager`: Reference to resolve tile names and codes.
  - `render_mode`: `"human"` (Pygame window) or `"ascii"` (console printout).

#### Render Invocation
* `render(game_state: dict, map_size: tuple, agents: list, recipe_order: list)`
  Redraws the frame based on the current state.
* `close()`
  Terminates display buffers and shuts down Pygame.

---

## Rendering Modes

### 1. ASCII Mode (`render_mode="ascii"`)
Fast and lightweight printout, perfect for remote execution or headless server cluster training.
* **Layout details:** Renders border walls (`_`, `|`), tables (`#`), dispensers, agents, and lowercase characters for chopped ingredients (`t`, `l`, `c`, `o`).
* Prints active steps, score metrics, and expected recipe orders.

### 2. Pygame Mode (`render_mode="human"`)
Interactive visual mode:
* Draws a structured color-coded grid representing the kitchen counters, tables, and walls.
* Renders agents as colored circles showing their current orientation direction.
* Shows held item icons (e.g. plate container showing ingredients inside).
* Renders a dedicated HUD displaying step count, individual rewards, and active expected orders.

---

## Verification & Testing

Verify visual outputs:
```bash
python -m unittest overcooked_marl_env/tests/test_renderer.py
```
To visually test ASCII output, run:
```bash
python example/default_map_test.py
```
To test Pygame graphical interactive gameplay, run:
```bash
python example/keyboard_play.py
```
