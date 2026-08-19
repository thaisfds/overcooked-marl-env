# Map System - Loading, Parsing, and Validation

The `map` module handles loading, parsing, and validating kitchen layouts from text configurations.

## Sub-package Contents

```
map/
├── __init__.py          # Module exports
├── map_config.py        # Map headers and validation dimensions constants
├── map_parser.py        # Converts plain text maps into structured grids
├── map_validator.py     # Structural validation (perimeter wall, agents spawn)
├── map_loader.py        # Facade loader that coordinates parsing & validation
└── maps/                # Pre-defined map layout text files
```

---

## Component Architecture

The module utilizes the **Facade** design pattern through `MapLoader`, shielding the caller from details about parsing and validation.

### 1. Parser (`map_parser.py`)
Parses `.txt` files containing `# Grid Map`, `# Dispensers`, and `# Recipes` sections:
* `parse_file(map_path: str) -> dict`
  Returns parsed data with grid array, agent starting locations, dispenser assignments, and recipe requests list.

### 2. Validator (`map_validator.py`)
Validates structural integrity and constraints:
* `validate_map(map_data: dict) -> bool`
  Returns `True` if map is valid, or prints validation errors.
* **Rules checked:** Perimeter surrounded by walls, exactly 2 agents, agents spawn on empty walkable tiles, existence of cutting boards (`C`), plate dispensers (`P`), delivery stations (`D`), and valid recipe names.

### 3. Facade Loader (`map_loader.py`)
Coordinates parsing and validation:
* `load_map(map_path: Optional[str] = None) -> MapData`
  Loads map, resolving default fallback maps if the file fails or doesn't exist.

---

## 🔍 Command-Line Map Validation Utility

We added a terminal script capability directly inside `map_loader.py`. You can validate any map file and see visual error reports from your shell:

```bash
python -m overcooked_marl_env.map.map_loader example/custom_map_template.txt
```

> [!TIP]
> If a `recipes.json` config exists in the same directory as the map file, it is automatically parsed to register custom ingredients/recipes before running validation.

---

## Included Maps
* **`default_map.txt` (7×7):** Small kitchen, ideal for debugging and quick RL training.
* **`tutorial.txt` (15×11):** Large kitchen with multiple cutting boards and open walkways for complex coordination.
* **`simple_map.txt` (7×7):** Simple layout containing essential preparation counters.

---

## Verification & Testing

Run map subsystem unit tests:
```bash
python -m unittest overcooked_marl_env/tests/unit/test_map_parser.py
python -m unittest overcooked_marl_env/tests/unit/test_map_validator.py
python -m unittest overcooked_marl_env/tests/unit/test_map_loader.py
```
