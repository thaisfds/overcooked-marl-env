<p align="center">
  <img src="./experiment_results/assets/banner_header.png" alt="Overcooked MARL Environment" width="100%" />
</p>

# Overcooked MARL Environment

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/downloads/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.24.0+-4b5563.svg)](https://pettingzoo.farama.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1+-4b5563.svg)](https://gymnasium.farama.org/)

A cooperative Multi-Agent Reinforcement Learning (MARL) environment inspired by Overcooked, developed as part of an undergraduate thesis (DCC/UFMG). Agents work together in a grid-based kitchen to collect ingredients, chop them, assemble plates, and deliver recipes in order.

Experiment results, learning curves, and algorithm comparisons (`PPO`, `DQN`, `IMPALA`, `APPO`, `BC`) are organized in [`experiment_results/`](./experiment_results/README.md).

<table>
  <tr>
    <th align="center">2-Agent Cooperation (Divided Kitchen)</th>
    <th align="center">4-Agent Team (Divided Kitchen)</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="./experiment_results/assets/gameplay_2_agentes_cooperacao.gif" alt="2-Agent Cooperative Gameplay" width="100%" />
    </td>
    <td align="center" width="50%">
      <img src="./experiment_results/assets/gameplay_4_agentes_equipe.gif" alt="4-Agent Cooperative Gameplay" width="100%" />
    </td>
  </tr>
</table>

---

## Overview

In this PettingZoo/Gymnasium environment, agents must cooperate to:

1. **Collect ingredients** (tomato, lettuce, cucumber, onion, etc.) from dispensers.
2. **Chop ingredients** on cutting boards (requiring multiple chop interactions).
3. **Assemble dishes** by combining chopped ingredients on clean plates.
4. **Deliver completed orders** at delivery stations following the order queue.

Agents receive shared rewards upon successful deliveries, encouraging coordination strategies such as division of labor, passing items across counters, and parallel preparation.

<div align="center">
  <img src="./experiment_results/01_visao_geral_e_evolucao/evolucao_ambiente_poc1_vs_poc2.png" alt="Environment Evolution" width="90%" />
</div>

---

## Installation

### 1. System packages (Linux/Ubuntu)
```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv python3-full
```

### 2. Virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Dependencies and editable install
```bash
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### 1. Interactive Playtesting (Keyboard)
```bash
python example/keyboard_play.py
```
* **Agent 0 (Blue):** Move with `W`, `A`, `S`, `D` | Interact with `Spacebar`
* **Agent 1 (Orange):** Move with Arrow Keys | Interact with `Enter`
* **Controls:** `R` to reset, `Esc` to quit.

### 2. Scripted Cooperation Tests
```bash
python example/default_map_test.py
python example/tutorial_map_test.py
```

### 3. Basic PettingZoo / Gymnasium Loop
```python
from overcooked_marl_env import OvercookedMARL

env = OvercookedMARL(render_mode="human")
observations, infos = env.reset()

for step in range(500):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    env.render()

    if any(terminations.values()) or any(truncations.values()):
        break

env.close()
```

---

## Custom Maps & Recipes

Maps, ingredients, and recipes can be defined via configuration files without modifying Python source code.

### 1. Custom Maps
Create a `.txt` layout file (e.g., `my_kitchen.txt`):
```text
# Grid Map
_________
|#C#P#D#|
|# 0   #|
|#   1 #|
|#I#I#I#|
_________

# Dispensers (name x y)
tomato 1 4
lettuce 3 4
onion 5 4

# Recipes
SIMPLE_SALAD, ONION_SOUP, ONION_SALAD
```
* **Legend:** `_`/`|` = Wall | `#` = Counter | `C` = Chopping Board | `P` = Plate Dispenser | `D` = Delivery | `I` = Ingredient Dispenser | `0`/`1` = Agent Spawns | Space = Floor.

### 2. Custom Recipes (`recipes.json`)
Place a `recipes.json` file in the same directory as the map file:
```json
{
  "ingredients": [
    {
      "name": "onion",
      "color": [235, 150, 235],
      "cuts_needed": 3,
      "char": "o"
    }
  ],
  "recipes": [
    {
      "name": "ONION_SOUP",
      "ingredients": ["onion", "onion", "onion"],
      "reward": 180.0
    }
  ]
}
```

### 3. Validating Maps via CLI
```bash
python -m overcooked_marl_env.map.map_loader example/custom_map_template.txt
```

---

## Action and Observation Spaces

### Action Space
`Discrete(5)` per agent:
* **0**: Up (North)
* **1**: Down (South)
* **2**: Left (West)
* **3**: Right (East)
* **4**: Interact (pick up, chop, place, combine, deliver)

### Observation Space
`Box(height, width, 9)` in grid mode, or `Box(150,)` when `featurized_obs=True` (default for RL training):
* **Channel 0**: Static map walls and counters
* **Channel 1**: Items on counters/stations
* **Channel 2**: Self agent position
* **Channel 3**: Other agents' positions
* **Channel 4**: Delivery stations
* **Channel 5**: Recipe order queue
* **Channels 6–8**: Processing/cutting steps remaining

---

## Default Recipes

| Recipe | Ingredients | Reward | Notes |
| :--- | :--- | :---: | :--- |
| **SIMPLE_SALAD** | Lettuce (chopped) | `20.0` | 1 ingredient |
| **GARDEN_SALAD** | Lettuce + Tomato (chopped) | `60.0` | 2 ingredients |
| **FRESH_SALAD** | Lettuce + Tomato + Cucumber (chopped) | `180.0` | 3 ingredients |

---

## Tests

```bash
python -m unittest discover -s overcooked_marl_env/tests
```

---

## Project Structure

```text
overcooked-marl-env/
├── experiment_results/      # Experiment plots, summaries, and gameplay GIFs
├── overcooked_marl_env/     # Environment package (core, environment, map, mechanics, rendering, tests)
├── experiments/             # RL training, evaluation, analysis, and visualization scripts
├── scripts/                 # Server deployment, Docker, local runner, and multi-algorithm benchmark scripts
└── example/                 # Example scripts and map templates
```

---

## Documentation

| Section | Links |
| :--- | :--- |
| **Experiment Results** | [All Results](./experiment_results/README.md) · [01 — Evolution](./experiment_results/README.md#1-overview--environment-evolution) · [02 — 1-Agent](./experiment_results/README.md#2-1-agent-open-kitchen-baseline) · [03 — 2-Agent](./experiment_results/README.md#3-2-agent-divided-kitchen-cooperation) · [04 — 4-Agent](./experiment_results/README.md#4-4-agent-brigade--multi-algorithm-benchmark) |
| **Training & Evaluation** | [Experiments Pipeline](./experiments/README.md) |
| **Environment Package** | [Package](./overcooked_marl_env/README.md) · [Core](./overcooked_marl_env/core/README.md) · [Environment](./overcooked_marl_env/environment/README.md) · [Map](./overcooked_marl_env/map/README.md) · [Mechanics](./overcooked_marl_env/mechanics/README.md) · [Rendering](./overcooked_marl_env/rendering/README.md) · [Utils](./overcooked_marl_env/utils/README.md) |
