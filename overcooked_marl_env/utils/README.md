<div align="center">

![Utils](https://img.shields.io/badge/🛠️_Module-utils-9333ea?style=for-the-badge&labelColor=1e1b4b)
![Status](https://img.shields.io/badge/Status-Reserved_for_Shared_Helpers-3b82f6?style=for-the-badge&labelColor=1e1b4b)

# 🛠️ Utils Module — Shared Utility Functions

Reserved namespace for general-purpose, pure utility functions shared across environment components.

</div>

<img src="../../resultados_experimentos/assets/divider_gamer.svg" width="100%" />

## 📁 Sub-package Contents

```text
utils/
└── __init__.py     # Module initialization
```

## 📌 Status & Guidelines

Currently empty and reserved for future cross-cutting utilities. New helpers should be **pure**, **type-hinted**, **documented**, and paired with unit tests in `tests/unit/test_utils.py`.

<details>
<summary><b>💡 Planned Utility Categories</b></summary>

* **Validation:** `validate_action(action)`, `validate_position(pos, height, width)`
* **Coordinate Conversions:** `position_to_index(pos, width)`, `index_to_position(index, width)`
* **Distances:** `manhattan_distance(pos1, pos2)`, `euclidean_distance(pos1, pos2)`
* **Observation Helpers:** `normalize_observation(obs)`, `flatten_observation(obs)`
* **Logging & Config:** `log_episode_stats(...)`, `print_game_state_summary(...)`, `load_config(...)`
</details>

<img src="../../resultados_experimentos/assets/divider_gamer.svg" width="100%" />

⬅️ **[Back to Package Documentation](../README.md)**
