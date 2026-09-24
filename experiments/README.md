# Reinforcement Learning Experiments

This directory contains the reinforcement learning experiments for the Overcooked MARL environment, implemented using Ray RLlib (PPO).

## 📁 Directory Structure

```text
experiments/
├── training/                                # 🧪 Model training scripts & automated cycles
│   ├── train_ppo.py                         # Parameterized PPO training script (RLlib + PyTorch)
│   └── run_experiment_cycle.sh              # 1-command automated pipeline (train + evaluate)
│
├── analysis/                                # 🔬 Model analysis & quantitative evaluation
│   ├── evaluate_ppo.py                      # Multi-episode deterministic evaluator with seed control
│   └── find_best_params.py                  # Checkpoint discovery tool for ray_results
│
├── visualization/                           # 🎨 Visualization notebooks & plotting tools
│   ├── visualize_best_model.ipynb           # Interactive Pygame graphical simulation notebook
│   ├── curves_and_performance.ipynb         # Unified performance suite (curves, ranking, logs, vision)
│   ├── plot_curves.py                       # CLI training curves PNG generator
│   └── learning_curves.png                  # Exported training curve image
│
└── logs/                                    # 📊 Experiment logs & summaries
    └── vision_analysis/                     # Step-by-step vision log files (.log)
```

---

## 🛠️ Prerequisites & Installation

To install all dependencies required to train models and visualize results via Jupyter Notebook:

```bash
# Install Ray RLlib (PPO engine) and Jupyter Notebook
venv/bin/pip install "ray[rllib]" notebook seaborn
```

---

## 🚀 How to Run

### 1. Training New Models (`experiments/training/`)

Execute an automated training + evaluation cycle:
```bash
./experiments/training/run_experiment_cycle.sh
```

Or run `train_ppo.py` directly with custom CLI parameters:
```bash
venv/bin/python experiments/training/train_ppo.py --map default_map.txt --num-agents 1 --workers 2
```

---

### 2. Model Analysis & Evaluation (`experiments/analysis/`)

Run multi-episode statistical evaluation on the latest checkpoint:
```bash
venv/bin/python experiments/analysis/evaluate_ppo.py --num-episodes 5
```

Enable step-by-step vision analysis logging during evaluation:
```bash
venv/bin/python experiments/analysis/evaluate_ppo.py --num-episodes 5 --log-vision
```

---

### 3. Visualization Notebooks (`experiments/visualization/`)

Launch Jupyter Notebook:
```bash
venv/bin/jupyter notebook --ip=0.0.0.0 --no-browser
```

- **`visualize_best_model.ipynb`**: Renders real-time Pygame graphical rollouts of the top checkpoint.
- **`curves_and_performance.ipynb`**: Renders training reward curves, log ranking tables, action distributions, and step-by-step vision log focus/score trajectory charts.
