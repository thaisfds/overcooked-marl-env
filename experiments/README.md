# Reinforcement Learning Experiments

This directory contains the reinforcement learning experiments for the Overcooked MARL environment, implemented using Ray RLlib (PPO).

## Files Overview
- **[ppo_overcookedRLlib.py](file:///home/thaiswsl/NewOvercookedEnv/experiments/ppo_overcookedRLlib.py)**: Training script for single-agent configuration.
- **[ppo_overcookedRLlib2Agents.py](file:///home/thaiswsl/NewOvercookedEnv/experiments/ppo_overcookedRLlib2Agents.py)**: Training script for two-agent cooperative configurations.
- **[ppo_model_evaluation.ipynb](file:///home/thaiswsl/NewOvercookedEnv/experiments/ppo_model_evaluation.ipynb)**: Jupyter notebook to evaluate single-agent policies.
- **[ppo_model_evaluation2Agents.ipynb](file:///home/thaiswsl/NewOvercookedEnv/experiments/ppo_model_evaluation2Agents.ipynb)**: Jupyter notebook to evaluate two-agent cooperative policies.

---

## 🛠️ Prerequisites & Installation

To install all the dependencies required to train models and visualize results via Jupyter Notebook:

```bash
# Install Ray RLlib (PPO engine) and Jupyter Notebook
venv/bin/pip install "ray[rllib]" notebook

# Install Seaborn (used for plotting evaluation graphs in notebooks)
venv/bin/pip install seaborn
```

---

## 🚀 How to Run the Experiments

First, make sure you are in the project root directory (`/home/thaiswsl/NewOvercookedEnv`) and have the virtual environment activated or use the direct virtual environment paths as shown below.

### 1. Running Training Scripts (Python)

To run the single-agent PPO training experiment:
```bash
venv/bin/python experiments/ppo_overcookedRLlib.py
```

To run the two-agent cooperative PPO training experiment:
```bash
venv/bin/python experiments/ppo_overcookedRLlib2Agents.py
```

*Note: Ray RLlib will automatically log tensorboard summaries and checkpoint files in `~/ray_results/`.*

---

### 2. Running Jupyter Notebooks (WSL Compatible)

Since you are running inside WSL, you can launch Jupyter Notebook in WSL and access it from your Windows browser. 

Launch the server with the following command:
```bash
venv/bin/jupyter notebook --ip=0.0.0.0 --no-browser
```

1. Copy the URL containing the token printed in the terminal (e.g., `http://127.0.0.1:8888/?token=...`).
2. Paste it in your Windows web browser (Chrome, Edge, etc.) to access the notebook interface.
3. Open `experiments/ppo_model_evaluation.ipynb` or `experiments/ppo_model_evaluation2Agents.ipynb` to evaluate and visualize your trained agents.
