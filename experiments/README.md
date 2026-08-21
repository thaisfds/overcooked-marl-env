# Reinforcement Learning Experiments

This directory contains the reinforcement learning experiments for the Overcooked MARL environment, implemented using Ray RLlib (PPO).

## Files Overview
- **[train_ppo.py](file:///home/thaiswsl/NewOvercookedEnv/experiments/train_ppo.py)**: **[NEW & OPTIMIZED]** Parameterized unified training script supporting parallel env runners and native parallel grid search.
- **[ppo_overcookedRLlib.py](file:///home/thaiswsl/NewOvercookedEnv/experiments/ppo_overcookedRLlib.py)**: Training script for single-agent configuration.
- **[ppo_overcookedRLlib2Agents.py](file:///home/thaiswsl/NewOvercookedEnv/experiments/ppo_overcookedRLlib2Agents.py)**: Training script for two-agent cooperative configurations.
- **[ppo_model_evaluation.ipynb](file:///home/thaiswsl/NewOvercookedEnv/experiments/ppo_model_evaluation.ipynb)**: Jupyter notebook to evaluate single-agent policies.
- **[ppo_model_evaluation2Agents.ipynb](file:///home/thaiswsl/NewOvercookedEnv/experiments/ppo_model_evaluation2Agents.ipynb)**: Jupyter notebook to evaluate two-agent cooperative policies.
- **[ppo_plot_training_curves.ipynb](file:///home/thaiswsl/NewOvercookedEnv/experiments/ppo_plot_training_curves.ipynb)**: **[NEW]** Jupyter notebook to instantly compare learning curves of all trained models.

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

### 1. Running the Unified Optimized Training Script (`train_ppo.py`)

This script supports training configurations dynamically via CLI arguments or parameter searches in the code.

**CLI Options:**
- `--map`: Map name (e.g. `default_map.txt`, `tutorial.txt`).
- `--num-agents`: Number of agents to train (e.g. `2`, `3`).
- `--episodes`: Total episodes to train.
- `--workers`: Number of parallel environment rollout workers (rollout parallelization).
- `--lr`: Learning rate override.
- `--gamma`: Discount factor override.
- `--batch-size`: Training batch size.
- `--minibatch-size`: SGD minibatch size.
- `--epochs`: Number of SGD epochs per batch update.
- `--entropy-coeff`: Entropy coefficient.
- `--gpus`: Number of GPUs to allocate (auto-detects CUDA devices).

**Run single trial via CLI:**
```bash
venv/bin/python experiments/train_ppo.py --map default_map.txt --num-agents 2 --workers 3 --episodes 300
```

**Run native parallel Grid Search:**
Edit the `HYPERPARAMETER_GRID` dictionary inside [`train_ppo.py`](file:///home/thaiswsl/NewOvercookedEnv/experiments/train_ppo.py):
```python
HYPERPARAMETER_GRID = {
    "lr": [3e-4, 1e-3],          # Ray Tune runs both in parallel
    "gamma": [0.90, 0.95],        # Ray Tune runs both in parallel
    "entropy_coeff": [0.01],
    "train_batch_size": [4000],
    "sgd_minibatch_size": [128],
    "update_epochs": [8],
    "num_episodes": [300, 400],   # Handled dynamically per trial via stop_fn
    "seed": [42]
}
```
Then, execute the script. Ray Tune will orchestrate concurrent trial execution based on system resources, saving checkpoints and `model_info.json` directly to the persistent `ray_results/` storage.

*Tip: You can combine the grid search with CLI arguments to override constant parameters (like map name or agent count) for all concurrent trials:*
```bash
venv/bin/python experiments/train_ppo.py --num-agents 3 --map tutorial.txt
```

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
