#!/usr/bin/env python3
import os
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if os.environ.get("HOME") in (None, "", "/"):
    os.environ["HOME"] = str(PROJECT_ROOT)
os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
os.environ["AIR_VERBOSITY"] = "0"
import argparse
import time
import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
from collections import defaultdict

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env

try:
    from overcooked_marl_env import OvercookedMARL
except ImportError:
    from env import OvercookedMARL

torch, nn = try_import_torch()

# ======================================================================
# HYPERPARAMETER GRID SEARCH CONFIGURATION (LONG-RUN / 1-WEEK SERVER GRID)
# ======================================================================
# Optimized for VeRLab 'magritte' server:
# CPU: Intel Core i7-3770K (8 threads) | GPU: GTX Titan X 12GB | RAM: 32GB
HYPERPARAMETER_GRID = {
    "lr": [1e-4, 3e-4, 5e-4],            # Taxa de aprendizado: Estável (1e-4) -> Padrão (3e-4) -> Rápido (5e-4)
    "gamma": [0.98, 0.99, 0.995],        # Fator de desconto: Visão de longo prazo para receitas complexas
    "entropy_coeff": [0.01, 0.03, 0.05], # Exploração sustentada em layouts difíceis
    "train_batch_size": [4000, 8000],    # Volume de amostragem por atualização
    "sgd_minibatch_size": [256],         # Tamanho de minibatch PPO
    "update_epochs": [8, 12],            # Passos de atualização PPO por amostragem
    "num_episodes": [2000],              # Episódios por trial (garante convergência completa do agente)
    "seed": [42]                         # Semente aleatória
}


class ParallelEnvToMultiAgent(MultiAgentEnv):
    """Wrapper to convert PettingZoo ParallelEnv to RLlib MultiAgentEnv."""
    def __init__(self, parallel_env):
        self.env = parallel_env
        self.agents = self.env.possible_agents
        super().__init__()
        
    @property
    def observation_space(self):
        return {agent: self.env.observation_space(agent) for agent in self.agents}
    
    @property
    def action_space(self):
        return {agent: self.env.action_space(agent) for agent in self.agents}
    
    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action_dict):
        obs, rewards, terminations, truncations, infos = self.env.step(action_dict)
        scaled_rewards = {agent: r for agent, r in rewards.items()}
        
        all_terminated = all(terminations.get(agent, False) for agent in self.agents)
        all_truncated = all(truncations.get(agent, False) for agent in self.agents)
        
        terminations['__all__'] = all_terminated
        truncations['__all__'] = all_truncated
        
        return obs, scaled_rewards, terminations, truncations, infos
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        if name.startswith('_') or name in ['env', 'agents', 'observation_space', 'action_space']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            env = object.__getattribute__(self, 'env')
            return getattr(env, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

def parse_args():
    default_workers = max(2, min(12, os.cpu_count() - 2)) if hasattr(os, 'cpu_count') and os.cpu_count() else 4
    parser = argparse.ArgumentParser(description="Optimized, parameterized RLlib PPO training for Overcooked MARL.")
    parser.add_argument("--map", type=str, default="default_map.txt", help="Map name (e.g. default_map.txt, tutorial.txt)")
    parser.add_argument("--num-agents", type=int, default=2, help="Number of agents to train with")
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--gamma", type=float, default=None, help="Override discount factor")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--minibatch-size", type=int, default=None, help="Override minibatch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override updates per batch")
    parser.add_argument("--entropy-coeff", type=float, default=None, help="Override entropy coefficient")
    parser.add_argument("--workers", type=int, default=default_workers, help="Number of parallel env rollout workers")
    parser.add_argument("--gpus", type=float, default=1.0, help="Total GPUs to allocate across concurrent trials (default: 1.0)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Checkpoint frequency in iterations (default: 10)")
    parser.add_argument("--keep-checkpoints-num", type=int, default=5, help="Number of recent checkpoints to keep (default: 5)")
    parser.add_argument("--resume", type=str, default="AUTO", choices=["AUTO", "LOCAL", "FALSE"], help="Auto-resume crashed trials (default: AUTO)")
    parser.add_argument("--concurrent", type=int, default=2, help="Number of concurrent trials to train in parallel (default: 2)")
    return parser.parse_args()

class TrainingProgressCallback(tune.Callback):
    """Callback to compute training progress percentage and update trial results for Ray Tune reporting."""
    def on_trial_result(self, iteration, trials, trial, result, **info):
        config = result.get("config", {})
        env_config_dict = config.get("env_config", {})
        episodes_limit = env_config_dict.get("num_episodes", 100)
        target_timesteps = episodes_limit * 3000
        timesteps_total = result.get("timesteps_total", 0)
        if target_timesteps > 0:
            pct = min(100.0, (timesteps_total / target_timesteps) * 100.0)
            result["progress_pct"] = f"{pct:.1f}%"
        else:
            result["progress_pct"] = "0.0%"


def main():
    args = parse_args()
    
    # Apply CLI overrides to parameter grid
    if args.episodes is not None:
        HYPERPARAMETER_GRID['num_episodes'] = [args.episodes]
    if args.lr is not None:
        HYPERPARAMETER_GRID['lr'] = [args.lr]
    if args.gamma is not None:
        HYPERPARAMETER_GRID['gamma'] = [args.gamma]
    if args.batch_size is not None:
        HYPERPARAMETER_GRID['train_batch_size'] = [args.batch_size]
    if args.minibatch_size is not None:
        HYPERPARAMETER_GRID['sgd_minibatch_size'] = [args.minibatch_size]
    if args.epochs is not None:
        HYPERPARAMETER_GRID['update_epochs'] = [args.epochs]
    if args.entropy_coeff is not None:
        HYPERPARAMETER_GRID['entropy_coeff'] = [args.entropy_coeff]
    if args.seed is not None:
        HYPERPARAMETER_GRID['seed'] = [args.seed]
        
    print("\n" + "="*50)
    print("STARTING OPTIMIZED MULTI-AGENT PPO TRAINING")
    print(f"  Map:          {args.map}")
    print(f"  Agents:       {args.num_agents}")
    print(f"  Workers:      {args.workers}")
    print(f"  GPUs:         {args.gpus}")
    print(f"  Grid Options: {HYPERPARAMETER_GRID}")
    print("="*50 + "\n")
    
    ray.init(ignore_reinit_error=True)
    
    # Environment Creator Fn
    def env_creator_fn(num_episodes_val):
        return OvercookedMARL(
            render_mode=None,
            map_path=args.map,
            num_fixed_agents=args.num_agents,
            verbose=False,
            featurized_obs=True
        )
        
    env_name = "overcooked_marl_parallel"
    # We pass the env_creator with dummy config mapping
    register_env(env_name, lambda config: ParallelEnvToMultiAgent(env_creator_fn(config.get("num_episodes", 300))))
    
    # Query observation & action spaces from temporary env
    temp_env = ParallelEnvToMultiAgent(env_creator_fn(300))
    agents = temp_env.agents
    obs_space = temp_env.observation_space["agent_0"]
    act_space = temp_env.action_space["agent_0"]
    temp_env.close()
    
    # Dynamic independent policy creation for each active agent index
    policies = {
        f"policy_{agent_id}": (None, obs_space, act_space, {})
        for agent_id in agents
    }
    
    # Map hyperparameters grid to Ray Tune's search space dynamically
    tune_lr = tune.grid_search(HYPERPARAMETER_GRID['lr']) if len(HYPERPARAMETER_GRID['lr']) > 1 else HYPERPARAMETER_GRID['lr'][0]
    tune_gamma = tune.grid_search(HYPERPARAMETER_GRID['gamma']) if len(HYPERPARAMETER_GRID['gamma']) > 1 else HYPERPARAMETER_GRID['gamma'][0]
    tune_entropy = tune.grid_search(HYPERPARAMETER_GRID['entropy_coeff']) if len(HYPERPARAMETER_GRID['entropy_coeff']) > 1 else HYPERPARAMETER_GRID['entropy_coeff'][0]
    tune_batch_size = tune.grid_search(HYPERPARAMETER_GRID['train_batch_size']) if len(HYPERPARAMETER_GRID['train_batch_size']) > 1 else HYPERPARAMETER_GRID['train_batch_size'][0]
    tune_minibatch_size = tune.grid_search(HYPERPARAMETER_GRID['sgd_minibatch_size']) if len(HYPERPARAMETER_GRID['sgd_minibatch_size']) > 1 else HYPERPARAMETER_GRID['sgd_minibatch_size'][0]
    tune_epochs = tune.grid_search(HYPERPARAMETER_GRID['update_epochs']) if len(HYPERPARAMETER_GRID['update_epochs']) > 1 else HYPERPARAMETER_GRID['update_epochs'][0]
    tune_episodes = tune.grid_search(HYPERPARAMETER_GRID['num_episodes']) if len(HYPERPARAMETER_GRID['num_episodes']) > 1 else HYPERPARAMETER_GRID['num_episodes'][0]
    tune_seed = tune.grid_search(HYPERPARAMETER_GRID['seed']) if len(HYPERPARAMETER_GRID['seed']) > 1 else HYPERPARAMETER_GRID['seed'][0]
    
    base_batch_size = HYPERPARAMETER_GRID['train_batch_size'][0]
    rollout_fragment_length = base_batch_size // max(1, args.workers)
    
    ppo_config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config={"num_episodes": tune_episodes}  # Pass num_episodes in env_config safely
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .training(
            lr=tune_lr,
            gamma=tune_gamma,
            num_epochs=tune_epochs,
            entropy_coeff=tune_entropy,
            grad_clip=1.0,
            model={
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
            }
        )
        .resources(
            num_gpus=(args.gpus / max(1, args.concurrent)) if args.gpus > 0 else 0
        )
        .env_runners(
            num_env_runners=max(1, args.workers // max(1, args.concurrent)),
            rollout_fragment_length="auto",
            batch_mode="truncate_episodes",
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"policy_{agent_id}",
            policies_to_train=list(policies.keys())
        )
    )
    
    config_dict = ppo_config.to_dict()
    config_dict['sgd_minibatch_size'] = tune_minibatch_size
    config_dict['train_batch_size'] = tune_batch_size
    config_dict['seed'] = tune_seed

    # Custom stopping criteria evaluator based on dynamic trial configuration episodes
    def stop_fn(trial_id, result):
        config = result.get("config", {})
        env_config_dict = config.get("env_config", {})
        episodes_limit = env_config_dict.get("num_episodes", 100)
        target_timesteps = episodes_limit * 3000
        return result.get("timesteps_total", 0) >= target_timesteps
        
    # Output path creation
    exp_dir = Path(__file__).resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir_name = f"PPO_Marl_{timestamp}"
    ray_results_base = exp_dir / "ray_results" / timestamp_dir_name
    ray_results_base.mkdir(parents=True, exist_ok=True)
    
    params_str = f"map-{args.map.replace('.txt', '')}_agents-{args.num_agents}_workers{args.workers}"
    params_dir_name = f"PPO_Marl_{params_str}"
    
    # Configure custom CLI progress reporter
    progress_reporter = tune.CLIReporter(
        metric_columns={
            "training_iteration": "iter",
            "time_total_s": "total time (s)",
            "timesteps_total": "ts",
            "progress_pct": "progress (%)",
            "env_runners/episode_reward_mean": "reward_mean",
        }
    )

    start_time = time.time()
    resume_val = False if args.resume == "FALSE" else (True if args.resume == "LOCAL" else "AUTO")
    
    try:
        analysis = tune.run(
            "PPO",
            name=params_dir_name,
            stop=stop_fn,
            config=config_dict,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_at_end=True,
            keep_checkpoints_num=args.keep_checkpoints_num,
            max_concurrent_trials=args.concurrent,
            resume=resume_val,
            verbose=1,
            progress_reporter=progress_reporter,
            callbacks=[TrainingProgressCallback()],
            storage_path=str(ray_results_base.resolve())
        )
        
        print("\n" + "="*50)
        print("ALL TRAINING TRIALS COMPLETED")
        print(f"Total duration: {(time.time() - start_time)/60:.2f} minutes")
        print("="*50 + "\n")
        
        # Collect and compare all trials in Grid Search
        all_trials_summary = []
        for trial in analysis.trials:
            trial_dir = None
            for attr in ['local_path', 'logdir', 'path']:
                try:
                    val = getattr(trial, attr, None)
                    if val:
                        trial_dir = Path(val)
                        break
                except Exception:
                    pass
                    
            if not trial_dir or not trial_dir.exists():
                continue
                
            target_dir = Path(ray_results_base) / params_dir_name / trial_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
                
            trial_config = trial.config
            trial_env_config = trial_config.get("env_config", {})
            
            learning_curves_data = []
            progress_csv = trial_dir / "progress.csv"
            if progress_csv.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(progress_csv)
                    if 'env_runners/episode_reward_mean' in df.columns:
                        learning_curves_data = [
                            {
                                'iteration': int(row['training_iteration']),
                                'reward_mean': float(row['env_runners/episode_reward_mean'])
                            }
                            for _, row in df.iterrows()
                            if pd.notna(row.get('env_runners/episode_reward_mean'))
                        ]
                except Exception:
                    pass
                    
            checkpoint_path = None
            if hasattr(trial, 'checkpoint') and trial.checkpoint:
                try:
                    if hasattr(trial.checkpoint, 'path'):
                        checkpoint_path = trial.checkpoint.path
                    elif hasattr(trial.checkpoint, 'to_directory'):
                        checkpoint_path = trial.checkpoint.to_directory()
                except Exception:
                    pass
                    
            if checkpoint_path:
                if "/tmp/" in checkpoint_path or not os.path.exists(checkpoint_path):
                    checkpoint_dir_name = Path(checkpoint_path).name
                    persistent_checkpoint_path = target_dir / checkpoint_dir_name
                    if persistent_checkpoint_path.exists():
                        checkpoint_path = str(persistent_checkpoint_path.resolve())
                    
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                checkpoints = sorted(
                    [d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint_')],
                    key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else 0,
                    reverse=True
                )
                if checkpoints:
                    checkpoint_path = str(checkpoints[0].resolve())
                else:
                    checkpoints = sorted(
                        [d for d in trial_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint_')],
                        key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else 0,
                        reverse=True
                    )
                    if checkpoints:
                        checkpoint_path = str(checkpoints[0].resolve())
                    
            avg_reward_last = None
            if hasattr(trial, 'last_result') and trial.last_result:
                if 'env_runners' in trial.last_result and isinstance(trial.last_result['env_runners'], dict):
                    env_runners = trial.last_result['env_runners']
                    avg_reward_last = env_runners.get('episode_reward_mean', env_runners.get('episode_return_mean', None))
                elif 'episode_reward_mean' in trial.last_result:
                    avg_reward_last = trial.last_result['episode_reward_mean']
                    
            p_lr = trial_config.get('lr', 3e-4)
            p_gamma = trial_config.get('gamma', 0.90)
            p_batch = trial_config.get('train_batch_size', 4000)
            p_seed = trial_config.get('seed', 42)
            p_epochs = trial_config.get('num_epochs', 8)
            p_entropy = trial_config.get('entropy_coeff', 0.01)
            p_episodes = trial_env_config.get('num_episodes', 300)
            
            config_str = f"lr{p_lr}_gamma{p_gamma}_batch{p_batch}_epochs{p_epochs}_episodes{p_episodes}_entropy{p_entropy}_seed{p_seed}"
            config_name = f"PPO_Config_{config_str}"
            
            model_info = {
                'trial_id': trial.trial_id,
                'config_name': config_name,
                'checkpoint_path': checkpoint_path,
                'avg_reward': float(avg_reward_last) if avg_reward_last is not None and not np.isnan(avg_reward_last) else None,
                'learning_curves': learning_curves_data,
                'training_config': {
                    'learning_rate': float(p_lr),
                    'gamma': float(p_gamma),
                    'train_batch_size': int(p_batch),
                    'sgd_minibatch_size': int(trial_config.get('sgd_minibatch_size', 128)),
                    'num_epochs': int(p_epochs),
                    'entropy_coeff': float(p_entropy),
                    'num_episodes': int(p_episodes),
                    'timesteps_total': int(trial.last_result.get('timesteps_total', 0) if trial.last_result else 0)
                }
            }
            
            with open(target_dir / "model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)
                
            all_trials_summary.append(model_info)

            # Record in persistent experiment results history
            try:
                from experiments.utils.results_tracker import record_experiment_entry
                record_experiment_entry({
                    "timestamp": timestamp,
                    "experiment_id": timestamp_dir_name,
                    "trial_id": trial.trial_id,
                    "map_name": args.map,
                    "num_agents": args.num_agents,
                    "learning_rate": float(p_lr),
                    "gamma": float(p_gamma),
                    "entropy_coeff": float(p_entropy),
                    "batch_size": int(p_batch),
                    "num_episodes": int(p_episodes),
                    "train_reward_mean": float(avg_reward_last) if avg_reward_last is not None and not np.isnan(avg_reward_last) else "N/A",
                    "status": "COMPLETED",
                    "checkpoint_path": checkpoint_path or ""
                })
            except Exception as tracker_err:
                print(f"[WARNING] Could not update results tracker: {tracker_err}")

        # Sort trials by avg_reward (highest first, handling None/NaN)
        all_trials_summary.sort(key=lambda t: t['avg_reward'] if t['avg_reward'] is not None else -999999.0, reverse=True)
        best_trial = all_trials_summary[0] if all_trials_summary else None

        # Save logs and unified export bundle
        exp_logs_dir = exp_dir / "logs"
        exp_logs_dir.mkdir(parents=True, exist_ok=True)
        exp_log_file = exp_logs_dir / f"{timestamp_dir_name}.json"

        exp_summary_data = {
            "ray_results_folder": timestamp_dir_name,
            "experiment_timestamp": timestamp,
            "total_trials": len(all_trials_summary),
            "best_trial": best_trial,
            "all_trials": all_trials_summary,
            "evaluation": None
        }

        with open(exp_log_file, "w") as f:
            json.dump(exp_summary_data, f, indent=2)

        # Build comprehensive Markdown comparison table for all Grid Search trials
        table_rows = []
        for rank, t in enumerate(all_trials_summary, 1):
            cfg = t['training_config']
            reward_str = f"{t['avg_reward']:+.2f}" if t['avg_reward'] is not None else "N/A"
            cp_str = f"`.../{Path(t['checkpoint_path']).name}`" if t['checkpoint_path'] else "N/A"
            table_rows.append(f"| {rank} | `{t['trial_id']}` | {cfg['learning_rate']} | {cfg['gamma']} | {cfg['entropy_coeff']} | **{reward_str}** | {cp_str} |")

        table_md = "\n".join(table_rows)
        best_reward_str = f"{best_trial['avg_reward']:+.2f}" if (best_trial and best_trial.get('avg_reward') is not None) else "N/A"
        best_trial_id = best_trial['trial_id'] if best_trial else "N/A"
        best_cfg_json = json.dumps(best_trial['training_config'], indent=2) if best_trial else "{}"
        best_cp_path = best_trial.get('checkpoint_path', 'N/A') if best_trial else "N/A"

        summary_md = f"""# Relatório de Experimento - Grid Search PPO Overcooked
* **Data/Hora:** {timestamp}
* **Sessão:** `{timestamp_dir_name}`
* **Total de Combinações/Trials:** {len(all_trials_summary)}
* **Melhor Recompensa de Treino:** {best_reward_str} (Trial `{best_trial_id}`)

## Tabela Comparativa dos Hiperparâmetros (Ordenado pelos Melhores Resultados)
| Rank | Trial ID | LR | Gamma | Entropy | Recompensa Média | Checkpoint |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{table_md}

## Detalhes do Melhor Modelo Encontrado
* **Checkpoint Path:** `{best_cp_path}`
* **Comando para Visualização:**
  ```bash
  venv/bin/python experiments/visualization/visualize_model.py {best_cp_path}
  ```

### Hiperparâmetros do Melhor Modelo:
```json
{best_cfg_json}
```
"""

        # Save summary report in logs directory
        summary_md_file = exp_logs_dir / f"{timestamp_dir_name}_summary.md"
        with open(summary_md_file, "w") as f:
            f.write(summary_md)
        with open(exp_logs_dir / "latest_summary.md", "w") as f:
            f.write(summary_md)

        print(f"Grid search completed with {len(all_trials_summary)} trials.")
        print(f"Best Trial: {best_trial['trial_id']} | Reward: {best_trial['avg_reward']}")
        print(f"📊 Summary report created at: {summary_md_file}")
            
    except Exception as e:
        print(f"Error during training session: {e}")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
