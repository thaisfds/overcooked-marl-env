#!/usr/bin/env python3
import os
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
# HYPERPARAMETER GRID SEARCH CONFIGURATION
# ======================================================================
# Edit this dictionary to define search spaces. Ray Tune will automatically
# run trial combinations in parallel based on available CPU/GPU resources.
# Passing CLI flags (e.g. --episodes 100) overrides these parameters.
HYPERPARAMETER_GRID = {
    "lr": [8e-4],                 # Reduced learning rate for logit sharpening
    "gamma": [0.98],              # High discount factor
    "entropy_coeff": [0.01],      # Low entropy coefficient to enforce deterministic logits
    "train_batch_size": [8000],   # Large batch
    "sgd_minibatch_size": [256],  # Proportional minibatch
    "update_epochs": [10],        # 10 epochs for fine-grained policy updates
    "num_episodes": [1000],       # Extended 1000 episodes for full deterministic consolidation
    "seed": [42]
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
        # Reward scaling to prevent gradient explosion and stabilize learning
        scaled_rewards = {agent: r * 0.01 for agent, r in rewards.items()}
        
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
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel env rollout workers")
    parser.add_argument("--gpus", type=int, default=1 if torch.cuda.is_available() else 0, help="Number of GPUs to allocate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()

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
            num_gpus=args.gpus
        )
        .env_runners(
            num_env_runners=args.workers,
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
        episodes_limit = env_config_dict.get("num_episodes", 300)
        target_timesteps = episodes_limit * 800
        return result.get("timesteps_total", 0) >= target_timesteps
        
    # Output path creation
    script_dir = Path(__file__).parent.absolute()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir_name = f"PPO_Marl_{timestamp}"
    ray_results_base = script_dir / "ray_results" / timestamp_dir_name
    ray_results_base.mkdir(parents=True, exist_ok=True)
    
    params_str = f"map-{args.map.replace('.txt', '')}_agents-{args.num_agents}_workers{args.workers}"
    params_dir_name = f"PPO_Marl_{params_str}"
    
    start_time = time.time()
    
    try:
        analysis = tune.run(
            "PPO",
            name=params_dir_name,
            stop=stop_fn,
            config=config_dict,
            checkpoint_freq=10,
            checkpoint_at_end=True,
            verbose=1,
            storage_path=str(ray_results_base.resolve())
        )
        
        print("\n" + "="*50)
        print("ALL TRAINING TRIALS COMPLETED")
        print(f"Total duration: {(time.time() - start_time)/60:.2f} minutes")
        print("="*50 + "\n")
        
        # Save model_info.json for each completed trial
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
                
            # Construct persistent target directory inside the workspace results folder
            target_dir = Path(ray_results_base) / params_dir_name / trial_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
                
            trial_config = trial.config
            trial_env_config = trial_config.get("env_config", {})
            
            # Read progress curves from trial_dir (where Tune keeps active log files)
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
                    
            # Resolve trial checkpoint path
            checkpoint_path = None
            if hasattr(trial, 'checkpoint') and trial.checkpoint:
                try:
                    if hasattr(trial.checkpoint, 'path'):
                        checkpoint_path = trial.checkpoint.path
                    elif hasattr(trial.checkpoint, 'to_directory'):
                        checkpoint_path = trial.checkpoint.to_directory()
                except Exception:
                    pass
                    
            # Map checkpoint_path to the persistent target directory if it contains /tmp
            if checkpoint_path:
                if "/tmp/" in checkpoint_path or not os.path.exists(checkpoint_path):
                    checkpoint_dir_name = Path(checkpoint_path).name
                    persistent_checkpoint_path = target_dir / checkpoint_dir_name
                    if persistent_checkpoint_path.exists():
                        checkpoint_path = str(persistent_checkpoint_path.resolve())
                    
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                # Search inside the persistent target directory first
                checkpoints = sorted(
                    [d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint_')],
                    key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else 0,
                    reverse=True
                )
                if checkpoints:
                    checkpoint_path = str(checkpoints[0].resolve())
                else:
                    # Fallback to the temp trial dir
                    checkpoints = sorted(
                        [d for d in trial_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint_')],
                        key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else 0,
                        reverse=True
                    )
                    if checkpoints:
                        checkpoint_path = str(checkpoints[0].resolve())
                    
            # Get reward mean metric
            avg_reward_last = 0.0
            if hasattr(trial, 'last_result') and trial.last_result:
                if 'env_runners' in trial.last_result and isinstance(trial.last_result['env_runners'], dict):
                    env_runners = trial.last_result['env_runners']
                    avg_reward_last = env_runners.get('episode_reward_mean', env_runners.get('episode_return_mean', 0.0))
                elif 'episode_reward_mean' in trial.last_result:
                    avg_reward_last = trial.last_result['episode_reward_mean']
                    
            # Format custom config name
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
                'config_name': config_name,
                'checkpoint_path': checkpoint_path,
                'avg_reward': float(avg_reward_last),
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
            
            # Save model_info.json in the persistent target_dir
            with open(target_dir / "model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)
                
            print(f"Trial {trial.trial_id} completed: reward={avg_reward_last:.2f} | Metadata saved: {target_dir / 'model_info.json'}")
            
    except Exception as e:
        print(f"Error during training session: {e}")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
