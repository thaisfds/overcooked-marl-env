"""
PPO (Proximal Policy Optimization) for Overcooked environment
Simplified version - clean code focused on essentials
"""

import os
import time
import random
from typing import Dict, Any
from itertools import product

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False

import gymnasium as gym
from overcooked_marl_env import OvercookedMARL


def get_best_device():
    """Automatically detects the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device, f"CUDA ({torch.cuda.get_device_name(0)})"
    
    if DIRECTML_AVAILABLE:
        device = torch_directml.device()
        return device, "DirectML (AMD GPU)"
    
    device = torch.device("cpu")
    return device, "CPU"


class PettingZooToGymWrapper(gym.Env):
    """Converts PettingZoo environment with 1 agent to Gymnasium interface."""
    
    def __init__(self, pettingzoo_env):
        super().__init__()
        self.env = pettingzoo_env
        
        if len(self.env.possible_agents) != 1:
            raise ValueError(f"Wrapper only works with 1 agent. Found: {len(self.env.possible_agents)}")
        
        self.agent_id = self.env.possible_agents[0]
        self.action_space = self.env.action_spaces[self.agent_id]
        self.observation_space = self.env.observation_spaces[self.agent_id]
        
    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return observations[self.agent_id], infos[self.agent_id]
    
    def step(self, action):
        actions = {self.agent_id: action}
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return (
            observations[self.agent_id],
            rewards[self.agent_id],
            terminations[self.agent_id],
            truncations[self.agent_id],
            infos[self.agent_id]
        )
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO with CNN for spatial observations."""
    
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        h, w, c = observation_space.shape
        n_actions = action_space.n
        
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            sample = torch.zeros(1, c, h, w)
            n_flatten = self.cnn(sample).shape[1]
        
        self.shared = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
        )
        
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        features = self.shared(self.cnn(x))
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value


def generate_hyperparameter_combinations(param_grid: dict) -> list:
    """Generates all possible hyperparameter combinations."""
    keys = param_grid.keys()
    values = [param_grid[key] for key in keys]
    combinations_raw = product(*values)
    return [dict(zip(keys, v)) for v in combinations_raw]


# Configuration
ENVIRONMENT_CONFIG = {
    'map_path': "overcooked_marl_env/map/maps/simple_map.txt",
    'num_fixed_agents': 1,
    'render_mode': None,
    'verbose': False,
    'max_episode_steps': 800,
}

param_grid = {
    'learning_rate': [3e-4, 1e-3],
    'num_steps': [512],
    'update_epochs': [4, 8, 16],
    'ent_coef': [0.05, 0.1, 0.15],
    'clip_coef': [0.1, 0.2],
    'num_minibatches': [4],
    'gamma': [0.995],
    'gae_lambda': [0.95, 0.97],
    'vf_coef': [0.5],
    'max_grad_norm': [0.5],
    'total_timesteps': [60000],
    'seed': [42],
    'exploration_epsilon_start': [0.5],
    'exploration_epsilon_end': [0.0],
}

if __name__ == "__main__":
    # Initialization
    device, device_name = get_best_device()
    print(f"Device: {device_name}")

    combinations = generate_hyperparameter_combinations(param_grid)
    print(f"Total configurations: {len(combinations)}\n")

    # Directory to save models
    MODEL_SAVE_DIR = "models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Tracking variables
    results = {}
    best_ppo_avg_reward = -np.inf
    best_ppo_std_reward = 0.0
    best_ppo_params = {}
    top_10_models = []

    start_time_total = time.time()

    # Grid search
    for i, params_original in enumerate(combinations):
        config = params_original.copy()
        config_name = f"config_{i+1:02d}"
        
        # Separator and title with hyperparameters
        print(f"\n{'='*70}")
        print(f"CONFIGURATION {i+1}/{len(combinations)}")
        print(f"{'='*70}")
        print(f"Update Epochs: {config['update_epochs']}")
        print(f"Entropy Coef: {config['ent_coef']}")
        print(f"GAE Lambda: {config['gae_lambda']}")
        print(f"Total Timesteps: {config['total_timesteps']}")
        print(f"Exploration: {config['exploration_epsilon_start']:.1f} -> {config['exploration_epsilon_end']:.1f}")
        print(f"{'='*70}\n")
        
        # Create environment
        base_env = OvercookedMARL(
            render_mode=ENVIRONMENT_CONFIG['render_mode'],
            map_path=ENVIRONMENT_CONFIG['map_path'],
            num_fixed_agents=ENVIRONMENT_CONFIG['num_fixed_agents'],
            verbose=ENVIRONMENT_CONFIG['verbose']
        )
        env = PettingZooToGymWrapper(base_env)
        
        # Seed for reproducibility
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        
        # Create agent and optimizer
        agent = ActorCritic(env.observation_space, env.action_space).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=config['learning_rate'], eps=1e-5)
        
        # Storage for rollouts
        num_steps = config['num_steps']
        obs_storage = torch.zeros((num_steps,) + env.observation_space.shape).to(device)
        actions_storage = torch.zeros(num_steps).to(device)
        logprobs_storage = torch.zeros(num_steps).to(device)
        rewards_storage = torch.zeros(num_steps).to(device)
        dones_storage = torch.zeros(num_steps).to(device)
        values_storage = torch.zeros(num_steps).to(device)
        
        # Metrics
        episode_rewards = []
        current_episode_reward = 0
        forced_explorations = 0
        total_actions_taken = 0
        
        # Reset environment
        obs, _ = env.reset()
        obs = torch.Tensor(obs).to(device)
        
        global_step = 0
        num_updates = config['total_timesteps'] // num_steps
        
        pbar = tqdm(range(1, num_updates + 1), desc=f"Config {i+1}/{len(combinations)}")
        
        for update in pbar:
            # PHASE 1: Data Collection (Rollout)
            agent.eval()
            for step in range(num_steps):
                global_step += 1
                total_actions_taken += 1
                obs_storage[step] = obs
                
                # Calculate current epsilon (linear decay)
                epsilon_start = config['exploration_epsilon_start']
                epsilon_end = config['exploration_epsilon_end']
                decay_steps = config['total_timesteps']
                
                if global_step < decay_steps:
                    epsilon = epsilon_start - (epsilon_start - epsilon_end) * (global_step / decay_steps)
                else:
                    epsilon = epsilon_end
                
                # Select action
                with torch.no_grad():
                    action_policy, logprob, _, value = agent.get_action_and_value(obs.unsqueeze(0))
                
                # Epsilon-greedy exploration
                if random.random() < epsilon:
                    action = torch.tensor(env.action_space.sample())
                    forced_explorations += 1
                    logits, _ = agent.forward(obs.unsqueeze(0))
                    probs = Categorical(logits=logits)
                    logprob = probs.log_prob(action)
                else:
                    action = action_policy
                
                actions_storage[step] = action
                logprobs_storage[step] = logprob
                values_storage[step] = value.flatten()
                
                # Step in environment
                next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
                done = terminated or truncated
                
                rewards_storage[step] = torch.tensor(reward).to(device)
                dones_storage[step] = torch.tensor(done).to(device)
                
                obs = torch.Tensor(next_obs).to(device)
                
                # Episode metrics
                current_episode_reward += reward
                
                if done:
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0
                    obs, _ = env.reset()
                    obs = torch.Tensor(obs).to(device)
            
            # PHASE 2: Calculate Advantages (GAE)
            with torch.no_grad():
                next_value = agent.get_action_and_value(obs.unsqueeze(0))[3]
                advantages = torch.zeros_like(rewards_storage).to(device)
                lastgaelam = 0
                
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - dones_storage[t]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_storage[t]
                        nextvalues = values_storage[t + 1]
                    
                    delta = rewards_storage[t] + config['gamma'] * nextvalues * nextnonterminal - values_storage[t]
                    advantages[t] = lastgaelam = delta + config['gamma'] * config['gae_lambda'] * nextnonterminal * lastgaelam
                
                returns = advantages + values_storage
            
            # PHASE 3: PPO Optimization
            agent.train()
            
            b_obs = obs_storage.reshape((-1,) + env.observation_space.shape)
            b_logprobs = logprobs_storage.reshape(-1).detach()
            b_actions = actions_storage.reshape(-1).detach()
            b_advantages = advantages.reshape(-1).detach()
            b_returns = returns.reshape(-1).detach()
            b_values = values_storage.reshape(-1).detach()
            
            # Normalize advantages
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
            # Minibatch optimization
            batch_size = num_steps
            minibatch_size = batch_size // config['num_minibatches']
            
            for epoch in range(config['update_epochs']):
                b_inds = torch.randperm(batch_size)
                
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    mb_advantages = b_advantages[mb_inds]
                    
                    # Policy loss (PPO clipping)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config['clip_coef'], 1 + config['clip_coef'])
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config['clip_coef'],
                        config['clip_coef']
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                    
                    # Entropy loss
                    entropy_loss = entropy.mean()
                    
                    # Total loss
                    loss = pg_loss - config['ent_coef'] * entropy_loss + v_loss * config['vf_coef']
                    
                    # Skip if NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), config['max_grad_norm'])
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
            
            # Update progress bar
            if len(episode_rewards) > 0:
                recent_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                pbar.set_postfix({
                    'reward': f"{recent_reward:.1f}",
                    'episodes': len(episode_rewards)
                })
        
        # Close environment
        env.close()
        
        # Calculate final average reward
        avg_reward_last_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        
        # Save results
        results[config_name] = {
            'params': params_original.copy(),
            'rewards': episode_rewards.copy(),
            'avg_reward': avg_reward_last_100,
        }
        
        # Manage top 10 models
        if len(top_10_models) < 10:
            top_10_models.append((avg_reward_last_100, params_original.copy(), episode_rewards.copy()))
            top_10_models.sort(key=lambda x: x[0], reverse=True)
        else:
            if avg_reward_last_100 > top_10_models[-1][0]:
                top_10_models.pop()
                top_10_models.append((avg_reward_last_100, params_original.copy(), episode_rewards.copy()))
                top_10_models.sort(key=lambda x: x[0], reverse=True)
        
        # Save model
        params_str = (
            f"lr{config['learning_rate']:.0e}_"
            f"steps{config['num_steps']}_"
            f"epochs{config['update_epochs']}_"
            f"ent{config['ent_coef']:.2f}_"
            f"clip{config['clip_coef']}_"
            f"mb{config['num_minibatches']}_"
            f"g{config['gamma']}_"
            f"lam{config['gae_lambda']}"
        )
        model_path = os.path.join(MODEL_SAVE_DIR, f"{config_name}_{params_str}_reward_{avg_reward_last_100:.2f}.pth")
        
        torch.save({
            'model_state_dict': agent.state_dict(),
            'config': config,
            'reward': avg_reward_last_100,
            'episode_rewards': episode_rewards,
        }, model_path)
        
        # Update best model
        if avg_reward_last_100 > best_ppo_avg_reward:
            best_ppo_avg_reward = avg_reward_last_100
            best_ppo_params = params_original.copy()
            best_ppo_std_reward = np.std(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.std(episode_rewards)
        
        # Summary per configuration
        print(f"Config {i+1}/{len(combinations)}: reward={avg_reward_last_100:.2f}, episodes={len(episode_rewards)}")

    # Final summary
    training_time_total = time.time() - start_time_total

    print(f"\n{'='*70}")
    print("TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total time: {training_time_total/60:.1f} min ({training_time_total/3600:.2f}h)")
    print(f"Configurations tested: {len(combinations)}")
    print(f"Best reward: {best_ppo_avg_reward:.2f} ± {best_ppo_std_reward:.2f}")
    print(f"Models saved in: {MODEL_SAVE_DIR}")

