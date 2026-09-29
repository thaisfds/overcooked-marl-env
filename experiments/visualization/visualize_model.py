#!/usr/bin/env python3
"""
1. Passando o caminho do checkpoint diretamente:
python experiments/visualization/visualize_model.py experiments/ray_results/PPO_Marl_20260928_220232/PPO_Marl_map-default_map_agents-1_workers4/PPO_overcooked_marl_parallel_73716_00000_0_entropy_coeff=0.0200,gamma=0.9800,lr=0.0003_2026-09-28_22-02-32/checkpoint_000000

2. Usando a flag --checkpoint:
python experiments/visualization/visualize_model.py --checkpoint caminho/para/checkpoint_000000

3. Rodando sem argumentos (Auto-detecta o modelo mais recente):
python experiments/visualization/visualize_model.py
"""
import os
import json
import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Enable render log recording
os.environ["OVERCOOKED_VISION_LOG"] = "render_rollout.log"
os.environ["RAY_ACCELERATOR_MANAGER_LOG_LEVEL"] = "ERROR"

import pygame
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

try:
    from overcooked_marl_env import OvercookedMARL
except ImportError:
    from env import OvercookedMARL


class ParallelEnvToMultiAgent(MultiAgentEnv):
    def __init__(self, parallel_env):
        self.env = parallel_env
        self.agents = self.env.possible_agents
        super().__init__()
        
    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action_dict):
        return self.env.step(action_dict)

    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()


def discover_latest_or_best_checkpoint(ray_results_dir="experiments/ray_results"):
    results_path = Path(ray_results_dir).resolve()
    if not results_path.exists():
        results_path = project_root / "experiments" / "ray_results"
        
    checkpoints = []
    for cp_path in results_path.glob("**/checkpoint_*"):
        if cp_path.is_dir() and (cp_path / "rllib_checkpoint.json").exists():
            avg_reward = -999999.0
            num_agents = 1
            map_name = "default_map.txt"
            
            for info_dir in [cp_path.parent, cp_path.parents[1]]:
                info_file = info_dir / "model_info.json"
                if info_file.exists():
                    try:
                        with open(info_file, "r") as f:
                            meta = json.load(f)
                            avg_reward = float(meta.get("avg_reward", -999999.0))
                            num_agents = int(meta.get("num_agents", 1))
                            map_name = meta.get("map", "default_map.txt")
                            break
                    except Exception:
                        pass
                        
            iter_num = int(cp_path.name.split("_")[-1]) if cp_path.name.split("_")[-1].isdigit() else 0
            mtime = cp_path.stat().st_mtime
            checkpoints.append((cp_path.resolve(), avg_reward, iter_num, mtime, num_agents, map_name))
                
    # Sort by modification time (mtime) and iteration number descending to pick the most recent checkpoint
    checkpoints.sort(key=lambda x: (x[3], x[2]), reverse=True)
    if not checkpoints:
        return None, 1, "default_map.txt"
    latest = checkpoints[0]
    return latest[0], latest[4], latest[5]


def run_pygame_visual_rollout(checkpoint_path, num_agents=1, map_name="default_map.txt", num_episodes=1, max_steps=800, step_delay_ms=50, explore=True):
    log_path = project_root / "experiments" / "logs" / "vision_analysis" / "render_rollout.log"
    cp_abs_path = Path(checkpoint_path).resolve()
    
    print(f"\n" + "="*60)
    print(f"🎮 RUNNING PYGAME VISUAL ROLLOUT")
    print(f"  Checkpoint: {cp_abs_path}")
    print(f"  Map:        {map_name} | Agents: {num_agents} | Explore: {explore}")
    print(f"  Vision Log: {log_path}")
    print("="*60 + "\n")
    
    def env_creator_fn():
        return OvercookedMARL(
            render_mode="human",
            map_path=map_name,
            num_fixed_agents=num_agents,
            verbose=False,
            featurized_obs=True
        )
        
    try:
        register_env("overcooked_marl_parallel", lambda config: ParallelEnvToMultiAgent(env_creator_fn()))
    except Exception:
        pass
    
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    algorithm = Algorithm.from_checkpoint(str(cp_abs_path))
    env = ParallelEnvToMultiAgent(env_creator_fn())
    
    try:
        for episode in range(num_episodes):
            obs_dict, infos = env.reset(seed=42 + episode)
            episode_reward = 0.0
            
            for step in range(max_steps):
                # Handle Pygame window events so window stays active and responsive
                try:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("[INFO] Pygame window closed by user.")
                            return
                except Exception:
                    pass

                actions = {}
                for agent_id, obs in obs_dict.items():
                    try:
                        raw_action = algorithm.compute_single_action(obs, policy_id=f"policy_{agent_id}", explore=explore)
                        if isinstance(raw_action, (tuple, list, np.ndarray)):
                            raw_action = raw_action[0]
                        action = int(raw_action)
                    except Exception:
                        try:
                            raw_action = algorithm.compute_single_action(obs, explore=explore)
                            if isinstance(raw_action, (tuple, list, np.ndarray)):
                                raw_action = raw_action[0]
                            action = int(raw_action)
                        except Exception:
                            action = 0
                    actions[agent_id] = action
                    
                obs_dict, rewards, terminations, truncations, infos = env.step(actions)
                episode_reward += sum(rewards.values())
                
                try:
                    env.render()
                except Exception:
                    pass
                    
                pygame.time.wait(step_delay_ms)
                
                if any(terminations.values()) or any(truncations.values()):
                    break
                    
            print(f"[SUCCESS] Episode {episode+1} finished. Total Reward: {episode_reward:+.2f}")
            print(f"[LOG] Render vision log stored at: {log_path}")
    finally:
        env.close()
        if ray.is_initialized():
            ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Render visual Pygame rollout of any specified RLlib PPO checkpoint.")
    parser.add_argument("checkpoint_path", type=str, nargs="?", default=None, help="Path to checkpoint folder (positional argument)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to checkpoint folder (optional flag)")
    parser.add_argument("--map", type=str, default="default_map.txt", help="Map name (default: default_map.txt)")
    parser.add_argument("--num-agents", type=int, default=1, help="Number of agents (default: 1)")
    parser.add_argument("--delay", type=int, default=50, help="Step delay in ms (e.g. 50 = 20 FPS)")
    parser.add_argument("--deterministic", action="store_true", help="Disable exploration sampling (greedy actions)")
    args = parser.parse_args()

    target_cp = args.checkpoint_path or args.checkpoint
    num_agents = args.num_agents
    map_name = args.map

    if target_cp:
        cp_path = Path(target_cp).resolve()
        # Try reading model_info metadata if available in parent directories
        for info_dir in [cp_path, cp_path.parent, cp_path.parents[1]]:
            info_file = info_dir / "model_info.json"
            if info_file.exists():
                try:
                    with open(info_file, "r") as f:
                        meta = json.load(f)
                        num_agents = int(meta.get("num_agents", num_agents))
                        map_name = meta.get("map", map_name)
                        break
                except Exception:
                    pass
    else:
        cp_path, num_agents, map_name = discover_latest_or_best_checkpoint()

    if cp_path and cp_path.exists():
        run_pygame_visual_rollout(
            checkpoint_path=cp_path,
            num_agents=num_agents,
            map_name=map_name,
            num_episodes=1,
            step_delay_ms=args.delay,
            explore=not args.deterministic
        )
    else:
        print(f"[ERROR] Checkpoint not found: {target_cp or 'No checkpoints in experiments/ray_results/'}")
        sys.exit(1)

if __name__ == "__main__":
    main()
