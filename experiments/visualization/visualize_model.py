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
        return {agent: self.env.observation_space(agent) for agent in self.agents}

    @property
    def action_space(self):
        return {agent: self.env.action_space(agent) for agent in self.agents}

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action_dict):
        return self.env.step(action_dict)

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


from experiments.utils.results_tracker import detect_map_and_agents_from_checkpoint

def discover_latest_or_best_checkpoint(ray_results_dir="experiments/ray_results"):
    exp_dir = project_root / "experiments"
    summary_candidates = [
        exp_dir / "logs" / "experiments_summary.json"
    ]
    
    best_cp = None
    best_score = -float('inf')
    num_agents = None
    map_name = None
    
    # 1. Prioritize reading persistent experiment history from logs/experiments_summary.json
    for summary_json in summary_candidates:
        if summary_json.exists():
            try:
                with open(summary_json, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                    
                for e in entries:
                    cp_str = e.get("checkpoint_path", "")
                    if not cp_str:
                        continue
                    cp_path = Path(cp_str).resolve()
                    if not cp_path.exists():
                        continue
                        
                    score = -float('inf')
                    eval_rw = e.get("eval_reward_mean")
                    train_rw = e.get("train_reward_mean")
                    if isinstance(eval_rw, (int, float)):
                        score = float(eval_rw)
                    elif isinstance(train_rw, (int, float)):
                        score = float(train_rw)
                        
                    if score > best_score or best_cp is None:
                        best_score = score
                        best_cp = cp_path
                        num_agents = int(e.get("num_agents", 1)) if e.get("num_agents") else None
                        map_name = e.get("map_name")
            except Exception:
                pass

    if best_cp:
        det_map, det_agents = detect_map_and_agents_from_checkpoint(best_cp)
        map_name = map_name or det_map
        num_agents = num_agents or det_agents
        print(f"🏆 Best checkpoint discovered from history:")
        print(f"   Checkpoint: {best_cp}")
        print(f"   Score:      {best_score:+.2f} | Map: {map_name} | Agents: {num_agents}")
        return best_cp, num_agents, map_name

    # 2. Fallback: Scan ray_results directories for checkpoint folders
    results_path = Path(ray_results_dir).resolve()
    if not results_path.exists():
        results_path = exp_dir / "ray_results"

    search_dirs = [results_path]

    checkpoints = []
    seen_paths = set()
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for cp_path in s_dir.glob("**/checkpoint_*"):
            if cp_path.is_dir() and (cp_path / "rllib_checkpoint.json").exists():
                r_cp = cp_path.resolve()
                if r_cp in seen_paths:
                    continue
                seen_paths.add(r_cp)
                
                avg_reward = -999999.0
                m_map, n_agents = detect_map_and_agents_from_checkpoint(r_cp)
                
                for info_dir in [r_cp.parent, r_cp.parents[1]]:
                    info_file = info_dir / "model_info.json"
                    if info_file.exists():
                        try:
                            with open(info_file, "r") as f:
                                meta = json.load(f)
                                avg_reward = float(meta.get("avg_reward", -999999.0))
                                break
                        except Exception:
                            pass
                            
                iter_num = int(cp_path.name.split("_")[-1]) if cp_path.name.split("_")[-1].isdigit() else 0
                mtime = cp_path.stat().st_mtime
                checkpoints.append((r_cp, avg_reward, iter_num, mtime, n_agents, m_map))
                
    checkpoints.sort(key=lambda x: (x[1], x[3]), reverse=True)
    if not checkpoints:
        return None, 2, "tutorial.txt"
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

    algorithm = None
    policies = None
    try:
        algorithm = Algorithm.from_checkpoint(str(cp_abs_path))
    except Exception as e:
        print(f"[INFO] Algorithm loading fallback to direct Policy loading: {e}")
        from ray.rllib.policy.policy import Policy
        policies = Policy.from_checkpoint(str(cp_abs_path))

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
                        policy_id = f"policy_{agent_id}"
                        if algorithm is not None:
                            try:
                                raw_action = algorithm.compute_single_action(obs, policy_id=policy_id, explore=explore)
                            except Exception:
                                raw_action = algorithm.compute_single_action(obs, policy_id="policy_agent_0", explore=explore)
                        else:
                            pol = policies.get(policy_id) or policies.get("policy_agent_0") or next(iter(policies.values()))
                            res = pol.compute_single_action(obs, explore=explore)
                            raw_action = res[0]
                        if isinstance(raw_action, (tuple, list, np.ndarray)):
                            raw_action = raw_action[0]
                        action = int(raw_action)
                    except Exception as err:
                        print(f"Error computing action for {agent_id}: {err}")
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
    parser.add_argument("--map", type=str, default=None, help="Map name (default: auto-detected from checkpoint)")
    parser.add_argument("--num-agents", type=int, default=None, help="Number of agents (default: auto-detected from checkpoint)")
    parser.add_argument("--delay", type=int, default=50, help="Step delay in ms (e.g. 50 = 20 FPS)")
    parser.add_argument("--deterministic", action="store_true", help="Disable exploration sampling (greedy actions)")
    args = parser.parse_args()

    target_cp = args.checkpoint_path or args.checkpoint

    if target_cp:
        cp_path = Path(target_cp).resolve()
        det_map, det_agents = detect_map_and_agents_from_checkpoint(cp_path)
        map_name = args.map if args.map is not None else det_map
        num_agents = args.num_agents if args.num_agents is not None else det_agents
    else:
        cp_path, det_agents, det_map = discover_latest_or_best_checkpoint()
        map_name = args.map if args.map is not None else det_map
        num_agents = args.num_agents if args.num_agents is not None else det_agents

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

