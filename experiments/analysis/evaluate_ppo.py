#!/usr/bin/env python3
import os
import sys
import argparse
import json
import glob
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set headless environment for Pygame before loading imports
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Ensure HOME is writable for non-root container users
if os.environ.get("HOME") in (None, "", "/"):
    os.environ["HOME"] = str(PROJECT_ROOT)

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from overcooked_marl_env import OvercookedMARL
from experiments.training.train_ppo import ParallelEnvToMultiAgent

def find_latest_checkpoint():
    exp_dir = Path(__file__).resolve().parent.parent
    ray_results = exp_dir / "ray_results"
    if not ray_results.exists():
        raise FileNotFoundError("ray_results directory not found")
        
    checkpoints = sorted(
        glob.glob(str(ray_results / "**" / "checkpoint_*"), recursive=True),
        key=lambda p: os.path.getmtime(p),
        reverse=True
    )
    checkpoints = [c for c in checkpoints if os.path.isdir(c)]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in ray_results")
    return checkpoints[0]

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-episode statistical evaluation for trained PPO models.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory (default: latest in ray_results)")
    parser.add_argument("--num-episodes", type=int, default=20, help="Number of evaluation episodes to run (default: 20)")
    parser.add_argument("--map", type=str, default="default_map.txt", help="Map name (default: default_map.txt)")
    parser.add_argument("--num-agents", type=int, default=1, help="Number of agents (default: 1)")
    parser.add_argument("--max-steps", type=int, default=800, help="Max steps per episode (default: 800)")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--log-vision", action="store_true", default=False, help="Enable step-by-step vision analysis logging (default: False)")
    return parser.parse_args()

def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    print("\n" + "="*60)
    print("STARTING PPO MULTI-EPISODE EVALUATION")
    print(f"  Checkpoint:   {checkpoint_path}")
    print(f"  Episodes:     {args.num_episodes}")
    print(f"  Map:          {args.map}")
    print(f"  Agents:       {args.num_agents}")
    print(f"  Vision Log:   {'ENABLED' if args.log_vision else 'DISABLED'}")
    print("="*60 + "\n")
    
    # Determine experiment name from checkpoint path
    exp_name = "evaluation"
    parts = Path(checkpoint_path).parts
    if "ray_results" in parts:
        idx = parts.index("ray_results")
        if idx + 1 < len(parts):
            exp_name = parts[idx + 1]
            
    exp_dir = Path(__file__).resolve().parent.parent
    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if args.log_vision:
        vision_dir = log_dir / "vision_analysis"
        vision_dir.mkdir(parents=True, exist_ok=True)
        vision_log_filename = f"vision_analysis_{exp_name}.log"
        os.environ["OVERCOOKED_VISION_LOG"] = vision_log_filename
        vision_log_path = vision_dir / vision_log_filename
        if vision_log_path.exists():
            vision_log_path.unlink()
        print(f"  Vision Log:   {vision_log_path.name}")
    else:
        os.environ.pop("OVERCOOKED_VISION_LOG", None)
        vision_log_path = None
        print("  Vision Log:   DISABLED")
    print("="*60 + "\n")
    
    ray.init(ignore_reinit_error=True)
    
    # Environment Creator Fn
    def env_creator_fn():
        return OvercookedMARL(
            render_mode="human",
            map_path=args.map,
            num_fixed_agents=args.num_agents,
            verbose=False,
            featurized_obs=True
        )
        
    env_name = "overcooked_marl_parallel"
    register_env(env_name, lambda config: ParallelEnvToMultiAgent(env_creator_fn()))
    
    algo = None
    policies = None
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"[INFO] Algorithm loading fallback to direct Policy loading: {e}")
        from ray.rllib.policy.policy import Policy
        policies = Policy.from_checkpoint(checkpoint_path)
        
    episode_results = []
    
    for ep_idx in range(args.num_episodes):
        seed = args.base_seed + ep_idx
        env = env_creator_fn()
        obs, infos = env.reset(seed=seed)
        
        ep_rewards = {agent: 0.0 for agent in env.possible_agents}
        done = False
        step_count = 0
        
        while not done and step_count < args.max_steps:
            # Render step (writes step info to vision_analysis_<EXP_NAME>.log)
            env.render()
            
            # Compute actions from trained policy
            actions = {}
            for agent_id, agent_obs in obs.items():
                if algo is not None:
                    action = algo.compute_single_action(agent_obs, policy_id=f"policy_{agent_id}", explore=False)
                else:
                    policy_id = f"policy_{agent_id}"
                    pol = policies.get(policy_id) or next(iter(policies.values()))
                    res = pol.compute_single_action(agent_obs, explore=False)
                    action = res[0]
                    if isinstance(action, (tuple, list, np.ndarray)):
                        action = action[0]
                    action = int(action)
                actions[agent_id] = action

                
            obs, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id, r in rewards.items():
                ep_rewards[agent_id] += r
                
            done = any(terminations.values()) or any(truncations.values())
            step_count += 1
            
        env.render() # Final step render
        delivered = list(env.state.get("delivered_recipes", []))
        delivered_count = len(delivered)
        total_raw_reward = sum(ep_rewards.values())
        
        ep_summary = {
            "episode": ep_idx + 1,
            "seed": seed,
            "steps": step_count,
            "total_raw_reward": total_raw_reward,
            "agent_rewards": ep_rewards,
            "delivered_count": delivered_count,
            "delivered_recipes": delivered
        }
        episode_results.append(ep_summary)
        
        print(f"Episode {ep_idx+1:02d}/{args.num_episodes:02d} | Seed: {seed} | Steps: {step_count:03d} | Reward: {total_raw_reward:+7.2f} | Delivered: {delivered_count} {delivered}", flush=True)
        env.close()
        
    ray.shutdown()
    
    # Statistical Aggregation
    rewards_list = [ep["total_raw_reward"] for ep in episode_results]
    deliveries_list = [ep["delivered_count"] for ep in episode_results]
    steps_list = [ep["steps"] for ep in episode_results]
    
    stats = {
        "num_episodes": args.num_episodes,
        "reward_mean": float(np.mean(rewards_list)),
        "reward_std": float(np.std(rewards_list)),
        "reward_min": float(np.min(rewards_list)),
        "reward_max": float(np.max(rewards_list)),
        "deliveries_mean": float(np.mean(deliveries_list)),
        "deliveries_total": int(np.sum(deliveries_list)),
        "steps_mean": float(np.mean(steps_list)),
        "checkpoint_path": checkpoint_path,
        "episodes": episode_results
    }
    
    # Save default evaluation summary
    summary_json_path = log_dir / "evaluation_results.json"
    with open(summary_json_path, "w") as f:
        json.dump(stats, f, indent=2)
        
    # Also update/write experiment-specific log file in experiments/logs/
    exp_log_file = log_dir / f"{exp_name}.json"
    if exp_log_file.exists():
        try:
            with open(exp_log_file, "r") as f:
                exp_data = json.load(f)
            exp_data["evaluation"] = stats
            with open(exp_log_file, "w") as f:
                json.dump(exp_data, f, indent=2)
        except Exception:
            pass
    else:
        with open(log_dir / f"{exp_name}_evaluation.json", "w") as f:
            json.dump(stats, f, indent=2)
            
    # Update summary report in log_dir if present
    for report_file in [log_dir / f"{exp_name}_summary.md", log_dir / "latest_summary.md"]:
        if report_file.exists():
            with open(report_file, "a") as f:
                f.write(f"\n\n## Avaliação Pós-Treino ({args.num_episodes} episódios)\n")
                f.write(f"* **Recompensa Média:** {stats['reward_mean']:+.2f} (± {stats['reward_std']:.2f})\n")
                f.write(f"* **Mín / Máx Recompensa:** {stats['reward_min']:+.2f} / {stats['reward_max']:+.2f}\n")
                f.write(f"* **Média de Entregas/Ep:** {stats['deliveries_mean']:.2f} (Total: {stats['deliveries_total']})\n")

    # Update persistent results tracker with evaluation metrics
    try:
        from experiments.utils.results_tracker import record_experiment_entry
        record_experiment_entry({
            "experiment_id": exp_name,
            "map_name": args.map,
            "num_agents": args.num_agents,
            "eval_reward_mean": stats['reward_mean'],
            "eval_reward_std": stats['reward_std'],
            "eval_deliveries_mean": stats['deliveries_mean'],
            "eval_steps_mean": stats['steps_mean'],
            "status": "EVALUATED",
            "checkpoint_path": checkpoint_path
        })
    except Exception as tracker_err:
        print(f"[WARNING] Could not update results tracker: {tracker_err}")
        
    print("\n" + "="*60, flush=True)
    print("STATISTICAL EVALUATION SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"  Episodes Evaluated:   {args.num_episodes}", flush=True)
    print(f"  Mean Raw Reward:      {stats['reward_mean']:+.2f} (± {stats['reward_std']:.2f})", flush=True)
    print(f"  Min / Max Reward:     {stats['reward_min']:+.2f} / {stats['reward_max']:+.2f}", flush=True)
    print(f"  Mean Deliveries/Ep:   {stats['deliveries_mean']:.2f} (Total: {stats['deliveries_total']})", flush=True)
    print(f"  Mean Steps/Episode:   {stats['steps_mean']:.1f}", flush=True)
    if vision_log_path:
        print(f"  Vision Log saved to:  {vision_log_path}", flush=True)
    print(f"  Summary saved to:     {summary_json_path}", flush=True)
    print("="*60 + "\n", flush=True)

if __name__ == "__main__":
    main()
