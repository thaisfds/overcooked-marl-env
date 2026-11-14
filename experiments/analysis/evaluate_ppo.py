#!/usr/bin/env python3
import os
import sys
import argparse
import json
import glob
from pathlib import Path
from collections import Counter
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

from experiments.utils.results_tracker import detect_map_and_agents_from_checkpoint

def find_best_or_latest_checkpoint():
    exp_dir = Path(__file__).resolve().parent.parent
    ray_results = exp_dir / "ray_results"
    summary_json = exp_dir / "logs" / "experiments_summary.json"

    # 1. Prioritize selecting the best performing checkpoint from latest_summary.md or experiments_summary.json
    latest_md = exp_dir / "logs" / "latest_summary.md"
    if latest_md.exists():
        try:
            import re
            with open(latest_md, "r", encoding="utf-8") as f:
                content = f.read()
            m = re.search(r"\*\s*\*\*Checkpoint Path:\*\*\s*`([^`]+)`", content)
            if m:
                cand_path = Path(m.group(1)).resolve()
                if cand_path.exists():
                    print(f"🏆 Auto-selected best checkpoint from latest_summary.md: {cand_path.parent.name}")
                    return str(cand_path)
        except Exception:
            pass

    if summary_json.exists():

        try:
            with open(summary_json, "r", encoding="utf-8") as f:
                entries = json.load(f)
            if entries:
                latest_exp_id = entries[-1].get("experiment_id")
                latest_trials = [e for e in entries if e.get("experiment_id") == latest_exp_id and e.get("checkpoint_path")]
                if latest_trials:
                    def get_score(e):
                        val = e.get("train_reward_mean")
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return -float('inf')
                    latest_trials.sort(key=get_score, reverse=True)
                    best_cp = latest_trials[0].get("checkpoint_path")
                    if best_cp and Path(best_cp).exists():
                        print(f"🏆 Auto-selected best trial checkpoint from latest experiment '{latest_exp_id}': {Path(best_cp).parent.name}")
                        return str(Path(best_cp).resolve())
        except Exception:
            pass

    # 2. Fallback: scan experiments/ray_results directory
    if not ray_results.exists():
        raise FileNotFoundError("Directory experiments/ray_results not found. Pass --checkpoint <path> to evaluate a specific model.")

    checkpoints = [c for c in ray_results.glob("**/checkpoint_*") if c.is_dir() and (c / "rllib_checkpoint.json").exists()]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in experiments/ray_results. Pass --checkpoint <path> to evaluate a specific checkpoint.")

    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(checkpoints[0].resolve())


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-episode statistical evaluation for trained PPO models.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory (default: best in latest training)")
    parser.add_argument("--num-episodes", type=int, default=20, help="Number of evaluation episodes to run (default: 20)")
    parser.add_argument("--map", type=str, default=None, help="Map name (default: auto-detected from checkpoint)")
    parser.add_argument("--num-agents", type=int, default=None, help="Number of agents (default: auto-detected from checkpoint)")
    parser.add_argument("--max-steps", type=int, default=800, help="Max steps per episode (default: 800)")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--log-vision", action="store_true", default=False, help="Enable step-by-step vision analysis logging (default: False)")
    return parser.parse_args()

def main():
    args = parse_args()
    cp_p = Path(args.checkpoint or find_best_or_latest_checkpoint()).resolve()
    checkpoint_path = str(cp_p)

    # Auto-detect map and num_agents from checkpoint metadata if not explicitly provided
    det_map, det_agents = detect_map_and_agents_from_checkpoint(cp_p)
    map_name = args.map if args.map is not None else det_map
    num_agents = args.num_agents if args.num_agents is not None else det_agents

    # Determine experiment name and trial_id from checkpoint path
    exp_name = "evaluation"
    trial_id = "default"
    parts = cp_p.parts
    if "ray_results" in parts:
        idx = parts.index("ray_results")
        if idx + 1 < len(parts):
            exp_name = parts[idx + 1]

    for p in [cp_p, cp_p.parent, cp_p.parents[1]]:
        m_info_file = p / "model_info.json"
        if m_info_file.exists():
            try:
                with open(m_info_file, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                if "trial_id" in m_data:
                    trial_id = m_data["trial_id"]
                    break
            except Exception:
                pass

            
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
            map_path=map_name,
            num_fixed_agents=num_agents,
            verbose=False,
            featurized_obs=True
        )
        
    for env_n in [
        "overcooked_marl_parallel",
        "overcooked_marl_parallel_impala",
        "overcooked_marl_parallel_appo",
        "overcooked_marl_parallel_dqn"
    ]:
        register_env(env_n, lambda config: ParallelEnvToMultiAgent(env_creator_fn()))
    
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
                    try:
                        action = algo.compute_single_action(agent_obs, policy_id=f"policy_{agent_id}", explore=False)
                    except Exception:
                        action = algo.compute_single_action(agent_obs, policy_id="policy_agent_0", explore=False)
                else:
                    policy_id = f"policy_{agent_id}"
                    pol = policies.get(policy_id) or policies.get("policy_agent_0") or next(iter(policies.values()))
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
    
    # Per-agent performance metrics
    all_agents = sorted(list(episode_results[0]["agent_rewards"].keys())) if episode_results else []
    agent_stats = {}
    for ag in all_agents:
        ag_rewards = [ep["agent_rewards"].get(ag, 0.0) for ep in episode_results]
        agent_stats[ag] = {
            "mean": float(np.mean(ag_rewards)),
            "std": float(np.std(ag_rewards)),
            "min": float(np.min(ag_rewards)),
            "max": float(np.max(ag_rewards))
        }

    # Recipe breakdown
    all_delivered = [r for ep in episode_results for r in ep["delivered_recipes"]]
    recipe_counts = dict(Counter(all_delivered))
    total_deliveries = int(np.sum(deliveries_list))
    max_deliveries_single_ep = int(np.max(deliveries_list)) if deliveries_list else 0

    # Success rates
    episodes_with_delivery = sum(1 for ep in episode_results if ep["delivered_count"] > 0)
    episodes_multi_delivery = sum(1 for ep in episode_results if ep["delivered_count"] > 1)
    delivery_success_rate = (episodes_with_delivery / args.num_episodes * 100.0) if args.num_episodes > 0 else 0.0
    multi_delivery_rate = (episodes_multi_delivery / args.num_episodes * 100.0) if args.num_episodes > 0 else 0.0

    stats = {
        "num_episodes": args.num_episodes,
        "reward_mean": float(np.mean(rewards_list)),
        "reward_std": float(np.std(rewards_list)),
        "reward_min": float(np.min(rewards_list)),
        "reward_max": float(np.max(rewards_list)),
        "deliveries_mean": float(np.mean(deliveries_list)),
        "deliveries_total": total_deliveries,
        "deliveries_max_single_ep": max_deliveries_single_ep,
        "delivery_success_rate": float(delivery_success_rate),
        "multi_delivery_rate": float(multi_delivery_rate),
        "recipe_breakdown": recipe_counts,
        "agent_rewards": agent_stats,
        "steps_mean": float(np.mean(steps_list)),
        "checkpoint_path": checkpoint_path,
        "map_name": map_name,
        "num_agents": num_agents,
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
            
    # Format per-agent and recipe text for reports
    agent_md_lines = []
    for ag, a_stat in agent_stats.items():
        agent_md_lines.append(f"* **{ag}:** {a_stat['mean']:+.2f} (± {a_stat['std']:.2f}) [Mín: {a_stat['min']:+.2f} | Máx: {a_stat['max']:+.2f}]")
    agent_md_text = "\n".join(agent_md_lines) if agent_md_lines else "* N/A"

    recipe_md_lines = []
    for r_name, r_cnt in sorted(recipe_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (r_cnt / total_deliveries * 100.0) if total_deliveries > 0 else 0.0
        recipe_md_lines.append(f"* **{r_name}:** {r_cnt} ({pct:.1f}%)")
    recipe_md_text = "\n".join(recipe_md_lines) if recipe_md_lines else "* Nenhuma receita entregue"

    eval_md_block = f"""

## Avaliação Pós-Treino ({args.num_episodes} episódios)
* **Recompensa Média Global:** {stats['reward_mean']:+.2f} (± {stats['reward_std']:.2f}) [Mín: {stats['reward_min']:+.2f} | Máx: {stats['reward_max']:+.2f}]
* **Média de Entregas/Ep:** {stats['deliveries_mean']:.2f} (Total: {total_deliveries} | Máx em 1 ep: {max_deliveries_single_ep})
* **Taxa de Sucesso (≥1 entrega):** {delivery_success_rate:.1f}% ({episodes_with_delivery}/{args.num_episodes} episódios)
* **Taxa de Entregas Múltiplas (≥2):** {multi_delivery_rate:.1f}% ({episodes_multi_delivery}/{args.num_episodes} episódios)
* **Duração Média:** {stats['steps_mean']:.1f} passos/episódio

### Desempenho por Agente:
{agent_md_text}

### Receitas Entregues:
{recipe_md_text}
"""
    # Update summary report in log_dir if present
    for report_file in [log_dir / f"{exp_name}_summary.md", log_dir / "latest_summary.md"]:
        if report_file.exists():
            with open(report_file, "a") as f:
                f.write(eval_md_block)

    # Update persistent results tracker with evaluation metrics
    try:
        from experiments.utils.results_tracker import record_experiment_entry
        record_experiment_entry({
            "experiment_id": exp_name,
            "trial_id": trial_id,
            "map_name": map_name,
            "num_agents": num_agents,
            "eval_reward_mean": stats['reward_mean'],
            "eval_reward_std": stats['reward_std'],
            "eval_deliveries_mean": stats['deliveries_mean'],
            "eval_steps_mean": stats['steps_mean'],
            "status": "EVALUATED",
            "checkpoint_path": checkpoint_path
        })
    except Exception as tracker_err:
        print(f"[WARNING] Could not update results tracker: {tracker_err}")
        
    print("\n" + "="*70, flush=True)
    print("           RELATÓRIO ESTATÍSTICO DE AVALIAÇÃO MULTIAGENTE", flush=True)
    print("="*70, flush=True)
    print(f"  Modelo / Checkpoint: {checkpoint_path}", flush=True)
    print(f"  Mapa: {map_name} | Agentes: {num_agents} | Episódios: {args.num_episodes}", flush=True)
    print("-"*70, flush=True)
    print("  DESEMPENHO GLOBAL:", flush=True)
    print(f"  • Recompensa Média:   {stats['reward_mean']:+.2f} (± {stats['reward_std']:.2f}) [Mín: {stats['reward_min']:+.2f} | Máx: {stats['reward_max']:+.2f}]", flush=True)
    print(f"  • Entregas por Ep:    {stats['deliveries_mean']:.2f} (Total: {total_deliveries} pratos | Máx/Ep: {max_deliveries_single_ep})", flush=True)
    print(f"  • Taxa de Sucesso:    {delivery_success_rate:.1f}% dos episódios com ≥1 entrega", flush=True)
    print(f"  • Entregas Múltiplas: {multi_delivery_rate:.1f}% dos episódios com ≥2 entregas", flush=True)
    print(f"  • Passos Médios/Ep:   {stats['steps_mean']:.1f}", flush=True)
    print("-"*70, flush=True)
    print("  RECOMPENSA INDIVIDUAL POR AGENTE:", flush=True)
    for ag, a_stat in agent_stats.items():
        print(f"  • {ag:<10}: {a_stat['mean']:+7.2f} (± {a_stat['std']:.2f}) [Mín: {a_stat['min']:+7.2f} | Máx: {a_stat['max']:+7.2f}]", flush=True)
    print("-"*70, flush=True)
    print("  DISTRIBUIÇÃO DE RECEITAS ENTREGUES:", flush=True)
    if recipe_counts:
        for r_name, r_cnt in sorted(recipe_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (r_cnt / total_deliveries * 100.0) if total_deliveries > 0 else 0.0
            print(f"  • {r_name:<16}: {r_cnt:3d} pratos ({pct:5.1f}%)", flush=True)
    else:
        print("  • Nenhuma receita entregue no período avaliado.", flush=True)
    if vision_log_path:
        print("-"*70, flush=True)
        print(f"  Vision Log salvo em:  {vision_log_path}", flush=True)
    print(f"  JSON de Métricas em:  {summary_json_path}", flush=True)
    print("="*70, flush=True)
    print("  Para reproduzir visualmente no Pygame:", flush=True)
    print(f"  venv/bin/python experiments/visualization/visualize_model.py {checkpoint_path}", flush=True)
    print("="*70 + "\n", flush=True)

if __name__ == "__main__":
    main()
