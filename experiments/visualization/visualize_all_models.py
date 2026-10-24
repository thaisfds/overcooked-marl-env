#!/usr/bin/env python3
"""
Multi-Model Visualizer for Overcooked MARL
Visualizes the first N seconds (default: 20s) of each trained model found in experiments/ray_results.

Usage:
  # Visualize all models (20s each, 20 FPS):
  python experiments/visualization/visualize_all_models.py

  # Custom duration and step delay:
  python experiments/visualization/visualize_all_models.py --seconds 20 --delay 50

  # Limit to top 5 models sorted by rank:
  python experiments/visualization/visualize_all_models.py --top-n 5

  # Specify custom summary markdown file:
  python experiments/visualization/visualize_all_models.py --summary experiments/logs/PPO_Marl_20261006_215831_summary.md

Controls:
  SPACE         : Pause / Resume simulation
  N / RIGHT ARROW: Skip to Next model
  P / LEFT ARROW : Go back to Previous model
  R             : Restart current model rollout
  Q / ESC       : Quit visualizer
"""

import os
import re
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ["OVERCOOKED_VISION_LOG"] = "batch_rollout.log"
os.environ["RAY_ACCELERATOR_MANAGER_LOG_LEVEL"] = "ERROR"

import pygame
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

try:
    from overcooked_marl_env import OvercookedMARL
    from overcooked_marl_env.core.recipes import OrderQueue
except ImportError:
    from env import OvercookedMARL
    from core.recipes import OrderQueue

from experiments.utils.results_tracker import detect_map_and_agents_from_checkpoint


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


def find_latest_checkpoint(trial_dir: Path) -> Path:
    """Returns the highest-indexed checkpoint directory inside a trial directory."""
    checkpoints = list(trial_dir.glob("checkpoint_*"))
    if not checkpoints:
        return None

    def cp_index(cp_path):
        parts = cp_path.name.split("_")
        if parts[-1].isdigit():
            return int(parts[-1])
        return 0

    checkpoints.sort(key=cp_index)
    return checkpoints[-1]


def parse_summary_markdown(summary_path: Path, ray_results_dir: Path):
    """Parses a session summary markdown file to extract ranked trial metadata."""
    models = []
    if not summary_path.exists():
        return models

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            # Match table line: | Rank | Trial ID | LR | Gamma | Entropy | Recompensa Média | Checkpoint |
            m = re.search(
                r'\|\s*(\d+)\s*\|\s*`?([\w_]+)`?\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*\*\*([^\*]+)\*\*\s*\|\s*`?(?:\.\.\./)?(checkpoint_\d+)`?',
                line
            )
            if m:
                rank_str, trial_id, lr, gamma, entropy, reward_str, cp_name = m.groups()

                # Find matching trial folder inside ray_results_dir
                matching_dirs = list(ray_results_dir.glob(f"**/*_{trial_id}_*"))
                cp_path = None
                if matching_dirs:
                    trial_dir = matching_dirs[0]
                    target_cp = trial_dir / cp_name
                    if target_cp.exists():
                        cp_path = target_cp
                    else:
                        cp_path = find_latest_checkpoint(trial_dir)

                if cp_path and cp_path.exists():
                    m_map, m_agents = detect_map_and_agents_from_checkpoint(cp_path)
                    models.append({
                        "rank": int(rank_str),
                        "trial_id": trial_id,
                        "lr": float(lr),
                        "gamma": float(gamma),
                        "entropy": float(entropy),
                        "reward_str": reward_str.strip(),
                        "reward_val": float(reward_str.replace("+", "").strip()) if reward_str.replace("+", "").replace(".", "").replace("-", "").isdigit() else 0.0,
                        "cp_path": cp_path,
                        "trial_dir": cp_path.parent,
                        "map": m_map,
                        "num_agents": m_agents
                    })
    except Exception as e:
        print(f"[WARN] Error parsing summary markdown ({summary_path.name}): {e}")

    return models


def discover_all_models(ray_results_dir: Path, summary_file: Path = None, sort_by: str = "rank"):
    """Discovers all trained models across ray_results directory."""
    exp_logs_dir = project_root / "experiments" / "logs"

    # 1. Try parsing specified or recent summary markdown
    summary_candidates = []
    if summary_file:
        summary_candidates.append(Path(summary_file))

    for summary_md in sorted(exp_logs_dir.glob("*_summary.md"), reverse=True):
        summary_candidates.append(summary_md)

    models = []
    for sc in summary_candidates:
        if sc.exists():
            models = parse_summary_markdown(sc, ray_results_dir)
            if models:
                print(f"[INFO] Successfully loaded {len(models)} models from summary report: {sc.name}")
                break

    # 2. Fallback or scan all trial directories directly in ray_results
    search_roots = [ray_results_dir]


    discovered_trials = []
    seen_trials = set()
    for s_root in search_roots:
        if s_root.exists():
            for t_dir in s_root.glob("**/PPO_overcooked_marl_parallel_*"):
                if t_dir.is_dir() and t_dir.resolve() not in seen_trials:
                    seen_trials.add(t_dir.resolve())
                    discovered_trials.append(t_dir)

    known_trial_ids = {m["trial_id"] for m in models}

    for trial_dir in sorted(discovered_trials):
        if not trial_dir.is_dir():
            continue

        # Extract trial ID from directory name
        dir_name = trial_dir.name
        t_match = re.search(r'parallel_([a-zA-Z0-9]+_\d+)', dir_name)
        trial_id = t_match.group(1) if t_match else dir_name[:15]

        if trial_id in known_trial_ids:
            continue

        cp_path = find_latest_checkpoint(trial_dir)
        if not cp_path or not cp_path.exists():
            continue

        # Extract hyperparameters from directory name or params.json
        lr = 0.0001
        gamma = 0.98
        entropy = 0.01
        reward_val = -9999.0
        reward_str = "N/A"

        lr_m = re.search(r'lr=([\d\.]+)', dir_name)
        if lr_m:
            lr = float(lr_m.group(1))
        gamma_m = re.search(r'gamma=([\d\.]+)', dir_name)
        if gamma_m:
            gamma = float(gamma_m.group(1))
        ent_m = re.search(r'entropy_coeff=([\d\.]+)', dir_name)
        if ent_m:
            entropy = float(ent_m.group(1))

        info_json = trial_dir / "model_info.json"
        if info_json.exists():
            try:
                with open(info_json, "r") as f:
                    meta = json.load(f)
                    if "avg_reward" in meta:
                        reward_val = float(meta["avg_reward"])
                        reward_str = f"+{reward_val:.2f}"
            except Exception:
                pass

        m_map, m_agents = detect_map_and_agents_from_checkpoint(cp_path)

        models.append({
            "rank": len(models) + 1,
            "trial_id": trial_id,
            "lr": lr,
            "gamma": gamma,
            "entropy": entropy,
            "reward_str": reward_str,
            "reward_val": reward_val,
            "cp_path": cp_path,
            "trial_dir": trial_dir,
            "map": m_map,
            "num_agents": m_agents
        })

    # Sort models based on user preference
    if sort_by == "rank":
        models.sort(key=lambda x: x["rank"])
    elif sort_by == "reward":
        models.sort(key=lambda x: x["reward_val"], reverse=True)
    elif sort_by == "trial":
        models.sort(key=lambda x: x["trial_id"])

    return models



def draw_hud_overlay(surface, model_info, model_idx, total_models, elapsed_sec, max_seconds, current_step, max_steps, current_reward, delivered_list, pending_list, is_paused):
    """Draws a sleek visual HUD overlay on top of Pygame window."""
    width, height = surface.get_size()

    # Create semi-transparent overlay banner at the top (height: 80px)
    hud_bar = pygame.Surface((width, 82), pygame.SRCALPHA)
    hud_bar.fill((18, 22, 32, 230))  # Sleek dark glass background
    surface.blit(hud_bar, (0, 0))

    # Bottom controls bar (height: 28px)
    ctrl_bar = pygame.Surface((width, 28), pygame.SRCALPHA)
    ctrl_bar.fill((12, 14, 20, 230))
    surface.blit(ctrl_bar, (0, height - 28))

    try:
        font_title = pygame.font.SysFont("monospace", 14, bold=True)
        font_sub = pygame.font.SysFont("monospace", 12)
        font_ctrl = pygame.font.SysFont("monospace", 11)
    except Exception:
        font_title = pygame.font.Font(None, 18)
        font_sub = pygame.font.Font(None, 15)
        font_ctrl = pygame.font.Font(None, 14)

    # Line 1: Title & Rank
    rank_str = f"Rank #{model_info['rank']}" if model_info.get('rank') else ""
    title_text = f"[{model_idx + 1}/{total_models}] Trial: {model_info['trial_id']} | {rank_str} | Train Rw: {model_info['reward_str']}"
    t_surf = font_title.render(title_text, True, (255, 215, 0))  # Gold color
    surface.blit(t_surf, (10, 6))

    # Line 2: Hyperparams & Timer
    m_map_str = model_info.get("map", "default_map.txt")
    m_ag_str = model_info.get("num_agents", 1)
    hp_text = f"Map: {m_map_str} ({m_ag_str} ag) | LR: {model_info['lr']} | Gamma: {model_info['gamma']} | Entropy: {model_info['entropy']}"
    h_surf = font_sub.render(hp_text, True, (180, 220, 255))
    surface.blit(h_surf, (10, 26))

    # Line 3: Delivered & Pending Dishes
    del_str = ", ".join(delivered_list) if delivered_list else "None"
    pend_str = ", ".join(pending_list[:3]) if pending_list else "None"
    if len(pending_list) > 3:
        pend_str += f" (+{len(pending_list)-3} more)"

    dish_text = f"Delivered ({len(delivered_list)}): {del_str} | Pending ({len(pending_list)}): {pend_str}"
    d_surf = font_sub.render(dish_text, True, (255, 230, 150))
    surface.blit(d_surf, (10, 44))

    # Progress & Score (Right Aligned)
    pause_tag = " [PAUSED]" if is_paused else ""
    timer_text = f"Time: {elapsed_sec:04.1f}s / {max_seconds:04.1f}s | Step: {current_step}/{max_steps}{pause_tag}"
    score_text = f"Rollout Reward: +{current_reward:.2f}"
    
    timer_surf = font_sub.render(timer_text, True, (100, 255, 180) if not is_paused else (255, 100, 100))
    score_surf = font_sub.render(score_text, True, (255, 255, 255))
    
    surface.blit(timer_surf, (10, 62))
    surface.blit(score_surf, (width - score_surf.get_width() - 12, 62))

    # Line 4: Bottom Controls
    ctrl_text = "[SPACE] Pause | [N / Right] Next Model | [P / Left] Prev Model | [R] Restart | [Q / ESC] Quit"
    c_surf = font_ctrl.render(ctrl_text, True, (180, 180, 190))
    surface.blit(c_surf, (10, height - 22))


def generate_execution_summary_report(executed_logs, summary_md_path):
    """Parses executed log files and generates a comprehensive evaluation report."""
    results = []
    for log_path in executed_logs:
        if not log_path.exists():
            continue

        info = {
            "rank": "N/A",
            "trial_id": "N/A",
            "lr": "N/A",
            "gamma": "N/A",
            "entropy": "N/A",
            "train_reward": "N/A",
            "eval_reward": 0.0,
            "eval_reward_str": "0.00",
            "delivered_count": 0,
            "delivered_dishes": "None",
            "pending_dishes": "None",
            "log_path": log_path
        }

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse Header
            m_rank = re.search(r"Rank\s*:\s*#?([\w/]+)", content)
            if m_rank:
                info["rank"] = m_rank.group(1)

            m_trial = re.search(r"Trial ID\s*:\s*([\w_]+)", content)
            if m_trial:
                info["trial_id"] = m_trial.group(1)

            m_hp = re.search(r"LR=([\d\.\-]+)\s*\|\s*Gamma=([\d\.\-]+)\s*\|\s*Entropy=([\d\.\-]+)", content)
            if m_hp:
                info["lr"] = m_hp.group(1)
                info["gamma"] = m_hp.group(2)
                info["entropy"] = m_hp.group(3)

            m_trw = re.search(r"Train Score\s*:\s*([^\n]+)", content)
            if m_trw:
                info["train_reward"] = m_trw.group(1).strip()

            # Parse Footer Summary
            m_evrw = re.search(r"Total Reward\s*:\s*([^\n]+)", content)
            if m_evrw:
                info["eval_reward_str"] = m_evrw.group(1).strip()
                try:
                    info["eval_reward"] = float(info["eval_reward_str"].replace("+", ""))
                except Exception:
                    pass

            m_del = re.search(r"Delivered Dishes\s*\(([^)]+)\)\s*:\s*([^\n]+)", content)
            if m_del:
                try:
                    info["delivered_count"] = int(m_del.group(1))
                except Exception:
                    info["delivered_count"] = 0
                info["delivered_dishes"] = m_del.group(2).strip()

            m_pend = re.search(r"Pending Dishes\s*\(([^)]+)\)\s*:\s*([^\n]+)", content)
            if m_pend:
                info["pending_dishes"] = m_pend.group(2).strip()

            results.append(info)
        except Exception as e:
            print(f"[WARN] Failed to parse log file {log_path.name}: {e}")

    if not results:
        return

    # Sort results by eval_reward descending
    results_sorted = sorted(results, key=lambda x: x["eval_reward"], reverse=True)

    # 1. Generate Markdown Summary Report
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    md_lines = []
    md_lines.append("# Relatório de Visualização - Rollout 20s dos Modelos")
    md_lines.append(f"* **Data/Hora Execução:** `{now_str}`")
    md_lines.append(f"* **Total de Modelos Avaliados:** {len(results)}")
    md_lines.append(f"* **Diretório dos Logs:** [`experiments/logs/vision_analysis/models_20s/`](file://{project_root}/experiments/logs/vision_analysis/models_20s/)")
    md_lines.append("")
    md_lines.append("## 📊 Tabela Comparativa de Desempenho Visual (Ordenada pela Recompensa de 20s)")
    md_lines.append("")
    md_lines.append("| Rank Grid | Trial ID | LR | Gamma | Entropy | Recompensa Treino | **Recompensa 20s** | Pratos Entregues | Arquivo de Log |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for r in results_sorted:
        rel_log = r["log_path"].relative_to(project_root)
        md_lines.append(
            f"| #{r['rank']} | `{r['trial_id']}` | {r['lr']} | {r['gamma']} | {r['entropy']} | {r['train_reward']} | **{r['eval_reward_str']}** | {r['delivered_count']} ({r['delivered_dishes']}) | [`{r['log_path'].name}`](file://{r['log_path']}) |"
        )

    md_lines.append("")
    md_lines.append("## 🏆 Destaques do Rollout Visual")
    best_eval = results_sorted[0]
    most_del = max(results, key=lambda x: x["delivered_count"])

    md_lines.append(f"* **Melhor Desempenho (Score 20s):** Trial `{best_eval['trial_id']}` (Rank #{best_eval['rank']}) com Recompensa de **{best_eval['eval_reward_str']}**")
    md_lines.append(f"* **Maior Número de Entregas:** Trial `{most_del['trial_id']}` com **{most_del['delivered_count']} entregas** em 20s")
    md_lines.append("")

    try:
        summary_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"\n[INFO] Evaluation summary report saved at: {summary_md_path.relative_to(project_root)}")
    except Exception as e:
        print(f"[WARN] Failed to write summary markdown: {e}")

    # 2. Print Summary Table in Terminal
    print("\n" + "=" * 95)
    print(f"📊 RELATÓRIO FINAL DA EXECUÇÃO VISUAL ({len(results)} MODELOS AVALIADOS)")
    print("=" * 95)
    print(f"{'Rank':<6} | {'Trial ID':<15} | {'LR':<8} | {'Gamma':<6} | {'Ent':<6} | {'Train Rw':<10} | {'Score 20s':<10} | {'Entregas':<8}")
    print("-" * 95)
    for r in results_sorted:
        print(f"#{r['rank']:<5} | {r['trial_id']:<15} | {r['lr']:<8} | {r['gamma']:<6} | {r['entropy']:<6} | {r['train_reward']:<10} | {r['eval_reward_str']:<10} | {r['delivered_count']} prato(s)")
    print("=" * 95)
    print(f"🏆 Melhor Modelo no Rollout 20s : Trial {best_eval['trial_id']} (Score 20s: {best_eval['eval_reward_str']})")
    print(f"📁 Arquivo de Relatório Salvo  : {summary_md_path.relative_to(project_root)}")
    print("=" * 95 + "\n")


def run_models_visualizer(models, max_seconds=20.0, step_delay_ms=50, num_agents=None, map_name=None, explore=True, seed=42):
    """Runs Pygame visual rollout loop for each model for max_seconds."""
    if not models:
        print("[ERROR] No valid models to visualize!")
        return

    executed_log_files = []

    print("\n" + "="*75)
    print(f"🎮 STARTING MULTI-MODEL VISUALIZER ({len(models)} MODELS FOUND)")
    print(f"  Duration per model : {max_seconds:.1f} seconds")
    print(f"  Step delay         : {step_delay_ms} ms (~{1000 // step_delay_ms} FPS)")
    if map_name or num_agents:
        print(f"  Explicit Override  : Map={map_name} | Agents={num_agents}")
    else:
        print(f"  Dynamic Env Config : Auto-detecting per model")
    print(f"  Exploration        : {explore} | Evaluation Seed: {seed}")
    print("="*75)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    active_env = None
    active_map = None
    active_agents = None

    def get_or_create_env(target_map, target_agents):
        nonlocal active_env, active_map, active_agents
        if active_env is not None and active_map == target_map and active_agents == target_agents:
            return active_env

        if active_env is not None:
            try:
                active_env.close()
            except Exception:
                pass
            active_env = None

        def env_creator_fn():
            return OvercookedMARL(
                render_mode="human",
                map_path=target_map,
                num_fixed_agents=target_agents,
                verbose=False,
                featurized_obs=True
            )

        try:
            register_env("overcooked_marl_parallel", lambda config: ParallelEnvToMultiAgent(env_creator_fn()))
        except Exception:
            pass

        active_env = ParallelEnvToMultiAgent(env_creator_fn())
        active_map = target_map
        active_agents = target_agents
        return active_env

    max_steps = int(max_seconds * (1000.0 / step_delay_ms))
    current_model_idx = 0
    total_models = len(models)

    try:
        while 0 <= current_model_idx < total_models:
            model_info = models[current_model_idx]
            cp_path = model_info["cp_path"]

            m_target_map = map_name if map_name is not None else model_info.get("map", "default_map.txt")
            m_target_agents = num_agents if num_agents is not None else model_info.get("num_agents", 1)
            env = get_or_create_env(m_target_map, m_target_agents)

            log_dir = project_root / "experiments" / "logs" / "vision_analysis" / "models_20s"
            log_dir.mkdir(parents=True, exist_ok=True)
            # Use exact trial directory name from ray results
            log_filename = f"{cp_path.parent.name}.log"
            log_filepath = log_dir / log_filename

            # Initialize model-specific log file
            with open(log_filepath, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("📋 OVERCOOKED MARL - VISUAL ROLLOUT LOG (20 SECONDS)\n")
                f.write("=" * 80 + "\n")
                f.write(f"Model Index   : [{current_model_idx + 1} / {total_models}]\n")
                f.write(f"Rank          : #{model_info.get('rank', 'N/A')}\n")
                f.write(f"Trial ID      : {model_info['trial_id']}\n")
                f.write(f"Checkpoint    : {cp_path}\n")
                f.write(f"Hyperparams   : LR={model_info['lr']} | Gamma={model_info['gamma']} | Entropy={model_info['entropy']}\n")
                f.write(f"Train Score   : {model_info['reward_str']}\n")
                f.write(f"Evaluation    : Map={m_target_map} | Agents={m_target_agents} | Max Duration={max_seconds:.1f}s | Seed={seed}\n")
                f.write("=" * 80 + "\n\n")

            os.environ["OVERCOOKED_VISION_LOG"] = str(log_filepath)

            print(f"\n" + "-"*70)
            print(f"▶ [{current_model_idx + 1}/{total_models}] Loading Model Trial: {model_info['trial_id']}")
            print(f"  Checkpoint : {cp_path}")
            print(f"  Map/Agents : {m_target_map} | {m_target_agents} agent(s)")
            print(f"  Hyperparams: LR={model_info['lr']} | Gamma={model_info['gamma']} | Entropy={model_info['entropy']}")
            print(f"  Train Score: {model_info['reward_str']}")
            print(f"  Log File   : {log_filepath.name}")
            print("-" * 70)

            algorithm = None
            policies = None
            try:
                algorithm = Algorithm.from_checkpoint(str(cp_path.resolve()))
            except Exception as e:
                from ray.rllib.policy.policy import Policy
                try:
                    policies = Policy.from_checkpoint(str(cp_path.resolve()))
                except Exception as p_err:
                    print(f"[ERROR] Failed to load checkpoint {cp_path.name}: {p_err}")
                    current_model_idx += 1
                    continue

            obs_dict, infos = env.reset(seed=seed)
            episode_reward = 0.0
            step_count = 0
            is_paused = False
            user_nav_action = None  # 'next', 'prev', 'restart', 'quit'

            while step_count < max_steps:
                # Process Pygame window input events
                try:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            user_nav_action = 'quit'
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                                user_nav_action = 'quit'
                                break
                            elif event.key == pygame.K_SPACE:
                                is_paused = not is_paused
                                print(f"[INFO] Playback {'PAUSED' if is_paused else 'RESUMED'}")
                            elif event.key in (pygame.K_RIGHT, pygame.K_n):
                                user_nav_action = 'next'
                                break
                            elif event.key in (pygame.K_LEFT, pygame.K_p):
                                user_nav_action = 'prev'
                                break
                            elif event.key == pygame.K_r:
                                user_nav_action = 'restart'
                                break
                except Exception:
                    pass

                if user_nav_action:
                    break

                # Extract delivered and pending dishes
                raw_env = getattr(env, 'env', env)
                if hasattr(raw_env, 'env'):
                    raw_env = raw_env.env
                state = getattr(raw_env, 'state', {})
                delivered_list = list(state.get("delivered_recipes", []))
                recipe_order = list(state.get("recipe_order", getattr(raw_env, 'recipe_order', [])))
                pending_list = OrderQueue.get_pending(recipe_order, delivered_list)

                if is_paused:
                    # Draw HUD in paused state
                    try:
                        env.render()
                        surface = pygame.display.get_surface()
                        if surface:
                            elapsed = step_count * (step_delay_ms / 1000.0)
                            draw_hud_overlay(
                                surface, model_info, current_model_idx, total_models,
                                elapsed, max_seconds, step_count, max_steps, episode_reward,
                                delivered_list, pending_list, True
                            )
                            pygame.display.flip()
                    except Exception:
                        pass
                    pygame.time.wait(100)
                    continue

                # Compute actions for agents
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
                        action = 0
                    actions[agent_id] = action

                # Step environment
                obs_dict, rewards, terminations, truncations, infos = env.step(actions)
                episode_reward += sum(rewards.values())
                step_count += 1

                # Re-extract delivered/pending post step
                state = getattr(raw_env, 'state', {})
                delivered_list = list(state.get("delivered_recipes", []))
                pending_list = OrderQueue.get_pending(recipe_order, delivered_list)

                del_str = ", ".join(delivered_list) if delivered_list else "None"
                pend_str = ", ".join(pending_list) if pending_list else "None"

                # Record step details in model log file
                try:
                    with open(log_filepath, "a", encoding="utf-8") as f:
                        action_strs = ", ".join(f"{aid}:{act}" for aid, act in actions.items())
                        step_rew = sum(rewards.values())
                        f.write(f"[Step {step_count:04d}] Actions: [{action_strs}] | Step Rwd: {step_rew:+.2f} | Total Rwd: {episode_reward:+.2f} | Delivered ({len(delivered_list)}): [{del_str}] | Pending ({len(pending_list)}): [{pend_str}]\n")
                except Exception:
                    pass

                # Render Pygame Frame + HUD Overlay
                try:
                    env.render()
                    surface = pygame.display.get_surface()
                    if surface:
                        elapsed = step_count * (step_delay_ms / 1000.0)
                        draw_hud_overlay(
                            surface, model_info, current_model_idx, total_models,
                            elapsed, max_seconds, step_count, max_steps, episode_reward,
                            delivered_list, pending_list, False
                        )
                        pygame.display.flip()
                except Exception:
                    pass

                pygame.time.wait(step_delay_ms)

                if any(terminations.values()) or any(truncations.values()):
                    print(f"[INFO] Trial episode ended naturally at step {step_count}.")
                    break

            # Write summary footer to model log file
            try:
                with open(log_filepath, "a", encoding="utf-8") as f:
                    del_str = ", ".join(delivered_list) if delivered_list else "None"
                    pend_str = ", ".join(pending_list) if pending_list else "None"
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("🏁 ROLLOUT SUMMARY\n")
                    f.write(f"Steps Completed         : {step_count} / {max_steps}\n")
                    f.write(f"Duration                : {step_count * (step_delay_ms / 1000.0):.1f} seconds\n")
                    f.write(f"Total Reward            : {episode_reward:+.2f}\n")
                    f.write(f"Delivered Dishes ({len(delivered_list)}): {del_str}\n")
                    f.write(f"Pending Dishes   ({len(pending_list)}): {pend_str}\n")
                    f.write("=" * 80 + "\n")
            except Exception:
                pass

            print(f"[SUMMARY] Model [{current_model_idx + 1}/{total_models}] {model_info['trial_id']} finished 20s rollout. Total Reward: {episode_reward:+.2f}")
            print(f"  Log File saved : {log_filepath.relative_to(project_root)}")
            executed_log_files.append(log_filepath)

            # Navigation handling
            if user_nav_action == 'quit':
                print("[INFO] User requested exit.")
                break
            elif user_nav_action == 'prev':
                current_model_idx = max(0, current_model_idx - 1)
            elif user_nav_action == 'restart':
                pass  # Keep current_model_idx unchanged
            else:  # 'next' or finished naturally
                current_model_idx += 1

    finally:
        if active_env is not None:
            try:
                active_env.close()
            except Exception:
                pass
        if ray.is_initialized():
            ray.shutdown()
        print("\n[FINISHED] Multi-Model visualization completed cleanly.")
        
        # Generate and display execution summary report from log files
        summary_md_path = project_root / "experiments" / "logs" / "vision_analysis" / "visualization_summary.md"
        generate_execution_summary_report(executed_log_files, summary_md_path)


def main():
    parser = argparse.ArgumentParser(description="Multi-Model 20-Second Rollout Visualizer for Overcooked MARL.")
    parser.add_argument("path", type=str, nargs="?", default=None, help="Directory containing trials, specific experiment folder, or summary markdown file (positional argument)")
    parser.add_argument("--seconds", type=float, default=20.0, help="Duration to visualize each model in seconds (default: 20.0)")
    parser.add_argument("--delay", type=int, default=50, help="Step delay in ms (default: 50 ms, ~20 FPS)")
    parser.add_argument("--ray-results", type=str, default="experiments/ray_results", help="Directory containing Ray experiment results")
    parser.add_argument("--summary", type=str, default=None, help="Path to summary markdown file (e.g. PPO_Marl_20261006_215831_summary.md)")
    parser.add_argument("--sort", type=str, choices=["rank", "reward", "trial"], default="rank", help="Sorting criteria for models (default: rank)")
    parser.add_argument("--top-n", type=int, default=None, help="Limit visualization to top N models")
    parser.add_argument("--map", type=str, default=None, help="Map name override (default: auto-detected per model)")
    parser.add_argument("--num-agents", type=int, default=None, help="Number of agents override (default: auto-detected per model)")
    parser.add_argument("--deterministic", action="store_true", help="Disable exploration sampling")
    args = parser.parse_args()

    # Determine ray_results_dir and summary_file from positional path or explicit flags
    summary_file = Path(args.summary).resolve() if args.summary else None

    if args.path:
        p = Path(args.path).resolve()
        if p.is_file() and p.suffix == ".md":
            summary_file = p
            ray_results_dir = (project_root / args.ray_results).resolve()
        elif p.is_dir():
            ray_results_dir = p
            if not summary_file:
                # Detect timestamp from path (e.g. 20261024_121315) to auto-select matching summary report if available
                m_ts = re.search(r'(\d{8}_\d{6})', str(p))
                if m_ts:
                    ts = m_ts.group(1)
                    match_summary = project_root / "experiments" / "logs" / f"PPO_Marl_{ts}_summary.md"
                    if match_summary.exists():
                        summary_file = match_summary
        else:
            ray_results_dir = p
    else:
        ray_results_dir = (project_root / args.ray_results).resolve()

    if not ray_results_dir.exists():
        print(f"[ERROR] Specified directory does not exist: {ray_results_dir}")
        sys.exit(1)

    # 1. Discover models
    models = discover_all_models(ray_results_dir, summary_file=summary_file, sort_by=args.sort)

    if args.top_n and args.top_n > 0:
        models = models[:args.top_n]

    if not models:
        print(f"[ERROR] No trained models with valid checkpoints found in {ray_results_dir}")
        sys.exit(1)

    # 2. Print discovered models table in console
    print("\n" + "="*85)
    print(f"📋 DISCOVERED {len(models)} TRAINED MODELS FOR VISUALIZATION:")
    print("="*85)
    print(f"{'Index':<6} | {'Rank':<6} | {'Trial ID':<15} | {'Map':<16} | {'Ag':<3} | {'LR':<8} | {'Gamma':<6} | {'Entropy':<7} | {'Train Reward':<12}")
    print("-" * 85)
    for idx, m in enumerate(models):
        print(f"{idx+1:<6} | #{m['rank']:<5} | {m['trial_id']:<15} | {m.get('map','N/A'):<16} | {m.get('num_agents','N/A'):<3} | {m['lr']:<8} | {m['gamma']:<6} | {m['entropy']:<7} | {m['reward_str']:<12}")
    print("="*85 + "\n")

    # 3. Run visualizer loop
    run_models_visualizer(
        models=models,
        max_seconds=args.seconds,
        step_delay_ms=args.delay,
        num_agents=args.num_agents,
        map_name=args.map,
        explore=not args.deterministic
    )



if __name__ == "__main__":
    main()
