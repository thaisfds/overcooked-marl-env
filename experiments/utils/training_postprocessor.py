#!/usr/bin/env python3
"""
Reusable post-processor, progress callbacks, and reporting engine for Overcooked MARL training.

Generates:
  - model_info.json in each trial directory (compatible with visualize_all_models.py & plot_curves.py)
  - Unified JSON experiment log in experiments/logs/{timestamp}.json
  - Ranked comparison Markdown report in experiments/logs/{timestamp}_summary.md and latest_summary.md
  - Persistent record in experiments/logs/experiments_summary.csv & experiments_summary.json
  - Checkpoint resolution for dynamic warm-starts
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
from ray import tune

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from experiments.utils.results_tracker import record_experiment_entry
except ImportError:
    record_experiment_entry = None


def convert_to_serializable(obj):
    """Converts numpy scalars and arrays to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


class TrainingProgressCallback(tune.Callback):
    """Callback to compute training progress percentage and update trial results for Ray Tune."""
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


def create_cli_progress_reporter() -> tune.CLIReporter:
    """Creates a standardized CLI progress reporter for Ray Tune."""
    return tune.CLIReporter(
        metric_columns={
            "training_iteration": "iter",
            "time_total_s": "total time (s)",
            "timesteps_total": "ts",
            "progress_pct": "progress (%)",
            "env_runners/episode_reward_mean": "reward_mean",
        }
    )


def resolve_restore_checkpoint(arg_val: Optional[str]) -> Optional[str]:
    """
    Resolves checkpoint path:
    1. Explicit path given by user (file or directory).
    2. Dynamically finds best checkpoint by reward if 'auto', 'best', or 'latest'.
    """
    if not arg_val:
        return None
    val_str = str(arg_val).strip()

    # 1. Explicit path
    p = Path(val_str)
    if p.exists():
        if (p / "rllib_checkpoint.json").exists():
            return str(p.resolve())
        sub_cps = [c for c in p.glob("**/checkpoint_*") if (c / "rllib_checkpoint.json").exists()]
        if sub_cps:
            return str(sorted(sub_cps, key=lambda x: str(x), reverse=True)[0].resolve())

    # 2. Dynamic search across experiment outputs
    if val_str.lower() in ("auto", "best", "latest"):
        search_roots = [
            PROJECT_ROOT / "experiments" / "ray_results",
            Path("/app/experiments/ray_results")
        ]

        all_trials_scored = []
        for r_dir in search_roots:
            if not r_dir.exists():
                continue
            for info_file in r_dir.glob("**/model_info.json"):
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cp = data.get("checkpoint_path")
                    score = data.get("avg_reward")
                    if cp and Path(cp).exists() and (Path(cp) / "rllib_checkpoint.json").exists():
                        all_trials_scored.append((float(score) if score is not None else -float("inf"), Path(cp)))
                except Exception:
                    pass

        if all_trials_scored:
            all_trials_scored.sort(key=lambda t: t[0], reverse=True)
            best_cp = all_trials_scored[0][1]
            return str(best_cp.resolve())

        # Fallback: newest checkpoint by modification time
        all_cps = []
        for r_dir in search_roots:
            if r_dir.exists():
                all_cps.extend([c for c in r_dir.glob("**/checkpoint_*") if (c / "rllib_checkpoint.json").exists()])
        if all_cps:
            all_cps.sort(key=lambda c: c.stat().st_mtime, reverse=True)
            return str(all_cps[0].resolve())

    return None


def postprocess_experiment_trials(
    analysis,
    algorithm_name: str,
    args: Any,
    ray_results_base: Path,
    params_dir_name: str,
    timestamp_dir_name: str,
    timestamp: str,
    start_time: float,
    primary_metric: str = "env_runners/episode_reward_mean"
) -> List[Dict[str, Any]]:
    """
    Standardized post-processing of all trials in a Ray Tune analysis object.
    Generates model_info.json, updates persistent tracker, and creates ranked Markdown reports.
    """
    exp_dir = Path(ray_results_base).parent.parent
    all_trials_summary = []

    print("\n" + "="*50)
    print(f"ALL {algorithm_name} TRAINING TRIALS COMPLETED")
    print(f"Total duration: {(time.time() - start_time)/60:.2f} minutes")
    print("="*50 + "\n")

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
                reward_col = None
                for candidate in [
                    'env_runners/episode_reward_mean',
                    'episode_reward_mean',
                    'sampler_results/episode_reward_mean',
                    'evaluation/episode_reward_mean'
                ]:
                    if candidate in df.columns:
                        reward_col = candidate
                        break

                if reward_col:
                    learning_curves_data = [
                        {
                            'iteration': int(row['training_iteration']),
                            'reward_mean': float(row[reward_col])
                        }
                        for _, row in df.iterrows()
                        if pd.notna(row.get(reward_col))
                    ]
            except Exception as e:
                print(f"[WARN] Failed to parse progress.csv for trial {trial.trial_id}: {e}")

        # 1. Best checkpoint by metric from Ray Tune
        checkpoint_path = None
        try:
            best_cp_obj = analysis.get_best_checkpoint(
                trial,
                metric=primary_metric,
                mode="max"
            )
            if best_cp_obj:
                if hasattr(best_cp_obj, 'path'):
                    checkpoint_path = best_cp_obj.path
                elif hasattr(best_cp_obj, 'to_directory'):
                    checkpoint_path = best_cp_obj.to_directory()
        except Exception:
            pass

        # 2. Peak of learning curve from CSV
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            if learning_curves_data:
                valid_pts = [pt for pt in learning_curves_data if pt.get('reward_mean') is not None]
                if valid_pts:
                    best_pt = max(valid_pts, key=lambda x: x['reward_mean'])
                    best_iter = best_pt['iteration']
                    cand_cp_name = f"checkpoint_{best_iter:06d}"
                    for d in [target_dir / cand_cp_name, trial_dir / cand_cp_name]:
                        if d.exists() and (d / "rllib_checkpoint.json").exists():
                            checkpoint_path = str(d.resolve())
                            break

        # 3. trial.checkpoint fallback
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            if hasattr(trial, 'checkpoint') and trial.checkpoint:
                try:
                    if hasattr(trial.checkpoint, 'path'):
                        checkpoint_path = trial.checkpoint.path
                    elif hasattr(trial.checkpoint, 'to_directory'):
                        checkpoint_path = trial.checkpoint.to_directory()
                except Exception:
                    pass

        if checkpoint_path:
            if "/tmp/" in str(checkpoint_path) or not os.path.exists(checkpoint_path):
                checkpoint_dir_name = Path(checkpoint_path).name
                persistent_checkpoint_path = target_dir / checkpoint_dir_name
                if persistent_checkpoint_path.exists():
                    checkpoint_path = str(persistent_checkpoint_path.resolve())

        # 4. Final fallback: search checkpoint directories
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

        # Extract last reward
        avg_reward_last = None
        if hasattr(trial, 'last_result') and trial.last_result:
            if 'env_runners' in trial.last_result and isinstance(trial.last_result['env_runners'], dict):
                env_runners = trial.last_result['env_runners']
                avg_reward_last = env_runners.get('episode_reward_mean', env_runners.get('episode_return_mean', None))
            elif 'episode_reward_mean' in trial.last_result:
                avg_reward_last = trial.last_result['episode_reward_mean']

        peak_reward = None
        if learning_curves_data:
            valid_rewards = [pt['reward_mean'] for pt in learning_curves_data if pt.get('reward_mean') is not None]
            if valid_rewards:
                peak_reward = float(np.max(valid_rewards))

        best_reward_val = peak_reward if peak_reward is not None else avg_reward_last

        # Parameters extraction
        p_lr = trial_config.get('lr', getattr(args, 'lr', 3e-4) or 3e-4)
        p_gamma = trial_config.get('gamma', getattr(args, 'gamma', 0.99) or 0.99)
        p_batch = trial_config.get('train_batch_size', getattr(args, 'batch_size', 4000) or 4000)
        p_seed = trial_config.get('seed', getattr(args, 'seed', 42) or 42)
        p_episodes = trial_env_config.get('num_episodes', getattr(args, 'episodes', 2000) or 2000)

        # 3rd parameter (entropy for PPO/IMPALA/APPO, or n_step for DQN)
        p_entropy = trial_config.get('entropy_coeff', getattr(args, 'entropy_coeff', 0.01) or 0.01)
        if algorithm_name == "DQN":
            p_n_step = trial_config.get('n_step', 3)
            p_param3_val = float(p_n_step)
            param3_name = "N-Step"
        else:
            p_param3_val = float(p_entropy)
            param3_name = "Entropy"

        config_str = f"lr{p_lr}_gamma{p_gamma}_batch{p_batch}_{param3_name.lower()}{p_param3_val}_episodes{p_episodes}_seed{p_seed}"
        config_name = f"{algorithm_name}_Config_{config_str}"

        model_info = {
            'trial_id': trial.trial_id,
            'algorithm': algorithm_name,
            'config_name': config_name,
            'checkpoint_path': checkpoint_path,
            'map': args.map,
            'num_agents': args.num_agents,
            'avg_reward': float(best_reward_val) if best_reward_val is not None and not np.isnan(best_reward_val) else None,
            'peak_reward': float(peak_reward) if peak_reward is not None and not np.isnan(peak_reward) else None,
            'last_reward': float(avg_reward_last) if avg_reward_last is not None and not np.isnan(avg_reward_last) else None,
            'learning_curves': learning_curves_data,
            'training_config': {
                'algorithm': algorithm_name,
                'learning_rate': float(p_lr),
                'gamma': float(p_gamma),
                'param3_name': param3_name,
                'param3_val': p_param3_val,
                'train_batch_size': int(p_batch),
                'num_episodes': int(p_episodes),
                'timesteps_total': int(trial.last_result.get('timesteps_total', 0) if trial.last_result else 0)
            }
        }

        # Save model_info.json in trial directory
        info_file = target_dir / "model_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(model_info), f, indent=2)

        all_trials_summary.append(model_info)

        # Record in persistent experiment results history
        if record_experiment_entry is not None:
            try:
                record_experiment_entry({
                    "timestamp": timestamp,
                    "experiment_id": timestamp_dir_name,
                    "trial_id": trial.trial_id,
                    "map_name": args.map,
                    "num_agents": args.num_agents,
                    "learning_rate": float(p_lr),
                    "gamma": float(p_gamma),
                    "entropy_coeff": p_param3_val,
                    "batch_size": int(p_batch),
                    "num_episodes": int(p_episodes),
                    "train_reward_mean": float(best_reward_val) if best_reward_val is not None and not np.isnan(best_reward_val) else "N/A",
                    "status": "COMPLETED",
                    "checkpoint_path": checkpoint_path or ""
                })
            except Exception as tracker_err:
                print(f"[WARNING] Could not update results tracker: {tracker_err}")

    # Sort trials by avg_reward descending
    all_trials_summary.sort(
        key=lambda t: t['avg_reward'] if t['avg_reward'] is not None else -999999.0,
        reverse=True
    )
    best_trial = all_trials_summary[0] if all_trials_summary else None

    # Save summary JSON in experiments/logs
    exp_logs_dir = exp_dir / "logs"
    exp_logs_dir.mkdir(parents=True, exist_ok=True)
    exp_log_file = exp_logs_dir / f"{timestamp_dir_name}.json"

    exp_summary_data = {
        "ray_results_folder": timestamp_dir_name,
        "algorithm": algorithm_name,
        "experiment_timestamp": timestamp,
        "total_trials": len(all_trials_summary),
        "best_trial": best_trial,
        "all_trials": all_trials_summary,
        "evaluation": None
    }

    with open(exp_log_file, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(exp_summary_data), f, indent=2)

    # Build Markdown table
    table_rows = []
    for rank, t in enumerate(all_trials_summary, 1):
        cfg = t['training_config']
        reward_str = f"{t['avg_reward']:+.2f}" if t['avg_reward'] is not None else "N/A"
        cp_str = f"`.../{Path(t['checkpoint_path']).name}`" if t['checkpoint_path'] else "N/A"
        # Table format matching visualize_all_models.py parser:
        table_rows.append(
            f"| {rank} | `{t['trial_id']}` | {cfg['learning_rate']} | {cfg['gamma']} | {cfg['param3_val']} | **{reward_str}** | {cp_str} |"
        )

    table_md = "\n".join(table_rows)
    best_reward_str = f"{best_trial['avg_reward']:+.2f}" if (best_trial and best_trial.get('avg_reward') is not None) else "N/A"
    best_trial_id = best_trial['trial_id'] if best_trial else "N/A"
    best_cfg_json = json.dumps(best_trial['training_config'], indent=2) if best_trial else "{}"
    best_cp_path = best_trial.get('checkpoint_path', 'N/A') if best_trial else "N/A"

    summary_md = f"""# Relatório de Experimento - Grid Search {algorithm_name} Overcooked
* **Data/Hora:** {timestamp}
* **Sessão:** `{timestamp_dir_name}`
* **Algoritmo:** {algorithm_name}
* **Total de Combinações/Trials:** {len(all_trials_summary)}
* **Melhor Recompensa de Treino:** {best_reward_str} (Trial `{best_trial_id}`)

## Tabela Comparativa dos Hiperparâmetros (Ordenado pelos Melhores Resultados)
| Rank | Trial ID | LR | Gamma | Param 3 | Recompensa Média | Checkpoint |
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

    summary_md_file = exp_logs_dir / f"{timestamp_dir_name}_summary.md"
    with open(summary_md_file, "w", encoding="utf-8") as f:
        f.write(summary_md)
    with open(exp_logs_dir / "latest_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"Grid search completed with {len(all_trials_summary)} trials.")
    if best_trial:
        print(f"Best Trial: {best_trial['trial_id']} | Reward: {best_trial['avg_reward']}")
        print(f"Checkpoint: {best_cp_path}")
    print(f"📊 Summary report created at: {summary_md_file}")

    return all_trials_summary
