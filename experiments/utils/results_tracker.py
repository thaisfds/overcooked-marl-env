#!/usr/bin/env python3
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "experiments" / "logs"
CSV_FILE = LOGS_DIR / "experiments_summary.csv"
JSON_FILE = LOGS_DIR / "experiments_summary.json"

CSV_HEADERS = [
    "timestamp",
    "experiment_id",
    "trial_id",
    "map_name",
    "num_agents",
    "learning_rate",
    "gamma",
    "entropy_coeff",
    "batch_size",
    "num_episodes",
    "train_reward_mean",
    "eval_reward_mean",
    "eval_reward_std",
    "eval_deliveries_mean",
    "eval_steps_mean",
    "status",
    "checkpoint_path"
]

def ensure_logs_dir():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

def record_experiment_entry(entry: Dict[str, Any]):
    """
    Appends or updates an experiment record in both JSON and CSV files.
    Keyed by (experiment_id, trial_id).
    """
    ensure_logs_dir()
    
    # Standardize data structure
    timestamp = entry.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    exp_id = str(entry.get("experiment_id", "unknown"))
    trial_id = str(entry.get("trial_id", "default"))
    
    row_data = {
        "timestamp": timestamp,
        "experiment_id": exp_id,
        "trial_id": trial_id,
        "map_name": entry.get("map_name", "default_map.txt"),
        "num_agents": entry.get("num_agents", 1),
        "learning_rate": entry.get("learning_rate", 3e-4),
        "gamma": entry.get("gamma", 0.99),
        "entropy_coeff": entry.get("entropy_coeff", 0.05),
        "batch_size": entry.get("batch_size", 8000),
        "num_episodes": entry.get("num_episodes", 300),
        "train_reward_mean": entry.get("train_reward_mean", "N/A"),
        "eval_reward_mean": entry.get("eval_reward_mean", "N/A"),
        "eval_reward_std": entry.get("eval_reward_std", "N/A"),
        "eval_deliveries_mean": entry.get("eval_deliveries_mean", "N/A"),
        "eval_steps_mean": entry.get("eval_steps_mean", "N/A"),
        "status": entry.get("status", "COMPLETED"),
        "checkpoint_path": entry.get("checkpoint_path", "")
    }

    # 1. Load existing JSON entries
    all_entries = []
    if JSON_FILE.exists():
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                all_entries = json.load(f)
        except Exception:
            all_entries = []

    # Update matching entry or append
    updated = False
    for item in all_entries:
        if item.get("experiment_id") == exp_id and item.get("trial_id") == trial_id:
            item.update(row_data)
            updated = True
            break
            
    if not updated:
        all_entries.append(row_data)

    # Save JSON
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    # 2. Write CSV file
    file_exists = CSV_FILE.exists()
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for item in all_entries:
            # Prepare row dict
            row = {h: item.get(h, "") for h in CSV_HEADERS}
            writer.writerow(row)

    print(f"📊 Results logged to persistent history: {CSV_FILE}")

def generate_markdown_summary_table() -> str:
    """Generates a clean Markdown table string comparing all recorded experiments."""
    if not JSON_FILE.exists():
        return "Nenhum histórico de experimentos encontrado."

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception:
        return "Erro ao carregar histórico de experimentos."

    if not entries:
        return "Nenhum experimento registrado no histórico."

    rows = []
    for rank, e in enumerate(entries, 1):
        lr = e.get("learning_rate", "-")
        gamma = e.get("gamma", "-")
        entropy = e.get("entropy_coeff", "-")
        tr_reward = e.get("train_reward_mean")
        tr_reward_str = f"{float(tr_reward):+.2f}" if isinstance(tr_reward, (int, float)) else str(tr_reward)
        ev_reward = e.get("eval_reward_mean")
        ev_reward_str = f"{float(ev_reward):+.2f}" if isinstance(ev_reward, (int, float)) else str(ev_reward)
        deliv = e.get("eval_deliveries_mean")
        deliv_str = f"{float(deliv):.2f}" if isinstance(deliv, (int, float)) else str(deliv)
        status = e.get("status", "COMPLETED")
        exp_id = e.get("experiment_id", "")
        
        rows.append(f"| {rank} | `{exp_id}` | {e.get('map_name')} | {e.get('num_agents')} | {lr} | {gamma} | {entropy} | **{tr_reward_str}** | **{ev_reward_str}** | {deliv_str} | `{status}` |")

    table_header = """# Tabela Comparativa de Experimentos MARL (Overcooked)

| # | Experiment ID | Mapa | Agentes | LR | Gamma | Entropy | Recompensa Treino | Recompensa Avaliação | Pratos Entregues/Ep | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    return table_header + "\n".join(rows)
