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

def detect_map_and_agents_from_checkpoint(checkpoint_path: Any) -> tuple:
    """
    Universally detects (map_name, num_agents) from any checkpoint directory or file.
    
    Checks in priority order:
    1. model_info.json metadata in checkpoint dir and its parent folders
    2. params.json (RLlib multiagent policies count / env_config)
    3. Path naming regex (e.g. map-<name>_agents-<N>)
    4. experiments_summary.json history lookup
    5. Fallback inspection of map file layout
    """
    import re
    if checkpoint_path is None:
        return "tutorial.txt", 2

    cp_p = Path(checkpoint_path).resolve()
    if cp_p.is_file():
        cp_p = cp_p.parent

    search_dirs = []
    curr = cp_p
    for _ in range(4):
        if curr.is_dir():
            search_dirs.append(curr)
        if curr.parent == curr:
            break
        curr = curr.parent

    map_name = None
    num_agents = None

    # 1. Check model_info.json
    for d in search_dirs:
        info_file = d / "model_info.json"
        if info_file.is_file():
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if map_name is None:
                    map_name = meta.get("map") or meta.get("map_name")
                if num_agents is None and "num_agents" in meta:
                    num_agents = int(meta["num_agents"])
            except Exception:
                pass
        if map_name and num_agents:
            break

    # 2. Check params.json
    if map_name is None or num_agents is None:
        for d in search_dirs:
            params_file = d / "params.json"
            if params_file.is_file():
                try:
                    with open(params_file, "r", encoding="utf-8") as f:
                        p = json.load(f)
                    if num_agents is None:
                        pols = p.get("policies") or p.get("multiagent", {}).get("policies", {})
                        if pols and isinstance(pols, dict):
                            num_agents = len(pols)
                        elif p.get("env_config", {}).get("num_agents"):
                            num_agents = int(p["env_config"]["num_agents"])
                    if map_name is None:
                        env_cfg = p.get("env_config", {})
                        map_name = env_cfg.get("map") or env_cfg.get("map_path") or env_cfg.get("map_name")
                except Exception:
                    pass
            if map_name and num_agents:
                break

    # 3. Check directory regex in path components
    if map_name is None or num_agents is None:
        for part in reversed(cp_p.parts):
            m_both = re.search(r"map-([a-zA-Z0-9_\.]+)_agents-(\d+)", part)
            if m_both:
                if map_name is None:
                    m_cand = m_both.group(1)
                    map_name = m_cand if m_cand.endswith(".txt") else f"{m_cand}.txt"
                if num_agents is None:
                    num_agents = int(m_both.group(2))
            if map_name is None:
                m_map = re.search(r"map-([a-zA-Z0-9_\.]+)", part)
                if m_map:
                    m_cand = m_map.group(1)
                    map_name = m_cand if m_cand.endswith(".txt") else f"{m_cand}.txt"
            if num_agents is None:
                m_ag = re.search(r"agents-(\d+)", part)
                if m_ag:
                    num_agents = int(m_ag.group(1))

    # 4. Check experiments_summary.json lookup
    if map_name is None or num_agents is None:
        summary_candidates = [JSON_FILE]
        for d in search_dirs:
            summary_candidates.append(d.parent / "logs" / "experiments_summary.json")

        for s_file in summary_candidates:
            if s_file.exists():
                try:
                    with open(s_file, "r", encoding="utf-8") as f:
                        records = json.load(f)
                    for r in records:
                        r_cp = r.get("checkpoint_path", "")
                        if r_cp and (r_cp in str(cp_p) or cp_p.name in r_cp or cp_p.parent.name in r_cp):
                            if map_name is None and r.get("map_name"):
                                map_name = r.get("map_name")
                            if num_agents is None and r.get("num_agents"):
                                num_agents = int(r.get("num_agents"))
                            break
                except Exception:
                    pass
            if map_name and num_agents:
                break

    # 5. Fallback inspection and normalization
    if map_name is None:
        map_name = "default_map.txt"
    elif not map_name.endswith(".txt"):
        map_name = f"{map_name}.txt"

    if num_agents is None:
        try:
            map_file = PROJECT_ROOT / "overcooked_marl_env" / "map" / "maps" / map_name
            if not map_file.exists():
                map_file = Path(map_name)
            if map_file.exists():
                with open(map_file, "r", encoding="utf-8") as f:
                    content = f.read()
                agent_digits = set(re.findall(r"[0-9]", content))
                if agent_digits:
                    num_agents = len(agent_digits)
        except Exception:
            pass

    if num_agents is None:
        num_agents = 2 if "tutorial" in map_name else 1

    return map_name, num_agents


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
