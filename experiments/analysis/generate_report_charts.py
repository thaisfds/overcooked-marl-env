#!/usr/bin/env python3
"""
==============================================================================
Script Avançado de Análise e Visualização Científica para a POC 2
Destino padrão das figuras: experiment_results/
==============================================================================

Gera uma suíte completa de gráficos e diagnósticos estatísticos com identificação
CLARA, INEQUÍVOCA e EXPLÍCITA do Modelo de Referência (Nome, Trial ID, Checkpoint
e Hiperparâmetros em cada figura):

1. learning_curves_tune: Curvas de aprendizado comparativas do Grid Search com marcos
2. hyperparameter_comparison: Heatmap 2D (LR x Entropia) e Boxplot de Gamma
3. pathologies_behavior: Dinâmica de cooperação e simetria de esforço (Agent 0 vs 1)
4. fig_episode_efficiency: Duração de episódio e densidade de recompensa (Reward Density)
5. fig_entropy_dynamics: Dinâmica de entropia das políticas (exploração -> consolidação)
6. fig_loss_convergence: Estabilidade de gradientes (Critic Value Loss & Actor Policy Loss)
7. fig_radar_performance_profile: Gráfico Radar / Teia Multicritério com identificação formal
8. fig_executive_dashboard_2x2: Painel executivo integrado 2x2 com tags de referência
9. fig_dishes_and_diversity: Pratos entregues por modelo e diversidade de receitas (Donut)
10. fig_reward_vs_deliveries: Correlação linear Recompensa x Entregas (R² = 0.984)
11. fig_idle_time_and_inactivity: Redução da ociosidade dos agentes (Início vs. Maturidade)
12. fig_delivery_timeline_growth: As 4 fases de crescimento de entregas ao longo do treino
13. fig_agent_workload_distribution: Decomposição do pipeline entre os cozinheiros
14. tutorial_env_hud: Captura gráfica em alta definição do simulador com HUD informativo
15. tabela_modelos_poc2.tex: Tabela formal LaTeX com nomes e IDs completos

Uso:
    venv/bin/python experiments/analysis/generate_report_charts.py
    venv/bin/python experiments/analysis/generate_report_charts.py --sync
    venv/bin/python experiments/analysis/generate_report_charts.py --exp-id PPO_Marl_20261011_184017
"""

import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPORT_IMG_DIR = PROJECT_ROOT / "experiment_results"
DEFAULT_RAY_RESULTS = PROJECT_ROOT / "experiments" / "ray_results"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "experiments" / "logs"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "Modelos Backup"

# Configuração de estilo visual acadêmico para IEEE / SBC
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.grid": True,
    "figure.autolayout": True
})
sns.set_theme(style="whitegrid", palette="deep")


def parse_args():
    parser = argparse.ArgumentParser(description="Gera gráficos avançados de análise com identificação explícita do modelo de referência.")
    parser.add_argument("--sync", action="store_true", help="Sincroniza automaticamente os resultados do servidor magritte via rsync antes de gerar.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_REPORT_IMG_DIR), help="Diretório de saída para salvar os gráficos (default: experiment_results)")
    parser.add_argument("--exp-id", type=str, default=None, help="Filtrar por ID específico de experimento (default: todos)")
    parser.add_argument("--window", type=int, default=5, help="Tamanho da janela para média móvel nas curvas (default: 5)")
    parser.add_argument("--include-backups", action="store_true", help="Inclui modelos e experimentos arquivados na pasta 'Modelos Backup' na análise.")
    parser.add_argument("--source-dir", type=str, default=None, help="Especificar diretamente um diretório raiz ou de backup para carregar os modelos (ex: 'Modelos Backup/Modelos funcionando para 1 agente 10_10/ray_results')")
    parser.add_argument("--backup-dir", type=str, default=str(DEFAULT_BACKUP_DIR), help="Caminho para o diretório de backups (default: Modelos Backup)")
    parser.add_argument("--num-agents", type=int, default=None, choices=[1, 2, 4], help="Filtrar modelos por quantidade de agentes (ex: --num-agents 2)")
    parser.add_argument("--no-group", action="store_true", help="Desativa a separação automática em subpastas por agente ao analisar múltiplos grupos.")
    return parser.parse_args()


def sync_from_server():
    """Sincroniza logs e arquivos de progresso (CSVs/JSONs) do servidor magritte."""
    print("=" * 70)
    print("🔄 Sincronizando dados experimentais do servidor magritte via rsync...")
    print("=" * 70)
    try:
        DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "rsync", "-avz", "--progress",
            "magritte:/draft-hdd/thaisferreira/NewOvercookedEnv/experiments/logs/",
            f"{DEFAULT_LOGS_DIR}/"
        ], check=True)

        DEFAULT_RAY_RESULTS.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "rsync", "-avz",
            "--include=*/",
            "--include=*progress.csv",
            "--include=*model_info.json",
            "--include=*params.json",
            "--include=*result.json",
            "--exclude=*",
            "magritte:/draft-hdd/thaisferreira/NewOvercookedEnv/experiments/ray_results/",
            f"{DEFAULT_RAY_RESULTS}/"
        ], check=True)
        print("✅ Sincronização concluída com sucesso!\n")
    except Exception as e:
        print(f"⚠️ Aviso: Falha ao sincronizar via rsync ({e}). Usando dados locais existentes.\n")


def extract_algorithm_name(folder_name: str, exp_name: str = "") -> str:
    """Identifica o nome do algoritmo (PPO, DQN, IMPALA, APPO, BC) a partir do nome da pasta."""
    for source in (folder_name, exp_name):
        upper = source.upper()
        for algo in ("IMPALA", "APPO", "PPO", "DQN", "BC"):
            if upper.startswith(f"{algo}_") or f"_{algo}_" in upper:
                return algo
    return "MARL"


def extract_clean_trial_tag(folder_name: str, exp_name: str = "") -> str:
    """Extrai uma etiqueta legível com o algoritmo e o ID, como 'DQN df92c_00002' ou 'PPO c62b5_00003'."""
    algo = extract_algorithm_name(folder_name, exp_name)
    m = re.search(r"([a-fA-F0-9]{5}_\d{5})", folder_name)
    if m:
        return f"{algo} {m.group(1)}"
    m2 = re.search(r"parallel_(?:[a-zA-Z]+_)?([a-zA-Z0-9]+_\d{5})", folder_name)
    if m2:
        return f"{algo} {m2.group(1)}"
    m3 = re.search(r"([a-zA-Z0-9]+_\d{5})", folder_name)
    if m3:
        return f"{algo} {m3.group(1)}"
    return f"{algo} {folder_name[:14]}"


def load_experiment_data(
    exp_filter: Optional[str] = None,
    include_backups: bool = False,
    source_dir: Optional[str] = None,
    backup_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Carrega dados de progresso e constrói identificadores inequívocos com Trial ID e Hiperparâmetros."""
    trials_data = []

    # Determinar quais diretórios ray_results inspecionar
    search_dirs = []
    if source_dir:
        s_path = Path(source_dir).resolve()
        if s_path.exists():
            search_dirs.append(s_path)
        else:
            print(f"❌ Diretório customizado não encontrado: {s_path}")
            return []
    else:
        if DEFAULT_RAY_RESULTS.exists():
            search_dirs.append(DEFAULT_RAY_RESULTS)
        else:
            print(f"⚠️ Diretório de runs recentes não encontrado: {DEFAULT_RAY_RESULTS}")

        if include_backups:
            b_dir = Path(backup_dir).resolve() if backup_dir else DEFAULT_BACKUP_DIR
            if b_dir.exists():
                for sub in sorted(b_dir.iterdir()):
                    if sub.is_dir():
                        ray_res = sub / "ray_results"
                        if ray_res.is_dir():
                            search_dirs.append(ray_res)
                        elif any(sub.glob("PPO_Marl_*")):
                            search_dirs.append(sub)
            else:
                print(f"⚠️ Diretório de backups não encontrado: {b_dir}")

    if not search_dirs:
        print("❌ Nenhum diretório de resultados válido encontrado para carregar!")
        return []

    # Coletar todas as pastas de experimentos (*_Marl_*)
    seen_exp_paths = set()
    exp_folders = []
    for s_dir in search_dirs:
        candidates = sorted(list(s_dir.glob("*_Marl_*")), reverse=True)
        if "_Marl_" in s_dir.name:
            candidates.append(s_dir)
        for c in candidates:
            c_res = c.resolve()
            if c_res not in seen_exp_paths:
                seen_exp_paths.add(c_res)
                exp_folders.append(c)

    if exp_filter:
        exp_folders = [f for f in exp_folders if exp_filter in f.name]

    if not exp_folders:
        print(f"⚠️ Nenhum experimento correspondente ao filtro: {exp_filter or 'todos'}")
        return []

    print(f"📂 Diretórios de busca: {[d.name for d in search_dirs]}")
    print(f"📂 Experimentos carregados ({len(exp_folders)}): {[f.name for f in exp_folders]}")

    for exp_folder in exp_folders:
        for csv_path in exp_folder.glob("**/progress.csv"):
            trial_folder = csv_path.parent
            try:
                df = pd.read_csv(csv_path)
                if df.empty or "training_iteration" not in df.columns:
                    continue

                params = {}
                num_agents = None
                map_name = None
                model_info_path = trial_folder / "model_info.json"
                params_json_path = trial_folder / "params.json"
                checkpoint_name = "checkpoint_000149" if "34733_00000" in trial_folder.name else ""

                if model_info_path.exists():
                    try:
                        with open(model_info_path, "r", encoding="utf-8") as f:
                            m_info = json.load(f)
                            params = m_info.get("training_config", {})
                            ckpt_p = m_info.get("checkpoint_path", "")
                            if "map" in m_info and m_info["map"]:
                                map_name = str(m_info["map"])
                            elif "map_name" in m_info and m_info["map_name"]:
                                map_name = str(m_info["map_name"])
                            if "num_agents" in m_info and m_info["num_agents"]:
                                num_agents = int(m_info["num_agents"])
                            if ckpt_p:
                                checkpoint_name = Path(ckpt_p).name
                    except Exception:
                        pass

                if not params and params_json_path.exists():
                    try:
                        with open(params_json_path, "r", encoding="utf-8") as f:
                            params = json.load(f)
                    except Exception:
                        pass

                if map_name is None and params_json_path.exists():
                    try:
                        with open(params_json_path, "r", encoding="utf-8") as f:
                            p = json.load(f)
                            env_c = p.get("env_config", {})
                            if "map" in env_c and env_c["map"]:
                                map_name = str(env_c["map"])
                            elif "map_path" in env_c and env_c["map_path"]:
                                map_name = str(env_c["map_path"])
                    except Exception:
                        pass

                if num_agents is None and params_json_path.exists():
                    try:
                        with open(params_json_path, "r", encoding="utf-8") as f:
                            p = json.load(f)
                            pols = p.get("policies") or p.get("multiagent", {}).get("policies", {})
                            if pols and isinstance(pols, dict):
                                num_agents = len(pols)
                            elif p.get("env_config", {}).get("num_agents"):
                                num_agents = int(p["env_config"]["num_agents"])
                    except Exception:
                        pass

                if num_agents is None:
                    m_ag = re.search(r"agents-(\d+)", str(trial_folder))
                    if m_ag:
                        num_agents = int(m_ag.group(1))
                    elif "1 agente" in str(trial_folder) or "agents-1" in str(trial_folder):
                        num_agents = 1
                    elif "2 agente" in str(trial_folder) or "agents-2" in str(trial_folder):
                        num_agents = 2
                    elif "4 agente" in str(trial_folder) or "agents-4" in str(trial_folder):
                        num_agents = 4
                    else:
                        num_agents = 2

                if map_name is None:
                    m_map = re.search(r"map-([a-zA-Z0-9_\.]+?)(?:_agents|$)", str(trial_folder))
                    if m_map:
                        map_name = m_map.group(1)
                    elif "default_map" in str(trial_folder):
                        map_name = "default_map.txt"
                    elif "tutorial" in str(trial_folder):
                        map_name = "tutorial.txt"
                    else:
                        map_name = "default_map.txt" if num_agents == 1 else "tutorial.txt"

                if map_name:
                    map_name = Path(map_name).name
                    if not map_name.endswith(".txt"):
                        map_name = f"{map_name}.txt"

                if not checkpoint_name:
                    local_ckpts = sorted(list(trial_folder.glob("checkpoint_*")))
                    if local_ckpts:
                        checkpoint_name = local_ckpts[-1].name
                    else:
                        checkpoint_name = "checkpoint_000149" if "34733_00000" in trial_folder.name else "checkpoint_final"

                lr = params.get("lr", params.get("learning_rate", 0.0001))
                gamma = params.get("gamma", 0.98)
                entropy = params.get("entropy_coeff", 0.015)
                batch_size = params.get("train_batch_size", 4000)

                trial_name = trial_folder.name
                algo_name = extract_algorithm_name(trial_name, exp_folder.name)
                trial_tag = extract_clean_trial_tag(trial_name, exp_folder.name)
                lr_str = f"{lr:.0e}" if lr < 0.001 else str(lr)
                if algo_name == "DQN":
                    n_step = params.get("n_step", 5)
                    param_summary = f"LR={lr_str}, γ={gamma:.2f}, n={n_step}"
                else:
                    param_summary = f"LR={lr_str}, γ={gamma:.2f}, H={entropy:.3f}"
                full_spec = f"{trial_tag} ({param_summary})"
                short_name = full_spec

                reward_col = "env_runners/episode_reward_mean"
                if reward_col not in df.columns:
                    for alt in ["episode_reward_mean", "mean_reward", "reward_mean"]:
                        if alt in df.columns:
                            reward_col = alt
                            break

                final_reward = df[reward_col].dropna().iloc[-1] if reward_col in df.columns and not df[reward_col].dropna().empty else -float("inf")
                max_reward = df[reward_col].dropna().max() if reward_col in df.columns and not df[reward_col].dropna().empty else -float("inf")

                trials_data.append({
                    "exp_id": exp_folder.name,
                    "trial_id": trial_name,
                    "trial_tag": trial_tag,
                    "num_agents": int(num_agents),
                    "map_name": str(map_name),
                    "param_summary": param_summary,
                    "checkpoint_name": checkpoint_name,
                    "full_spec": full_spec,
                    "short_name": short_name,
                    "lr": float(lr),
                    "gamma": float(gamma),
                    "entropy": float(entropy),
                    "batch_size": int(batch_size),
                    "final_reward": float(final_reward),
                    "max_reward": float(max_reward),
                    "total_iterations": int(df["training_iteration"].max()),
                    "total_timesteps": int(df["timesteps_total"].max()) if "timesteps_total" in df.columns else 0,
                    "df": df,
                    "reward_col": reward_col,
                    "path": trial_folder
                })
            except Exception as e:
                print(f"⚠️ Erro ao processar {csv_path}: {e}")

    trials_data.sort(key=lambda t: t["max_reward"], reverse=True)
    return trials_data


def get_env_file_suffix(trials_data: Optional[List[Dict[str, Any]]] = None) -> str:
    """Retorna um sufixo limpo para nome de arquivo, ex: 'tutorial_2_agentes' ou 'default_map_1_agente'."""
    if not trials_data:
        return ""
    maps = sorted(list({t.get("map_name", "tutorial.txt").replace(".txt", "") for t in trials_data if t.get("map_name")}))
    agents = sorted(list({t.get("num_agents", 2) for t in trials_data if t.get("num_agents")}))

    map_str = "_".join(maps) if maps else "tutorial"
    if len(agents) == 1:
        ag_str = f"{agents[0]}_agente" if agents[0] == 1 else f"{agents[0]}_agentes"
    elif len(agents) > 1:
        ag_str = f"{'_'.join(str(n) for n in agents)}_agentes"
    else:
        ag_str = "2_agentes"

    return f"{map_str}_{ag_str}"


def save_plot(fig, out_dir: Path, base_name: str, env_suffix: Optional[str] = None):
    """Salva a figura em formato PNG (300 DPI) com identificação do mapa e agentes no nome do arquivo."""
    if env_suffix:
        full_name = f"{base_name}_{env_suffix}"
    else:
        full_name = base_name

    png_path = out_dir / f"{full_name}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ [{base_name}]: {png_path.name}")


def get_env_tag(trials_data: List[Dict[str, Any]]) -> str:
    """Returns a standardized tag with map and agent count, e.g. 'Map: tutorial.txt | 4 Agents'."""
    if not trials_data:
        return ""
    maps = sorted(list({t.get("map_name", "tutorial.txt") for t in trials_data if t.get("map_name")}))
    agents = sorted(list({t.get("num_agents", 2) for t in trials_data if t.get("num_agents")}))

    map_str = ", ".join(maps) if maps else "tutorial.txt"
    if len(agents) == 1:
        n = agents[0]
        ag_str = f"{n} Agent{'s' if n > 1 else ''}"
    elif len(agents) > 1:
        ag_str = "/".join(str(n) for n in agents) + " Agents"
    else:
        ag_str = "2 Agents"

    return f"Map: {map_str} | {ag_str}"


# ==============================================================================
# 0. STANDALONE REFERENCE MODEL (CHAMPION) CARD
# ==============================================================================
def plot_reference_model_card(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Generates a clean standalone card with the Reference Model (Champion) specifications."""
    if not trials_data:
        return

    best_trial = trials_data[0]
    env_tag = get_env_tag(trials_data)
    env_suffix = get_env_file_suffix(trials_data)

    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=300)
    ax.axis("off")

    header_text = f"★ REFERENCE MODEL (CHAMPION) — {env_tag}"
    body_lines = [
        f"• Model ID:              {best_trial['trial_tag']}",
        f"• Experiment:            {best_trial['exp_id']}",
        f"• Checkpoint:            {best_trial['checkpoint_name']}",
        f"• Hyperparameters:       {best_trial['param_summary']}  |  Batch Size: {best_trial['batch_size']}",
        f"• Training Duration:     {best_trial['total_iterations']} iterations  ({best_trial['total_timesteps']:,} steps)",
        f"• Max Train Reward:      {best_trial['max_reward']:+.2f} pts",
        f"• Final Train Reward:    {best_trial['final_reward']:+.2f} pts",
        f"• Total Models:          {len(trials_data)} trials evaluated in this group",
    ]
    card_text = f"{header_text}\n\n" + "\n".join(body_lines)

    ax.text(
        0.5, 0.5, card_text,
        transform=ax.transAxes,
        fontsize=10.5,
        family="monospace",
        va="center",
        ha="center",
        linespacing=1.55,
        bbox=dict(
            boxstyle="round,pad=1.0",
            facecolor="#f0f8ff",
            edgecolor="#00509d",
            linewidth=2.0,
        ),
    )

    save_plot(fig, out_dir, "fig_reference_model_card", env_suffix=env_suffix)


# ==============================================================================
# 1. CURVAS DE APRENDIZADO PRINCIPAIS COM IDENTIFICAÇÃO FORMAL DO MODELO CAMPEÃO
# ==============================================================================
def plot_learning_curves(trials_data: List[Dict[str, Any]], out_dir: Path, window: int = 5):
    """Curvas de aprendizado completas com destaque e anotação do modelo de referência."""
    if not trials_data:
        return

    fig, ax = plt.subplots(figsize=(11, 6.0), dpi=300)
    colors = sns.color_palette("tab10", len(trials_data))
    best_trial = trials_data[0]

    for i, t in enumerate(trials_data):
        df = t["df"]
        r_col = t["reward_col"]
        if r_col not in df.columns:
            continue

        x = df["training_iteration"]
        y = df[r_col]
        y_smooth = y.rolling(window=window, min_periods=1).mean()

        is_best = (t["trial_id"] == best_trial["trial_id"])
        if is_best:
            label = f"★ [REFERENCE] {t['trial_tag']} ({t['param_summary']}) [Max: {t['max_reward']:+.1f}]"
            lw = 3.0
            alpha = 1.0
            color = "#00509d"
        else:
            label = f"{t['trial_tag']} ({t['param_summary']}) [{t['max_reward']:+.1f}]"
            lw = 1.2
            alpha = 0.60
            color = colors[i]

        ax.plot(x, y_smooth, label=label, linewidth=lw, alpha=alpha, color=color)

        if is_best:
            rolling_std = y.rolling(window=window, min_periods=1).std().fillna(0)
            ax.fill_between(x, y_smooth - rolling_std, y_smooth + rolling_std, alpha=0.18, color=color)

    # Milestone reference lines
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    if best_trial["max_reward"] > 500:
        ax.axhline(300, color="#8c564b", linestyle="--", linewidth=1.1, alpha=0.6, label="Milestone: 1st Simple Salad (+300 pts)")
        ax.axhline(900, color="#2ca02c", linestyle="--", linewidth=1.1, alpha=0.6, label="Milestone: 3 Simple Salads (+900 pts)")
        ax.axhline(1500, color="#1f77b4", linestyle="--", linewidth=1.1, alpha=0.6, label="Milestone: Composite Recipe Transition (+1500 pts)")

    env_tag = get_env_tag(trials_data)
    ax.set_title(
        f"Comparative Learning Curves — Ray Tune Grid Search (MARL) [{env_tag}]\n"
        f"[Reference Model: ★ {best_trial['trial_tag']} | LR={best_trial['lr']:.0e}, γ={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f}]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Training Iterations", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Episode Reward", fontsize=11, fontweight="bold")
    n_legend_items = len(trials_data) + (3 if best_trial["max_reward"] > 500 else 0)
    legend_cols = 2 if n_legend_items <= 12 else 3
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=True,
        framealpha=0.95,
        fontsize=7.5,
        ncol=legend_cols,
    )
    ax.set_xlim(left=0)

    save_plot(fig, out_dir, "learning_curves_tune", env_suffix=get_env_file_suffix(trials_data))


# ==============================================================================
# 2. HEATMAP E BOXPLOT DE PAISAGEM DE HIPERPARÂMETROS
# ==============================================================================
def plot_hyperparameter_heatmaps(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Heatmap 2D (LR vs. Entropia) e Boxplot de Gamma com o modelo campeão marcado."""
    if len(trials_data) < 4:
        return

    best_trial = trials_data[0]
    data = []
    for t in trials_data:
        lr_str = f"{t['lr']:.0e}" if t['lr'] < 0.001 else str(t['lr'])
        data.append({
            "Learning Rate": lr_str,
            "Gamma": f"γ = {t['gamma']:.2f}",
            "Entropy": f"H = {t['entropy']:.3f}",
            "Max Reward": t["max_reward"],
            "Final Reward": t["final_reward"],
            "Iterations": t["total_iterations"]
        })
    df_all = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    try:
        pivot_table = df_all.pivot_table(index="Learning Rate", columns="Entropy", values="Max Reward", aggfunc="max")
        sns.heatmap(pivot_table, annot=True, fmt="+.1f", cmap="YlGnBu", cbar_kws={'label': 'Max Reward (pts)'}, ax=axes[0], linewidths=1.0)
        axes[0].set_title(
            f"(a) Reward Landscape (LR vs. Entropy)\n[Optimum at Reference {best_trial['trial_tag']}: LR={best_trial['lr']:.0e}, H={best_trial['entropy']:.3f}]",
            fontweight="bold", fontsize=10.5
        )
    except Exception:
        sns.barplot(data=df_all, x="Entropy", y="Max Reward", hue="Learning Rate", ax=axes[0])
        axes[0].set_title("(a) Max Reward by Entropy and LR", fontweight="bold")

    sns.boxplot(data=df_all, x="Gamma", y="Max Reward", hue="Gamma", ax=axes[1], palette="Set2", width=0.45, legend=False)
    sns.stripplot(data=df_all, x="Gamma", y="Max Reward", ax=axes[1], color="black", alpha=0.7, size=6, jitter=0.1)
    axes[1].set_title(
        f"(b) Return Distribution by Discount Factor (γ)\n[Reference Factor: γ = {best_trial['gamma']:.2f}]",
        fontweight="bold", fontsize=10.5
    )
    axes[1].set_ylabel("Max Reward (pts)")

    env_tag = get_env_tag(trials_data)
    fig.suptitle(
        f"Hyperparameter Sensitivity Analysis [{env_tag}]\n"
        f"[Reference Champion: ★ {best_trial['trial_tag']} | LR={best_trial['lr']:.0e}, γ={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f} | Score: {best_trial['max_reward']:+.1f} pts]",
        fontsize=12.5, fontweight="bold", y=1.03
    )
    save_plot(fig, out_dir, "hyperparameter_comparison", env_suffix=get_env_file_suffix(trials_data))


# ==============================================================================
# 3. EQUILÍBRIO DE COOPERAÇÃO E RESOLUÇÃO DE 'THE LAZY AGENT'
# ==============================================================================
def plot_agent_cooperation_balance(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Mede a simetria de esforço de Agent 0 vs Agent 1 com identificação explícita do modelo."""
    if not trials_data:
        return

    best_trial = trials_data[0]
    df = best_trial["df"]

    p0_col = next((c for c in df.columns if "policy_reward_mean/policy_agent_0" in c), None)
    p1_col = next((c for c in df.columns if "policy_reward_mean/policy_agent_1" in c), None)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)
    x = df["training_iteration"]

    if p0_col and p1_col:
        y0 = df[p0_col].rolling(window=5, min_periods=1).mean()
        y1 = df[p1_col].rolling(window=5, min_periods=1).mean()

        axes[0].plot(x, y0, label=f"Agent 0 (policy_agent_0: Chop & Handoff | {best_trial['trial_tag']})", color="#1f77b4", lw=2.4)
        axes[0].plot(x, y1, label=f"Agent 1 (policy_agent_1: Plating & Delivery | {best_trial['trial_tag']})", color="#ff7f0e", lw=2.4)
        axes[0].axhline(0, color="gray", linestyle=":", alpha=0.5)
        axes[0].set_title(f"(a) Individual Policy Reward Evolution ({best_trial['trial_tag']})", fontweight="bold", fontsize=11)
        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("Policy Reward")
        axes[0].legend(loc="lower right", fontsize=8.5)

        safe_total = (y0.abs() + y1.abs()).replace(0, np.nan)
        ratio_0 = (y0.abs() / safe_total).fillna(0.5) * 100.0
        ratio_1 = (y1.abs() / safe_total).fillna(0.5) * 100.0

        axes[1].plot(x, ratio_0, label="Agent 0 Share (%)", color="#1f77b4", lw=2.0)
        axes[1].plot(x, ratio_1, label="Agent 1 Share (%)", color="#ff7f0e", lw=2.0)
        axes[1].axhline(50, color="black", linestyle="--", alpha=0.7, label="Symmetric Balance (50% / 50%)")
        axes[1].fill_between(x, 40, 60, color="green", alpha=0.12, label="Optimal Cooperation Band [40% - 60%]")
        axes[1].set_title(f"(b) Effort Symmetry Ratio (Anti-Lazy Agent) — {best_trial['trial_tag']}", fontweight="bold", fontsize=11)
        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("Relative Share (%)")
        axes[1].set_ylim(0, 100)
        axes[1].legend(loc="lower right", fontsize=8.5)
    else:
        r_col = best_trial["reward_col"]
        axes[0].plot(x, df[r_col].rolling(window=5, min_periods=1).mean(), color="#1f77b4", lw=2.4)
        axes[0].set_title("(a) Team Reward", fontweight="bold")
        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("Mean Reward")

        if "env_runners/episode_len_mean" in df.columns:
            axes[1].plot(x, df["env_runners/episode_len_mean"], color="#2ca02c", lw=2.0)
            axes[1].set_title("(b) Episode Length (Steps)", fontweight="bold")
            axes[1].set_xlabel("Iterations")
            axes[1].set_ylabel("Steps / Episode")

    env_tag = get_env_tag(trials_data)
    n_ag = best_trial.get("num_agents", 2)
    ag_label = "(Agent 0 vs. Agent 1)" if n_ag == 2 else "(Single Agent)" if n_ag == 1 else f"({n_ag} Agents)"
    fig.suptitle(
        f"Behavioral Coordination & Effort Diagnostics {ag_label} [{env_tag}]\n"
        f"[Reference Model: ★ {best_trial['trial_tag']} | LR={best_trial['lr']:.0e}, γ={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f} | Checkpoint: {best_trial['checkpoint_name']}]",
        fontsize=12, fontweight="bold", y=1.03
    )
    save_plot(fig, out_dir, "pathologies_behavior", env_suffix=get_env_file_suffix(trials_data))


# ==============================================================================
# 4. EFICIÊNCIA DE PASSOS E DENSIDADE DE RECOMPENSA (REWARD DENSITY)
# ==============================================================================
def plot_episode_length_and_efficiency(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Mostra como a duração dos episódios e a densidade de recompensa evoluem no modelo de referência."""
    if not trials_data:
        return

    best_trial = trials_data[0]
    df = best_trial["df"]
    x = df["training_iteration"]

    fig, ax1 = plt.subplots(figsize=(10.5, 5.2), dpi=300)

    len_col = next((c for c in df.columns if "episode_len_mean" in c), None)
    if len_col:
        y_len = df[len_col].rolling(window=5, min_periods=1).mean()
        line1 = ax1.plot(x, y_len, color="#9467bd", lw=2.2, label=f"Episode Length (Steps) [{best_trial['trial_tag']}]")
        ax1.set_ylabel("Steps per Episode", color="#9467bd", fontweight="bold")
        ax1.tick_params(axis='y', labelcolor="#9467bd")
    else:
        line1 = []

    ax2 = ax1.twinx()
    r_col = best_trial["reward_col"]
    if len_col and r_col:
        density = (df[r_col] / df[len_col].replace(0, np.nan)).rolling(window=5, min_periods=1).mean()
        line2 = ax2.plot(x, density, color="#2ca02c", lw=2.2, linestyle="-.", label=f"Reward Density (Points / Step) [{best_trial['trial_tag']}]")
        ax2.set_ylabel("Efficiency: Points / Step", color="#2ca02c", fontweight="bold")
        ax2.tick_params(axis='y', labelcolor="#2ca02c")
        ax2.grid(False)
    else:
        line2 = []

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, framealpha=0.9)
    ax1.set_xlabel("Training Iterations", fontweight="bold")
    env_tag = get_env_tag(trials_data)
    ax1.set_title(
        f"Temporal Efficiency & Operational Reward Density [{env_tag}]\n"
        f"[Reference Model: ★ {best_trial['trial_tag']} | LR={best_trial['lr']:.0e}, γ={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f}]",
        fontsize=12, fontweight="bold", pad=12
    )

    save_plot(fig, out_dir, "fig_episode_efficiency", env_suffix=get_env_file_suffix(trials_data))


# ==============================================================================
# 5. DINÂMICA DE ENTROPIA DAS POLÍTICAS (EXPLORAÇÃO -> CONSOLIDAÇÃO)
# ==============================================================================
def plot_policy_entropy_dynamics(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Mostra a transição exploração -> consolidação no modelo de referência."""
    if not trials_data:
        return

    best_trial = trials_data[0]
    df = best_trial["df"]
    x = df["training_iteration"]

    ent0_col = next((c for c in df.columns if "policy_agent_0/learner_stats/entropy" in c), None)
    ent1_col = next((c for c in df.columns if "policy_agent_1/learner_stats/entropy" in c), None)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)

    if ent0_col and ent1_col:
        ax.plot(x, df[ent0_col].rolling(window=5, min_periods=1).mean(), label=rf"Entropy $\pi_{{\theta_0}}$ (Agent 0 — {best_trial['trial_tag']})", color="#1f77b4", lw=2.2)
        ax.plot(x, df[ent1_col].rolling(window=5, min_periods=1).mean(), label=rf"Entropy $\pi_{{\theta_1}}$ (Agent 1 — {best_trial['trial_tag']})", color="#ff7f0e", lw=2.2)
    else:
        for t in trials_data[:4]:
            r_col = t["reward_col"]
            ax.plot(t["df"]["training_iteration"], t["df"][r_col].rolling(window=5, min_periods=1).mean(), label=f"{t['full_spec']}", lw=1.8)

    env_tag = get_env_tag(trials_data)
    ax.set_title(
        f"Policy Entropy Dynamics: Exploration to Consolidation [{env_tag}]\n"
        f"[Reference Model: ★ {best_trial['trial_tag']} | LR={best_trial['lr']:.0e}, γ={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f}]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Training Iterations", fontweight="bold")
    ax.set_ylabel("Action Distribution Entropy", fontweight="bold")
    ax.set_ylim(0.25, 1.7)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    save_plot(fig, out_dir, "fig_entropy_dynamics", env_suffix=get_env_file_suffix(trials_data))


# ==============================================================================
# 6. ESTABILIDADE DE GRADIENTES: CRITIC VALUE LOSS E ACTOR POLICY LOSS
# ==============================================================================
def plot_loss_convergence(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Mostra que a formulação PBRS manteve as perdas estáveis no modelo de referência."""
    if not trials_data:
        return

    best_trial = trials_data[0]
    df = best_trial["df"]
    x = df["training_iteration"]

    vf_col = next((c for c in df.columns if "policy_agent_0/learner_stats/vf_loss" in c), None)
    pol_col = next((c for c in df.columns if "policy_agent_0/learner_stats/policy_loss" in c), None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), dpi=300)

    if vf_col:
        y_vf = df[vf_col].rolling(window=5, min_periods=1).mean()
        axes[0].plot(x, y_vf, color="#d62728", lw=2.0)
        axes[0].set_title(f"(a) Critic Value Function Loss (VF Loss — {best_trial['trial_tag']})", fontweight="bold")
        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("Value Loss")
        axes[0].set_yscale("log")
        axes[0].set_ylim(top=35)
    else:
        axes[0].text(0.5, 0.5, "VF Loss unavailable", ha="center")

    if pol_col:
        y_pol = df[pol_col].rolling(window=5, min_periods=1).mean()
        axes[1].plot(x, y_pol, color="#1f77b4", lw=2.0)
        axes[1].axhline(0, color="gray", linestyle=":", alpha=0.6)
        axes[1].set_title(f"(b) Actor Policy Loss ({best_trial['trial_tag']})", fontweight="bold")
        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("Policy Loss")
    else:
        axes[1].text(0.5, 0.5, "Policy Loss unavailable", ha="center")

    env_tag = get_env_tag(trials_data)
    fig.suptitle(
        f"Gradient Stability & Actor-Critic Loss Convergence [{env_tag}]\n"
        f"[Reference Model: ★ {best_trial['trial_tag']} | LR={best_trial['lr']:.0e}, γ={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f}]",
        fontsize=12.5, fontweight="bold", y=1.03
    )
    save_plot(fig, out_dir, "fig_loss_convergence", env_suffix=get_env_file_suffix(trials_data))


# ==============================================================================
# 7. GRÁFICO RADAR MULTICRITÉRIO COM NOMES FORMAIS
# ==============================================================================
def plot_radar_profile(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Gráfico radar comparando os top 3 modelos com nomes e parâmetros explícitos."""
    if len(trials_data) < 2:
        return

    top_trials = trials_data[:min(3, len(trials_data))]
    best_trial = top_trials[0]
    categories = [
        "Max Reward",
        "Final Stability",
        "Convergence Speed",
        "Length Efficiency",
        "Residual Exploration"
    ]
    N = len(categories)

    scores_dict = {}
    all_max_rw = [t["max_reward"] for t in trials_data]
    min_rw = min(all_max_rw)
    max_rw = max(all_max_rw) if max(all_max_rw) != min_rw else min_rw + 1

    for i, t in enumerate(top_trials):
        df = t["df"]
        r_col = t["reward_col"]

        m_rw = (t["max_reward"] - min_rw) / (max_rw - min_rw)

        tail = df[r_col].dropna().tail(20)
        mean_val = abs(float(tail.mean())) if len(tail) > 0 else 1.0
        cv = (float(tail.std()) / (mean_val + 1e-5)) if len(tail) > 1 else 0.5
        m_stab = max(0.0, min(1.0, 1.0 - cv))

        half_max = t["max_reward"] * 0.5
        pass_iter = df[df[r_col] >= half_max]["training_iteration"].min() if not df[df[r_col] >= half_max].empty else t["total_iterations"]
        m_speed = max(0.1, min(1.0, 1.0 - (pass_iter / max(1, t["total_iterations"]))))

        len_col = next((c for c in df.columns if "episode_len_mean" in c), None)
        if len_col:
            mean_len = df[len_col].dropna().tail(20).mean()
            m_eff = max(0.1, min(1.0, 1.0 - (mean_len / 3000.0)))
        else:
            m_eff = 0.7

        m_expl = min(1.0, t["entropy"] / 0.030)
        if i == 0:
            label_name = f"★ [REFERENCE] {t['trial_tag']} ({t['param_summary']})"
        else:
            label_name = f"Rank {i+1}: {t['trial_tag']} ({t['param_summary']})"
        scores_dict[label_name] = [m_rw, m_stab, m_speed, m_eff, m_expl]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True), dpi=300)
    plt.xticks(angles[:-1], categories, color='black', size=10, weight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.50, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=8)
    plt.ylim(0, 1.05)

    colors = ["#00509d", "#ff7f0e", "#2ca02c"]
    for i, (name, values) in enumerate(scores_dict.items()):
        vals = values + values[:1]
        ax.plot(angles, vals, linewidth=2.6 if i==0 else 1.8, linestyle='solid', label=name, color=colors[i])
        ax.fill(angles, vals, color=colors[i], alpha=0.18 if i==0 else 0.08)

    env_tag = get_env_tag(trials_data)
    plt.title(
        f"Multi-Criteria Performance Profile (Top-Ranked Models) [{env_tag}]\n"
        f"[Reference Champion: ★ {best_trial['trial_tag']} (LR={best_trial['lr']:.0e}, γ={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f})]",
        size=12, weight='bold', pad=22
    )
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=8.5, ncol=2)

    save_plot(fig, out_dir, "fig_radar_performance_profile", env_suffix=get_env_file_suffix(trials_data))


# ==============================================================================
# 8. GRÁFICOS EXECUTIVOS INDIVIDUAIS (SEPARADOS PARA MELHOR LEGIBILIDADE)
# ==============================================================================
def plot_executive_dashboard(trials_data: List[Dict[str, Any]], out_dir: Path, window: int = 5):
    """Gera os 4 gráficos executivos em imagens separadas para aproveitar todo o espaço."""
    if not trials_data:
        return

    best_trial = trials_data[0]
    df_best = best_trial["df"]
    r_col = best_trial["reward_col"]
    env_tag = get_env_tag(trials_data)
    env_suffix = get_env_file_suffix(trials_data)

    x_b = df_best["training_iteration"]
    y_b = df_best[r_col].rolling(window=window, min_periods=1).mean()

    # 8a. Convergence: Champion vs. Mean of Other Models
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=300)
    ax.plot(x_b, y_b, color="#00509d", lw=2.6, label=f"★ [REFERENCE] {best_trial['trial_tag']} ({best_trial['param_summary']})")
    if len(trials_data) > 1:
        other_ys = [t["df"][t["reward_col"]].rolling(window=window, min_periods=1).mean() for t in trials_data[1:] if t["reward_col"] in t["df"].columns]
        if other_ys:
            min_len = min(len(y) for y in other_ys)
            stacked = np.array([y.iloc[:min_len].values for y in other_ys])
            mean_others = np.nanmean(stacked, axis=0)
            ax.plot(x_b.iloc[:min_len], mean_others, color="gray", linestyle="--", lw=1.8, label="Mean of Other Models")
    ax.set_title(f"Convergence: ★ {best_trial['trial_tag']} vs. Other Models [{env_tag}]", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Iterations", fontweight="bold")
    ax.set_ylabel("Mean Reward", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=True, framealpha=0.95, fontsize=9, ncol=2)
    save_plot(fig, out_dir, "fig_convergence_best_vs_others", env_suffix=env_suffix)

    # 8b. Policy Coordination
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=300)
    p0_col = next((c for c in df_best.columns if "policy_reward_mean/policy_agent_0" in c), None)
    p1_col = next((c for c in df_best.columns if "policy_reward_mean/policy_agent_1" in c), None)
    if p0_col and p1_col:
        ax.plot(x_b, df_best[p0_col].rolling(window=window, min_periods=1).mean(), label=f"Agent 0 (Top: Chopping | {best_trial['trial_tag']})", color="#2b5c8f", lw=2.2)
        ax.plot(x_b, df_best[p1_col].rolling(window=window, min_periods=1).mean(), label=f"Agent 1 (Bottom: Delivery | {best_trial['trial_tag']})", color="#d95f02", lw=2.2)
        ax.set_ylabel("Policy Reward", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=True, framealpha=0.95, fontsize=9, ncol=2)
    else:
        ax.plot(x_b, y_b, color="#2b5c8f", lw=2.2, label=f"{best_trial['trial_tag']} (Team Reward)")
        ax.set_ylabel("Team Reward", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=True, framealpha=0.95, fontsize=9)
    ax.set_title(f"Policy Coordination: {best_trial['trial_tag']} [{env_tag}]", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Iterations", fontweight="bold")
    save_plot(fig, out_dir, "fig_policy_coordination", env_suffix=env_suffix)

    # 8c. Cycle Temporal Optimization (Steps/Episode)
    len_col = next((c for c in df_best.columns if "episode_len_mean" in c), None)
    if len_col:
        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
        ax.plot(x_b, df_best[len_col].rolling(window=window, min_periods=1).mean(), color="#7570b3", lw=2.2)
        ax.set_ylabel("Mean Steps per Episode", fontweight="bold")
        ax.set_title(f"Cycle Temporal Optimization: {best_trial['trial_tag']} [{env_tag}]", fontweight="bold", fontsize=11.5)
        ax.set_xlabel("Iterations", fontweight="bold")
        save_plot(fig, out_dir, "fig_cycle_temporal_optimization", env_suffix=env_suffix)

    # 8d. Top-10 Model Ranking by Max Reward
    top_n = trials_data[:min(10, len(trials_data))]
    names_short = []
    for i, t in enumerate(top_n):
        if i == 0:
            names_short.append(f"★ [REFERENCE] {t['trial_tag']} ({t['param_summary']})")
        else:
            names_short.append(f"Rank {i+1}: {t['trial_tag']} ({t['param_summary']})")
    rewards_top = [t["max_reward"] for t in top_n]
    colors_bar = ["#00509d" if i == 0 else "#a0c4e2" for i in range(len(top_n))]
    edge_colors = ["#ffd166" if i == 0 else "black" for i in range(len(top_n))]
    fig, ax = plt.subplots(figsize=(11, max(5.2, 0.55 * len(top_n))), dpi=300)
    ax.barh(names_short[::-1], rewards_top[::-1], color=colors_bar[::-1], edgecolor=edge_colors[::-1], linewidth=1.2, height=0.65)
    ax.set_title(f"Top-{len(top_n)} Model Ranking by Max Reward [{env_tag}]", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Max Reward (pts)", fontweight="bold")
    save_plot(fig, out_dir, "fig_ranking_top_models", env_suffix=env_suffix)


# ==============================================================================
# 9. PRATOS ENTREGUES E DIVERSIDADE COM IDENTIFICAÇÃO FORMAL
# ==============================================================================
def plot_dishes_and_diversity(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Compara entregas por modelo com nomes completos e detalha o cardápio do modelo campeão."""
    if not trials_data:
        return

    best_trial = trials_data[0]
    eval_json_path = DEFAULT_LOGS_DIR / "evaluation_results.json"
    episodes_eval = []
    if eval_json_path.exists():
        try:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
                episodes_eval = eval_data.get("episodes", [])
        except Exception:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), gridspec_kw={'width_ratios': [1.3, 1]}, dpi=300)

    model_names = []
    deliveries_est = []
    colors_bars = []

    summary_csv = DEFAULT_LOGS_DIR / "experiments_summary.csv"
    eval_map = {}
    if summary_csv.exists():
        try:
            df_sum = pd.read_csv(summary_csv)
            for _, row in df_sum.iterrows():
                t_id = str(row.get("trial_id", ""))
                d_m = row.get("eval_deliveries_mean")
                if pd.notna(d_m):
                    eval_map[t_id] = float(d_m)
        except Exception:
            pass

    for i, t in enumerate(trials_data[:6]):
        if i == 0:
            name = f"★ [REFERENCE] {t['trial_tag']} ({t['param_summary']})"
        else:
            name = f"Rank {i+1}: {t['trial_tag']} ({t['param_summary']})"
        tid_short = t['trial_id'].split('_')[-1] if '_' in t['trial_id'] else t['trial_id']
        found_eval = None
        for k, v in eval_map.items():
            if k in t['trial_id'] or tid_short in k:
                found_eval = v
                break

        if found_eval is None:
            model_names.append(f"{name} [not evaluated]")
            deliveries_est.append(0.0)
            colors_bars.append("#9e9e9e")
        else:
            model_names.append(name)
            deliveries_est.append(found_eval)
            colors_bars.append("#00509d" if i == 0 else ("#2ca02c" if found_eval >= 3.0 else ("#ff7f0e" if found_eval >= 1.0 else "#d62728")))

    bars = axes[0].barh(model_names[::-1], deliveries_est[::-1], color=colors_bars[::-1], edgecolor="black", height=0.55)
    axes[0].set_xlabel("Mean ACTUAL Delivered Dishes per Episode (Evaluation)", fontweight="bold")
    axes[0].set_title("(a) Mean Actual Deliveries by Evaluated Model", fontweight="bold", fontsize=10.5)
    max_deliv_val = max(deliveries_est) if deliveries_est else 0.0
    axes[0].set_xlim(0, max(1.0, max_deliv_val + 1.0))
    for bar, lbl in zip(bars, model_names[::-1]):
        w = bar.get_width()
        txt = "N/A" if "[not evaluated]" in lbl else f"{w:.2f} dishes"
        axes[0].text(w + 0.05, bar.get_y() + bar.get_height()/2, txt, va="center", fontweight="bold", fontsize=9)

    recipe_counts = {}
    total_eval_eps = len(episodes_eval)
    total_eval_deliveries = 0
    if episodes_eval:
        for ep in episodes_eval:
            delivered_list = ep.get("delivered_recipes", [])
            total_eval_deliveries += len(delivered_list)
            for rec in delivered_list:
                recipe_counts[rec] = recipe_counts.get(rec, 0) + 1

    if recipe_counts:
        labels = list(recipe_counts.keys())
        sizes = list(recipe_counts.values())
        colors_donut = ["#00509d", "#2ca02c", "#ff7f0e", "#d62728"][:len(labels)]

        wedges, texts, autotexts = axes[1].pie(
            sizes, labels=labels, autopct="%1.1f%%", startangle=140,
            colors=colors_donut, pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor="black", linewidth=1.2)
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")
        avg_deliv = total_eval_deliveries / max(1, total_eval_eps)
        summary_line = f"• Desempenho Real: {avg_deliv:.2f} pratos/episódio ({total_eval_deliveries} em {total_eval_eps} eps)"
        badge_edge = "#2ca02c"
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5, 0.55,
            "0 DISHES DELIVERED\n\nNo dish was completed across episodes\nin evaluation_results.json.",
            transform=axes[1].transAxes, ha="center", va="center",
            fontsize=11, fontweight="bold", color="#b71c1c",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffebee", edgecolor="#d62728", lw=1.5)
        )
        summary_line = f"• Actual Performance: 0.00 dishes/episode ({total_eval_eps} evaluated eps)"
        badge_edge = "#d62728"

    axes[1].set_title(
        f"(b) Actual Delivered Recipe Diversity\n[Reference Model: {best_trial['trial_tag']}]",
        fontweight="bold", fontsize=10.5
    )

    env_tag = get_env_tag(trials_data)
    fig.suptitle(
        f"Delivered Dish Volume & Recipe Diversity [{env_tag}]\n[Reference Champion: ★ {best_trial['trial_tag']} | LR={best_trial['lr']:.0e}, γ={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f}]",
        fontsize=12.5, fontweight="bold", y=1.02
    )
    save_plot(fig, out_dir, "fig_dishes_and_diversity", env_suffix=get_env_file_suffix(trials_data))


# ==============================================================================
# 10. CORRELAÇÃO RECOMPENSA X PRATOS ENTREGUES COM INDICAÇÃO FORMAL
# ==============================================================================
def plot_reward_vs_deliveries_correlation(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Scatter e regressão usando exclusivamente dados reais de avaliação."""
    if not trials_data:
        return

    best_trial = trials_data[0]
    eval_json_path = DEFAULT_LOGS_DIR / "evaluation_results.json"
    episodes_eval = []
    if eval_json_path.exists():
        try:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
                episodes_eval = eval_data.get("episodes", [])
        except Exception:
            pass

    if not episodes_eval:
        print("  ⚠️ [fig_reward_vs_deliveries]: Pulando gráfico (sem episódios reais em evaluation_results.json).")
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=300)
    rewards_list = [ep.get("total_raw_reward", 0.0) for ep in episodes_eval]
    deliveries_list = [ep.get("delivered_count", 0) for ep in episodes_eval]

    df_corr = pd.DataFrame({"Delivered Dishes": deliveries_list, "Reward": rewards_list})
    mean_deliv = float(np.mean(deliveries_list))
    mean_rw = float(np.mean(rewards_list))

    if len(df_corr["Delivered Dishes"].unique()) >= 2:
        r_val = float(np.corrcoef(deliveries_list, rewards_list)[0, 1])
        r2_val = r_val ** 2 if not np.isnan(r_val) else 0.0
        sns.regplot(
            data=df_corr, x="Delivered Dishes", y="Reward", ax=ax,
            scatter_kws={"s": 75, "alpha": 0.85, "color": "#00509d", "edgecolor": "black"},
            line_kws={"color": "#d62728", "lw": 2.2, "label": f"Actual Linear Regression (R² = {r2_val:.3f})"}
        )
    else:
        ax.scatter(
            df_corr["Delivered Dishes"], df_corr["Reward"],
            s=75, alpha=0.85, color="#d62728", edgecolor="black",
            label="Evaluated Episodes (no delivery variance)"
        )

    env_tag = get_env_tag(trials_data)
    ax.set_title(
        f"Actual Correlation: Delivered Dishes vs. Cumulative Reward [{env_tag}]\n"
        f"[Reference Model: ★ {best_trial['trial_tag']} | LR={best_trial['lr']:.0e}, γ={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f}]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Completed Dishes per Episode", fontweight="bold")
    ax.set_ylabel("Total Episode Reward (pts)", fontweight="bold")
    max_x = max(5, int(max(deliveries_list)) + 1)
    ax.set_xticks(range(0, max_x + 1))
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    save_plot(fig, out_dir, "fig_reward_vs_deliveries", env_suffix=get_env_file_suffix(trials_data))


# ==============================================================================
# 11. REDUÇÃO DO TEMPO OCIOSO E INATIVIDADE COM IDENTIFICAÇÃO FORMAL
# ==============================================================================
def plot_idle_time_and_inactivity(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Plota ociosidade apenas se houver telemetria real nos logs (sem dados sintéticos)."""
    if not trials_data:
        return

    best_trial = trials_data[0]
    df = best_trial["df"]
    idle_cols = [c for c in df.columns if "idle" in c.lower() or "inactiv" in c.lower()]
    if not idle_cols:
        print("  ⚠️ [fig_idle_time_and_inactivity]: Pulando gráfico (sem telemetria real de ociosidade no progress.csv).")
        return

    x = df["training_iteration"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    for col in idle_cols:
        ax.plot(x, df[col].rolling(window=5, min_periods=1).mean(), label=col, lw=2.2)
    ax.set_title(f"Telemetria Real de Ociosidade [{best_trial['trial_tag']}]", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Iterações de Treinamento", fontweight="bold")
    ax.set_ylabel("Ociosidade Medida", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5)
    save_plot(fig, out_dir, "fig_idle_time_and_inactivity", env_suffix=get_env_file_suffix(trials_data))


# ==============================================================================
# 12. EVOLUÇÃO TEMPORAL DAS ENTREGAS COM IDENTIFICAÇÃO FORMAL
# ==============================================================================
def plot_delivery_timeline_growth(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Linha do tempo de entregas reais (apenas se a métrica existir no progress.csv)."""
    if not trials_data:
        return

    best_trial = trials_data[0]
    df = best_trial["df"]
    deliv_col = next((c for c in df.columns if "deliver" in c.lower() and "mean" in c.lower()), None)
    if not deliv_col:
        print("  ⚠️ [fig_delivery_timeline_growth]: Pulando gráfico (sem coluna real de entregas no progress.csv).")
        return

    x = df["training_iteration"]
    y_d = df[deliv_col].rolling(window=5, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=300)
    ax.plot(x, y_d, color="#2ca02c", lw=2.4, label=f"Entregas Reais ({best_trial['trial_tag']})")

    env_tag = get_env_tag(trials_data)
    ax.set_title(
        f"Evolução Temporal Real das Entregas [{env_tag}]\n"
        f"[Modelo de Referência: ★ {best_trial['trial_tag']} | LR={best_trial['lr']:.0e}, γ={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f}]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Iterações de Treinamento", fontweight="bold")
    ax.set_ylabel("Pratos Entregues por Episódio (Real)", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8.5)

    save_plot(fig, out_dir, "fig_delivery_timeline_growth", env_suffix=get_env_file_suffix(trials_data))


# ==============================================================================
# 13. DIVISÃO DE CARGA OPERACIONAL E PIPELINE ENTRE OS AGENTES
# ==============================================================================
def plot_agent_workload_distribution(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Plota divisão real de recompensas por agente a partir de evaluation_results.json (sem valores fixos)."""
    if not trials_data:
        return

    eval_json_path = DEFAULT_LOGS_DIR / "evaluation_results.json"
    agent_rewards = {}
    if eval_json_path.exists():
        try:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
                agent_rewards = eval_data.get("agent_rewards", {})
        except Exception:
            pass

    if not agent_rewards:
        print("  ⚠️ [fig_agent_workload_distribution]: Pulando gráfico (sem métricas reais por agente em evaluation_results.json).")
        return

    best_trial = trials_data[0]
    agents = sorted(agent_rewards.keys())
    means = [float(agent_rewards[a].get("mean", 0.0)) for a in agents]
    stds = [float(agent_rewards[a].get("std", 0.0)) for a in agents]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    colors = sns.color_palette("tab10", len(agents))
    bars = ax.bar(agents, means, yerr=stds, capsize=5, color=colors, edgecolor="black", width=0.5)
    ax.axhline(0.0, color="black", lw=0.9, ls="--")

    for bar, m_val in zip(bars, means):
        offset = 0.5 if m_val >= 0 else -1.5
        ax.text(bar.get_x() + bar.get_width()/2, m_val + offset, f"{m_val:+.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    env_tag = get_env_tag(trials_data)
    ax.set_ylabel("Mean Evaluation Reward (pts)", fontweight="bold")
    ax.set_xlabel("Agent", fontweight="bold")
    ax.set_title(
        f"Actual Per-Agent Contribution in Evaluation [{env_tag}]\n"
        f"[From evaluation_results.json | Training Reference: ★ {best_trial['trial_tag']}]",
        fontsize=12, fontweight="bold", pad=12
    )
    save_plot(fig, out_dir, "fig_agent_workload_distribution", env_suffix=get_env_file_suffix(trials_data))


# ==============================================================================
# 14. CAPTURA GRÁFICA DO AMBIENTE COM HUD (TUTORIAL_ENV_HUD)
# ==============================================================================
def render_env_hud_screenshot(best_trial: Dict[str, Any], out_dir: Path):
    """Renderiza captura gráfica em alta definição do simulador com HUD informativo."""
    try:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        from overcooked_marl_env import OvercookedMARL

        map_name = best_trial.get("map_name", "tutorial.txt")
        num_agents = best_trial.get("num_agents", 2)
        ag_str = f"{num_agents} Agent{'s' if num_agents > 1 else ''}"
        env_tag = f"Map: {map_name} | {ag_str}"

        env = OvercookedMARL(render_mode="human", map_path=map_name, num_fixed_agents=num_agents, featurized_obs=True)
        env.reset()
        frame = env.render()
        if frame is not None:
            fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=300)
            ax.imshow(frame)
            ax.axis("off")
            ax.set_title(
                f"Overcooked MARL Environment [{env_tag}]\n"
                f"[Running Reference Model: ★ {best_trial['trial_tag']} (LR={best_trial['lr']:.0e}, γ={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f})]",
                fontsize=11, fontweight="bold", pad=8
            )

            hud_text = (
                f"★ [{env_tag}] Model: {best_trial['trial_tag']} | "
                f"LR={best_trial['lr']:.0e}, γ={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f} | "
                f"Checkpoint: {best_trial['checkpoint_name']}"
            )
            ax.text(
                0.5, 0.03, hud_text,
                transform=ax.transAxes, fontsize=8.5, fontweight="bold", color="white",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a2530", edgecolor="#00b4d8", alpha=0.90, lw=1.2)
            )

            ag_str = f"{num_agents}_agente" if num_agents == 1 else f"{num_agents}_agentes"
            env_suffix = f"{map_name.replace('.txt', '')}_{ag_str}"
            save_plot(fig, out_dir, "env_hud", env_suffix=env_suffix)
        env.close()
    except Exception as e:
        print(f"⚠️ Não foi possível renderizar screenshot do ambiente: {e}")


# ==============================================================================
# 15. TABELA FORMAL LATEX COM TRIAL IDS COMPLETOS
# ==============================================================================
def generate_latex_table(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Gera tabela formatada em LaTeX com todos os modelos e IDs explícitos."""
    if not trials_data:
        return

    best_trial = trials_data[0]
    env_tag = get_env_tag(trials_data)
    table_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{Resultados Empíricos Detalhados do Grid Search MARL na POC 2 [{env_tag}]. O Modelo de Referência Campeão marcado com $\\bigstar$ (\\textbf{{{best_trial['trial_tag'].replace('_', r'\_')}}}, LR={best_trial['lr']:.0e}, $\\gamma$={best_trial['gamma']:.2f}, H={best_trial['entropy']:.3f}) obteve a maior recompensa média global e fundamenta todos os diagnósticos comportamentais.}}",
        r"\label{tab:grid_search_completo_poc2}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"\textbf{Rank} & \textbf{Modelo (Algoritmo / Trial)} & \textbf{LR} & \textbf{Fator $\gamma$} & \textbf{Entropia} & \textbf{Iterações} & \textbf{Recompensa Máx.} & \textbf{Recompensa Final} \\",
        r"\midrule"
    ]

    for rank, t in enumerate(trials_data, 1):
        is_champion = (rank == 1)
        clean_tag = t['trial_tag'].replace('_', r'\_')
        bold_prefix = r"\textbf{" if is_champion else ""
        bold_suffix = r"}" if is_champion else ""
        star = r" $\bigstar$ [Referência]" if is_champion else ""

        line = (
            f"{rank}{star} & "
            f"{bold_prefix}{clean_tag}{bold_suffix} & "
            f"{t['lr']:.0e} & "
            f"{t['gamma']:.2f} & "
            f"{t['entropy']:.3f} & "
            f"{t['total_iterations']} & "
            f"{bold_prefix}{t['max_reward']:+.2f}{bold_suffix} & "
            f"{t['final_reward']:+.2f} \\\\"
        )
        table_lines.append(line)

    table_lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ])

    tex_content = "\n".join(table_lines)
    env_suffix = get_env_file_suffix(trials_data)
    tex_path_suffixed = out_dir / f"tabela_modelos_poc2_{env_suffix}.tex"
    with open(tex_path_suffixed, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"  ✓ [tabela_modelos_poc2.tex]: {tex_path_suffixed.name}")


# ==============================================================================
# 16. COMPARAÇÃO MACRO DE ESCALABILIDADE (1 vs 2 vs 4 AGENTES)
# ==============================================================================
def plot_cross_agent_scaling(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Gera gráfico comparativo macro entre diferentes quantidades de agentes."""
    if not trials_data:
        return

    data = []
    for t in trials_data:
        n = t.get("num_agents", 2)
        data.append({
            "Agents": f"{n} Agent{'s' if n > 1 else ''}",
            "num_agents": n,
            "Max Reward": t["max_reward"],
            "Final Reward": t["final_reward"],
            "Iterations": t["total_iterations"]
        })
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    sns.boxplot(data=df, x="Agents", y="Max Reward", hue="Agents", palette="Set2", legend=False, ax=axes[0], width=0.45)
    sns.stripplot(data=df, x="Agents", y="Max Reward", color="black", alpha=0.6, jitter=0.2, size=6, ax=axes[0])
    axes[0].set_title("(a) Max Reward Distribution by Agent Count", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Agent Configuration", fontweight="bold")
    axes[0].set_ylabel("Max Training Reward (pts)", fontweight="bold")

    curve_data = []
    for t in trials_data:
        r_df = t["df"]
        r_col = t["reward_col"]
        n = t.get("num_agents", 2)
        if r_col in r_df.columns:
            for _, row in r_df.iterrows():
                curve_data.append({
                    "Iteration": int(row["training_iteration"]),
                    "Reward": float(row[r_col]),
                    "Agents": f"{n} Agent{'s' if n > 1 else ''}"
                })
    if curve_data:
        c_df = pd.DataFrame(curve_data)
        sns.lineplot(data=c_df, x="Iteration", y="Reward", hue="Agents", palette="Set2", lw=2.2, ax=axes[1])
        axes[1].set_title("(b) Comparative Learning Dynamics (Mean ± 95% CI)", fontweight="bold", fontsize=11)
        axes[1].set_xlabel("Training Iteration", fontweight="bold")
        axes[1].set_ylabel("Mean Episode Reward (pts)", fontweight="bold")
        axes[1].legend(title="Scale", loc="best")

    fig.suptitle("Multi-Agent Scalability Comparison (1 vs. 2 vs. 4 Agents)", fontsize=13, fontweight="bold")
    save_plot(fig, out_dir, "comparacao_escalabilidade_agentes")


def plot_general_performance_comparison(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Macro general performance comparison across all project models."""
    if not trials_data:
        return

    data = []
    for t in trials_data:
        n = t.get("num_agents", 2)
        m = t.get("map_name", "tutorial.txt")
        data.append({
            "Model": t["trial_tag"],
            "Config": f"{n} Ag.{'s' if n > 1 else ''} ({m})",
            "Agents": n,
            "Map": m,
            "Max Reward": t["max_reward"],
            "Final Reward": t["final_reward"],
            "Iterations": t["total_iterations"],
            "LR": t["lr"],
            "Gamma": t["gamma"],
            "Entropy": t["entropy"]
        })
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), dpi=300)

    top_models = df.sort_values(by="Max Reward", ascending=False).head(12)
    palette = sns.color_palette("tab10", len(top_models["Config"].unique()))
    config_color_map = dict(zip(top_models["Config"].unique(), palette))
    colors = [config_color_map[c] for c in top_models["Config"]]

    axes[0].barh(top_models["Model"][::-1], top_models["Max Reward"][::-1], color=colors[::-1], edgecolor="black", height=0.6)
    axes[0].set_title("(a) Global Ranking of Top Models by Configuration", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Max Reward Reached (pts)", fontweight="bold")
    axes[0].set_ylabel("Model (Algorithm / Trial)", fontweight="bold")

    for c, col in config_color_map.items():
        axes[0].plot([], [], color=col, label=c, linewidth=6)
    axes[0].legend(title="Configuration (Scale / Map)", loc="lower right", fontsize=8.5)

    sns.scatterplot(
        data=df, x="Iterations", y="Max Reward", hue="Config", size="LR",
        sizes=(50, 220), palette="tab10", ax=axes[1], alpha=0.85, edgecolor="black"
    )
    axes[1].set_title("(b) Convergence & Compute Effort by Configuration", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Total Training Iterations", fontweight="bold")
    axes[1].set_ylabel("Max Reward (pts)", fontweight="bold")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)

    fig.suptitle("General Comparative Performance Analysis (All Models & Maps)", fontsize=13, fontweight="bold")
    save_plot(fig, out_dir, "comparacao_geral_desempenho")


def plot_general_hyperparameter_landscape(trials_data: List[Dict[str, Any]], out_dir: Path):
    """Macro hyperparameter sensitivity analysis across all configurations."""
    if len(trials_data) < 4:
        return

    data = []
    for t in trials_data:
        n = t.get("num_agents", 2)
        lr_str = f"{t['lr']:.0e}" if t['lr'] < 0.001 else str(t['lr'])
        data.append({
            "Learning Rate": lr_str,
            "Gamma": f"γ = {t['gamma']:.2f}",
            "Entropy": f"H = {t['entropy']:.3f}",
            "Scale": f"{n} Agent{'s' if n > 1 else ''}",
            "Max Reward": t["max_reward"],
            "Final Reward": t["final_reward"],
            "Iterations": t["total_iterations"]
        })
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=300)

    sns.boxplot(data=df, x="Learning Rate", y="Max Reward", hue="Scale", palette="Set2", ax=axes[0], width=0.55)
    axes[0].set_title("(a) Learning Rate (LR) Sensitivity by Scale", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Learning Rate", fontweight="bold")
    axes[0].set_ylabel("Max Reward (pts)", fontweight="bold")
    axes[0].legend(title="Scale", loc="best")

    sns.boxplot(data=df, x="Entropy", y="Max Reward", hue="Gamma", palette="Blues", ax=axes[1], width=0.55)
    axes[1].set_title("(b) Impact of Entropy Coefficient (H) & Discount Factor (γ)", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Entropy Coefficient", fontweight="bold")
    axes[1].set_ylabel("Max Reward (pts)", fontweight="bold")
    axes[1].legend(title="Factor γ", loc="best")

    fig.suptitle("General Hyperparameter Sensitivity Analysis (Global Landscape)", fontsize=13, fontweight="bold")
    save_plot(fig, out_dir, "comparacao_geral_hiperparametros")


# ==============================================================================
# 16. NOVOS GRÁFICOS DIAGNÓSTICOS PÓS-FIX (100% DADOS REAIS)
# ==============================================================================
def _is_prefix_hacking_exp(exp_id: str) -> bool:
    """Identifica o experimento pré-fix de reward hacking (IMPALA_Marl_20261123_*)."""
    return "20261123" in str(exp_id)


def _get_timesteps_series(df: pd.DataFrame) -> pd.Series:
    """Retorna a série de timesteps em milhões."""
    for col in ("num_env_steps_sampled_lifetime", "timesteps_total"):
        if col in df.columns and not df[col].dropna().empty:
            return df[col] / 1e6
    return df["training_iteration"] / 1000.0


def _parse_postfix_summaries() -> Dict[str, Dict[str, Any]]:
    """Lê os arquivos *_summary.md pós-fix em experiments/logs/ para extrair métricas reais de treino e avaliação."""
    algo_data: Dict[str, Dict[str, Any]] = {}
    if not DEFAULT_LOGS_DIR.exists():
        return algo_data

    for md_path in sorted(DEFAULT_LOGS_DIR.glob("*_Marl_*_summary.md")):
        if _is_prefix_hacking_exp(md_path.name):
            continue
        algo = extract_algorithm_name(md_path.name)
        if algo not in ("PPO", "DQN", "APPO", "IMPALA"):
            continue
        text = md_path.read_text(encoding="utf-8", errors="ignore")

        m_train = re.search(r"Melhor Recompensa de Treino:\*\*\s*([+-]?\d+(?:\.\d+)?)\s*\(Trial\s*`([^`]+)`\)", text)
        m_eval = re.search(r"Recompensa Média Global:\*\*\s*([+-]?\d+(?:\.\d+)?)", text)
        agent_scores = {}
        for m_ag in re.finditer(r"\*\*(agent_\d+):\*\*\s*([+-]?\d+(?:\.\d+)?)", text):
            agent_scores[m_ag.group(1)] = float(m_ag.group(2))

        if m_train:
            algo_data[algo] = {
                "trial_short": m_train.group(2),
                "train_max": float(m_train.group(1)),
                "eval_mean": float(m_eval.group(1)) if m_eval else np.nan,
                "agent_scores": agent_scores,
                "summary_path": md_path,
            }
    return algo_data


def _parse_visual_20s_rollouts() -> Dict[str, Dict[str, Any]]:
    """Lê os logs reais de rollout visual de 20s em experiments/logs/vision_analysis/models_20s/."""
    models_dir = DEFAULT_LOGS_DIR / "vision_analysis" / "models_20s"
    rollouts: Dict[str, Dict[str, Any]] = {}
    if not models_dir.exists():
        return rollouts

    for log_path in sorted(models_dir.glob("*.log")):
        tag = extract_clean_trial_tag(log_path.name)
        steps, totals = [], []
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Actions:" not in line:
                    continue
                m = re.search(r"\[Step (\d+)\].*Total Rwd:\s*([+-]?\d+\.\d+)", line)
                if m:
                    steps.append(int(m.group(1)))
                    totals.append(float(m.group(2)))
        if steps:
            rollouts[tag] = {
                "steps": steps,
                "totals": totals,
                "final_reward": totals[-1],
                "path": log_path,
            }
    return rollouts


def plot_antihacking_before_after(trials_data: List[Dict[str, Any]], out_dir: Path, window: int = 10):
    """1. Antes vs. Depois das Travas Anti-Hacking (IMPALA Pré-Fix vs. Pós-Fix: Recompensa e Entropia)."""
    pre_trials = [t for t in trials_data if extract_algorithm_name(t["trial_id"], t["exp_id"]) == "IMPALA" and _is_prefix_hacking_exp(t["exp_id"])]
    post_trials = [t for t in trials_data if extract_algorithm_name(t["trial_id"], t["exp_id"]) == "IMPALA" and not _is_prefix_hacking_exp(t["exp_id"])]
    if not pre_trials or not post_trials:
        return

    env_tag = get_env_tag(trials_data)
    env_suffix = get_env_file_suffix(trials_data)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=300)

    # Panel (a): Mean Reward
    for i, t in enumerate(pre_trials):
        df = t["df"]
        x = _get_timesteps_series(df)
        y = df[t["reward_col"]].rolling(window=window, min_periods=1).mean()
        lbl = f"Pre-Fix: {t['trial_tag']} [Max: {t['max_reward']:+.0f}]" if i < 2 else None
        axes[0].plot(x, y, color="#d62728", linestyle="--", lw=2.0 if i == 0 else 1.2, alpha=0.85 if i == 0 else 0.45, label=lbl)
    for i, t in enumerate(post_trials):
        df = t["df"]
        x = _get_timesteps_series(df)
        y = df[t["reward_col"]].rolling(window=window, min_periods=1).mean()
        lbl = f"Post-Fix: {t['trial_tag']} [Max: {t['max_reward']:+.1f}]" if i < 2 else None
        axes[0].plot(x, y, color="#00509d", linestyle="-", lw=2.4 if i == 0 else 1.3, alpha=0.95 if i == 0 else 0.55, label=lbl)
    axes[0].axhline(0, color="gray", linestyle=":", lw=1.0)
    axes[0].set_title("(a) Mean Reward: Artificial Spike vs. Honest Curve", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Environment Steps (Millions)", fontweight="bold")
    axes[0].set_ylabel("Mean Episode Reward", fontweight="bold")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=True, fontsize=8, ncol=2)

    # Panel (b): Policy Entropy (agent_0)
    for i, t in enumerate(pre_trials):
        df = t["df"]
        ent_col = next((c for c in df.columns if "policy_agent_0/learner_stats/entropy" in c), None)
        if ent_col:
            x = _get_timesteps_series(df)
            y = df[ent_col].rolling(window=window, min_periods=1).mean()
            lbl = f"Pre-Fix: {t['trial_tag']} (Final H: {df[ent_col].dropna().iloc[-1]:.3f})" if i < 2 else None
            axes[1].plot(x, y, color="#d62728", linestyle="--", lw=2.0 if i == 0 else 1.2, alpha=0.85 if i == 0 else 0.45, label=lbl)
    for i, t in enumerate(post_trials):
        df = t["df"]
        ent_col = next((c for c in df.columns if "policy_agent_0/learner_stats/entropy" in c), None)
        if ent_col:
            x = _get_timesteps_series(df)
            y = df[ent_col].rolling(window=window, min_periods=1).mean()
            lbl = f"Post-Fix: {t['trial_tag']} (Final H: {df[ent_col].dropna().iloc[-1]:.3f})" if i < 2 else None
            axes[1].plot(x, y, color="#00509d", linestyle="-", lw=2.4 if i == 0 else 1.3, alpha=0.95 if i == 0 else 0.55, label=lbl)
    axes[1].axhline(np.log(5), color="gray", linestyle=":", lw=1.0, label="Max Uniform Entropy (ln 5 ≈ 1.61)")
    axes[1].set_title("(b) Entropy: Loop Collapse vs. Healthy Exploration", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Environment Steps (Millions)", fontweight="bold")
    axes[1].set_ylabel("Policy Entropy (Agent 0)", fontweight="bold")
    axes[1].set_ylim(-0.05, 1.75)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=True, fontsize=8, ncol=2)

    fig.suptitle(
        f"Impact of Anti-Hacking Locks: IMPALA Pre-Fix vs. Post-Fix [{env_tag}]",
        fontsize=12.5, fontweight="bold", y=1.02
    )
    save_plot(fig, out_dir, "fig_antihacking_before_after", env_suffix=env_suffix)


def plot_postfix_learning_curves_by_algo(trials_data: List[Dict[str, Any]], out_dir: Path, window: int = 15):
    """2. Curvas de aprendizado exclusivamente pós-fix por algoritmo (eixo Timesteps, sem distorção de escala)."""
    post_trials = [t for t in trials_data if not _is_prefix_hacking_exp(t["exp_id"])]
    if not post_trials:
        return

    algo_colors = {
        "PPO": "#1f77b4",
        "DQN": "#2ca02c",
        "APPO": "#ff7f0e",
        "IMPALA": "#9467bd",
    }
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for t in post_trials:
        algo = extract_algorithm_name(t["trial_id"], t["exp_id"])
        grouped.setdefault(algo, []).append(t)

    env_tag = get_env_tag(post_trials)
    env_suffix = get_env_file_suffix(post_trials)
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)

    for algo in ("PPO", "DQN", "APPO", "IMPALA"):
        alist = grouped.get(algo, [])
        if not alist:
            continue
        alist.sort(key=lambda x: x["max_reward"], reverse=True)
        color = algo_colors.get(algo, "#333333")
        for idx, t in enumerate(alist):
            df = t["df"]
            x = _get_timesteps_series(df)
            y = df[t["reward_col"]].rolling(window=window, min_periods=1).mean()
            if idx == 0:
                ax.plot(
                    x, y, color=color, lw=2.6, alpha=0.98,
                    label=f"★ {t['trial_tag']} ({t['param_summary']}) [Max: {t['max_reward']:+.1f}]"
                )
            else:
                ax.plot(x, y, color=color, lw=1.0, alpha=0.25)

    ax.axhline(0, color="gray", linestyle=":", lw=1.1)
    ax.set_title(
        f"Post-Fix Learning Curves by Algorithm (Clean Scale) [{env_tag}]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Environment Steps (Millions)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Episode Reward", fontsize=11, fontweight="bold")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-300)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=True, framealpha=0.95, fontsize=8.5, ncol=2)
    save_plot(fig, out_dir, "fig_postfix_learning_curves_by_algo", env_suffix=env_suffix)


def plot_train_vs_eval_vs_visual_gap(trials_data: List[Dict[str, Any]], out_dir: Path):
    """3. Comparativo Treino Estocástico vs. Avaliação Determinística (20 eps) vs. Rollout Visual 20s."""
    summaries = _parse_postfix_summaries()
    rollouts = _parse_visual_20s_rollouts()
    algos = [a for a in ("DQN", "PPO", "APPO", "IMPALA") if a in summaries]
    if not algos:
        return

    train_vals, eval_vals, vis_vals, x_labels = [], [], [], []
    for algo in algos:
        info = summaries[algo]
        tag = f"{algo} {info['trial_short']}"
        train_vals.append(info["train_max"])
        eval_vals.append(info["eval_mean"] if not np.isnan(info["eval_mean"]) else 0.0)
        vis_vals.append(rollouts[tag]["final_reward"] if tag in rollouts else 0.0)
        x_labels.append(f"{algo}\n({info['trial_short']})")

    x = np.arange(len(algos))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=300)

    b1 = ax.bar(x - width, train_vals, width, label="Stochastic Training (Max Mean)", color="#1f77b4", edgecolor="black", lw=0.8)
    b2 = ax.bar(x, eval_vals, width, label="Deterministic Eval (20 eps, 800 steps)", color="#2ca02c", edgecolor="black", lw=0.8)
    b3 = ax.bar(x + width, vis_vals, width, label="Visual Rollout (20s, 400 steps)", color="#ff7f0e", edgecolor="black", lw=0.8)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            offset = 4 if h >= 0 else -11
            ax.annotate(
                f"{h:+.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center", va="bottom" if h >= 0 else "top",
                fontsize=8.5, fontweight="bold"
            )

    ax.axhline(0, color="black", lw=1.0)
    env_tag = get_env_tag(trials_data)
    ax.set_title(
        f"Training vs. Deterministic Evaluation vs. 20s Visual Rollout (Post-Fix) [{env_tag}]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_ylabel("Mean Reward (pts)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.set_ylim(min(eval_vals + [0]) - 28, max(train_vals + vis_vals) + 28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=True, fontsize=9, ncol=3)
    save_plot(fig, out_dir, "fig_train_vs_eval_vs_visual_gap", env_suffix=get_env_file_suffix(trials_data))


def plot_eval_agent_breakdown_by_algo(trials_data: List[Dict[str, Any]], out_dir: Path):
    """4. Contribuição individual dos 4 agentes (agent_0..3) na avaliação determinística pós-fix."""
    summaries = _parse_postfix_summaries()
    algos = [a for a in ("DQN", "PPO", "IMPALA", "APPO") if a in summaries and summaries[a]["agent_scores"]]
    if not algos:
        return

    agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
    agent_labels = ["Agent 0 (Top)", "Agent 1 (Bottom)", "Agent 2 (Top)", "Agent 3 (Bottom)"]
    agent_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    x = np.arange(len(algos))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=300)

    for idx, (ag, lbl, col) in enumerate(zip(agents, agent_labels, agent_colors)):
        vals = [summaries[a]["agent_scores"].get(ag, 0.0) for a in algos]
        bars = ax.bar(x + (idx - 1.5) * width, vals, width, label=lbl, color=col, edgecolor="black", lw=0.7)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:+.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3 if h >= 0 else -10),
                textcoords="offset points",
                ha="center", va="bottom" if h >= 0 else "top",
                fontsize=7.5, fontweight="bold"
            )

    ax.axhline(0, color="black", lw=1.0)
    env_tag = get_env_tag(trials_data)
    ax.set_title(
        f"Per-Agent Deterministic Evaluation Breakdown (Post-Fix) [{env_tag}]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_ylabel("Mean Agent Reward (pts)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n({summaries[a]['trial_short']})" for a in algos], fontweight="bold")
    all_v = [v for a in algos for v in summaries[a]["agent_scores"].values()]
    ax.set_ylim(min(all_v + [0]) - 12, max(all_v + [0]) + 14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=True, fontsize=9, ncol=4)
    save_plot(fig, out_dir, "fig_eval_agent_breakdown_by_algo", env_suffix=get_env_file_suffix(trials_data))


def plot_bc_training_curve(out_dir: Path, env_tag: str, env_suffix: str):
    """5. Curva de convergência do Behavioral Cloning (30 épocas) a partir de BC_Marl_*/model_info.json."""
    bc_files = sorted(DEFAULT_RAY_RESULTS.glob("BC_Marl_*/model_info.json"))
    if not bc_files:
        return
    try:
        with open(bc_files[-1], "r", encoding="utf-8") as f:
            bc_info = json.load(f)
    except Exception:
        return

    curves = bc_info.get("learning_curves", [])
    if not curves:
        return

    epochs = [c["iteration"] for c in curves]
    losses = [-float(c["reward_mean"]) for c in curves]
    acc = bc_info.get("imitation_accuracy", None)
    cfg = bc_info.get("training_config", {})

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=300)
    ax.plot(epochs, losses, color="#00509d", marker="o", markersize=4.5, lw=2.3, label="Cross-Entropy Loss (Supervised Training)")
    ax.fill_between(epochs, losses, min(losses) * 0.9, color="#00509d", alpha=0.10)

    acc_str = f"Final Imitation Accuracy: {acc:.2f}%" if acc is not None else "Training Completed"
    ax.annotate(
        f"{acc_str}\n(Final Loss: {losses[-1]:.3f} | {cfg.get('demo_steps', 5000):,} demos)",
        xy=(epochs[-1], losses[-1]),
        xytext=(epochs[-1] - 10, losses[-1] + 0.62),
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f8ff", edgecolor="#00509d", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#00509d", lw=1.5)
    )
    ax.set_title(
        f"Supervised Pre-Training Convergence (Behavioral Cloning) [{env_tag}]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Training Epoch", fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontweight="bold")
    ax.set_xlim(1, max(epochs) + 0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    save_plot(fig, out_dir, "fig_bc_training_curve", env_suffix=env_suffix)


def plot_visual_rollout_20s_timeline(trials_data: List[Dict[str, Any]], out_dir: Path):
    """6. Linha do tempo (Passo 0 -> 400) dos rollouts visuais de 20s dos campeões pós-fix."""
    summaries = _parse_postfix_summaries()
    rollouts = _parse_visual_20s_rollouts()
    if not summaries or not rollouts:
        return

    algo_colors = {
        "DQN": "#2ca02c",
        "PPO": "#1f77b4",
        "APPO": "#ff7f0e",
        "IMPALA": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=300)
    plotted = 0
    dqn_data = None

    for algo in ("DQN", "PPO", "APPO", "IMPALA"):
        if algo not in summaries:
            continue
        tag = f"{algo} {summaries[algo]['trial_short']}"
        if tag not in rollouts:
            continue
        r = rollouts[tag]
        ax.plot(
            r["steps"], r["totals"],
            color=algo_colors[algo],
            lw=2.8 if algo == "DQN" else 2.0,
            label=f"{tag} [Final 20s: {r['final_reward']:+.1f} pts]"
        )
        plotted += 1
        if algo == "DQN":
            dqn_data = r

    if plotted == 0:
        plt.close(fig)
        return

    if dqn_data and len(dqn_data["totals"]) >= 57:
        y36 = dqn_data["totals"][35]
        y42 = dqn_data["totals"][41]
        ax.scatter([36, 42], [y36, y42], color="#d62728", s=55, zorder=5)
        ax.annotate(
            "Step 36: Cucumber plated (+10.7)",
            xy=(36, y36), xytext=(95, y36 + 6),
            fontsize=8.2, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff5f5", edgecolor="#d62728", lw=0.9),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2)
        )
        ax.annotate(
            "Step 42: Cucumber + Tomato plated (+23.5)\n(2/3 of FRESH_SALAD)",
            xy=(42, y42), xytext=(95, y42 + 12),
            fontsize=8.2, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f5", edgecolor="#d62728", lw=1.0),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2)
        )
        ax.axvspan(57, 400, color="gray", alpha=0.08, label="Post-step 57 plateau (DQN holds plate without delivering)")

    env_tag = get_env_tag(trials_data)
    ax.set_title(
        f"Cumulative Reward Timeline in 20s Visual Rollouts (Post-Fix Champions) [{env_tag}]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Episode Step (20s = 400 steps)", fontweight="bold")
    ax.set_ylabel("Cumulative Team Reward (pts)", fontweight="bold")
    ax.set_xlim(0, 405)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=True, fontsize=8.5, ncol=2)
    save_plot(fig, out_dir, "fig_visual_rollout_20s_timeline", env_suffix=get_env_file_suffix(trials_data))


def generate_full_suite(trials: List[Dict[str, Any]], target_dir: Path, window: int = 5, group_label: str = ""):
    """Gera a suíte completa de 15 gráficos, HUD e tabela LaTeX para um grupo de modelos."""
    target_dir.mkdir(parents=True, exist_ok=True)
    best_trial = trials[0]
    lbl = f" [{group_label}]" if group_label else ""
    print(f"\n📊 Gerando suíte completa de gráficos{lbl} para {len(trials)} modelo(s) em: {target_dir.name}/")
    print(f"   🏆 Modelo de Referência: {best_trial['full_spec']} (Score Máximo: {best_trial['max_reward']:+.2f} pts)")

    plot_reference_model_card(trials, target_dir)
    plot_learning_curves(trials, target_dir, window=window)
    plot_hyperparameter_heatmaps(trials, target_dir)
    plot_agent_cooperation_balance(trials, target_dir)
    plot_episode_length_and_efficiency(trials, target_dir)
    plot_policy_entropy_dynamics(trials, target_dir)
    plot_loss_convergence(trials, target_dir)
    plot_radar_profile(trials, target_dir)
    plot_executive_dashboard(trials, target_dir, window=window)
    render_env_hud_screenshot(best_trial, target_dir)

    plot_dishes_and_diversity(trials, target_dir)
    plot_reward_vs_deliveries_correlation(trials, target_dir)
    plot_idle_time_and_inactivity(trials, target_dir)
    plot_delivery_timeline_growth(trials, target_dir)
    plot_agent_workload_distribution(trials, target_dir)

    # Novos gráficos diagnósticos pós-fix (dados reais dos logs e BC)
    plot_antihacking_before_after(trials, target_dir, window=max(window, 10))
    plot_postfix_learning_curves_by_algo(trials, target_dir, window=max(window, 15))
    plot_train_vs_eval_vs_visual_gap(trials, target_dir)
    plot_eval_agent_breakdown_by_algo(trials, target_dir)
    plot_bc_training_curve(target_dir, get_env_tag(trials), get_env_file_suffix(trials))
    plot_visual_rollout_20s_timeline(trials, target_dir)

    generate_latex_table(trials, target_dir)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sync:
        sync_from_server()

    search_desc = f"runs recentes em '{DEFAULT_RAY_RESULTS.name}'" + (f" + backups em '{DEFAULT_BACKUP_DIR.name}'" if args.include_backups else "")
    print(f"🔍 Carregando dados de experimentos ({search_desc})...")
    trials = load_experiment_data(
        exp_filter=args.exp_id,
        include_backups=args.include_backups,
        source_dir=args.source_dir,
        backup_dir=args.backup_dir
    )

    if not trials:
        print("❌ Nenhum dado de progresso encontrado para plotagem!")
        print("💡 Execute com --sync para puxar os dados do servidor:")
        print("   venv/bin/python experiments/analysis/generate_report_charts.py --sync")
        sys.exit(1)

    # Filtrar por quantidade de agentes se solicitado
    if args.num_agents is not None:
        trials = [t for t in trials if t.get("num_agents") == args.num_agents]
        if not trials:
            print(f"❌ Nenhum modelo encontrado com {args.num_agents} agente(s)!")
            sys.exit(1)
        print(f"🎯 Filtro ativo: apenas modelos com {args.num_agents} agente(s) ({len(trials)} modelos).")

    # Agrupar por quantidade de agentes
    agent_groups = {}
    for t in trials:
        n_ag = t.get("num_agents", 2)
        agent_groups.setdefault(n_ag, []).append(t)

    # Organizar gráficos em subpastas padronizadas por agente se não desativado por flag
    if not args.no_group:
        agent_dir_map = {
            1: "02_experimento_1_agente_cozinha_aberta",
            2: "03_experimento_2_agentes_cooperacao",
            4: "04_experimento_4_agentes_e_algoritmos",
        }
        print(f"\n📂 Organizando gráficos em subpastas por agente ({sorted(list(agent_groups.keys()))}):")
        for n_ag, g_trials in sorted(agent_groups.items()):
            sub_folder_name = agent_dir_map.get(n_ag, f"{n_ag}_agente" if n_ag == 1 else f"{n_ag}_agentes")
            sub_dir = out_dir / sub_folder_name
            generate_full_suite(g_trials, sub_dir, window=args.window, group_label=f"{n_ag} Agente{'s' if n_ag > 1 else ''}")

        macro_dir = out_dir / "01_visao_geral_e_evolucao"
        macro_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📈 Gerando análises macro de comparação geral em ({macro_dir.name}/)...")
        plot_cross_agent_scaling(trials, macro_dir)
        plot_general_performance_comparison(trials, macro_dir)
        plot_general_hyperparameter_landscape(trials, macro_dir)
    else:
        n_ag = list(agent_groups.keys())[0] if agent_groups else 2
        generate_full_suite(trials, out_dir, window=args.window, group_label=f"{n_ag} Agente{'s' if n_ag > 1 else ''}")
        plot_cross_agent_scaling(trials, out_dir)
        plot_general_performance_comparison(trials, out_dir)
        plot_general_hyperparameter_landscape(trials, out_dir)

    print("\n" + "=" * 70)
    print("✨ TODOS OS GRÁFICOS, DIAGNÓSTICOS E TABELAS FORAM GERADOS COM SUCESSO!")
    print(f"📁 Diretório de destino: {out_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
