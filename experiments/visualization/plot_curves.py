#!/usr/bin/env python3
"""
Utility script to plot learning curves directly to an image.
Reads progress.csv from experiments/ray_results and plots reward vs. episodes,
highlighting the exact peak reward and the episode where each model reached its maximum.
Always includes DQN, IMPALA, APPO and PPO.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp"
import re
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Gera gráficos de curvas de aprendizado dos modelos treinados.")
    parser.add_argument("--output", "-o", default=None, help="Caminho/nome da imagem de saída (default: learning_curves.png)")
    parser.add_argument("--algo", "-a", choices=["DQN", "IMPALA", "APPO", "PPO", "ALL"], default="ALL", help="Filtra por algoritmo específico ou gera todos (default: ALL)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    ray_dir = project_root / "experiments" / "ray_results"
    
    if not ray_dir.exists():
        print(f"[ERROR] Ray results directory not found at {ray_dir}")
        return

    # Always include all 4 reinforcement learning algorithms
    target_algos = ["DQN", "IMPALA", "APPO", "PPO"] if args.algo == "ALL" else [args.algo]
    models_by_algo = {k: [] for k in target_algos}

    # Search for progress.csv across all trials
    for csv_path in ray_dir.glob("**/progress.csv"):
        s = str(csv_path)
            
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
            
        rw_col = "env_runners/episode_return_mean" if "env_runners/episode_return_mean" in df.columns else "env_runners/episode_reward_mean"
        if rw_col not in df.columns:
            rw_cols = [c for c in df.columns if "reward_mean" in c or "return_mean" in c]
            if rw_cols:
                rw_col = rw_cols[0]
            else:
                continue
            
        valid_df = df.dropna(subset=[rw_col]).copy()
        if valid_df.empty:
            continue
            
        if "env_runners/episodes_this_iter" in valid_df.columns:
            valid_df["cum_episodes"] = valid_df["env_runners/episodes_this_iter"].cumsum()
        else:
            valid_df["cum_episodes"] = valid_df.get("env_runners/num_episodes", valid_df["training_iteration"])
            
        dir_name = csv_path.parent.name
        algo = None
        if "/DQN_Marl_" in s:
            algo = "DQN"
        elif "/IMPALA_Marl_" in s:
            algo = "IMPALA"
        elif "/APPO_Marl_" in s:
            algo = "APPO"
        elif "/PPO_Marl_" in s:
            algo = "PPO"
            
        if not algo or algo not in models_by_algo:
            continue
        
        m_tid = re.search(r'([a-f0-9]{5}_\d{5})', dir_name)
        tid = m_tid.group(1) if m_tid else dir_name[:12]
        
        lr = "N/A"
        gamma = "N/A"
        ent = "N/A"
        m_lr = re.search(r'lr[=_]([0-9]+(?:_[0-9]+)+|\d+\.\d+)', dir_name)
        if m_lr:
            lr = m_lr.group(1).replace("_", ".")
        m_g = re.search(r'gamma[=_]([0-9]+(?:_[0-9]+)+|\d+\.\d+)', dir_name)
        if m_g:
            gamma = m_g.group(1).replace("_", ".")
        m_e = re.search(r'entropy_coeff[=_]([0-9]+(?:_[0-9]+)+|\d+\.\d+)', dir_name)
        if m_e:
            ent = m_e.group(1).replace("_", ".")
        
        peak_idx = valid_df[rw_col].idxmax()
        peak_row = valid_df.loc[peak_idx]
        
        models_by_algo[algo].append({
            "algo": algo,
            "trial_id": tid,
            "lr": lr,
            "gamma": gamma,
            "ent": ent,
            "df": valid_df[["training_iteration", "cum_episodes", rw_col]],
            "rw_col": rw_col,
            "peak_rw": float(peak_row[rw_col]),
            "peak_ep": int(peak_row["cum_episodes"]),
            "peak_iter": int(peak_row["training_iteration"]),
            "total_ep": int(valid_df["cum_episodes"].iloc[-1])
        })

    total_models = sum(len(v) for v in models_by_algo.values())
    if total_models == 0:
        print("[ERROR] No valid models found in ray_results.")
        return

    # Print summary table in terminal
    print("\n" + "=" * 115)
    print(f"{'Algoritmo':<10} | {'Trial ID':<13} | {'LR':<7} | {'Gamma':<6} | {'Ent':<6} | {'Pico Reward':<16} | {'Episódio do Pico':<24} | {'Iteração Pico'}")
    print("=" * 115)
    for algo in target_algos:
        for m in sorted(models_by_algo[algo], key=lambda x: -x["peak_rw"]):
            ep_str = f"{m['peak_ep']} de {m['total_ep']} ({(m['peak_ep']/m['total_ep'])*100:.1f}%)"
            it_str = f"{m['peak_iter']}"
            print(f"{m['algo']:<10} | {m['trial_id']:<13} | {m['lr']:<7} | {m['gamma']:<6} | {m['ent']:<6} | {m['peak_rw']:+16.2f} | {ep_str:<24} | {it_str}")
    print("=" * 115 + "\n")

    palette = {
        "DQN": ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"],
        "IMPALA": ["#e377c2", "#17becf", "#ff7f0e", "#bcbd22"],
        "APPO": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        "PPO": ["#e6005c", "#8c1aff", "#009999", "#ff9900"]
    }

    # If single algorithm requested, plot single panel
    if args.algo != "ALL":
        fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
        algo = args.algo
        for idx, m in enumerate(sorted(models_by_algo[algo], key=lambda x: -x["peak_rw"])):
            c = palette[algo][idx % len(palette[algo])]
            lbl = f"{m['trial_id']} (LR={m['lr']}, γ={m['gamma']}) | Pico: {m['peak_rw']:+.1f} @ ep {m['peak_ep']}"
            ax.plot(m["df"]["cum_episodes"], m["df"][m["rw_col"]], label=lbl, color=c, linewidth=2.0)
            ax.scatter([m["peak_ep"]], [m["peak_rw"]], color=c, marker="*", s=160, zorder=5)

        ax.set_title(f"{algo} - Curvas de Aprendizado (Pico vs. Episódios)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Episódios de Treinamento", fontsize=11)
        ax.set_ylabel("Recompensa Média", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="best")
    else:
        # Full 3x2 Grid Figure for all 4 Algorithms + Champions Comparison + Summary Table
        fig, axes = plt.subplots(3, 2, figsize=(17, 14), dpi=130)
        plt.subplots_adjust(hspace=0.32, wspace=0.22)

        # 1. DQN Plot (Top-Left)
        ax_dqn = axes[0, 0]
        for idx, m in enumerate(sorted(models_by_algo["DQN"], key=lambda x: -x["peak_rw"])):
            c = palette["DQN"][idx % len(palette["DQN"])]
            lbl = f"{m['trial_id']} (LR={m['lr']}, γ={m['gamma']}) | Pico: {m['peak_rw']:+.1f} @ ep {m['peak_ep']}"
            ax_dqn.plot(m["df"]["cum_episodes"], m["df"][m["rw_col"]], label=lbl, color=c, linewidth=1.8, alpha=0.9)
            ax_dqn.scatter([m["peak_ep"]], [m["peak_rw"]], color=c, marker="*", s=140, zorder=5)

        ax_dqn.set_title("DQN - Curvas de Aprendizado (Pico vs. Episódios)", fontsize=12, fontweight="bold")
        ax_dqn.set_xlabel("Episódios de Treinamento", fontsize=10)
        ax_dqn.set_ylabel("Recompensa Média", fontsize=10)
        ax_dqn.grid(True, linestyle="--", alpha=0.5)
        ax_dqn.legend(fontsize=8, loc="lower right")

        # 2. IMPALA Plot (Top-Right)
        ax_imp = axes[0, 1]
        for idx, m in enumerate(sorted(models_by_algo["IMPALA"], key=lambda x: -x["peak_rw"])):
            c = palette["IMPALA"][idx % len(palette["IMPALA"])]
            lbl = f"{m['trial_id']} (LR={m['lr']}, Ent={m['ent']}) | Pico: {m['peak_rw']:+.1f} @ ep {m['peak_ep']}"
            ax_imp.plot(m["df"]["cum_episodes"], m["df"][m["rw_col"]], label=lbl, color=c, linewidth=1.8, alpha=0.9)
            ax_imp.scatter([m["peak_ep"]], [m["peak_rw"]], color=c, marker="*", s=140, zorder=5)

        ax_imp.set_title("IMPALA - Curvas de Aprendizado (Pico vs. Episódios)", fontsize=12, fontweight="bold")
        ax_imp.set_xlabel("Episódios de Treinamento", fontsize=10)
        ax_imp.set_ylabel("Recompensa Média", fontsize=10)
        ax_imp.grid(True, linestyle="--", alpha=0.5)
        ax_imp.legend(fontsize=8, loc="upper left")

        # 3. APPO Plot (Middle-Left)
        ax_app = axes[1, 0]
        for idx, m in enumerate(sorted(models_by_algo["APPO"], key=lambda x: -x["peak_rw"])):
            c = palette["APPO"][idx % len(palette["APPO"])]
            lbl = f"{m['trial_id']} (γ={m['gamma']}, Ent={m['ent']}) | Pico: {m['peak_rw']:+.1f} @ ep {m['peak_ep']}"
            ax_app.plot(m["df"]["cum_episodes"], m["df"][m["rw_col"]], label=lbl, color=c, linewidth=1.8, alpha=0.9)
            ax_app.scatter([m["peak_ep"]], [m["peak_rw"]], color=c, marker="*", s=140, zorder=5)

        ax_app.set_title("APPO - Curvas de Aprendizado (Pico vs. Episódios)", fontsize=12, fontweight="bold")
        ax_app.set_xlabel("Episódios de Treinamento", fontsize=10)
        ax_app.set_ylabel("Recompensa Média", fontsize=10)
        ax_app.grid(True, linestyle="--", alpha=0.5)
        ax_app.legend(fontsize=8, loc="upper left")

        # 4. PPO Plot (Middle-Right)
        ax_ppo = axes[1, 1]
        for idx, m in enumerate(sorted(models_by_algo["PPO"], key=lambda x: -x["peak_rw"])):
            c = palette["PPO"][idx % len(palette["PPO"])]
            lbl = f"{m['trial_id']} (γ={m['gamma']}, Ent={m['ent']}) | Pico: {m['peak_rw']:+.1f} @ ep {m['peak_ep']}"
            ax_ppo.plot(m["df"]["cum_episodes"], m["df"][m["rw_col"]], label=lbl, color=c, linewidth=1.8, alpha=0.9)
            ax_ppo.scatter([m["peak_ep"]], [m["peak_rw"]], color=c, marker="*", s=140, zorder=5)

        ax_ppo.set_title("PPO - Curvas de Aprendizado (Pico vs. Episódios)", fontsize=12, fontweight="bold")
        ax_ppo.set_xlabel("Episódios de Treinamento", fontsize=10)
        ax_ppo.set_ylabel("Recompensa Média", fontsize=10)
        ax_ppo.grid(True, linestyle="--", alpha=0.5)
        ax_ppo.legend(fontsize=8, loc="upper left")

        # 5. Comparison of Best Models of Each Algorithm (Bottom-Left)
        ax_cmp = axes[2, 0]
        best_models = []
        for algo in ["IMPALA", "APPO", "DQN", "PPO"]:
            if models_by_algo[algo]:
                best_models.append(max(models_by_algo[algo], key=lambda x: x["peak_rw"]))

        cmp_colors = ["#17becf", "#ff7f0e", "#1f77b4", "#e6005c"]
        for idx, m in enumerate(best_models):
            c = cmp_colors[idx % len(cmp_colors)]
            lbl = f"{m['algo']}: {m['trial_id']} | Pico: {m['peak_rw']:+.1f} @ ep {m['peak_ep']}"
            ax_cmp.plot(m["df"]["cum_episodes"], m["df"][m["rw_col"]], label=lbl, color=c, linewidth=2.2)
            ax_cmp.scatter([m["peak_ep"]], [m["peak_rw"]], color=c, marker="*", s=180, zorder=5)

        ax_cmp.set_title("Comparativo dos Melhores Modelos de Cada Algoritmo", fontsize=12, fontweight="bold")
        ax_cmp.set_xlabel("Episódios de Treinamento", fontsize=10)
        ax_cmp.set_ylabel("Recompensa Média", fontsize=10)
        ax_cmp.grid(True, linestyle="--", alpha=0.5)
        ax_cmp.legend(fontsize=8.5, loc="upper left")

        # 6. Summary Table Panel (Bottom-Right)
        ax_tbl = axes[2, 1]
        ax_tbl.axis("off")
        
        table_rows = []
        for algo in ["DQN", "IMPALA", "APPO", "PPO"]:
            for m in sorted(models_by_algo[algo], key=lambda x: -x["peak_rw"]):
                table_rows.append([
                    m["algo"],
                    m["trial_id"],
                    f"{m['peak_rw']:+.1f}",
                    f"Ep {m['peak_ep']}",
                    f"{(m['peak_ep']/m['total_ep'])*100:.0f}%"
                ])
                
        col_labels = ["Algoritmo", "Trial ID", "Pico Reward", "Episódio Pico", "% Treino"]
        table = ax_tbl.table(
            cellText=table_rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.25)

        # Style header cells
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")

        ax_tbl.set_title("Tabela Resumo dos Picos por Episódio", fontsize=12, fontweight="bold", pad=12)

        fig.suptitle("Overcooked MARL (4 Agentes - tutorial.txt) - Evolução de Recompensa e Picos por Episódio (DQN, IMPALA, APPO, PPO)", fontsize=13, fontweight="bold", y=0.985)

    output_path = Path(args.output).resolve() if args.output else Path(__file__).parent / "learning_curves.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"[SUCESSO] Gráfico salvo em: {output_path}")

if __name__ == "__main__":
    main()
