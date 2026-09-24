#!/usr/bin/env python3
"""
Utility script to plot learning curves directly to an image or display them.
Reads model_info.json from experiments/ray_results.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def main():
    base_dir = Path(__file__).parent / "ray_results"
    all_runs = []
    
    for model_info_file in base_dir.glob("**/model_info.json"):
        try:
            with open(model_info_file, 'r') as f:
                data = json.load(f)
            config_name = data.get('config_name', model_info_file.parent.name)
            curves = data.get('learning_curves', [])
            for pt in curves:
                all_runs.append({
                    'Config': config_name,
                    'Iteration': pt['iteration'],
                    'Mean Reward': pt['reward_mean']
                })
        except Exception as e:
            print(f"Error reading {model_info_file}: {e}")

    if not all_runs:
        print(f"No model_info.json found in {base_dir}")
        return

    df = pd.DataFrame(all_runs)
    plt.figure(figsize=(10, 5), dpi=120)
    for name, group in df.groupby('Config'):
        plt.plot(group['Iteration'], group['Mean Reward'], marker='o', linewidth=2, label=name)

    plt.title("Overcooked MARL - Curva de Aprendizado PPO (Ciclo 3)", fontsize=13, fontweight='bold')
    plt.xlabel("Iteração de Treinamento", fontsize=11)
    plt.ylabel("Recompensa Média por Episódio (Scaled)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()

    output_path = Path(__file__).parent / "learning_curves.png"
    plt.savefig(output_path)
    print(f"Gráfico salvo com sucesso em: {output_path}")

if __name__ == "__main__":
    main()
