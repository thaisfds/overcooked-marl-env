<div align="center">

![Overview](https://img.shields.io/badge/🌟_01_Overview-POC_I_vs_POC_II-9333ea?style=for-the-badge&labelColor=1e1b4b)
![Benchmark](https://img.shields.io/badge/🧠_Benchmarks-PPO_|_DQN_|_IMPALA_|_APPO-3b82f6?style=for-the-badge&labelColor=1e1b4b)

# 🌟 01 — Overview & Environment Evolution

Visual comparison between **POC I** (open kitchen + CNN observation) and **POC II** (divided kitchen + 150D featurized observation with focus rays), alongside multi-algorithm and scaling benchmarks.

</div>

<img src="../assets/divider_gamer.svg" width="100%" />

<div align="center">
  <img src="./evolucao_ambiente_poc1_vs_poc2.png" alt="POC I vs POC II Evolution" width="95%" />
</div>

<br/>

<div align="center">
<table>
  <tr>
    <td align="center" width="50%">
      <img src="./comparacao_algoritmos_marl.png" alt="Algorithm Comparison" width="100%" />
      <br/><sub><b>Algorithm Benchmark (4 Agents):</b> IMPALA (+3,990.6 pts), APPO (+1,584.8 pts), DQN (+177.6 pts), and PPO (+131.0 pts).</sub>
    </td>
    <td align="center" width="50%">
      <img src="./tempo_treino_e_eficiencia_algoritmos.png" alt="Training Time & Efficiency" width="100%" />
      <br/><sub><b>Training Efficiency:</b> Wall-clock time (hours) and sampling throughput (steps/s).</sub>
    </td>
  </tr>
</table>
</div>

<details>
<summary><b>📑 All Figures in this Folder (9 files)</b></summary>

* [`evolucao_ambiente_poc1_vs_poc2.png`](./evolucao_ambiente_poc1_vs_poc2.png) — Side-by-side comparison of POC I vs. POC II.
* [`cozinha_aberta_poc1.png`](./cozinha_aberta_poc1.png) — POC I open kitchen screenshot.
* [`cozinha_dividida_2_agentes.png`](./cozinha_dividida_2_agentes.png) — POC II divided kitchen (`tutorial.txt`) with 2 agents.
* [`cozinha_dividida_4_agentes.png`](./cozinha_dividida_4_agentes.png) — POC II divided kitchen (`tutorial.txt`) with 4 agents.
* [`comparacao_algoritmos_marl.png`](./comparacao_algoritmos_marl.png) — Reward distribution across MARL algorithm families.
* [`tempo_treino_e_eficiencia_algoritmos.png`](./tempo_treino_e_eficiencia_algoritmos.png) — Training duration and throughput comparison.
* [`comparacao_escalabilidade_agentes.png`](./comparacao_escalabilidade_agentes.png) — Agent scaling distribution and learning dynamics.
* [`comparacao_geral_desempenho.png`](./comparacao_geral_desempenho.png) — Overall model ranking and convergence effort.
* [`comparacao_geral_hiperparametros.png`](./comparacao_geral_hiperparametros.png) — Global hyperparameter sensitivity (LR, Entropy, Gamma).
</details>

<img src="../assets/divider_gamer.svg" width="100%" />

⬅️ **[Back to Results Index](../README.md)**
