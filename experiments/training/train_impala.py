#!/usr/bin/env python3
"""
====================================================================================================
⚡ OVERCOOKED MARL - TREINAMENTO MULTIAGENTE COM IMPALA (V-TRACE DISTRIBUÍDO)
====================================================================================================

📌 POR QUE ESTE TREINAMENTO É EXCELENTE PARA O PROJETO OVERCOOKED:
----------------------------------------------------------------------------------------------------
1. VAZÃO MASSIVA DE EXPERIÊNCIA (5x A 10x MAIS PASSOS POR SEGUNDO):
   O PPO padrão é estritamente síncrono: todos os workers precisam parar e esperar o fim da coleta
   do lote para que o modelo calcule os gradientes. Se um worker se atrasar, todos os outros ficam ociosos.
   O IMPALA (Importance Weighted Actor-Learner Architecture) desacopla completamente os Workers do Learner:
   os workers enviam trajetórias sem parar e o learner processa em fila contínua, permitindo gerar
   20M a 50M de passos no servidor no mesmo tempo que o PPO levaria para fazer 3M.

2. EXPLORAÇÃO RÁPIDA DE ÁRVORES DE ESTADOS COMPLEXAS:
   Com uma taxa de amostragem tão superior, a chance estatística de os agentes tropeçarem na montagem
   de saladas compostas (GARDEN_SALAD e FRESH_SALAD) por exploração estocástica cresce exponencialmente.

3. ESTABILIDADE MATEMÁTICA VIA V-TRACE:
   Como os dados chegam de múltiplos workers rodando em paralelo, algumas trajetórias podem ter sido
   coletadas com uma política ligeiramente anterior à atual. O algoritmo V-Trace corrige o desvio
   de importância (off-policy correction) com corte de pesos, garantindo convergência estável sem
   divergência de gradiente.

4. SINERGIA COM COORDENAÇÃO MULTIAGENTE:
   Permite treinar tanto 2 quanto 4 agentes mantendo políticas independentes com altíssimo paralelismo.
====================================================================================================
"""

import os
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if os.environ.get("HOME") in (None, "", "/"):
    os.environ["HOME"] = str(PROJECT_ROOT)
os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
os.environ["AIR_VERBOSITY"] = "0"

import argparse
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env

try:
    from overcooked_marl_env import OvercookedMARL
except ImportError:
    from env import OvercookedMARL

torch, nn = try_import_torch()

from experiments.utils.training_postprocessor import (
    TrainingProgressCallback,
    create_cli_progress_reporter,
    postprocess_experiment_trials,
    resolve_restore_checkpoint
)

# ======================================================================
# HYPERPARAMETER GRID SEARCH CONFIGURATION (IMPALA)
# ======================================================================
HYPERPARAMETER_GRID = {
    "lr": [3e-4, 5e-4],                   # Taxa de aprendizado para RMSprop / Adam
    "gamma": [0.98, 0.99],                # Fator de desconto
    "entropy_coeff": [0.01, 0.02],        # Coeficiente de entropia para exploração contínua
    "train_batch_size": [500, 1000],      # Lote de atualização do learner
    "num_episodes": [2500],               # Total de episódios
    "seed": [42]
}

class ParallelEnvToMultiAgent(MultiAgentEnv):
    """Wrapper para converter PettingZoo ParallelEnv em RLlib MultiAgentEnv."""
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
        obs, rewards, terminations, truncations, infos = self.env.step(action_dict)
        all_terminated = all(terminations.get(agent, False) for agent in self.agents)
        all_truncated = all(truncations.get(agent, False) for agent in self.agents)
        terminations['__all__'] = all_terminated
        truncations['__all__'] = all_truncated
        return obs, rewards, terminations, truncations, infos
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        if name.startswith('_') or name in ['env', 'agents', 'observation_space', 'action_space']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return getattr(object.__getattribute__(self, 'env'), name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

def parse_args():
    default_workers = max(2, min(12, os.cpu_count() - 2)) if hasattr(os, 'cpu_count') and os.cpu_count() else 4
    parser = argparse.ArgumentParser(description="High-Throughput Distributed IMPALA Training for Overcooked MARL.")
    parser.add_argument("--map", type=str, default="tutorial.txt", help="Map file name (default: tutorial.txt)")
    parser.add_argument("--num-agents", type=int, default=2, help="Number of agents to train (default: 2)")
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--gamma", type=float, default=None, help="Override discount factor")
    parser.add_argument("--workers", type=int, default=default_workers, help="Number of parallel rollout workers")
    parser.add_argument("--gpus", type=float, default=1.0, help="Total GPUs allocated (default: 1.0)")
    parser.add_argument("--concurrent", type=int, default=2, help="Max concurrent trials in Grid Search")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Checkpoint frequency in iterations")
    parser.add_argument("--keep-checkpoints-num", type=int, default=15, help="Number of checkpoints to keep")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size")
    parser.add_argument("--restore-checkpoint", type=str, default=None, help="Path to checkpoint or 'auto' to warm-start policies")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.episodes is not None:
        HYPERPARAMETER_GRID["num_episodes"] = [args.episodes]
    if args.lr is not None:
        HYPERPARAMETER_GRID["lr"] = [args.lr]
    if args.gamma is not None:
        HYPERPARAMETER_GRID["gamma"] = [args.gamma]

    print("\n" + "="*70)
    print("⚡ INICIANDO TREINAMENTO DISTRIBUÍDO COM IMPALA (V-TRACE)")
    print(f"  Mapa:           {args.map}")
    print(f"  Agentes:        {args.num_agents}")
    print(f"  Workers:        {args.workers}")
    print(f"  Concorrência:   {args.concurrent}")
    print(f"  Opções do Grid: {HYPERPARAMETER_GRID}")
    print("="*70 + "\n")

    ray.init(ignore_reinit_error=True)

    def env_creator_fn(num_episodes_val):
        return OvercookedMARL(
            render_mode=None,
            map_path=args.map,
            num_fixed_agents=args.num_agents,
            verbose=False,
            featurized_obs=True
        )

    env_name = "overcooked_marl_parallel_impala"
    register_env(env_name, lambda config: ParallelEnvToMultiAgent(env_creator_fn(config.get("num_episodes", 300))))

    temp_env = ParallelEnvToMultiAgent(env_creator_fn(300))
    agents = temp_env.agents
    obs_space = temp_env.observation_space["agent_0"]
    act_space = temp_env.action_space["agent_0"]
    temp_env.close()

    policies = {
        f"policy_{agent_id}": (None, obs_space, act_space, {})
        for agent_id in agents
    }

    tune_lr = tune.grid_search(HYPERPARAMETER_GRID["lr"]) if len(HYPERPARAMETER_GRID["lr"]) > 1 else HYPERPARAMETER_GRID["lr"][0]
    tune_gamma = tune.grid_search(HYPERPARAMETER_GRID["gamma"]) if len(HYPERPARAMETER_GRID["gamma"]) > 1 else HYPERPARAMETER_GRID["gamma"][0]
    tune_entropy = tune.grid_search(HYPERPARAMETER_GRID["entropy_coeff"]) if len(HYPERPARAMETER_GRID["entropy_coeff"]) > 1 else HYPERPARAMETER_GRID["entropy_coeff"][0]
    tune_batch_size = tune.grid_search(HYPERPARAMETER_GRID["train_batch_size"]) if len(HYPERPARAMETER_GRID["train_batch_size"]) > 1 else HYPERPARAMETER_GRID["train_batch_size"][0]
    tune_episodes = tune.grid_search(HYPERPARAMETER_GRID["num_episodes"]) if len(HYPERPARAMETER_GRID["num_episodes"]) > 1 else HYPERPARAMETER_GRID["num_episodes"][0]

    impala_config = (
        ImpalaConfig()
        .environment(
            env=env_name,
            env_config={
                "num_episodes": tune_episodes,
                "map": args.map,
                "num_agents": args.num_agents,
            }
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .training(
            lr=tune_lr,
            gamma=tune_gamma,
            entropy_coeff=tune_entropy,
            train_batch_size=tune_batch_size,
            grad_clip=40.0,
            vtrace=True,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }
        )
        .resources(
            num_gpus=(args.gpus / max(1, args.concurrent)) if (args.gpus > 0 and torch.cuda.is_available()) else 0
        )
        .env_runners(
            num_env_runners=max(1, args.workers // max(1, args.concurrent)),
            rollout_fragment_length=50,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"policy_{agent_id}",
            policies_to_train=list(policies.keys())
        )
    )

    config_dict = impala_config.to_dict()

    def stop_fn(trial_id, result):
        config = result.get("config", {})
        env_config_dict = config.get("env_config", {})
        episodes_limit = env_config_dict.get("num_episodes", 2500)
        target_timesteps = episodes_limit * 2000
        return result.get("timesteps_total", 0) >= target_timesteps

    exp_dir = Path(__file__).resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir_name = f"IMPALA_Marl_{timestamp}"
    ray_results_base = exp_dir / "ray_results" / timestamp_dir_name
    ray_results_base.mkdir(parents=True, exist_ok=True)

    params_str = f"map-{args.map.replace('.txt', '')}_agents-{args.num_agents}_workers{args.workers}"
    params_dir_name = f"IMPALA_Marl_{params_str}"

    start_time = time.time()
    try:
        analysis = tune.run(
            "IMPALA",
            name=params_dir_name,
            stop=stop_fn,
            config=config_dict,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_at_end=True,
            checkpoint_score_attr="env_runners/episode_reward_mean",
            keep_checkpoints_num=args.keep_checkpoints_num,
            max_concurrent_trials=args.concurrent,
            verbose=1,
            progress_reporter=create_cli_progress_reporter(),
            callbacks=[TrainingProgressCallback()],
            storage_path=str(ray_results_base.resolve())
        )
        postprocess_experiment_trials(
            analysis=analysis,
            algorithm_name="IMPALA",
            args=args,
            ray_results_base=ray_results_base,
            params_dir_name=params_dir_name,
            timestamp_dir_name=timestamp_dir_name,
            timestamp=timestamp,
            start_time=start_time
        )
        print("\n✅ Treinamento IMPALA finalizado com sucesso!")
    except Exception as e:
        print(f"Erro durante sessão de treino IMPALA: {e}")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
