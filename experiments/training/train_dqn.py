#!/usr/bin/env python3
"""
====================================================================================================
🍳 OVERCOOKED MARL - TREINAMENTO MULTIAGENTE COM DQN / RAINBOW (OFF-POLICY)
====================================================================================================

📌 POR QUE ESTE TREINAMENTO É EXCELENTE PARA O PROJETO OVERCOOKED:
----------------------------------------------------------------------------------------------------
1. OFF-POLICY COM REPLAY BUFFER PRIORITIZADO (PER):
   O PPO (on-policy) descarta trajetórias logo após a atualização de gradientes. Se os agentes
   descobrem por acaso a sequência complexa de uma GARDEN_SALAD ou FRESH_SALAD (pegar, passar pelo
   balcão, picar, empratar e entregar), o PPO pode esquecer esse evento raro se a amostragem for descartada.
   No DQN, TODAS as transições bem-sucedidas são armazenadas em um Replay Buffer e reamostradas
   centenas de vezes, solidificando a política de receitas compostas sem esquecimento catastrófico.

2. ESPAÇO DE AÇÕES DISCRETO IDEAL:
   O Overcooked possui um conjunto pequeno de 5 ações discretas (Mover Cima, Baixo, Esquerda, Direita e
   Interagir). Q-Learning é matematicamente comprovado e altamente eficiente para estimar Q(s, a)
   em espaços discretos bem delimitados.

3. DOUBLE DQN + DUELING ARCHITECTURE:
   - Double Q-Learning evita a superestimação sistemática de valores de recompensa.
   - Dueling Networks decompõem o valor do estado V(s) e a vantagem A(s, a), permitindo ao agente
     reconhecer se um estado é promissor (ex: prato quase pronto no balcão) independentemente de qual
     movimento imediato for executado.

4. N-STEP BOOTSTRAPPING (n=3):
   Propaga o retorno futuro de 3 passos para trás, acelerando dramaticamente a atribuição de crédito
   para sequências encadeadas (ex: pegar ingrediente -> colocar na bancada -> picar).
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
import random
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig, DQN
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
# HYPERPARAMETER GRID SEARCH CONFIGURATION (DQN / RAINBOW)
# ======================================================================
HYPERPARAMETER_GRID = {
    "lr": [1e-4, 2e-4],                   # Taxas estáveis (1e-4 foi campeão +177.6; 3e-4 divergia Q-values)
    "gamma": [0.98, 0.99],                # Fatores de desconto para horizonte curto e longo
    "n_step": [5],                        # 5-step bootstrapping propaga crédito de corte (3 cortes + passe)
    "train_batch_size": [256],            # Lote maior com menos overhead por passo
    "target_network_update_freq": [1000], # Sincronização estável da rede alvo
    "num_episodes": [1000],               # 1000 episódios / iterações (<24h benchmark)
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

from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import MultiAgentPrioritizedReplayBuffer
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer

def _patch_prioritized_replay_buffer():
    """Patches Ray RLlib PrioritizedReplayBuffer to eliminate IndexError when sampling at boundaries."""
    if getattr(PrioritizedReplayBuffer, "_bounds_safe_patched", False):
        return

    orig_sample_proportional = PrioritizedReplayBuffer._sample_proportional

    def _safe_sample_proportional(self, num_items: int):
        storage_len = len(self._storage)
        if storage_len == 0:
            return []
        res = []
        for _ in range(num_items):
            total_mass = self._it_sum.sum(0, storage_len)
            mass = random.random() * total_mass
            idx = self._it_sum.find_prefixsum_idx(mass)
            # Clamping guarantees idx is strictly within [0, storage_len - 1]
            idx = max(0, min(int(idx), storage_len - 1))
            res.append(idx)
        return res

    PrioritizedReplayBuffer._sample_proportional = _safe_sample_proportional

    orig_sample = PrioritizedReplayBuffer.sample

    def _safe_sample(self, num_items: int, beta: float = 0.4, **kwargs):
        if len(self) == 0 or len(self._storage) == 0:
            return None
        return orig_sample(self, num_items, beta=beta, **kwargs)

    PrioritizedReplayBuffer.sample = _safe_sample
    PrioritizedReplayBuffer._bounds_safe_patched = True

_patch_prioritized_replay_buffer()

class MetaWithContains(type(MultiAgentPrioritizedReplayBuffer)):
    def __contains__(cls, item):
        return str(item) in cls.__name__

class SafeMAPRB(MultiAgentPrioritizedReplayBuffer, metaclass=MetaWithContains):
    """Subclass patching Ray 2.40+ ABCMeta containment bug and PrioritizedReplayBuffer bounds bug."""
    def __init__(self, *args, **kwargs):
        _patch_prioritized_replay_buffer()
        super().__init__(*args, **kwargs)

    def sample(self, *args, **kwargs):
        _patch_prioritized_replay_buffer()
        return super().sample(*args, **kwargs)

def parse_args():
    default_workers = max(2, min(8, os.cpu_count() - 2)) if hasattr(os, 'cpu_count') and os.cpu_count() else 4
    parser = argparse.ArgumentParser(description="Multi-Agent DQN / Rainbow Training for Overcooked MARL.")
    parser.add_argument("--map", type=str, default="tutorial.txt", help="Map file name (default: tutorial.txt)")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents to train (default: 4)")
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--gamma", type=float, default=None, help="Override discount factor")
    parser.add_argument("--workers", type=int, default=default_workers, help="Number of rollout workers")
    parser.add_argument("--gpus", type=float, default=1.0, help="Total GPUs allocated (default: 1.0)")
    parser.add_argument("--concurrent", type=int, default=2, help="Max concurrent trials in Grid Search")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Checkpoint frequency in iterations")
    parser.add_argument("--keep-checkpoints-num", type=int, default=15, help="Number of checkpoints to keep")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size")
    parser.add_argument("--restore-checkpoint", type=str, default=None, help="Path to checkpoint or 'auto' to warm-start policies")
    parser.add_argument("--resume", type=str, default=None, help="Resume Ray Tune experiment ('AUTO', 'LOCAL', etc.)")
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
    print("🚀 INICIANDO TREINAMENTO MULTIAGENTE COM DQN / RAINBOW (OFF-POLICY)")
    print(f"  Mapa:           {args.map}")
    print(f"  Agentes:        {args.num_agents}")
    print(f"  Workers:        {args.workers}")
    print(f"  Concorrência:   {args.concurrent}")
    print(f"  Opções do Grid: {HYPERPARAMETER_GRID}")
    print("="*70 + "\n")

    avail_cpus = os.cpu_count() or 4
    max_runners_by_cpu = max(1, (avail_cpus // max(1, args.concurrent)) - 1)
    runners_per_trial = max(1, min(args.workers // max(1, args.concurrent), max_runners_by_cpu))
    ray_total_cpus = max(avail_cpus, args.concurrent * (runners_per_trial + 1))
    ray.init(num_cpus=ray_total_cpus, ignore_reinit_error=True)

    def env_creator_fn(num_episodes_val):
        return OvercookedMARL(
            render_mode=None,
            map_path=args.map,
            num_fixed_agents=args.num_agents,
            verbose=False,
            featurized_obs=True
        )

    env_name = "overcooked_marl_parallel_dqn"
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
    tune_n_step = tune.grid_search(HYPERPARAMETER_GRID["n_step"]) if len(HYPERPARAMETER_GRID["n_step"]) > 1 else HYPERPARAMETER_GRID["n_step"][0]
    tune_batch_size = tune.grid_search(HYPERPARAMETER_GRID["train_batch_size"]) if len(HYPERPARAMETER_GRID["train_batch_size"]) > 1 else HYPERPARAMETER_GRID["train_batch_size"][0]
    tune_target_sync = tune.grid_search(HYPERPARAMETER_GRID["target_network_update_freq"]) if len(HYPERPARAMETER_GRID["target_network_update_freq"]) > 1 else HYPERPARAMETER_GRID["target_network_update_freq"][0]
    tune_episodes = tune.grid_search(HYPERPARAMETER_GRID["num_episodes"]) if len(HYPERPARAMETER_GRID["num_episodes"]) > 1 else HYPERPARAMETER_GRID["num_episodes"][0]

    dqn_config = (
        DQNConfig()
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
            double_q=True,
            dueling=True,
            n_step=tune_n_step,
            train_batch_size=tune_batch_size,
            target_network_update_freq=tune_target_sync,
            num_steps_sampled_before_learning_starts=2000,
            replay_buffer_config={
                "type": SafeMAPRB,
                "capacity": 100000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
            },
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }
        )
        .resources(
            num_gpus=max(0.0, round((args.gpus / max(1, args.concurrent)) - 0.001, 4)) if (args.gpus > 0 and torch.cuda.is_available()) else 0
        )
        .env_runners(
            num_env_runners=runners_per_trial,
            rollout_fragment_length=20,
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 150000,
            },
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"policy_{agent_id}",
            policies_to_train=list(policies.keys())
        )
    )

    config_dict = dqn_config.to_dict()

    def stop_fn(trial_id, result):
        config = result.get("config", {})
        env_config_dict = config.get("env_config", {})
        episodes_limit = env_config_dict.get("num_episodes", 1000)
        target_timesteps = episodes_limit * 1000
        return result.get("timesteps_total", 0) >= target_timesteps or result.get("training_iteration", 0) >= episodes_limit

    exp_dir = Path(__file__).resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir_name = f"DQN_Marl_{timestamp}"
    ray_results_base = exp_dir / "ray_results" / timestamp_dir_name
    ray_results_base.mkdir(parents=True, exist_ok=True)

    params_str = f"map-{args.map.replace('.txt', '')}_agents-{args.num_agents}_workers{args.workers}"
    params_dir_name = f"DQN_Marl_{params_str}"

    start_time = time.time()
    try:
        analysis = tune.run(
            "DQN",
            name=params_dir_name,
            stop=stop_fn,
            config=config_dict,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_at_end=True,
            checkpoint_score_attr="env_runners/episode_reward_mean",
            keep_checkpoints_num=args.keep_checkpoints_num,
            max_concurrent_trials=args.concurrent,
            max_failures=3,
            resume=args.resume if args.resume else False,
            verbose=1,
            progress_reporter=create_cli_progress_reporter(),
            callbacks=[TrainingProgressCallback()],
            storage_path=str(ray_results_base.resolve())
        )
        postprocess_experiment_trials(
            analysis=analysis,
            algorithm_name="DQN",
            args=args,
            ray_results_base=ray_results_base,
            params_dir_name=params_dir_name,
            timestamp_dir_name=timestamp_dir_name,
            timestamp=timestamp,
            start_time=start_time
        )
        print("\n✅ Treinamento DQN finalizado com sucesso!")
    except Exception as e:
        print(f"Erro durante sessão de treino DQN: {e}")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
