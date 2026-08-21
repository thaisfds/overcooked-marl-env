# ppo_overcooked_marl.py (CÓDIGO FINAL E COMPLETO)

import os
import time
import json
from datetime import datetime
from typing import Dict, Any
from itertools import product
from pathlib import Path

import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig 
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env

import gymnasium as gym
try:
    from overcooked_marl_env import OvercookedMARL 
except ImportError:
    from env import OvercookedMARL 

torch, nn = try_import_torch()


# ----------------------------------------------------------------------
# WRAPPER PARA CONVERTER ParallelEnv EM MultiAgentEnv DO RLLIB
# ----------------------------------------------------------------------

class ParallelEnvToMultiAgent(MultiAgentEnv):
    """Wrapper para converter PettingZoo ParallelEnv em RLLib MultiAgentEnv."""
    
    def __init__(self, parallel_env):
        self.env = parallel_env
        self.agents = self.env.possible_agents
        # Chamar super().__init__() depois de inicializar self.env para evitar recursão
        super().__init__()
        
    @property
    def observation_space(self):
        # RLLib espera um dicionário de espaços - usar métodos do ambiente
        return {
            agent: self.env.observation_space(agent) 
            for agent in self.agents
        }
    
    @property
    def action_space(self):
        # RLLib espera um dicionário de espaços - usar métodos do ambiente
        return {
            agent: self.env.action_space(agent) 
            for agent in self.agents
        }
    
    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return obs, infos
    
    def step(self, action_dict):
        obs, rewards, terminations, truncations, infos = self.env.step(action_dict)
        # Scale rewards to prevent gradient explosion (NaN/Inf)
        scaled_rewards = {agent: r * 0.01 for agent, r in rewards.items()}
        
        all_terminated = all(terminations.get(agent, False) for agent in self.agents)
        all_truncated = all(truncations.get(agent, False) for agent in self.agents)
        
        terminations['__all__'] = all_terminated
        truncations['__all__'] = all_truncated
        
        return obs, scaled_rewards, terminations, truncations, infos
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    def __getattr__(self, name):
        # Evitar recursão: só delegar se o atributo não existir neste objeto
        # Não delegar atributos privados ou que já existem
        if name.startswith('_') or name in ['env', 'agents', 'observation_space', 'action_space']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # Delegar outros atributos para o ambiente original apenas se env existir
        try:
            env = object.__getattribute__(self, 'env')
            return getattr(env, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# ----------------------------------------------------------------------
# MODELO CNN CUSTOMIZADO PARA PPO (Opção 2 - Fallback)
# ----------------------------------------------------------------------

class CNNPPOModel(TorchModelV2, nn.Module):
    """Custom CNN model for PPO to process spatial observations (Legacy API)."""
    
    def __init__(
        self,
        obs_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        num_outputs: int,
        model_config: Dict,
        name: str,
        **kw
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )
        
        # Extract observation shape (H, W, C)
        h, w, c = obs_space.shape
        
        # CNN layers similar to ActorCritic in ppo_overcooked.py
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            sample = torch.zeros(1, c, h, w)
            n_flatten = self.cnn(sample).shape[1]
        
        # Fully connected layers
        self.shared = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
        )
        
        # Output layer for action logits
        self.action_head = nn.Linear(128, num_outputs)
        
        # Value function head
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, input_dict, state, seq_lens):
        """Forward pass through the network."""
        obs = input_dict["obs"]
        
        # Convert from (batch, H, W, C) to (batch, C, H, W)
        if len(obs.shape) == 4:
            obs = obs.permute(0, 3, 1, 2)
        elif len(obs.shape) == 3:
            obs = obs.permute(2, 0, 1).unsqueeze(0)
        
        # Normalize to [0, 1]
        obs = obs.float() / 255.0
        
        # Forward through CNN
        features = self.cnn(obs)
        features = self.shared(features)
        
        # Store features for value function
        self._features = features
        
        # Action logits
        logits = self.action_head(features)
        
        return logits, state
    
    def value_function(self):
        """Return the value function output."""
        return self.value_head(self._features).squeeze(1)


# ----------------------------------------------------------------------
# FUNÇÕES UTILITÁRIAS
# ----------------------------------------------------------------------

def generate_hyperparameter_combinations(param_grid: dict) -> list:
    keys = param_grid.keys()
    values = [param_grid[key] for key in keys]
    combinations_raw = product(*values)
    return [dict(zip(keys, v)) for v in combinations_raw]


def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    else:
        return str(obj)

def generate_params_string(params: Dict[str, Any]) -> str:
    """
    Gera string descritiva dos parâmetros para usar em nomes de diretórios.
    
    Args:
        params: Dicionário com parâmetros de treinamento
    
    Returns:
        String formatada: "lr1e-3_gamma0.90_batch4000_mb128_epochs16_episodes200_entropy0.20_seed42"
    """
    def format_value(val):
        if isinstance(val, (int, float)):
            if isinstance(val, float) and (val < 0.001 or val >= 1000):
                return f"{val:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
            elif isinstance(val, float):
                return f"{val:.3f}".rstrip('0').rstrip('.')
            else:
                return str(val)
        return str(val)
    
    parts = []
    if 'learning_rate' in params:
        parts.append(f"lr{format_value(params['learning_rate'])}")
    if 'gamma' in params:
        parts.append(f"gamma{format_value(params['gamma'])}")
    if 'train_batch_size' in params:
        parts.append(f"batch{params['train_batch_size']}")
    if 'sgd_minibatch_size' in params:
        parts.append(f"mb{params['sgd_minibatch_size']}")
    if 'update_epochs' in params:
        parts.append(f"epochs{params['update_epochs']}")
    elif 'num_epochs' in params:
        parts.append(f"epochs{params['num_epochs']}")
    if 'num_episodes' in params:
        parts.append(f"episodes{params['num_episodes']}")
    if 'entropy_coeff' in params:
        parts.append(f"entropy{format_value(params['entropy_coeff'])}")
    if 'seed' in params:
        parts.append(f"seed{params['seed']}")
    
    return "_".join(parts)

# ----------------------------------------------------------------------
# CONFIGURAÇÕES DO AMBIENTE E PARÂMETROS
# ----------------------------------------------------------------------

# Calcular caminho do mapa (default_map.txt) - funciona tanto quando executado quanto quando importado
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MAP_PATH_ABS = os.path.join(_SCRIPT_DIR, "overcooked_marl_env", "map", "maps", "default_map.txt")

ENVIRONMENT_CONFIG = {
    'map_path': _MAP_PATH_ABS,  # Usar default_map.txt por padrão
    'num_fixed_agents': 1, 
    'render_mode': None,
    'verbose': False,
    'max_episode_steps': 1200,
    'featurized_obs': True,  # ADICIONADO: observações 1D vetorizadas
}

param_grid = {
    'learning_rate': [3e-4, 1e-3], 
    'gamma': [0.90, 0.95, 0.995],
    'train_batch_size': [4000], 
    'sgd_minibatch_size': [128],
    'update_epochs': [4, 8, 16], 
    'num_episodes': [100, 200, 400], 
    'entropy_coeff': [0.15, 0.20, 0.25],
    'seed': [42],
}


def env_creator(env_config=None):
    return OvercookedMARL(
        render_mode=ENVIRONMENT_CONFIG['render_mode'],
        map_path=ENVIRONMENT_CONFIG['map_path'], 
        num_fixed_agents=ENVIRONMENT_CONFIG['num_fixed_agents'],
        verbose=ENVIRONMENT_CONFIG['verbose'],
        featurized_obs=ENVIRONMENT_CONFIG.get('featurized_obs', False)
    )

# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # O caminho do mapa já está configurado em ENVIRONMENT_CONFIG acima
    # Verificar se o arquivo existe
    if not os.path.exists(ENVIRONMENT_CONFIG['map_path']):
        print(f"AVISO: Mapa não encontrado em {ENVIRONMENT_CONFIG['map_path']}")
        print(f"Tentando usar caminho relativo...")
        # Fallback para caminho relativo
        ENVIRONMENT_CONFIG['map_path'] = os.path.join("overcooked_marl_env", "map", "maps", "default_map.txt")
    
    print(f"Mapa configurado: {ENVIRONMENT_CONFIG['map_path']}")
    
    ray.init(ignore_reinit_error=True)
    
    # Registrar modelo customizado (Opção 2 - Fallback)
    ModelCatalog.register_custom_model("cnn_ppo_model", CNNPPOModel)
    
    env_name = "overcooked_marl_parallel"
    # Usar wrapper para converter ParallelEnv em MultiAgentEnv do RLLib
    register_env(env_name, lambda config: ParallelEnvToMultiAgent(env_creator())) 
    
    test_env = ParallelEnvToMultiAgent(env_creator())
    obs_space_single = test_env.observation_space["agent_0"]
    act_space_single = test_env.action_space["agent_0"]
    test_env.close()
    
    print(f"Observation space (Single Agent): {obs_space_single}")
    print(f"Action space (Single Agent): {act_space_single}")
    
    combinations = generate_hyperparameter_combinations(param_grid)
    
    # Criar timestamp único para esta sessão de treinamento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TIMESTAMP_DIR = f"PPO_Marl_{timestamp}"
    
    # Criar pasta para salvar resultados do Ray
    RAY_RESULTS_BASE_DIR = os.path.join(_SCRIPT_DIR, "ray_results", TIMESTAMP_DIR)
    os.makedirs(RAY_RESULTS_BASE_DIR, exist_ok=True)
    print(f"Resultados do Ray serão salvos em: {RAY_RESULTS_BASE_DIR}\n")
    
    start_time_total = time.time()

    for i, params_original in enumerate(combinations):
        config = params_original.copy()
        
        num_episodes = config.pop('num_episodes')
        avg_episode_length = ENVIRONMENT_CONFIG.get('max_episode_steps', 1200) 
        timesteps_total = num_episodes * avg_episode_length
        
        # PPO Parâmetros
        train_batch_size = config.pop('train_batch_size')
        sgd_minibatch_size = config.pop('sgd_minibatch_size') 
        num_epochs = config.pop('update_epochs')
        entropy_coeff = config.pop('entropy_coeff', 0.01)  # Padrão: 0.01 se não especificado
        
        # 6. Configuração do PPO (IDPPO Multi-Agent)
        ppo_config = (
            PPOConfig()
            .environment(env=env_name)
            .api_stack(
                enable_rl_module_and_learner=False, 
                enable_env_runner_and_connector_v2=False
            )
            .training(
                lr=config['learning_rate'],
                gamma=config['gamma'],
                num_epochs=num_epochs,
                entropy_coeff=entropy_coeff,  # Coeficiente de entropia para aumentar exploração 
                grad_clip=1.0,  # Clip gradients to prevent explosion (NaN/Inf)
                model={
                    "fcnet_hiddens": [64, 64],
                    "fcnet_activation": "tanh",
                }
            )
            .env_runners( 
                num_env_runners=0, # DEBUG: Roda no driver principal
                rollout_fragment_length=train_batch_size, 
                batch_mode="truncate_episodes",
            )
            .multi_agent(
                policies={
                    "shared_policy": (
                        None, 
                        obs_space_single, 
                        act_space_single, 
                        {},
                    ),
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
                policies_to_train=["shared_policy"],
            )
        )
        
        config_dict = ppo_config.to_dict()
        
        config_dict['sgd_minibatch_size'] = sgd_minibatch_size 
        config_dict['train_batch_size'] = train_batch_size 

        # 7. Execução do Treinamento
        try:
            # Gerar string de parâmetros para o diretório
            params_for_dir = params_original.copy()
            params_for_dir['train_batch_size'] = train_batch_size
            params_for_dir['sgd_minibatch_size'] = sgd_minibatch_size
            params_for_dir['update_epochs'] = num_epochs
            params_str = generate_params_string(params_for_dir)
            
            # Adicionar prefixo PPO_Marl_ ao nome do diretório
            params_dir_name = f"PPO_Marl_{params_str}"
            
            # Usar storage_path ao invés de local_dir (deprecado)
            # Usar o nome do diretório de parâmetros como name do experimento
            # Isso faz com que o Ray Tune crie: RAY_RESULTS_BASE_DIR/params_dir_name/PPO_overcooked_marl_parallel_.../
            # Sem nível intermediário adicional
            analysis = tune.run(
                "PPO",
                name=params_dir_name,  # Usar nome do diretório de parâmetros como name do experimento
                stop={"timesteps_total": timesteps_total},
                config=config_dict,
                checkpoint_freq=10,
                checkpoint_at_end=True,
                verbose=1,
                storage_path=RAY_RESULTS_BASE_DIR,  # Usar diretório base (timestamp)
            )
            
            # O diretório do experimento será criado pelo Ray em: RAY_RESULTS_BASE_DIR/params_dir_name/
            params_dir = os.path.join(RAY_RESULTS_BASE_DIR, params_dir_name)
            
            trial = analysis.trials[0]
            
            # A métrica está em env_runners (dicionário aninhado em trial.last_result)
            avg_reward_last = 0.0
            
            if hasattr(trial, 'last_result') and trial.last_result:
                # Acessar env_runners que é um dicionário aninhado
                if 'env_runners' in trial.last_result and isinstance(trial.last_result['env_runners'], dict):
                    env_runners = trial.last_result['env_runners']
                    if 'episode_return_mean' in env_runners:
                        avg_reward_last = env_runners['episode_return_mean']
                    elif 'episode_reward_mean' in env_runners:
                        avg_reward_last = env_runners['episode_reward_mean']
                # Fallback: tentar acesso direto (caso a estrutura seja diferente)
                elif 'episode_reward_mean' in trial.last_result:
                    avg_reward_last = trial.last_result['episode_reward_mean']
            
            # Extrair learning curves do progress.csv ou result.json
            learning_curves_data = []
            
            if hasattr(trial, 'logdir') and trial.logdir:
                trial_dir = Path(trial.logdir)
                
                # Tentar ler progress.csv
                progress_csv = trial_dir / "progress.csv"
                if progress_csv.exists():
                    try:
                        import pandas as pd
                        df = pd.read_csv(progress_csv)
                        if 'env_runners/episode_reward_mean' in df.columns:
                            learning_curves_data = [
                                {
                                    'iteration': int(row['training_iteration']), 
                                    'reward_mean': float(row['env_runners/episode_reward_mean'])
                                }
                                for _, row in df.iterrows()
                                if pd.notna(row.get('env_runners/episode_reward_mean'))
                            ]
                    except Exception as e:
                        print(f"DEBUG: Erro ao ler progress.csv: {e}")
                
                # Fallback: tentar ler result.json
                if not learning_curves_data:
                    result_json = trial_dir / "result.json"
                    if result_json.exists():
                        try:
                            with open(result_json, 'r') as f:
                                for line in f:
                                    if line.strip():
                                        data = json.loads(line)
                                        if 'env_runners' in data and isinstance(data['env_runners'], dict):
                                            if 'episode_reward_mean' in data['env_runners']:
                                                learning_curves_data.append({
                                                    'iteration': int(data.get('training_iteration', 0)),
                                                    'reward_mean': float(data['env_runners']['episode_reward_mean'])
                                                })
                                            elif 'episode_return_mean' in data['env_runners']:
                                                learning_curves_data.append({
                                                    'iteration': int(data.get('training_iteration', 0)),
                                                    'reward_mean': float(data['env_runners']['episode_return_mean'])
                                                })
                        except Exception as e:
                            print(f"DEBUG: Erro ao ler result.json: {e}")
            
            # Salvar checkpoint do modelo
            checkpoint_path = None
            
            # Tentar acessar o checkpoint de diferentes formas
            if hasattr(trial, 'checkpoint') and trial.checkpoint:
                try:
                    # Forma 1: Se for um objeto Checkpoint do Ray Air
                    if hasattr(trial.checkpoint, 'to_directory'):
                        checkpoint_path = trial.checkpoint.to_directory()
                    # Forma 2: Se tiver atributo path
                    elif hasattr(trial.checkpoint, 'path'):
                        checkpoint_path = trial.checkpoint.path
                    # Forma 3: Se tiver atributo value (API antiga)
                    elif hasattr(trial.checkpoint, 'value'):
                        checkpoint_path = trial.checkpoint.value
                    # Forma 4: Se for string direto
                    else:
                        checkpoint_path = str(trial.checkpoint)
                except Exception as e:
                    print(f"DEBUG: Erro ao acessar checkpoint: {e}")
                    checkpoint_path = None
            
            # Se ainda não encontrou, buscar no diretório do trial
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                # Buscar no diretório do trial
                trial_logdir = getattr(trial, 'logdir', None)
                if not trial_logdir:
                    # Tentar obter do analysis
                    trial_logdir = getattr(analysis, 'trial_dataframes', None)
                    if trial_logdir and len(analysis.trials) > 0:
                        trial_logdir = analysis.trials[0].logdir
                
                if trial_logdir:
                    trial_dir = str(trial_logdir)
                    # Procurar por checkpoint_* no diretório do trial
                    if os.path.exists(trial_dir):
                        try:
                            checkpoints = [d for d in os.listdir(trial_dir) 
                                         if d.startswith('checkpoint_') and os.path.isdir(os.path.join(trial_dir, d))]
                            if checkpoints:
                                # Pegar o checkpoint mais recente (maior número)
                                checkpoints.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0, reverse=True)
                                checkpoint_path = os.path.join(trial_dir, checkpoints[0])
                                print(f"DEBUG: Checkpoint encontrado no diretório: {checkpoint_path}")
                        except Exception as e:
                            print(f"DEBUG: Erro ao buscar checkpoint no diretório: {e}")
                
                # Se ainda não encontrou, buscar na pasta local ray_results com nova estrutura
                if not checkpoint_path or not os.path.exists(checkpoint_path):
                    if os.path.exists(params_dir):
                        # Procurar diretamente no diretório de parâmetros (sem nível intermediário)
                        for root, dirs, files in os.walk(params_dir):
                            for d in dirs:
                                if d.startswith('checkpoint_'):
                                    potential_path = os.path.join(root, d)
                                    if os.path.exists(potential_path):
                                        checkpoint_path = potential_path
                                        print(f"DEBUG: Checkpoint encontrado em ray_results local: {checkpoint_path}")
                                        break
                            if checkpoint_path and os.path.exists(checkpoint_path):
                                break
            
            # Salvar informações do modelo em JSON no diretório do experimento
            if checkpoint_path and os.path.exists(checkpoint_path):
                config_name = f"PPO_Config_{i+1:02d}"
                
                # Salvar informações do modelo em JSON no diretório do experimento
                model_info = {
                    'config_name': config_name,
                    'checkpoint_path': checkpoint_path,
                    'params': convert_to_serializable(params_original.copy()),
                    'avg_reward': float(avg_reward_last),
                    'learning_curves': learning_curves_data,
                    'training_config': {
                        'learning_rate': float(config['learning_rate']),
                        'gamma': float(config['gamma']),
                        'train_batch_size': int(train_batch_size),
                        'sgd_minibatch_size': int(sgd_minibatch_size),
                        'num_epochs': int(num_epochs),
                        'entropy_coeff': float(entropy_coeff),
                        'num_episodes': int(num_episodes),
                        'timesteps_total': int(timesteps_total),
                    }
                }
                
                # Salvar model_info.json no diretório do experimento (params_dir)
                # Garantir que o diretório existe (foi criado pelo Ray Tune)
                os.makedirs(params_dir, exist_ok=True)
                model_info_path = os.path.join(params_dir, "model_info.json")
                with open(model_info_path, 'w') as f:
                    json.dump(model_info, f, indent=2)
                
                print(f"Config {i+1}/{len(combinations)}: reward={avg_reward_last:.2f} | Checkpoint em: {checkpoint_path}")
            else:
                print(f"Config {i+1}/{len(combinations)}: reward={avg_reward_last:.2f} | Checkpoint não encontrado")

        except Exception as e:
            print(f"Erro na config {i+1}: {e}")
            continue
    
    # Final summary
    training_time_total = time.time() - start_time_total
    
    print("\n" + "="*70)
    print("TREINAMENTO CONCLUÍDO (IDPPO com CNN)")
    print(f"Total time: {training_time_total/60:.1f} min")
    print(f"Resultados salvos em: {RAY_RESULTS_BASE_DIR}")
    print("="*70)
    
    # 8. Shutdown Ray
    ray.shutdown()