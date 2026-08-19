# Notes

sudo apt update
sudo apt upgrade -y

sudo apt install python3 python3-pip python3-venv python3-full -y

python3 --version
pip3 --version

python3 -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip

pip install gymnasium numpy pettingzoo matplotlib numpy torch-directml tqdm

pip install -e .

python example/tutorial_map_test.py  > test_log.txt

deactivate

# Decisions
Summary: gymnasium environment for MARL based on overcooked

### MARL Environment Design
##### Define Agents
Variable Agents - In the implementation of step(), you'll need to manage a dictionary of active agents and return data only for them. PettingZoo is the ideal library for this, as it's built to handle the sequential or parallel nature of games with variable agents/turns.

##### Define Observation Spaces
Observation Spaces (Global Vision) - You'll use a gymnasium.spaces.Box or gymnasium.spaces.Dict for the observation. I recommend a 3D Box (width x height x state layers) where each layer represents the state (e.g., 1st layer: agent location, 2nd layer: tomato location, 3rd layer: cutting station, etc.).

##### Define Action Spaces
Action Spaces (Walk + Interact) - You'll use a gymnasium.spaces.Discrete where the first 4 values are directions and the remaining values are interactions (e.g., 0-3: Walk, 4: Interact).

Collision and Restrictions - The challenge here is Action Masking. In each step(), the agent will try an action, but you'll need to check if it's valid (e.g., no wall, station or agent in the way). If the action is invalid, you can:
1) Not move the agent (stay still), or
2) Apply a small reward penalty.

##### Reward Design
Hybrid Reward - This is excellent. In the step() method, you'll calculate:

Individual Reward (local): If the action was a successful interaction (e.g., picked up an item, placed an item at the station).

Collective Reward (global): If the action resulted in the delivery of a dish. This reward should be added to the individual value and distributed to all agents in the rewards: dict.

##### Termination Design
Time Limit - This translates to the truncated signal. It's Gymnasium's way of saying the episode ended for a reason not directly related to the task's failure or success (e.g., time out).

Delivered Dishes - This translates to the terminated signal. It's the natural conclusion of the task (success or critical failure).

### Environment Implementation


# AI TIPS: MARL Environment Implementation Order

Phase 1: Skeleton and Spaces (The Gymnasium/PettingZoo Structure)

This phase is about defining the "shape" of your environment. If you get this wrong, nothing else will work.

    Create the Base Class: Define the OvercookedMARL class inheriting from ParallelEnv (from PettingZoo).

    Define Static Variables: In __init__, define the list of agent IDs (self.agents) and the map size (self.map_size).

    Define Observation & Action Spaces: Implement self.observation_spaces and self.action_spaces as dictionaries of gymnasium.spaces.

        Observation: Define the exact dimensions (width x height x state layers) of your 3D Box.

        Action: Define the total number of discrete actions (4 movements + 1 interact) in your Discrete(5).

    Implement _initialize_game_state and _get_obs: Create these helper functions. At first, they can return dummy data (np.zeros or spaces.sample()) just to ensure the input/output data format is correct.

    Implement reset(): Implement reset() to initialize the game state and return the first observations and information using your helper methods.

Phase 2: Core Logic (The Game Engine)

This is the most complex phase, where the simulation takes place.

    Implement step() (Input Logic): Set up the step(actions) method to receive the action dictionary and initialize the 5 return dictionaries (obs, rewards, term, trunc, info).

    Movement and Collision Logic: Within the step() loop, implement the agent movement logic, ensuring collision rules (with walls, stations, and other agents) are respected.

    Interaction Logic: Implement the main Overcooked logic: what happens when an agent uses the "Interact" action in different states (pick up item, cut item, combine item).

    Hybrid Reward Logic:

        Calculate small individual rewards (for progression).

        Calculate the large global reward (for dish delivery) and distribute to all agents.

    Termination Logic: Implement the conditions for terminated (dishes delivered) and truncated (time limit), ensuring that the return dictionary __all__ is marked as True when the episode ends.

Phase 3: Finalization and Testing

    Implement render() (Optional but Recommended): Add logic to render the map (simple ASCII if using render_mode="human", or a pixel matrix for render_mode="rgb_array").

    Run check_env: Use the check_env function from Gymnasium (and parallel_api_test from PettingZoo, if desired) to formally validate that you followed the API protocol.

    Run the Sanity Test: Run the enviroment_tests/test.py script to ensure the environment runs 100 steps without crashing, confirming that the game logic is stable.

    Finalize the Setup: Make sure your setup.py and registration in __init__.py are correct so others can use gym.make().

The Golden Rule: Don't move to the next phase until the code for the current phase works perfectly with your test code.

---

# Sistema Refatorado (Outubro 2025)

## Visão Geral da Refatoração

O sistema de itens e receitas foi completamente refatorado, migrando de um sistema baseado em enums e códigos numéricos para um sistema flexível baseado em dicionários e classes Python.

## Principais Mudanças

### 1. Sistema de Itens (items.py)
- **Antes**: ItemType enum com valores fixos
- **Depois**: Dicionários com metadados e estados nomeados
  - Estados: `RAW` → `CHOPPING_1` → `CHOPPING_2` → `CHOPPING_3` → `READY`
  - Processamento progressivo: 3 interações para processar
  - Progresso armazenado no item (permite corte cooperativo)

### 2. Sistema de Receitas (recipes.py)
- **Antes**: Códigos bitwise e validação numérica
- **Depois**: Classes Python e RecipeManager
  - `Recipe` base class com método `matches()`
  - `RecipeManager.validate_delivery()` para validação
  - Fácil adicionar novas receitas

### 3. Sistema de Instâncias (env.py)
- Cada item tem ID único e dicionário completo
- Progresso de corte mantido ao mover items
- Suporte a corte cooperativo entre agentes

### 4. Interações Atualizadas
- **Processamento**: 3 cortes necessários (0.1 pts cada, 1.0 ao completar)
- **Prato**: Pode pegar com mãos vazias ou com item READY (combina automaticamente)
- **Bancadas**: Combinação flexível de pratos e ingredientes READY
- **Entrega**: Validação baseada em lista de ingredientes (receitas inválidas = 0 pontos)

### 5. Visualização Melhorada
- **ASCII**: Letras indicam estado (minúscula=RAW, maiúscula=READY, números=processando)
- **Pygame**: Cores gradientes por estado, progresso visual de cortes, ingredientes em pratos

## Arquivos Modificados
- `overcooked_marl_env/core/items.py` - Sistema novo de itens
- `overcooked_marl_env/core/recipes.py` - Classes de receitas
- `overcooked_marl_env/env.py` - Lógica de interação atualizada
- `maps/default_map.txt` - Novo formato de dispensadores (nomes em vez de IDs)

## Arquivos Criados
- `enviroment_tests/test_new_system.py` - Testes do sistema refatorado
- `REFACTORING_SUMMARY.md` - Documentação completa da refatoração

## Como Usar

### Executar Testes
```bash
venv\Scripts\activate
python enviroment_tests/test_new_system.py
```

### Exemplo de Código
```python
from overcooked_marl_env.env import OvercookedMARL

env = OvercookedMARL(render_mode="human", verbose=True)
obs, info = env.reset()

# Agente 0 pega tomate e corta 3 vezes
actions = {
    "agent_0": 3  # RIGHT
}
obs, rewards, dones, truncs, info = env.step(actions)

# Info agora contém dados completos do item
print(info["agent_0"]["carrying_item_data"])
# {'name': 'tomato', 'state': 'RAW', 'cuts_done': 0, ...}
```

## Compatibilidade

O sistema mantém backwards compatibility:
- IDs numéricos de dispensadores ainda funcionam
- Código antigo deve funcionar sem alterações
- Novos recursos são opcionais

Ver `REFACTORING_SUMMARY.md` para documentação completa.

