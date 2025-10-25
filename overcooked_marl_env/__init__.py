from gymnasium.envs.registration import register
from .env import OvercookedMARL

# Registration for Gymnasium compatibility
# Note: For PettingZoo environments use direct instantiation
register(
    id="OvercookedMARL-v0",
    entry_point="overcooked_marl_env.env:OvercookedMARL",
    max_episode_steps=500,
)