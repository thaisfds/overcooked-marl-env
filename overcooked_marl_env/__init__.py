from gymnasium.envs.registration import register
from .env import OvercookedMARL

register(
    id="OvercookedMARL-v0",
    entry_point="overcooked_marl_env.env:OvercookedMARL",
    max_episode_steps=500,
)