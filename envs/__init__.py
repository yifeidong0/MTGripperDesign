from .vpush_env import VPushSimulationEnv
from gymnasium.envs.registration import register

register(
    id='VPushSimulationEnv-v0',
    entry_point='envs.vpush_env:VPushSimulationEnv',
)