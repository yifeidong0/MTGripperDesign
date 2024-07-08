from .vpush_env import VPushSimulationEnv
from .vpush_pb_env import VPushPbSimulationEnv
from gymnasium.envs.registration import register

register(
    id='VPushSimulationEnv-v0',
    entry_point='envs.vpush_env:VPushSimulationEnv',
)

register(
    id='VPushPbSimulationEnv-v0',
    entry_point='envs.vpush_pb_env:VPushPbSimulationEnv',
)
