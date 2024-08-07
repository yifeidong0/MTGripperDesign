from .vpush_env import VPushSimulationEnv
from .vpush_pb_env import VPushPbSimulationEnv
from .ucatch_env import UCatchSimulationEnv
from .panda.panda_push_env import PandaPushEnv
from gymnasium.envs.registration import register

register(
    id='VPushSimulationEnv-v0',
    entry_point='envs.vpush_env:VPushSimulationEnv',
)

register(
    id='VPushPbSimulationEnv-v0',
    entry_point='envs.vpush_pb_env:VPushPbSimulationEnv',
)

register(
    id='UCatchSimulationEnv-v0',
    entry_point='envs.ucatch_env:UCatchSimulationEnv',
)

register(
    id='ScoopSimulationEnv-v0',
    entry_point='envs.scoop_env:ScoopSimulationEnv',
)

register(
    id='PandaPushEnv-v0',
    entry_point='envs.panda.panda_push_env:PandaPushEnv',
)