import argparse
import datetime

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description="RL co-design project")
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility. Default is 42.')
    parser.add_argument('--time_stamp', type=str, default=get_timestamp(), help='Current time of the script execution')
    parser.add_argument('--env_id', type=str, choices=[
                                'VPushSimulationEnv-v0',
                                'VPushPbSimulationEnv-v0',
                                'UCatchSimulationEnv-v0',
                                'ScoopSimulationEnv-v0',
                                'DLRSimulationEnv-v0',
                                'PandaUPushEnv-v0'  # Assuming you want to keep this as a default or possible choice
                            ], default='PandaUPushEnv-v0', help='Environment ID for the simulation')
    parser.add_argument('--total_timesteps', type=int, default=int(3e6), help='Total number of timesteps for training')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='auto', help='Computational device to use (auto, cuda, cpu)')
    parser.add_argument('--obs_type', type=str, choices=['pose', 'image'], default='pose', help='Type of observations for the training')
    parser.add_argument('--using_robustness_reward', type=str2bool, nargs='?', const=True, default=False, help='Enable or disable the robustness reward')
    parser.add_argument('--checkpoint_freq', type=int, default=int(1e3), help='Frequency of saving checkpoints')
    parser.add_argument('--n_envs', type=int, default=1, help='Number of environments to run in parallel')
    parser.add_argument('--render_mode', type=str, choices=['rgb_array', 'human'], default='human', help='Rendering mode for the simulation')
    parser.add_argument('--algo', type=str, choices=['ppo', 'tqc', 'sac'], default='ppo', help='RL algorithm to use for training')
    return parser.parse_args()
