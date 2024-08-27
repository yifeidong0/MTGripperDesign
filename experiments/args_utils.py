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
    parser.add_argument('--env_id', type=str, choices=['vpush', 'catch', 'dlr', 'panda',], default='panda', help='Environment ID for the simulation')
    parser.add_argument('--total_timesteps', type=int, default=int(3e6), help='Total number of timesteps for training')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='auto', help='Computational device to use (auto, cuda, cpu)')
    parser.add_argument('--obs_type', type=str, choices=['pose', 'image'], default='pose', help='Type of observations for the training')
    parser.add_argument('--using_robustness_reward', type=str2bool, nargs='?', const=True, default=True, help='Enable or disable the robustness reward')
    parser.add_argument('--perturb', type=str2bool, nargs='?', const=False, default=False, help='Add random perturbations to the target object')
    parser.add_argument('--checkpoint_freq', type=int, default=int(1e3), help='Frequency of saving checkpoints')
    parser.add_argument('--n_envs', type=int, default=1, help='Number of environments to run in parallel')
    parser.add_argument('--render_mode', type=str, choices=['rgb_array', 'human'], default='human', help='Rendering mode for the simulation')
    parser.add_argument('--algo', type=str, choices=['ppo', 'tqc', 'sac'], default='ppo', help='RL algorithm to use for training')
    parser.add_argument('--reward_weights', type=float, nargs='+', default=[0.1, 0.001, -0.03, 0.1, 10.0, 50.0, 5e-3, 100.0], help='List of reward weights to use during training')
    parser.add_argument('--wandb_mode', type=str, choices=['online', 'offline', 'disabled'], default='disabled', help='Wandb mode for logging')
    return parser.parse_args()

# Aug. 26: 
# catch PPO 19-22: get 1/250 if in_polygon
# catch PPO 23-26: get 100 if success
# catch PPO 27-30: get 100 if success
# catch PPO 31-34: get 100 if success
# target: 
# for 4 scenarios:
    # 5 training curves with robustness reward - policy a
    # 5 training curves without robustness reward - policy b
    # 5 BO runs+test with disturbance using policy a
    # 5 BO runs+test with disturbance using policy b
    # 5 BO runs+test without disturbance using policy a
    # 5 BO runs+test without disturbance using policy b

# Aug. 27:
# catch PPO 36-39: get 100 if success, 3e6 timesteps, cutoff at around 2.3M steps
# BO running with disturbance using policy a
# BO running with disturbance using policy b