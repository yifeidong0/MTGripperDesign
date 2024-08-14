import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import HParam

import envs
from experiments.args_utils import get_args

class CustomCallback(CheckpointCallback):
    def __init__(self, args,  save_freq: int, save_path: str, name_prefix: str = "rl_model", save_replay_buffer: bool = False, save_vecnormalize: bool = False, verbose: int = 0):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)
        self.args = args
    
    def _on_training_start(self) -> None:
        hparam_dict = vars(self.args)
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )
    
    def _on_step(self) -> bool:
        super()._on_step()
        info = self.locals['infos'][0]
        if 'cost' in info:
            self.logger.record('cost', info['cost'])
        return True


def main():
    args = get_args()

    paths = {
        "tensorboard_log": f"results/runs/{args.env_id}/",
        "log_path": f"results/logs/{args.env_id}/",
        "model_save_path": f"results/models/{args.env_id}/",
        "monitor_dir": f"results/monitor/{args.env_id}/",
    }       
    
    env_kwargs = {'obs_type': args.obs_type, 'using_robustness_reward': args.using_robustness_reward, 'render_mode': args.rander_mode}
    if args.n_envs > 1:
        env = make_vec_env(args.env_id, n_envs=args.n_envs, seed=args.random_seed, env_kwargs=env_kwargs)
    else:
        env = gym.make(args.env_id, **env_kwargs)
        # check_env(env)
        
    custom_callback = CustomCallback(
        args=args,
        save_freq=args.checkpoint_freq,
        save_path=paths["model_save_path"],
        name_prefix=f"{args.env_id}_{args.time_stamp}",
        # verbose=2,
    )

    total_timesteps = int(1e6)
    if args.env_id == 'PandaPushEnv-v0':
        policy_name = "MultiInputPolicy"
    else:
        if args.obs_type == 'pose':
            policy_name = "MlpPolicy"
        elif args.obs_type == 'image':
            policy_name = "CnnPolicy"
    model = PPO(policy_name, 
                env, 
                verbose=1, 
                tensorboard_log=paths["tensorboard_log"],
                device=args.device,
                seed=args.random_seed,
                )

    model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=5, callback=custom_callback)

    model.save(f"results/models/{args.env_id}_{total_timesteps}_{args.time_stamp}_final") # last model

if __name__ == "__main__":
    main()