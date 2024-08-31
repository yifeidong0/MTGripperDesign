import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import HParam
import wandb
from wandb.integration.sb3 import WandbCallback

import envs
from experiments.args_utils import get_args
from policy.her_replay_buffer_mod import HerReplayBufferMod

class CustomCallback(CheckpointCallback):
    def __init__(self, args,  save_freq: int, save_path: str, name_prefix: str = "rl_model", 
                 save_replay_buffer: bool = False, save_vecnormalize: bool = False, verbose: int = 0):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)
        self.args = args
    
    def _on_training_start(self) -> None:
        hparam_dict = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in vars(self.args).items()}
        
        if hasattr(self.training_env.envs[0], 'reward_weights'):
            reward_weights = self.training_env.envs[0].reward_weights
            hparam_dict.update({'reward_weights': str(reward_weights)})  # Convert to string to avoid type issues
        
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
    
    env_ids = {'vpush':'VPushPbSimulationEnv-v0', 
              'catch':'UCatchSimulationEnv-v0',
              'dlr':'DLRSimulationEnv-v0',
              'panda':'PandaUPushEnv-v0'}
    env_id = env_ids[args.env_id]

    paths = {
        "tensorboard_log": f"results/runs/{env_id}/",
        "log_path": f"results/logs/{env_id}/",
        "model_save_path": f"results/models/{env_id}/",
        "monitor_dir": f"results/monitor/{env_id}/",
    }
    
    os.system(f"mkdir asset/{args.time_stamp}")

    env_kwargs = {'obs_type': args.obs_type,
                  'using_robustness_reward': args.using_robustness_reward,
                  'perturb': args.perturb,
                  'perturb_sigma': args.perturb_sigma,
                  'render_mode': args.render_mode, 
                  'time_stamp': args.time_stamp,
                  'reward_type': 'dense', # dense, sparse
                  'reward_weights': args.reward_weights,
                  }
    
    if args.wandb_mode != 'disabled':
        wandb.init(
            project="MTGripperDesign",
            name=f"{env_id}_{args.time_stamp}",
            config=vars(args),
            sync_tensorboard=True,
            save_code=True,
            mode=args.wandb_mode,
        )
    
    if args.n_envs > 1:
        env = make_vec_env(env_id, n_envs=args.n_envs, seed=args.random_seed, env_kwargs=env_kwargs)
    else:
        env = gym.make(env_id, **env_kwargs)
        if env_id != 'PandaUPushEnv-v0':
            check_env(env)
        
    custom_callback = CustomCallback(
        args=args,
        save_freq=args.checkpoint_freq,
        save_path=paths["model_save_path"],
        name_prefix=f"{env_id}_{args.time_stamp}",
        # verbose=2,
    )

    if env_id == 'PandaPushEnv-v0':
        policy_name = "MultiInputPolicy"
    else:
        if args.obs_type == 'pose':
            policy_name = "MultiInputPolicy" # MlpPolicy
        elif args.obs_type == 'image':
            policy_name = "CnnPolicy"
    if args.algo == 'ppo':
        model = PPO(policy_name, 
                    env, 
                    verbose=1, 
                    tensorboard_log=paths["tensorboard_log"],
                    device=args.device,
                    seed=args.random_seed,
                    )            
    elif args.algo == 'tqc':
        model = TQC('MultiInputPolicy', 
                    env,
                    buffer_size=1000000,
                    batch_size=2048,
                    gamma=0.95,
                    learning_rate=1e-3,
                    learning_starts=10000,
                    tau=0.05,
                    replay_buffer_class=HerReplayBufferMod,
                    replay_buffer_kwargs={
                        "goal_selection_strategy": 'future',
                        "n_sampled_goal": 4
                    },
                    policy_kwargs={
                        "net_arch": [512, 512, 512],
                        "n_critics": 2
                    },
                    verbose=1,
                    tensorboard_log=paths["tensorboard_log"],
                    device="auto",
                    seed=args.random_seed,
                    )
    elif args.algo == 'sac':
        model = SAC('MultiInputPolicy', 
                    env,
                    buffer_size=1000000,
                    batch_size=2048,
                    gamma=0.95,
                    learning_rate=1e-3,
                    learning_starts=10000,
                    tau=0.05,
                    replay_buffer_class=HerReplayBufferMod,
                    replay_buffer_kwargs={
                        "goal_selection_strategy": 'future',
                        "n_sampled_goal": 4
                    },
                    policy_kwargs={
                        "net_arch": [512, 512, 512],
                        "n_critics": 2
                    },
                    verbose=0,
                    tensorboard_log=paths["tensorboard_log"],
                    device="auto",
                    seed=args.random_seed,
                    )  
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            progress_bar=1,
            log_interval=5,
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_freq=args.checkpoint_freq,
                model_save_path=paths["model_save_path"],
            ) if args.wandb_mode != 'disabled' else custom_callback,
        )
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        model.save(f"results/models/{env_id}_{args.total_timesteps}_{args.time_stamp}_final_{args.using_robustness_reward}_{args.perturb}") # last model
        os.system(f"rm -rf asset/{args.time_stamp}")

if __name__ == "__main__":
    main()