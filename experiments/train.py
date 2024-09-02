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


from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from collections import deque

class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path, verbose=0, n_episodes=100):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.best_success_rate = -np.inf
        self.ep_success_buffer = deque(maxlen=n_episodes)  # Only keep the last `n_episodes`
        self.best_model_path = None  # Track the best model path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if an episode has ended
        if self.locals['dones'][0]:
            # Fetch the 'is_success' values from the current environment step
            success = self.locals['infos'][0].get('is_success', 0)
            self.ep_success_buffer.append(success)

            # Calculate the average success rate over the last `n_episodes` episodes
            avg_success_rate = np.mean(self.ep_success_buffer)
            
            if avg_success_rate > self.best_success_rate and len(self.ep_success_buffer) > 10:
                self.best_success_rate = avg_success_rate
                # Create a new model file name with the success rate included
                model_save_path = os.path.join(self.save_path, f"best_model_{self.num_timesteps}_steps_{avg_success_rate:.4f}.zip")
                
                # Save the new best model
                self.model.save(model_save_path)
                wandb.save(model_save_path)  # Upload the best model to WandB
                
                if self.verbose > 0:
                    print(f"New best model saved with avg success rate: {avg_success_rate:.4f} at step {self.num_timesteps}")

                # Remove the previous best model if it exists
                if self.best_model_path is not None and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                
                # Update the path to the new best model
                self.best_model_path = model_save_path

        return True


def main():
    args = get_args()
    
    run_id = wandb.util.generate_id()
    
    env_ids = {'vpush':'VPushPbSimulationEnv-v0', 
              'catch':'UCatchSimulationEnv-v0',
              'dlr':'DLRSimulationEnv-v0',
              'panda':'PandaUPushEnv-v0'}
    env_id = env_ids[args.env_id]

    paths = {
        "tensorboard_log": f"results/runs/{env_id}/",
        "log_path": f"results/logs/{env_id}/",
        "model_save_path": f"results/models/{env_id}/{args.time_stamp}_{run_id}_{args.random_seed}_{args.using_robustness_reward}_{args.perturb}/",
        "monitor_dir": f"results/monitor/{env_id}/",
    }
    
    os.system(f"mkdir asset/{run_id}")

    env_kwargs = {'obs_type': args.obs_type,
                  'using_robustness_reward': args.using_robustness_reward,
                  'perturb': args.perturb,
                  'perturb_sigma': args.perturb_sigma,
                  'render_mode': args.render_mode, 
                  'run_id': run_id,
                  'reward_type': 'dense', # dense, sparse
                  'reward_weights': args.reward_weights,
                  }
    
    if args.wandb_mode != 'disabled':
        os.environ["WANDB_RUN_GROUP"] = args.wandb_group_name
        custom_run_name = f"{args.env_id}_seed{args.random_seed}_robust{args.using_robustness_reward}_perturb{args.perturb}_{run_id}"
        wandb.init(
            project="MTGripperDesign",
            group=args.wandb_group_name,
            name=custom_run_name,  # Set the custom run name here
            config=vars(args),
            sync_tensorboard=True,
            save_code=True,
            mode=args.wandb_mode,
        )
    
    env = gym.make(env_id, **env_kwargs)
    if args.wandb_mode == 'disabled':
        try:
            if env_id != 'PandaUPushEnv-v0':
                check_env(env)
        finally:
            os.system(f"rm -rf asset/{run_id}")
        
    custom_callback = CustomCallback(
        args=args,
        save_freq=args.checkpoint_freq,
        save_path=paths["model_save_path"],
        name_prefix=run_id,
        verbose=2,
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
    elif args.algo == 'sac':
        assert args.env_id == 'panda', "Only PandaUPushEnv-v0 is supported for SAC and HER"
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
                    verbose=1,
                    tensorboard_log=paths["tensorboard_log"],
                    device=args.device,
                    seed=args.random_seed,
                    )  

    best_model_callback = SaveBestModelCallback(
        save_path=paths["model_save_path"],
        verbose=1,
        n_episodes=100,  # Track the last 100 episodes
    )
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            progress_bar=1,
            log_interval=5,
            callback=[best_model_callback] if args.wandb_mode != 'disabled' else custom_callback,
        )
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        # model.save(f"results/models/{env_id}_{args.total_timesteps}_{run_id}_final_{args.using_robustness_reward}_{args.perturb}") # last model
        model.save(os.path.join(paths["model_save_path"], f"final_model_{args.total_timesteps}_steps.zip"))
        os.system(f"rm -rf asset/{run_id}")

if __name__ == "__main__":
    main()
