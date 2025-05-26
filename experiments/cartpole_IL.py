import collections

import gymnasium as gym
import numpy as np
from stable_baselines3.common import vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.util.util import make_vec_env
from imitation.algorithms import bc
from imitation.policies import interactive
from imitation.data import rollout

class CartPoleInteractivePolicy(interactive.DiscreteInteractivePolicy):
    """Interactive policy for CartPole with discrete actions."""
    
    def __init__(self, env: vec_env.VecEnv):
        """Builds CartPoleInteractivePolicy.

        Args:
            env: Environment to interact with.
        """
        action_keys_names = collections.OrderedDict(
            [
                ("a", "left"),   # action 0: push cart left
                ("d", "right"),  # action 1: push cart right
            ]
        )
        super().__init__(
            observation_space=env.observation_space,
            action_space=env.action_space,
            action_keys_names=action_keys_names,
        )
        self.env = env

    def _render(self, obs):
        """Renders the environment with the current observation."""
        return

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    env = make_vec_env(
        "CartPole-v1",
        n_envs=1,
        max_episode_steps=20,
        rng=rng,
        env_make_kwargs={
            "render_mode": "human",  # Render mode for interactive policy
        },
    )
    
    expert = CartPoleInteractivePolicy(env)
    
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_episodes=5),
        unwrap=False,
        rng=rng,
    )
    
    transitions = rollout.flatten_trajectories(rollouts)
    
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )
    
    bc_trainer.train(n_epochs=1)
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward after training: {reward_after_training}")
    



