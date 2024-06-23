# import gymnasium as gym

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

# # Parallel environments
# vec_env = make_vec_env("CartPole-v1", n_envs=4)

# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")


import gymnasium as gym
env = gym.make("CarRacing-v2", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()