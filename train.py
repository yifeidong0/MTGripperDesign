import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.utils import get_linear_fn
import envs
import numpy as np
import math
import time
import pygame

def handle_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

def main():
    env_id = 'VPushSimulationEnv-v0'

    obs_type = 'pose'
    env = gym.make(env_id)
    check_env(env)
    # env = make_vec_env(env_id, n_envs=4)    
    
    total_timesteps = int(5e6)
    model = PPO("CnnPolicy" if obs_type == 'image' else "MlpPolicy", 
                env, 
                verbose=1, 
                tensorboard_log="./ppo_box2d_tensorboard/")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("ppo_box2d")

    import imageio  # Import the imageio library to save video    # Wrap the environment with Monitor to save video
    # Set up video writer
    video_filename = 'test_video.mp4'
    video_writer = imageio.get_writer(video_filename, fps=30)

    obs, _ = env.reset(seed=0)
    for episode in range(1):
        time.sleep(2)
        print(f"Episode {episode + 1} begins")
        done, truncated = False, False
        while not (done or truncated):
            action, _states = model.predict(np.array(obs))
            obs, reward, done, truncated, _ = env.step(action)
            env.render()
            handle_pygame_events()  # Ensure events are handled during each step

            # Capture frame
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))  # Convert to the shape (height, width, 3)
            video_writer.append_data(frame)

            time.sleep(.002)
        print("Done!" if done else "Truncated")
        print(f"Episode {episode + 1} finished")
    
    env.close()
    video_writer.close()
    print(f"Video saved as {video_filename}")
    

if __name__ == "__main__":
    main()