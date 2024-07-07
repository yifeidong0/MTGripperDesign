import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
import pygame
import random
import math
from sim.vpush_sim import VPushSimulation  # Assuming VPushSimulation is in a separate file.
import time
from stable_baselines3.common.utils import get_linear_fn

def handle_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

class VPushSimulationEnv(gym.Env):
    def __init__(self, object_type='circle', v_angle=np.pi/6, gui=True, img_size=(84, 84), obs_type='image'):
        super(VPushSimulationEnv, self).__init__()
        self.simulation = VPushSimulation(object_type, v_angle)
        self.gui = gui
        self.img_size = img_size  # New parameter for image size
        self.obs_type = obs_type  # New parameter for observation type
        # self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.action_res = 10
        # self.action_space = spaces.MultiDiscrete([self.action_res, self.action_res, ])
        # discrete action space
        self.action_space = spaces.Discrete(6)
        
        if self.obs_type == 'image':
            # Observation space: smaller RGB image of the simulation
            self.observation_space = spaces.Box(low=0, high=1.0, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.float64)
        else:
            # Observation space: low-dimensional pose (object pose, gripper pose)
            self.observation_space = spaces.Box(low=np.array([-1.0]*6), high=np.array([1.0]*6), dtype=np.float64)
        
        # Goal parameters
        self.goal_radius = 10.0
        self.goal_position = np.array([30, 30])
        self.simulation.goal_position = self.goal_position
        self.simulation.goal_radius = self.goal_radius

        # Pygame setup
        if self.gui:
            pygame.init()
            self.screen = pygame.display.set_mode((self.simulation.width, self.simulation.height))
            pygame.display.set_caption('Box2D with Pygame - Robot Pushing Object')

    def reset(self, seed=None, options=None):
        print("Resetting the environment")
        super().reset(seed=seed)
        self.simulation.step_count = 0

        # Set goal position randomly within specified range
        # self.goal_position = np.array([random.uniform(-10, 10), random.uniform(20, 40)])
        # self.simulation.goal_position = self.goal_position

        # Randomize robot position and orientation
        # self.simulation.robot_body.position = (random.uniform(-20, 20), random.uniform(10, 50))
        self.simulation.robot_body.position = (-30+random.normalvariate(0, 3), 30+random.normalvariate(0, 3))
        self.simulation.robot_body.angle = random.normalvariate(0, math.pi/12)

        # Ensure object does not penetrate gripper: place it at least some distance away from the gripper
        min_distance = 10  # Minimum distance to ensure no penetration
        while True:
            obj_x = random.normalvariate(0, 3)
            obj_y = random.normalvariate(30, 3)
            distance = np.linalg.norm(np.array([obj_x, obj_y]) - np.array(self.simulation.robot_body.position))
            if distance >= min_distance:
                break
        
        self.simulation.object_body.position = (obj_x, obj_y)
        self.simulation.object_body.angle = random.uniform(-math.pi, math.pi)

        # Reset velocities of robot and object
        self.simulation.robot_body.linearVelocity = (0, 0)
        self.simulation.robot_body.angularVelocity = 0
        self.simulation.object_body.linearVelocity = (0, 0)
        self.simulation.object_body.angularVelocity = 0
        self.simulation.world.ClearForces()

        obs = self._get_obs()
        return obs, {}

    def step(self, action, sim_steps=1):
        
        # noise = np.random.normal(0, 0.1, size=action.shape)
        # action = action + noise
        
        # Rescale action to desired range
        # action = 2*action / self.action_res - 1
        # action = action * np.array([.1, .1,])
        # action = action * np.array([.01, .01,])
        # action = action * np.array([.01, .01, .001])
        # action = np.clip(rescaled_action, self.action_space.low, self.action_space.high)
        if action == 0:
            action = (0.0, 0.02, 0)
        elif action == 1:
            action = (0.0, -0.02, 0)
        elif action == 2:
            action = (-0.02, 0.0, 0)
        elif action == 3:
            action = (0.02, 0.0, 0)
        elif action == 4:
            action = (0.0, 0.0, 0.001)
        elif action == 5:
            action = (0.0, 0.0, -0.001)

        # Get the current velocities
        current_linear_velocity = self.simulation.robot_body.linearVelocity
        current_angular_velocity = self.simulation.robot_body.angularVelocity

        # Apply the velocity offsets
        new_linear_velocity = (
            float(current_linear_velocity[0] + action[0]),
            float(current_linear_velocity[1] + action[1]),
            # float(action[0]),
            # float(action[1])
        )
        new_angular_velocity = float(current_angular_velocity + action[2])

        # Set the new velocities
        self.simulation.robot_body.linearVelocity = new_linear_velocity
        self.simulation.robot_body.angularVelocity = new_angular_velocity

        # Step the simulation
        for _ in range(sim_steps):
            self.simulation.world.Step(self.simulation.timeStep, self.simulation.vel_iters, self.simulation.pos_iters)
        self.simulation.world.ClearForces()
        self.simulation.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        truncated = self._is_truncated()

        return obs, reward, done, truncated, {}

    def _get_obs(self):
        # Render the screen and capture the image
        self.screen.fill(self.simulation.colors['background'])
        self.simulation.draw()
        pygame.display.flip()
        if self.obs_type == 'image':
            img = pygame.surfarray.array3d(pygame.display.get_surface())
            img = np.transpose(img, (1, 0, 2))  # Convert to the shape (height, width, 3)
            
            # Resize the image to the desired observation size
            img = pygame.transform.scale(pygame.surfarray.make_surface(img), self.img_size)
            img = pygame.surfarray.array3d(img)
            img = np.transpose(img, (1, 0, 2))  # Convert back to the shape (height, width, 3)
            
            # Normalize image observation
            img = img / 255.0
            return img
        else:
            # Low-dimensional observation: object pose, gripper pose, goal position
            object_pose = np.array(list(self.simulation.object_body.position) + [self.simulation.object_body.angle,])
            gripper_pose = np.array(list(self.simulation.robot_body.position) + [self.simulation.robot_body.angle,])
            # goal_position = self.goal_position

            obs = np.concatenate([object_pose, gripper_pose])
            
            # Normalize low-dimensional observation
            max_vals = np.array([self.simulation.width / 20, self.simulation.height / 10, np.pi] * 2 ) 
                                # + [self.simulation.width / 20, self.simulation.height / 10,])
            obs_normalized = obs / max_vals
            
            return obs_normalized

    def _draw_goal(self):
        pygame.draw.circle(self.screen, (255, 255, 0), self.simulation.to_pygame(self.goal_position), int(self.goal_radius * 10))

    def _compute_reward(self):
        object_pos = np.array(self.simulation.object_body.position)
        gripper_pos = np.array(self.simulation.robot_body.position)
        # object_vel = np.array(self.simulation.object_body.linearVelocity)
        # gripper_vel = np.array(self.simulation.robot_body.linearVelocity)
        distance_to_goal = np.linalg.norm(object_pos - self.goal_position)
        distance_to_object = np.linalg.norm(object_pos - gripper_pos)

        if distance_to_goal > self.goal_radius:
            # reward = -distance_to_object*0.01 + min(1000, -np.log(distance_to_goal*0.3))  # Negative reward proportional to distance
            # reward = 10/(distance_to_goal+3*distance_to_object)
            reward = np.exp(-0.1*distance_to_object) + 3*np.exp(-0.1*distance_to_goal)
        else:
            reward = 1000  # High positive reward for reaching the goal

        if abs(object_pos[0]) > 38 or (object_pos[1] < 2 or object_pos[1] > 58) or abs(gripper_pos[0]) > 38 or (gripper_pos[1] < 2 or gripper_pos[1] > 58):
            reward -= 100  # Negative reward for going out of bounds

        # Penalize high velocities
        # reward += np.exp((-np.linalg.norm(object_vel) * 10 - np.linalg.norm(gripper_vel) * 10))
        # print('reward', reward)

        return reward

    def _is_done(self):
        object_pos = np.array(self.simulation.object_body.position)
        distance_to_goal = np.linalg.norm(object_pos - self.goal_position)
        return bool(distance_to_goal < self.goal_radius)

    def _is_truncated(self):
        # Get the position of the gripper and object
        gripper_pos = np.array(self.simulation.robot_body.position)
        object_pos = np.array(self.simulation.object_body.position)
        
        # Define canvas boundaries
        canvas_min_x, canvas_max_x = -self.simulation.width / 20, self.simulation.width / 20  # -40,40
        canvas_min_y, canvas_max_y = 0, self.simulation.height / 10  # 0,60
        
        # Check if the gripper/object is out of the canvas
        gripper_out_of_canvas = not (canvas_min_x <= gripper_pos[0] <= canvas_max_x and canvas_min_y <= gripper_pos[1] <= canvas_max_y)
        object_out_of_canvas = not (canvas_min_x <= object_pos[0] <= canvas_max_x and canvas_min_y <= object_pos[1] <= canvas_max_y)
    
        time_ended = self.simulation.step_count >= 10000  # Maximum number of steps

        return bool(gripper_out_of_canvas or object_out_of_canvas or time_ended)

    def render(self, mode='human'):
        if self.gui:
            handle_pygame_events()
            pygame.display.flip()

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    v_angle = math.pi / 3  # Example angle in radians (30 degrees)
    object_type = 'circle'  # Can be 'circle' or 'polygon'
    obs_type = 'pose'  # Can be 'image' or 'pose'
    env = VPushSimulationEnv(object_type, v_angle, gui=True, img_size=(42, 42), obs_type=obs_type)  # Use smaller image size or low-dim pose
    check_env(env)

    total_timesteps = int(1e6)
    learning_rate_schedule = get_linear_fn(1e-1, 0, total_timesteps)
    # clip_range_schedule = get_linear_fn(2.5e-4, 0, total_timesteps)
    # model = PPO("CnnPolicy" if obs_type == 'image' else "MlpPolicy", 
    #             env, 
    #             verbose=1, 
    #             learning_rate=1e-1, 
    #             n_steps=2048, 
    #             batch_size=64, 
    #             n_epochs=10, 
    #             gamma=0.99, 
    #             gae_lambda=0.95, 
    #             clip_range=0.2, 
    #             ent_coef=0.0, 
    #             vf_coef=0.5, 
    #             max_grad_norm=0.5, 
    #             tensorboard_log="./ppo_box2d_tensorboard/")
    model = DQN("CnnPolicy" if obs_type == 'image' else "MlpPolicy", 
                env, 
                verbose=1, 
                learning_rate=learning_rate_schedule, 
                # buffer_size=100000,  # Reduced buffer size to handle computational limitations
                # learning_starts=1000,  # Start learning earlier for faster convergence
                # batch_size=64,  # Increased batch size for stable training
                # tau=1.0, 
                # gamma=0.99, 
                # train_freq=(4, 'step'),  # Ensure training frequency is correctly defined
                # gradient_steps=1, 
                exploration_fraction=0.6, 
                exploration_initial_eps=1.0, 
                exploration_final_eps=0.2,  # Lower final epsilon for better exploitation
                # max_grad_norm=10, 
                target_update_interval=250,
                tensorboard_log="./dqn_box2d_tensorboard/")
    # model = A2C("CnnPolicy" if obs_type == 'image' else "MlpPolicy", 
    #             env, 
    #             verbose=1, 
    #             learning_rate=learning_rate_schedule,  # Adjusted learning rate
    #             n_steps=10,  # Increase number of steps per update
    #             gamma=0.99, 
    #             gae_lambda=0.95, 
    #             ent_coef=.1,  # Increase entropy coefficient to encourage exploration
    #             vf_coef=0.5, 
    #             max_grad_norm=0.5, 
    #             tensorboard_log="./a2c_box2d_tensorboard/")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

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