import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from sim.ucatch_sim import UCatchSimulation  # Assuming UCatchSimulation is in a separate file

def handle_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

class UCatchSimulationEnv(gym.Env):
    def __init__(self, object_type='circle', design_param=[5, 5, 5, np.pi/2, np.pi/2], gui=True, img_size=(42, 42), obs_type='pose'):
        super(UCatchSimulationEnv, self).__init__()
        self.simulation = UCatchSimulation(object_type, design_param, use_gui=gui)
        self.gui = gui
        self.img_size = img_size
        self.obs_type = obs_type
        self.task_int = 0 if self.obs_type == 'circle' else 1
        self.design_param = design_param
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)

        if self.obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=1.0, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.float64)
        else:
            self.observation_space = spaces.Box(low=np.array([-1.0]*12), high=np.array([1.0]*12), dtype=np.float64)

        if self.gui:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption('Box2D with Pygame - U Catch')

        # self.last_robot_pose = None
        # self.last_object_pose = None
        # self.last_action = None

        self.canvas_min_x, self.canvas_max_x = -self.simulation.width / 20, self.simulation.width / 20  # -40,40
        self.canvas_min_y, self.canvas_max_y = 0, self.simulation.height / 10  # 0,60

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulation.step_count = 0
        # self.simulation.setup()
        self.obs_type = random.choice(['circle', 'polygon'])
        self.task_int = 0 if self.obs_type == 'circle' else 1
        self.design_param = [random.uniform(5, 10), random.uniform(5, 10), random.uniform(5, 10), 
                             random.uniform(np.pi/2, np.pi), random.uniform(np.pi/2, np.pi)]
        self.simulation.reset_task_and_design(self.obs_type, self.design_param)
        obs = self._get_obs()
        return obs, {}

    def step(self, action, sim_steps=1):
        # self.last_object_pose = np.array([self.simulation.object_body.position[0], self.simulation.object_body.position[1]])
        # self.last_robot_pose = np.array([self.simulation.robot_body.position[0], self.simulation.robot_body.position[1]])

        action = np.clip(action, self.action_space.low, self.action_space.high)
        new_velocity = [action[0] * 10, 0]  # Scale action to desired range

        self.simulation.robot_body.linearVelocity = new_velocity

        for _ in range(sim_steps):
            self.simulation.world.Step(self.simulation.timeStep, self.simulation.vel_iters, self.simulation.pos_iters)
        self.simulation.world.ClearForces()
        self.simulation.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        done = self._is_done()
        truncated = self._is_truncated()
        self.last_action = action

        return obs, reward, done, truncated, {}

    def _get_obs(self):
        if self.gui:
            self.screen.fill(self.simulation.colors['background'])
            self.simulation.draw()
        pygame.display.flip()

        if self.obs_type == 'image': # TODO: debug this
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
            object_pose = np.array(self.simulation.object_body.position)
            object_vel = np.array(self.simulation.object_body.linearVelocity)
            robot_pose = np.array(self.simulation.robot_body.position)
            
            obs = np.concatenate([object_pose, robot_pose])
            max_vals = np.array([self.simulation.width/20, self.simulation.height/10] * 2) 
            pos_normalized = obs / max_vals
            object_vel_normalized = object_vel / np.array([50, 50])  # Normalize object velocity
            task_design_params_normalized = np.array([self.task_int, self.design_param[0]/10, 
                                                      self.design_param[1]/10, self.design_param[2]/10, 
                                                      self.design_param[3]/np.pi, self.design_param[4]/np.pi])

            return np.concatenate([pos_normalized, object_vel_normalized, task_design_params_normalized])

    def _compute_reward(self, action):
        object_pos = np.array([self.simulation.object_body.position[0], self.simulation.object_body.position[1]])
        robot_pos = np.array([self.simulation.robot_body.position[0], self.simulation.robot_body.position[1]])
        distance_to_goal = np.linalg.norm(object_pos - robot_pos)

        reward = 0
        if self.simulation.check_end_condition():
            reward += 100
        else:
            reward -= distance_to_goal * 0.1

        return reward

    def _is_done(self):
        return self.simulation.check_end_condition()

    def _is_truncated(self):
        robot_pos = np.array(self.simulation.robot_body.position)
        object_pos = np.array(self.simulation.object_body.position)

        out_of_bounds = (
            robot_pos[0] < self.canvas_min_x or robot_pos[0] > self.canvas_max_x or 
            robot_pos[1] < self.canvas_min_y or robot_pos[1] > self.canvas_max_y or 
            object_pos[0] < self.canvas_min_x or object_pos[0] > self.canvas_max_x or 
            object_pos[1] < self.canvas_min_y or object_pos[1] > self.canvas_max_y
        )
        time_ended = self.simulation.step_count >= 2000  # Maximum number of steps

        return bool(out_of_bounds or time_ended)

    def render(self, mode='human'):
        if self.gui:
            handle_pygame_events()
            pygame.display.flip()

    def close(self):
        pygame.quit()

