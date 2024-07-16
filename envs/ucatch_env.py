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
            self.screen = pygame.display.set_mode((self.simulation.width, self.simulation.height))
            pygame.display.set_caption('Box2D with Pygame - U Catch')

        self.last_position_difference = None
        self.last_velocity_difference = None
        self.canvas_min_x, self.canvas_max_x = -self.simulation.width / 20, self.simulation.width / 20  # -40,40
        self.canvas_min_y, self.canvas_max_y = 0, self.simulation.height / 10  # 0,60
        self.num_end_steps = 0
        self.is_success = False

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
        self.num_end_steps = 0
        self.is_success = False
        return obs, {}

    def step(self, action, sim_steps=1):
        # self.last_object_pose = np.array([self.simulation.object_body.position[0], self.simulation.object_body.position[1]])
        # self.last_robot_pose = np.array([self.simulation.robot_body.position[0], self.simulation.robot_body.position[1]])

        action = np.clip(action, self.action_space.low, self.action_space.high)
        new_velocity = [action[0] * 20, 0]  # Scale action to desired range

        self.simulation.robot_body.linearVelocity = new_velocity

        for _ in range(sim_steps):
            self.simulation.world.Step(self.simulation.timeStep, self.simulation.vel_iters, self.simulation.pos_iters)
            # pygame.time.Clock().tick(260)
        self.simulation.world.ClearForces()
        self.simulation.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        done = self._is_done()
        truncated = self._is_truncated()
        self.last_action = action

        info = {}
        if done or truncated:
            info['is_success'] = self.is_success
        return obs, reward, done, truncated, info

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
        object_vel = np.array([self.simulation.object_body.linearVelocity[0], self.simulation.object_body.linearVelocity[1]])
        robot_pos = np.array([self.simulation.robot_body.position[0], self.simulation.robot_body.position[1]])
        current_position_difference = abs(self.simulation.object_landing_position_x - robot_pos[0])

        # Reward for decreasing the distance between object and the goal
        reward = 0
        if self.last_position_difference is not None:
            reward += self.last_position_difference - current_position_difference
        self.last_position_difference = current_position_difference
        
        # Reward for reaching desired velocity
        self.robot_distance_to_travel = robot_pos[0] - self.simulation.object_landing_position_x
        time_to_fall = abs(self.simulation.object_landing_velocity_y - object_vel[1]) / self.simulation.g
        self.robot_desired_vx = -self.robot_distance_to_travel / time_to_fall
        current_velocity_difference = abs(self.robot_desired_vx - action[0]) # - self.simulation.robot_body.linearVelocity[0]
        if self.last_velocity_difference is not None:
            reward += self.last_velocity_difference - current_velocity_difference
        self.last_velocity_difference = current_velocity_difference

        # TODO: Reward of caging robustness

        # Reward for catching the object
        if self.simulation.check_end_condition():
            reward += 1

        return reward

    def _is_done(self):
        if self.simulation.check_end_condition():
            self.num_end_steps += 1
        if self.num_end_steps >= 100: # end if object stays in the basket for 100 steps
            self.is_success = True
            return True
        return False

    def _is_truncated(self):
        robot_pos = np.array(self.simulation.robot_body.position)
        object_pos = np.array(self.simulation.object_body.position)

        out_of_bounds = (
            robot_pos[0] < self.canvas_min_x or robot_pos[0] > self.canvas_max_x or 
            robot_pos[1] < self.canvas_min_y or robot_pos[1] > self.canvas_max_y or 
            object_pos[0] < self.canvas_min_x or object_pos[0] > self.canvas_max_x or 
            object_pos[1] < self.canvas_min_y or object_pos[1] > self.canvas_max_y
        )
        time_ended = self.simulation.step_count >= 500  # Maximum number of steps
        object_on_ground = object_pos[1] < self.simulation.object_on_ground_height # object falls on the ground outside the basket

        return bool(out_of_bounds or time_ended or object_on_ground)

    def render(self, mode='human'):
        if self.gui:
            handle_pygame_events()
            pygame.display.flip()

    def close(self):
        pygame.quit()

