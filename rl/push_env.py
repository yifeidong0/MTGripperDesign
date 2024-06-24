import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from push_sim import ForwardSimulationPlanePush
import time

class VisuomotorPlanarPushEnv(gym.Env):
    def __init__(self, task_type, gripper_length, gui=False):
        super(VisuomotorPlanarPushEnv, self).__init__()
        self.simulation = ForwardSimulationPlanePush(task_type, gripper_length, gui)
        self.action_space = spaces.Box(low=-0.01, high=0.01, shape=(3,), dtype=np.float32)
        
        # observation space: object type (0 for box, 1 for circle), object pose (x, y, theta), gripper pose (x, y, theta)
        low = np.array([-4, -4, -np.pi, -4, -4, -np.pi], dtype=np.float32)
        high = np.array([4, 4, np.pi, 4, 4, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.simulation.reset_states()
        self.simulation.step_count = 0
        return self._get_obs(), {}
    
    def step(self, action):
        assert self.action_space.contains(action), f"{action} is not a valid action"
        
        # Get the current state of the gripper
        # current_pos, current_orn = p.getBasePositionAndOrientation(self.simulation.gripperUid)
        current_vel = np.array(p.getBaseVelocity(self.simulation.gripperUid)[0][:2])
        current_angular_vel = p.getBaseVelocity(self.simulation.gripperUid)[1][2]
        
        # Apply the velocity action to the gripper
        new_vel = current_vel + action[:2]
        new_angular_vel = current_angular_vel + action[2]
        
        # Set the velocity of the gripper
        p.resetBaseVelocity(self.simulation.gripperUid, linearVelocity=[new_vel[0], new_vel[1], 0], angularVelocity=[0, 0, new_angular_vel])
        
        p.stepSimulation()
        self.simulation.step_count += 1
        
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        truncated = self._is_truncated(obs)
        
        return obs, reward, done, truncated, {}

    def _get_obs(self):
        object_pos, object_orn = p.getBasePositionAndOrientation(self.simulation.objectUid)
        gripper_pos, gripper_orn = p.getBasePositionAndOrientation(self.simulation.gripperUid)
        
        # object_type = 0 if self.simulation.task_type == 'box' else 1
        object_orn_z = p.getEulerFromQuaternion(object_orn)[2]
        gripper_orn_z = p.getEulerFromQuaternion(gripper_orn)[2]
        return np.array([*object_pos[:2], object_orn_z, *gripper_pos[:2], gripper_orn_z], dtype=np.float32)
    
    def _compute_reward(self):
        object_pos, _ = p.getBasePositionAndOrientation(self.simulation.objectUid)
        object_vel = np.array(p.getBaseVelocity(self.simulation.objectUid)[0])
        # if object_pos[1] > 5:
        #     return 100
        # else:
        #     return -1
        return object_vel[1] if object_pos[1] < 3 else 100
    
    def _is_done(self):
        object_pos, _ = p.getBasePositionAndOrientation(self.simulation.objectUid)
        return object_pos[1] > 3
    
    def _is_truncated(self, obs):
        return self.simulation.step_count > 240 * 100 or \
            not (self.observation_space.low <= obs).all() or \
            not (obs <= self.observation_space.high).all()
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        self.simulation.finish_sim()

# Main script to run PPO training
if __name__ == "__main__":
    task_type = 'box'
    gripper_length = 0.5
    env = VisuomotorPlanarPushEnv(task_type, gripper_length, gui=True)
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1) # CnnPolicy, MlpPolicy
    model.learn(total_timesteps=300000, progress_bar=True)

    obs, _ = env.reset()
    for episode in range(1):
        time.sleep(2)
        print(f"Episode {episode + 1} begins")
        done, truncated = False, False
        while not (done or truncated):
            action, _states = model.predict(np.array(obs))
            obs, reward, done, truncated, _ = env.step(action)
            env.render()
            time.sleep(.002)
        print("Done!" if done else "Truncated")
        print(f"Episode {episode + 1} finished")
    
    env.close()
