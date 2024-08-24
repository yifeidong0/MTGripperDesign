import pickle
import time

import numpy as np
import rospy
import roslib.packages
import geometry_msgs.msg
import moveit_commander
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from stable_baselines3.sac.policies import MlpPolicy

class PandaInferenceNode:
    def __init__(self) -> None:
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)

        planning_frame = self.group.get_planning_frame()
        rospy.loginfo("Reference frame: %s" % planning_frame)

        ee = "panda_hand"
        self.group.set_end_effector_link(ee)
        eef_link = self.group.get_end_effector_link()
        rospy.loginfo("End effector: %s" % eef_link)

        group_names = self.robot.get_group_names()
        rospy.loginfo("Robot Groups: %s" % group_names)
        
        # Set up velocity and acceleration
        vel = 0.4
        acc = 0.4        
        self.group.set_max_velocity_scaling_factor(vel)
        self.group.set_max_acceleration_scaling_factor(acc)
        
        # Set up tolerance
        self.joint_tol = 0.0001
        self.orn_tol   = 0.001
        self.pos_tol   = 0.001
        self.group.set_goal_joint_tolerance(self.joint_tol)
        self.group.set_goal_orientation_tolerance(self.orn_tol)
        self.group.set_goal_position_tolerance(self.pos_tol)
        
        rospy.loginfo("Ready!")
        
        model_path = roslib.packages.get_pkg_dir("panda_inference") + "/model/policy_to_test"
        self.model = MlpPolicy.load(model_path)
        # self.control_timer = rospy.Timer(rospy.Duration(1), self.control_timer_cb)
        
        self.obs = None
        self.z_offset = 0.3
    
    def go_home(self, msg):
        target_pose = geometry_msgs.msg.Pose()
        home_euler = (np.pi, 0, 0)
        home_position = [0.3, 0.0, self.z_offset]
        target_pose.orientation = geometry_msgs.msg.Quaternion(*quaternion_from_euler(*home_euler))
        target_pose.position = geometry_msgs.msg.Point(*home_position)
        self.group.set_pose_target(target_pose)
        move_success = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        if move_success:
            rospy.loginfo("Going home successfully!")
        else:
            rospy.logwarn("Fail to go home!")
    
    def control_timer_cb(self, event):
        if self.obs is None:
            return
        action, _ = self.model.predict(self.obs)
        print("Action: ", action)
        current_pose = self.group.get_current_pose().pose
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = current_pose.position.x + float(action[0])
        target_pose.position.y = current_pose.position.y + float(action[1])
        target_pose.position.z = self.z_offset
        current_orientation_euler = euler_from_quaternion([current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w])
        # target_orientation_euler = (np.pi, 0, current_orientation_euler[2] + float(action[2]))
        target_orientation_euler = (np.pi, 0, 0)
        target_pose.orientation = geometry_msgs.msg.Quaternion(*quaternion_from_euler(*target_orientation_euler))
        self.group.set_pose_target(target_pose)
        start = time.time()
        move_success = self.group.go(wait=True) # TODO: wait=True
        print("Time for planning & execution: %.3f sec" % (time.time() - start))
        self.group.stop()
        self.group.clear_pose_targets()
        if move_success:
            rospy.loginfo("Move end effector successfully!")
        else:
            rospy.logwarn("Fail to move end effector!")

    def replay_obs(self):
        obs_path = roslib.packages.get_pkg_dir("panda_inference") + "/scripts/all_obs.pkl"
        with open(obs_path, "rb") as f:
            all_obs = pickle.load(f)
        for obs in all_obs:
            self.obs = obs
            self.control_timer_cb(None)
        self.obs = None

    def run(self):
        num_steps = 100
        self.obs = {}
        self.obs["observation"] = np.random.rand(28)
        self.obs["achieved_goal"] = np.random.rand(3)
        self.obs["desired_goal"] = np.random.rand(3)
        for i in range(num_steps):
            ee_pose = self.group.get_current_pose().pose
            self.obs["observation"][:3] = [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z]
            ee_euler = euler_from_quaternion([ee_pose.orientation.x, ee_pose.orientation.y, ee_pose.orientation.z, ee_pose.orientation.w])
            self.obs["observation"][3:6] = [0, 0, 0]
            self.obs["observation"][6:7] = ee_euler[2]
            self.obs["observation"][7:11] = [0, 0, 0, 0]
            self.obs["observation"][11:14] = [3, 0, 0]
            self.obs["observation"][14:17] = [0, 0, 0]
            self.obs["observation"][17:20] = [0, 0, 0]
            self.obs["observation"][20:23] = [0, 0, 0]
            self.obs["observation"][23:26] = [5, 0, 0]
            self.obs["observation"][26:27] = [0]
            self.obs["observation"][27:28] = [0.1]
            self.obs["achieved_goal"] = self.obs["observation"][11:14]
            self.obs["desired_goal"] = self.obs["observation"][23:26]
            self.control_timer_cb(None)

        # ee_position = observation[..., :3]
        # ee_velocity = observation[...,3:6]
        # ee_yaw = observation[...,6:7]
        # design_params = observation[...,7:11]
        # object_position = observation[...,11:14]
        # object_rotation = observation[...,14:17]
        # object_velocity = observation[...,17:20]
        # object_angular_velocity = observation[...,20:23]
        # target_position = observation[...,23:26]
        # task_int = observation[...,26:27]
        # object_rad = observation[...,27:28]

if __name__ == "__main__":
    rospy.init_node("panda_inference")
    panda_inference_node = PandaInferenceNode()
    rospy.loginfo("Going home...")
    panda_inference_node.go_home(None)
    # rospy.loginfo("Replaying observations...")
    # panda_inference_node.replay_obs()
    # rospy.loginfo("Replay all observations done!")
    rospy.loginfo("Went home, start to control!")
    panda_inference_node.run()