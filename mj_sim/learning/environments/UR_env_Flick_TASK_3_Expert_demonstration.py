



import argparse
import time
from threading import Lock
import threading

import queue


from typing import Tuple, Union

import mujoco as mj
import numpy as np
import torch


from utils.mj import get_mj_model, attach, detach,  get_joint_names, get_joint_q, get_joint_dq, get_joint_ddq, get_joint_torque, body_name2id, set_joint_q, get_body_pose
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns
from typing import Union, Tuple, List
from skrl.memories.torch import Memory

import pickle

from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from torch.distributions import Normal

from spatialmath.base import r2q, qqmul, q2r, eul2r, qconj
import spatialmath as sm

from robots.ur_robot import URRobot
from collections import deque


from utils.rtb import make_tf
from utils.math import angular_distance

import roboticstoolbox as rtb

import copy


from learning.airl_UR.airl import AIRL
from learning.airl_UR.utils.utils import log_prob_density
from dm_control import mjcf
from Gello.gello import GelloUR5
from pynput import keyboard


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class URSim_SKRL_env(MujocoEnv):

    # Not sure what this guy is
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        # "render_fps": 100,
    }


    # set default episode_len for truncate episodes
    def __init__(self, args, name, render_mode="human"):
        # For expert training set




        self.args = args
        self.actuated_joints = 6
        observation_space_size = 27


        # # change shape of observation to your observation space size
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(observation_space_size,), 
            dtype=np.float64
        )
        
        super().__init__(
            model_path=os.path.abspath(args.scene_path),
            frame_skip=7,
            observation_space=self.observation_space,
            render_mode=render_mode
        )

        # UR5e robot interface
        self.ur5e = URRobot(args, self.data, self.model, robot_type=URRobot.Type.UR5e)


        # # constrained action space
        self.action_space = Box(
            -np.pi*2, 
            np.pi*2, 
            shape=(self.actuated_joints,), 
            dtype=np.float64
        )


        self.step_number = 0
        self.all_steps = 0
        self.episode_len = args.episode_timeout
        
        

        self.log_current_joints = [[] for _ in range(self.actuated_joints)]
        self.log_current_vel = [[] for _ in range(self.actuated_joints)]
        self.log_current_acc = [[] for _ in range(self.actuated_joints)]
        self.log_actions = [[] for _ in range(self.actuated_joints)]


        # Log error value between target and current position

        self.error_x_list = []
        self.error_y_list = []
        self.error_z_list = []
        self.error_angle_list = []

        # Log position of the block, and joint value to see if its been flicked

        self.block_pos_list_x = []
        self.block_pos_list_y = []
        self.block_pos_list_z = []
        self.block_joint_value = []



        self.log_rewards = []

        self.out_of_bounds = np.pi*2 - 0.1

        self.plot_data_after_count = 3
        self.episode_count = 0
        self.name = name

        self.previous_action = [-1]*self.actuated_joints 


        self.robot_joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

        # For reward normalization
        self.reward_beta = 0.01
        self.reward_v = 10


        self.state_actions = []
        self.memory = deque()

        self.memory_single_episode = deque()





        # Load start state:
        qpos_dir = os.path.abspath("learning/config_files/F3_start_state/qpos_start_state.pkl")
        qvel_dir = os.path.abspath("learning/config_files/F3_start_state/qvel_start_state.pkl")

        qpos_file = open(qpos_dir, "rb")
        qvel_file = open(qvel_dir, "rb")

        self.Q_list = pickle.load(qpos_file)
        self.Q_vel_list = pickle.load(qvel_file)




            
        self.gello_bot = GelloUR5("/dev/ttyUSB0")
        self.ctrl_queue = queue.Queue(maxsize=1)
        thread = threading.Thread(target=self.set_gello_val, daemon=True)
        thread.start()

        listener = keyboard.Listener(
            on_press=self.keyboard_callback)
        
        listener.start()

        self.record_mode = False


        # Load expert data [AIRL]
        # path_expert_data_1 = os.path.abspath("state_action_data/25-04-18_12-59-56URSim_SKRL_env_PPO/expert_memory.pkl")


        # expert_data_file_1 = open(path_expert_data_1, "rb")
        # self.expert_memory = pickle.load(expert_data_file_1)



        # DEBUG; This is for getting the observation space size.. 
        # But mujoco must initialize before its possible to run it
        self.reset()
        print("Observation test: ", self._get_obs())
        assert len(self._get_obs()) == observation_space_size, f"Observation space size is not correct. Expected {observation_space_size}, got {len(self._get_obs())}"



    def set_gello_val(self):
        while True:
            try:
                q = self.gello_bot.get_q()
                self.ctrl_queue.put(q)
            except:
                self.gello_bot = GelloUR5("/dev/ttyUSB0")

            time.sleep(0.001)


    def keyboard_callback(self, key):
        try:
            if key.char == "r":
                print("\n")
                print("start expert demonstration recording!")
                self.record_mode = True


            if key.char == "s":
                print("\n")
                print("time step skipped forward")
                self.step_number += 2000
                self.record_mode = True


            if key.char == "p":
                print("\n")
                print("state saved")

                # Save the memory to a pickle file
                with open("learning/config_files/F3_start_state/"  + f"qpos_start_state.pkl", "wb") as file:
                    pickle.dump(self.data.qpos[:], file)



                # Save the memory to a pickle file
                with open( "learning/config_files/F3_start_state/" + f"qvel_start_state.pkl", "wb") as file:
                    pickle.dump(self.data.qvel[:], file)



                # Load start state:
                qpos_dir = os.path.abspath("learning/config_files/F3_start_state/qpos_start_state.pkl")
                qvel_dir = os.path.abspath("learning/config_files/F3_start_state/qvel_start_state.pkl")

                qpos_file = open(qpos_dir, "rb")
                qvel_file = open(qvel_dir, "rb")

                self.Q_list = pickle.load(qpos_file)
                self.Q_vel_list = pickle.load(qvel_file)




        except AttributeError:
            pass




            
    def _r(self) -> float:
        '''
        Summary: This is the reward function for the RL mujoco simulation task.
                    The reward is based on the proximity to 0

        ARGS:
            args: The arguments that is passed from the main script
            robot: The robot model that also contains all mujoco data and model information

        RETURNS:
            punishment: Quantitative value that represents the performance of the robot, based on intended task specified in this function.
        '''


        
          # Performance which will also be part of the observation space.
        performance_score = 0
        performance_score_vel = 0

    


        # Add block joint value as observation - Joint value 2
        block_joint_3 = get_joint_q(self.data, self.model, "f3_block_3_joint")



        # Add block joint value as observation - Joint value 2
        block_joint_4 = get_joint_q(self.data, self.model, "f3_block_4_joint")






        # Add block joint value as observation - Joint value 1
        block_joint = get_joint_dq(self.data, self.model, "f3_block_1_joint")
        performance_score_vel += abs(block_joint)

        # Add block joint value as observation - Joint value 2
        block_joint = get_joint_dq(self.data, self.model, "f3_block_2_joint")
        performance_score_vel += abs(block_joint)

        # Add block joint value as observation - Joint value 2
        block_joint = get_joint_dq(self.data, self.model, "f3_block_3_joint")
        performance_score_vel += abs(block_joint)

        # Add block joint value as observation - Joint value 2
        block_joint = get_joint_dq(self.data, self.model, "f3_block_4_joint")
        performance_score_vel += abs(block_joint)
        

        # Error based en distance


        F3_block_3_name = "f3_block_3"
        F3_block_5_name = "f3_block_5"


        F3_block_3_pose = get_body_pose(self.data, self.model, F3_block_3_name)
        F3_block_5_pose = get_body_pose(self.data, self.model, F3_block_5_name)

        z_position_error = 0
        desired_z_position = 0.1768

                
        z_position_error += abs(F3_block_3_pose.t[2] - desired_z_position)
        z_position_error += abs(F3_block_5_pose.t[2] - desired_z_position)



        desired_position_error = abs(block_joint_3 + block_joint_4)/ 4
        desired_velocity_error = abs(performance_score_vel) / (6 * 5)
        z_position_error = z_position_error**2


        error_sum = desired_position_error + desired_velocity_error + z_position_error

        reward = 1 - error_sum


        return  reward
        


    def _d(self) -> bool:

        
        # joints = self.get_robot_q()


        # for i in range(len(joints)):
        #     if abs(joints[i]) > self.out_of_bounds: # If the joint is outside the range
        #         return True

        #     if i >= self.actuated_joints - 1:
        #         break



        # If timeout
        if self.step_number >= self.episode_len:
            return True

        


        return False


    def _get_obs(self) -> np.ndarray:
        joint = self.get_robot_q()
        joint_vel = self.get_robot_dq()

        # Order as [joint1 pos, joint1 vel, joint2 pos, joint2 vel, ...]
        observation = np.array([])


        for i in range(self.actuated_joints):
            observation = np.append(observation, joint[i])
        
        for i in range(self.actuated_joints):
            observation = np.append(observation, joint_vel[i])
            
            # observation = np.append(observation, joint_acc[i])
        # observation = np.append(observation, self.gripper)
        # if self.object_attached == True:
        #     observation = np.append(observation, 0)
        # elif self.object_attached == False:
        #     observation = np.append(observation, 1)

            
        # # Add cartesian position of the end effector
        ee_pose = self.ur5e.get_ee_pose()

        cartesian_ee_pos = ee_pose.t
        cartesian_ee_rot = r2q(ee_pose.R)

        observation = np.append(observation, cartesian_ee_pos)
        observation = np.append(observation, cartesian_ee_rot)




        # Add block joint value as observation - Joint value 1
        block_joint = get_joint_q(self.data, self.model, "f3_block_1_joint")
        observation = np.append(observation, block_joint)


        # Add block joint value as observation - Joint value 2
        block_joint = get_joint_q(self.data, self.model, "f3_block_2_joint")
        observation = np.append(observation, block_joint)


        # Add block joint value as observation - Joint value 2
        block_joint = get_joint_q(self.data, self.model, "f3_block_3_joint")
        observation = np.append(observation, block_joint)


        # Add block joint value as observation - Joint value 2
        block_joint = get_joint_q(self.data, self.model, "f3_block_4_joint")
        observation = np.append(observation, block_joint)




        # Add block joint value as observation - Joint value 1
        block_joint = get_joint_dq(self.data, self.model, "f3_block_1_joint")
        observation = np.append(observation, block_joint)

        # Add block joint value as observation - Joint value 2
        block_joint = get_joint_dq(self.data, self.model, "f3_block_2_joint")
        observation = np.append(observation, block_joint)


        # Add block joint value as observation - Joint value 2
        block_joint = get_joint_dq(self.data, self.model, "f3_block_3_joint")
        observation = np.append(observation, block_joint)


        # Add block joint value as observation - Joint value 2
        block_joint = get_joint_dq(self.data, self.model, "f3_block_4_joint")
        observation = np.append(observation, block_joint)



        
        return observation






    # Get position of foldable block NOTE: task related function, should go somewhere else when cleaning up code.
    def get_box_position(self, block_pos, block_rot):
        """
        Summary: Get the position of the block in the robot frame
        ARGS:
            block_pos: The position of the block
            block_rot: The rotation of the block (quaternion)
        """
        

        # Get robot base position
        self.ur5e_base_cartesian = [0.22331901, 0.37537452, 0.08791326]
        self.ur5e_rot_quat = [ -0.19858483999999996, -0.00311175, 0.0012299899999999998, 0.98007799]
        self.ur5e_base_SE3 = make_tf(self.ur5e_base_cartesian, self.ur5e_rot_quat)





        block_pos = np.append(block_pos, 1)

        # position above the block to pick it up in box frame
        box_shift_translation_box_space = [-0.14,   0.2,   -0.075,  1.]

        # Convert from box space to world frame
        box_rotation = make_tf([0, 0, 0], block_rot)
        box_shift_translation = np.array(box_rotation) @ box_shift_translation_box_space

        # add the translation to the block position for correct pick up position
        block_pos[0] = block_pos[0] + box_shift_translation[0]
        block_pos[1] = block_pos[1] + box_shift_translation[1]
        block_pos[2] = block_pos[2] + abs(box_shift_translation[2]) # Never pick the box from beneath

        # Convert to robot frame
        point_translated = np.array(self.ur5e_base_SE3.inv()) @ block_pos
        point_translated = point_translated[0:3] / point_translated[3]

        # gained by designing rotation in world frame
        desired_rotation = [ 0.0000,   -0.7071,    0.7071,    0.0000]

        return point_translated, desired_rotation
    

    def joint_change_for_cartesian_pose(self, desired_position, desired_rotation):
        """
        Summary: 
            This function is used for the expert data generation where we get the 
            joint change value for the desired cartesian pose

        ARGS:
            desired_pos: The desired cartesian pose [x,y,z]
            desired_rot: The desired rotation in quaternion euler angles
        """
        
        # Position 2: charge flick downwards
        jac=np.zeros((6,self.model.nv))
        id=self.model.body("wrist_3_link").id


        mj.mj_jacBody(self.model, self.data, jac[:3], jac[3:], id) #Get geometric jacobian
        # Only the first 6 indexes are the ur5e robot joints

        jac_ur5e = jac[:, :6]

        pose = self.ur5e.get_ee_pose()

        quat_rot = r2q(pose.R)

        flick_rot = desired_rotation # in robot frame

        quat_flick = qqmul(quat_rot, r2q(eul2r(flick_rot)))


        def angular_velocities(q1, q2, dt): # https://mariogc.com/post/angular-velocity-quaternions/#the-angular-velocities
            return (2 / dt) * np.array([
                q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
                q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
                q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

        angular_velocity_flick = angular_velocities(quat_rot, quat_flick, 2.0)

        # Translation flick does not work


        
        world_to_robot = q2r([-0.19858483999999996, -0.00311175, 0.0012299899999999998, 0.98007799])

        ee_pose_world_frame = world_to_robot @ pose.R

        angular_velocity_flick = ee_pose_world_frame @ angular_velocity_flick
        desired_position = ee_pose_world_frame @ desired_position


        desired_cartesian_change = np.concatenate((desired_position, angular_velocity_flick))
        # Calculate the joint change
        joint_change = np.linalg.inv(jac_ur5e) @ (desired_cartesian_change)

        return joint_change


    def joint_space_control(self, input_val):
        
        # Remade into a velocity controller


        Kp = 10


        # Calculate the desired velocity based on a p controller



        q_tilde = np.array(input_val) - np.array(self.get_robot_q())


        u = Kp*(q_tilde)

        # print("desired acceleration: \n", desired_acceleration)
        # print("acceleration current: \n", self.get_robot_ddq())
        # print("acceleration error: \n", Kd*(dq_tilde))
        # print("\n")
        # print("desired velocity: \n", desired_velocity)
        # print("velocity current: \n", self.get_robot_dq())
        # print("velocity error: \n", Kp2*(q_tilde))
        # input()

        return u
        


    def step(
        self,
        a: Union[np.ndarray, list, torch.Tensor],
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Placeholder function.

        Returns:
            Tuple: A tuple.
        """
                
        # Expert data generation



        # semi reset 
        if self.record_mode == True:

            if self.step_number == 0:

                self.set_state(self.Q_list, self.Q_vel_list)

                self.started = True

        else:
            self.started = False


        if self.ctrl_queue.full():
        
            item = self.ctrl_queue.get()
            self.expert_action = item[0:6]
            #self.gripper = item[6]
            self.expert_action


        r = self._r()

        
        a = self.joint_space_control(self.expert_action)


        # saturate action values 
        for i, val in enumerate(a):
            a[i] = min(max(a[i], -5), 5)



        pre_state = self._get_obs()


        self.do_simulation(a, self.frame_skip)


        if self.record_mode == True and self.started == True:
            self.step_number += 1



        action = a



        reward = self._r()
        done = self._d()
        obs = self._get_obs()   


        # log_pis = log_prob_density(torch.Tensor(a), torch.Tensor(np.array([a])), torch.Tensor(np.array([[0.1]*self.actuated_joints])))
        # reward = self._r_IRL(obs=pre_state, done=done, log_pis=log_pis, next_state=obs)



        if self.args.save_state_actions and self.record_mode==True and self.started == True:
            state_action = np.concatenate((self._get_obs(), action))    
            self.state_actions.append(state_action)

            self.memory_single_episode.append([pre_state, action, reward, done, 0, obs]) # the log_pis should be 0, as it is calculated online in IRL



        robot_joints = self.get_robot_q()
        # robot_vel = self.get_robot_dq()
        # robot_acc = self.get_robot_ddq()





        infos =  {"Robot joint 0":  torch.tensor(robot_joints[0]),
                  }

        if self.render_mode == "human":
            self.render()
        
        truncated = self.step_number > self.episode_len





        return obs, reward, done, truncated, infos


    def reset(self): # overloading reset_model to fit more libraries 
        return self.reset_model(), {}
    

    def reset_model(
        self
    ):     
        self.all_steps += self.step_number




        if self.args.save_state_actions:

            print("reward: ", self._r())
            val = input("keep the expert demonstration? y/n")

            print("response: ", val)
            if val == 'ry' or val == 'rY':
                # Ensure the directory exists before saving the .npy file
                directory = f"./state_action_data/{self.name}"
                if not os.path.exists(directory):
                    os.makedirs(directory)  # Create the directory if it doesn't exist

                np.save(f"./state_action_data/{self.name}/all_state_actions.npy", np.array(self.state_actions))

                # Calculate the number of datasets (entries in self.state_actions)
                num_datasets = len(self.state_actions)
                
                # Write metadata to a .txt file, overwriting the old file
                with open(f"./state_action_data/{self.name}/metadata_states_and_actions.txt", "w") as file:
                    file.write(f"Number of datasets: {num_datasets}\n")

                for val in self.memory_single_episode:
                    self.memory.append(val)


                # Save the memory to a pickle file
                with open(f"./state_action_data/{self.name}/expert_memory.pkl", "wb") as file:
                    pickle.dump(self.memory, file)

                # Calculate the number of datasets (entries in self.state_actions)
                num_memory = len(self.memory)

                # Write metadata to a .txt file, overwriting the old file
                with open(f"./state_action_data/{self.name}/metadata_expert_memory.txt", "w") as file:
                    file.write(f"Number of datasets: {num_memory}\n")

                print("Files saved!")
                

        self.gripper = 0

        self.error_x_list = []
        self.error_y_list = []
        self.error_z_list = []
        self.error_angle_list = []
        self.memory_single_episode = deque()


        self.block_pos_list_x = []
        self.block_pos_list_y = []
        self.block_pos_list_z = []
        self.block_joint_value = []

        self.log_current_joints = [[] for _ in range(self.actuated_joints)]
        self.log_current_vel = [[] for _ in range(self.actuated_joints)]
        self.log_current_acc = [[] for _ in range(self.actuated_joints)]
        self.log_actions = [[] for _ in range(self.actuated_joints)]
        self.log_rewards = []
        self.episode_count += 1
        self.step_number = 0
    




        self.set_state(self.Q_list, self.Q_vel_list)

        # attach(
        #     self.data,
        #     self.model,
        #     "attach",
        #     "2f85",
        #     self.ur5e.T_world_base @ self.ur5e.get_ee_pose(),
        # )

        attach(
            self.data,
            self.model,
            "attach",
            "f3_block_free_joint",
            self.ur5e.T_world_base @ self.ur5e.get_ee_pose()
        )

        self.record_mode = False

        return  self._get_obs()

    def _plot_robot_data(self):
        '''
        Summary: Plots the joint data of the robot
        '''
        # Define the directory for saving plots
        plot_dir = f"./plot_data/{self.name}/"

        # Check if the directory exists, if not, create it
        os.makedirs(plot_dir, exist_ok=True)


        # Plot ddq values for each episode from sublist
        fig, ax = plt.subplots()
        clrs = sns.color_palette("husl", 20)
        ax.set_title(f"Episode: {self.episode_count} - steps episode start: {self.all_steps}")
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.axhline(np.pi, color='red', linestyle='--', linewidth=0.8, label="Target block joint value")

        # 243 optimal reward and 1/250 is reward scaling factor
        # ax.axhline(1, color='blue', linestyle='--', linewidth=0.8, label="Optimal reward")

        self.time_list = np.linspace(0, len(self.block_joint_value)*self.model.opt.timestep, len(self.block_joint_value))


        max_val = max(np.max(self.block_joint_value), np.max(self.block_pos_list_x), np.max(self.block_pos_list_y), np.max(self.block_pos_list_z))
        min_val = min(np.min(self.block_joint_value), np.min(self.block_pos_list_x), np.min(self.block_pos_list_y), np.min(self.block_pos_list_z))

        ax.set_ylim(min(min_val, -1)-0.1, max(max_val, np.pi + 0.1) + 0.1)


        ax.plot(self.time_list, np.array(self.block_joint_value), label=f"block joint value", linewidth=2)

        ax.plot(self.time_list, np.array(self.block_pos_list_x), label=f"block pos: x", linewidth=2)
        ax.plot(self.time_list, np.array(self.block_pos_list_y), label=f"block pos: y", linewidth=2)
        ax.plot(self.time_list, np.array(self.block_pos_list_z), label=f"block pos: z",  linewidth=2)



        # ax.plot(self.time_list, np.array(self.log_rewards), c=clrs[6], label="rewards", linewidth=1.3, linestyle="dotted")

        ax.legend(loc="lower right")
            # Save the plot
        plt.savefig(f"./plot_data/{self.name}/Latest_value_{self.episode_count}.png")
        plt.close()


    def get_robot_q(self):

        # Return the 6 first values of the qpos
        pos = []

        for i, joint_name in enumerate(self.robot_joint_names):
            pos.append(get_joint_q(self.data, self.model, joint_name)[0])

        return pos
    


    def get_robot_dq(self):

        # Return the 6 first values of the qpos
        vel = []

        for i, joint_name in enumerate(self.robot_joint_names):
            vel.append(get_joint_dq(self.data, self.model, joint_name)[0])
        return vel
    

    def get_robot_ddq(self):

        # Return the 6 first values of the qpos
        acc = []

        for i, joint_name in enumerate(self.robot_joint_names):
            acc.append(get_joint_ddq(self.data, self.model, joint_name)[0])
        return acc



    def get_robot_torque(self):

        # Return the 6 first values of the qpos
        Torque = []

        for i, joint_name in enumerate(self.robot_joint_names):
            Torque.append(get_joint_torque(self.data, self.model, joint_name)[0])
        return Torque


    def get_gravity_compensation(self):
        qf=[]

        for i, joint_name in enumerate(self.robot_joint_names):
            qf.append(float(self.data.joint(joint_name).qfrc_bias)) 

        return qf



    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]