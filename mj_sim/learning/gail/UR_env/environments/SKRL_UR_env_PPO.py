



import argparse
import time
from threading import Lock
from typing import Tuple, Union

import mujoco as mj
import numpy as np
import torch

from utils.mj import get_mj_data, get_mj_model, attach, get_joint_names, get_joint_q, get_joint_dq, get_joint_ddq, get_joint_torque
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



from robots.ur_robot import URRobot




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

        self.args = args
        self.actuated_joints = 6
        observation_space_size = 1 * self.actuated_joints


        # # change shape of observation to your observation space size
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(observation_space_size,), 
            dtype=np.float64
        )
        
        super().__init__(
            model_path=os.path.abspath("learning/scenes/RL_task.xml"),
            frame_skip=1,
            observation_space=self.observation_space,
            render_mode=render_mode
        )

        # UR5e robot interface
        self.ur5e = URRobot(args, self.data, self.model, robot_type=URRobot.Type.UR5e)



        # constrained action space
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

        self.log_rewards = []

        self.out_of_bounds = np.pi*2 - 0.1

        self.plot_data_after_count = 5
        self.episode_count = 0
        self.name = name

        self.previous_action = [-1]*self.actuated_joints 


        self.robot_joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

        # For reward normalization
        self.reward_beta = 0.01
        self.reward_v = 10


        self.state_actions = []
        

        # DEBUG; This is for getting the observation space size.. 
        # But mujoco must initialize before its possible to run it
        print("Observation test: ", self._get_obs())
        assert len(self._get_obs()) == observation_space_size, f"Observation space size is not correct. Expected {observation_space_size}, got {len(self._get_obs())}"



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
        # This value is the target that the joints should converge at:
        target = np.pi/2
        
        # Get the joint value closest in the list
        joints = self.get_robot_q()
        joints_vel = self.get_robot_dq()
        joint_acc = self.get_robot_ddq()
        joint_torque = self.get_robot_torque()


        proximity = 0.05
        deviation = np.pi / 6
        proximity_reward = 1/self.actuated_joints

        scaling_proximity_reward = (deviation * np.sqrt(2 * np.pi)) * proximity_reward

        punishment = 0

        # punishment += 10.0 # Alive bonus
        # print("joints_vel: ", joints_vel)
        # print("Alive bonus: ", 20.0)



        for i in range(len(joints)):  
            # punishment -= abs(np.power(joints[i], 3) / 2)
            # print(f"joint {i} - Punishemnt 0: ", -abs(np.power(joints[i], 2)))
            
            # Make a normal distribution around 0
            # proximity_func = (np.exp(-np.power(joints[i] / (2 * deviation), 2)) / (deviation * np.sqrt(2 * np.pi))) * scaling_proximity_reward

            # Linear proximity function
            proximity_func = (1 - (abs(joints[i] - target) / (np.pi)))*proximity_reward


            punishment += proximity_func


            # # Proximity based punishment
            # if joints[i] > 0:

            #     punishment -= max(0, np.sign(joints_vel[i]))* abs(joints[i]) * abs(joints_vel[i]) # Punish if positive
            #     # Positive velocity is moving in the wrong direction
            #     # Negative velocity is moving in the right direction
            #     # print(f"joint {i} - Punishemnt 1: ", max(0, joints_vel[i])*10)

            # else:
            #     punishment += min(0, np.sign(joints_vel[i]))* abs(joints[i]) * abs(joints_vel[i]) # Punish if negative
            #     # Positive velocity is moving in the right direction
            #     # Negative velocity is moving in the wrong direction
            #     # print(f"joint {i} - Punishemnt 2: ", min(0, joints_vel[i])*10)


            # At border penalty
            penalty_deviation = np.pi/10
            penalty_highest_val = 1/self.actuated_joints


            gaussian_penalty = np.exp(-np.power((abs(joints[i] - target) - self.out_of_bounds) / (2 * penalty_deviation), 2)) / (penalty_deviation * np.sqrt(2 * np.pi))
            gaussian_penalty *= penalty_highest_val * (penalty_deviation * np.sqrt(2 * np.pi))
            
            punishment -= abs(gaussian_penalty)


            gaussian_penalty = np.exp(-np.power((abs(joints[i] - target) + self.out_of_bounds) / (2 * penalty_deviation), 2)) / (penalty_deviation * np.sqrt(2 * np.pi))
            gaussian_penalty *= penalty_highest_val * (penalty_deviation * np.sqrt(2 * np.pi))
            punishment -= abs(gaussian_penalty)

                # print(f"joint {i} - Boundary punishment: ", -1000)



            # fast_punishment = 0
            # # Punish for going too fast with robot joints:
            # if joints_vel[i] > 20:
            #     fast_punishment -= abs(joints_vel[i]) * 0.01
            

            # # Punish for large acceleration
            # if joint_acc[i] > 5:
            #     fast_punishment -= abs(joint_acc[i]) * 0.01

            # punishment -= fast_punishment


            if i >= self.actuated_joints - 1:
                break




        # print(f"joint {i} - Total: Punishment: ", punishment)
        # input()
        return punishment





    def _d(self) -> bool:

        
        joints = self.get_robot_q()


        for i in range(len(joints)):
            if abs(joints[i]) > self.out_of_bounds: # If the joint is outside the range
                return True

            if i >= self.actuated_joints - 1:
                break



        # If timeout
        if self.step_number > self.episode_len:
            return True

        


        return False


    def _get_obs(self) -> np.ndarray:
        joint = self.get_robot_q()
        joint_vel = self.get_robot_dq()
        joint_acc = self.get_robot_ddq()

        # Order as [joint1 pos, joint1 vel, joint2 pos, joint2 vel, ...]
        observation = np.array([])
        for i in range(self.actuated_joints):
            observation = np.append(observation, joint[i])
            #observation = np.append(observation, joint_vel[i])
            # observation = np.append(observation, joint_acc[i])

    
        return observation


    def step(
        self,
        a: Union[np.ndarray, list, torch.Tensor],
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Placeholder function.

        Returns:
            Tuple: A tuple.
        """

        # Insert Manual control in the RL algorithm instead
        pose_1 = [0,0,0,0,0,0]
        pose_2 = [1, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2]
        pose_3 = [2.64, -1.38, 2.2, 3.64, -np.pi/2, 0]
        pose_4 = [3.89, -np.pi/2, 0, -np.pi/2, 0, 0]

        action_list = [pose_1, pose_2, pose_3, pose_4]
        time_marks = [0, 200, 400, 600, 800]



        # At given time marks set the action of the joint to the corresponding torque.

        for i, val in enumerate(action_list):
                if self.step_number >= time_marks[i]:
                    for j, joint_val in enumerate(action_list[i]):
                        a[j] = joint_val

        

        max_val = np.pi*2
        min_val = -np.pi*2


        action_padding = [0]*7 # Pad non actuated values
        

        for i in range(self.actuated_joints):
            action_padding[i] = max(min_val, min(max_val, a[i])) # Saturated values.
        
        action = np.array(action_padding)



        self.do_simulation(action,self.frame_skip)
        
        state_action = np.concatenate((self._get_obs(), action))
        self.state_actions.append(state_action)


        self.step_number += 1

        reward = self._r()
        done = self._d()
        obs = self._get_obs()


        robot_joints = self.get_robot_q()
        robot_vel = self.get_robot_dq()
        robot_acc = self.get_robot_ddq()
        # Log the joints
        for i in range(self.actuated_joints):
            self.log_current_joints[i].append(robot_joints[i])
            self.log_current_vel[i].append(robot_vel[i])
            self.log_current_acc[i].append(robot_acc[i])

            self.log_actions[i].append(max(min_val, min(max_val, action[i])))

        self.log_rewards.append(reward)



        infos =  {"Robot joint 0":  torch.tensor(robot_joints[0]),
                  }

        if self.render_mode == "human":
            self.render()
        
        truncated = self.step_number > self.episode_len


        return obs, reward, done, truncated, infos



    def reset_model(
        self, 
        set_random_seed: int = -1
    ):     
        
        np.random.seed(set_random_seed)
        
        self.all_steps += self.step_number

        if self.args.plot_data:
            if  (self.episode_count - 1) % self.plot_data_after_count == 0:
                self._plot_robot_data()



        if self.args.save_state_actions:

            # Ensure the directory exists before saving the .npy file
            directory = f"./state_action_data/{self.name}"
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory if it doesn't exist


            np.save(f"./state_action_data/{self.name}/all_state_actions.npy", np.array(self.state_actions))

            # Calculate the number of datasets (entries in self.state_actions)
            num_datasets = len(self.state_actions)
            
            # Write metadata to a .txt file, overwriting the old file
            with open(f"./state_action_data/{self.name}/metadata.txt", "w") as file:
                file.write(f"Number of datasets: {num_datasets}\n")


        self.log_current_joints = [[] for _ in range(self.actuated_joints)]
        self.log_current_vel = [[] for _ in range(self.actuated_joints)]
        self.log_current_acc = [[] for _ in range(self.actuated_joints)]
        self.log_actions = [[] for _ in range(self.actuated_joints)]
        self.log_rewards = []
        self.episode_count += 1
        self.step_number = 0
    

        #noise_max = np.deg2rad(5)
        noise_max_pos = 0.3
        noise_max_vel = 0.2

        noise_max_pos = np.pi/4 # Remove noise for debug
        noise_max_vel = np.pi/4 # Remove noise for debug
        
        sign = 1 if np.random.random() < 0.5 else -1
        #Random join positions
        rjp = sign * (np.random.rand(6) * noise_max_pos)
        rjv = sign * (np.random.rand(6) * noise_max_vel)


            



        home_pos_robot = np.array([2.8+rjp[0], -1.5708+rjp[1], np.pi/2+rjp[2], -1.5708+rjp[3], -1.5708+rjp[4], -np.pi/2+rjp[5]])
        home_gripper = np.array([0, 0, 8.00406330e-02 + 0.1, 0, 1, 0, 0])
        home_gripper_separate = np.array([0]*8)


        # Set the position of the boxes
        home_pos_boxes = np.array([])
        table_position_x = [0.5 + 0.03*2, 0.95 - 0.03*2]
        table_position_y = [0.1 + 0.03*2, 0.7 - 0.03*2]

        table_height = 8.00406330e-02 + 0.1 # Drop it 0.1 meters from the table


        boxes = ["r1", "r2", "r3", "g1", "b1"]
        for name in boxes:
            rand_x = np.random.random() * (table_position_x[1] - table_position_x[0]) + table_position_x[0]
            rand_y = np.random.random() * (table_position_y[1] - table_position_y[0]) + table_position_y[0]
            home_pos_boxes = np.append(home_pos_boxes, [rand_x,rand_y, table_height, 1, 0, 0, 0])



        Q_list = np.concatenate((home_pos_robot, home_gripper, home_gripper_separate, home_pos_boxes))
        #Q_list = home_pos_robot

        Q_vel_list = np.array([0]*len(self.data.qvel))

        for i in range(len(rjv)): # Add random noise to the joint velocities
            Q_vel_list[i] = rjv[i]


        self.set_state(Q_list, Q_vel_list)

        attach(
            self.data,
            self.model,
            "attach",
            "2f85",
            self.ur5e.T_world_base @ self.ur5e.get_ee_pose(),
        )

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
        ax.set_title(f"Episode: {self.episode_count} - steps episode start: {self.all_steps} - Normalized values")
        ax.set_ylim(-1.1, 1.1)
        ax.axhline((np.pi/2)/self.out_of_bounds, color='black', linestyle='--', linewidth=0.8, label="target Line")

        # 243 optimal reward and 1/250 is reward scaling factor
        ax.axhline(1, color='blue', linestyle='--', linewidth=0.8, label="Optimal reward")



        
        self.time_list = np.linspace(0, len(self.log_current_joints[0])*self.model.opt.timestep, len(self.log_current_joints[0]))


        for i in range(self.actuated_joints):
            ax.plot(self.time_list, np.array(self.log_current_joints[i])/self.out_of_bounds,    c=clrs[i], label=f"joint: {i}")
            #ax.plot(self.time_list, np.array(self.log_current_vel[i])/self.out_of_bounds,       c=clrs[i + self.actuated_joints], label=f"velocities: {i}")
            #ax.plot(self.time_list, np.array(self.log_current_acc[i])/np.pi*2,                  c=clrs[i + self.actuated_joints*2], label=f"Accelerations: {i}")
            #ax.plot(self.time_list, np.array(self.log_actions[i])/self.out_of_bounds,c=clrs[i + self.actuated_joints],  label=f"actions: {i}", linewidth=0.5)




        ax.plot(self.time_list, np.array(self.log_rewards), c=clrs[6], label="rewards", linewidth=1.3, linestyle="dotted")

        
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




    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]