import argparse
import time
from threading import Lock
from typing import Tuple, Union

import mujoco as mj
import numpy as np
import torch

from robots import BaseRobot
from utils.mj import get_mj_data, get_mj_model, attach, get_joint_names, get_joint_q, get_joint_dq, get_joint_ddq
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
import os

from learning.robots.ur_robot import URRobot
from learning.robots.twof85 import Twof85

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns
import itertools


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
    def __init__(self, args, render_mode="human"):


        self.args = args
        self.resolution = 50

        self.actuated_joints = 2
        self.possible_actions = 2


        self.observation_space = Discrete(np.power(self.resolution, self.actuated_joints*2))
        


        super().__init__(
            model_path=os.path.abspath("learning/scenes/RL_task_Q_learning.xml"),
            frame_skip=3,
            observation_space=self.observation_space,
            render_mode=render_mode
        )


        self.action_space = Discrete(self.actuated_joints*self.possible_actions)

        self.step_number = 0
        self.episode_len = args.episode_timeout
        


        self.log_current_joints = [[] for _ in range(7)]
        self.log_actions = []
        self.log_rewards = []

        self.episode_count = 0
        self.name = time.strftime("%y-%m-%d_%H-%M-%S") + "URSim_SKRL_env_Q_learning" + f"_joints_{self.actuated_joints}"

        self.robot_joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]



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
        # Get the joint value closest in the list
        joints = self.get_robot_q()
        joints_vel = self.get_robot_dq()

        punishment = -0.1

        for i in range(len(joints)):  

            punishment -= abs(joints[i])*2

            # Proximity based punishment
            if joints[i] > 0:
                punishment -= max(0, joints_vel[i])*10 # Punish if positive
                # Positive velocity is moving in the wrong direction
                # Negative velocity is moving in the right direction

            else:
                punishment += min(0, joints_vel[i])*10 # Punish if negative
                # Positive velocity is moving in the right direction
                # Negative velocity is moving in the wrong direction

            if i >= self.actuated_joints - 1:
                break


        return punishment



    def _d(self) -> bool:

        joints = self.get_robot_q()
        robot_vel = self.get_robot_dq()

        # If the robot is within proximity with low enough velocity then the task is done
        Task_done = True
        for i in range(len(joints)):
            if abs(joints[i]) > 0.1 or abs(robot_vel[i]) > 0.1:
                Task_done = False

            if i >= self.actuated_joints - 1:
                break


        # Disable finish task early
        # if Task_done:
        #     print("\n\n ------------------------ Finished task early!! ------------------------ \n\n")
        #     return True


        # If timeout
        if self.step_number > self.episode_len:
            return True
        


        return False


    def _get_obs(self) -> np.ndarray:
        # 200 values for each joint
        digit_resolution = self.resolution
        max_joint_val = 2*np.pi
        min_joint_val = -2*np.pi

        observation_list = np.linspace(min_joint_val, max_joint_val, digit_resolution)
        velocity_list = np.linspace(-50, 50, digit_resolution)

        # Get the joint value closest in the list
        robot_joints_scaled_down = [0]*self.actuated_joints
        robot_joints_vel_scaled_down = [0]*self.actuated_joints
        robot_joints = self.get_robot_q()
        robot_joint_vel = self.get_robot_dq()


        for i, val in enumerate(robot_joints):
            joint_val = val
            closest_val = min(observation_list, key=lambda x:abs(x-joint_val))

            robot_joints_scaled_down[i] = np.where(observation_list == closest_val)[0][0]
                
            if i >= self.actuated_joints - 1:
                break
        

        for i,val in enumerate(robot_joint_vel):
            joint_val = val
            closest_val = min(velocity_list, key=lambda x:abs(x-joint_val))

            robot_joints_vel_scaled_down[i] = np.where(velocity_list == closest_val)[0][0]


            # print(" ---------------  Observation function:  ---------------")
            # print("Robot joints vel: ", val)
            # print("Scaled down joint velocity: ", robot_joints_vel_scaled_down[i])
            # input()
                
            if i >= self.actuated_joints - 1:
                break
        


        obs_in_list = np.concatenate((np.array(robot_joints_scaled_down), np.array(robot_joints_vel_scaled_down)))

        list2scalar_obs = 0
        for i, obs in enumerate(obs_in_list):
            list2scalar_obs += obs * self.resolution**i

        obs_int = np.array([list2scalar_obs]).astype(int)


        # obs_int =  np.array(robot_joints_scaled_down).astype(int)


        return obs_int
        



        #  ---- with gripper ---- 
        # right_driver_joint = int(robot._gripper.q[0]*digit_resolution)
        # left_driver_joint = int(robot._gripper.q[4]*digit_resolution)

        #gripper_q = np.array([right_driver_joint, left_driver_joint])
        # return np.append(robot_joints, gripper_q)


        # ---- without gripper ---- 
        return robot_joints



    def step(
        self,
        a: Union[np.ndarray, list, torch.Tensor],
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Placeholder function.

        Returns:
            Tuple: A tuple.
        """


        actions_list = [-40, 40]
        all_combinations = list(itertools.product(actions_list, repeat=self.actuated_joints))


        complete_action = [0]*7
        action_set = all_combinations[a]

        for i, vals in enumerate(action_set):
            complete_action[i] = vals


        self.do_simulation(complete_action, self.frame_skip)



        self.step_number += 1

        reward = self._r()
        done = self._d()
        obs = self._get_obs()



        robot_joints = self.data.qpos
        robot_joint_vel = self.data.qvel


        for i in range(self.actuated_joints):
            
            self.log_current_joints[i].append(robot_joints[i])




        self.log_actions.append(complete_action[0])
        self.log_rewards.append(reward)


        infos =  {"Robot joint 0":  torch.tensor(robot_joints[0]),
                  }

        if self.render_mode == "human":
            self.render()

        truncated = self.step_number > self.episode_len  # No idea what this is for'



        return obs, reward, done, truncated, infos



    def reset_model(
        self, 
    ):
        
        self._plot_robot_data()
        self.log_current_joints = [[] for _ in range(7)]
        self.log_actions = []
        self.log_rewards = []   

        self.episode_count += 1


        #noise_max = np.deg2rad(5)
        noise_max = 0 # Remove noise for debug
        sign = 1 if np.random.random() < 0.5 else -1
        noise = sign * (np.random.random() * noise_max)


        home_pos_robot = np.array([2.8, -1.5708, 0, -1.5708, -1.5708, 0])
        home_gripper = np.array([0,0, 8.00406330e-02 + 0.1, 1, 0, 0, 0])
        home_gripper_separate = np.array([0]*8)


        # Set the position of the boxes
        home_pos_boxes = np.array([])
        table_position_x = [0.5 + 0.03*2, 0.95 - 0.03*2]
        table_position_y = [0.1 + 0.03*2, 0.7 - 0.03*2]

        table_height = 8.00406330e-02 + 0.1 # Drop it 0.1 meters from the table
        rand_x = np.random.random() * (table_position_x[1] - table_position_x[0]) + table_position_x[0]
        rand_y = np.random.random() * (table_position_y[1] - table_position_y[0]) + table_position_y[0]


        boxes = ["r1", "r2", "r3", "g1", "b1"]
        for name in boxes:
            rand_x = np.random.random() * (table_position_x[1] - table_position_x[0]) + table_position_x[0]
            rand_y = np.random.random() * (table_position_y[1] - table_position_y[0]) + table_position_y[0]
            home_pos_boxes = np.append(home_pos_boxes, [rand_x,rand_y, table_height, 1, 0, 0, 0])

        Q_list = np.concatenate((home_pos_robot, home_gripper,home_gripper_separate, home_pos_boxes))

        Q_vel_list = np.array([0]*50)
        
        self.set_state(Q_list, Q_vel_list)

        # # Force acceleration to be 0
        # self._step_mujoco_simulation([0]*7, 2000)

        self.step_number = 0
        

        obs = self._get_obs()


        # Plot the joint data


        return obs

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
        clrs = sns.color_palette("husl", 5)
        ax.set_title(f"Episode {self.episode_count} - Normalized values")
        # plot all joints
        
        for i in range(self.actuated_joints):
            ax.plot(np.array(self.log_current_joints[i])/(np.pi / 2),    c=clrs[i], label=f"Joint {i}")



        ax.plot(np.array(self.log_rewards)/200,             c="red", label="reward")
        ax.legend()
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



    # def _get_reset_info(self):
    #     """
    #     SUMMARY: Gets the reset information for the environment
    #     """

    #     obs, info = self._get_obs()

    #     return info

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)