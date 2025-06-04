


from typing import Tuple, Union

import mujoco as mj
import numpy as np
import torch


from utils.mj import get_joint_q, get_joint_dq, get_joint_ddq, get_joint_torque, get_body_pose, get_body_vel
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os

import matplotlib
matplotlib.use('Agg')


import pickle

from typing import Tuple, Union



from spatialmath.base import r2q, qqmul, q2r, eul2r, qconj


from robots.ur_robot import URRobot
from collections import deque


from utils.rtb import make_tf



from shapely import Polygon



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
        observation_space_size = 55


        # # change shape of observation to your observation space size
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(observation_space_size,), 
            dtype=np.float64
        )
        
        super().__init__(
            model_path=os.path.abspath(args.scene_path),
            frame_skip=14,
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
        
        



        # Log error value between target and current position

        # Log position of the block, and joint value to see if its been flicked

        self.block_joint_value = []




        self.episode_count = 0
        self.name = name

        self.previous_action = [-1]*self.actuated_joints 


        self.robot_joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]



        # # Inserting the reward function for the AIRL algorithm
        # self.AIRL_trainer = AIRL(
        #     state_shape=self.observation_space.shape,
        #     action_shape=self.action_space.shape,
        #     device=torch.device("cuda"), #  "cuda" if args.cuda else "cpu"
        #     seed=np.random.randint(0, 1000),
        #     #rollout_length=args.rollout_length
        # )


        # # trained Reward function for discriminator
        # disc_model_path =  "/home/peter/OneDrive/Skrivebord/Uni stuff/Masters/masters_code/mj_sim/learning/airl_UR/Saved_models/TRAINED_16_HOURS_AIRL_UR_FLICK_TASK_25-02-19_16-36-05"
        # disc_model_candidate = "latest"

        # print("Loading trained discriminator")

        # discrim_state_dict = torch.load(disc_model_path + "/" + disc_model_candidate + "_AIRL_discrim.pkl")
        # self.AIRL_trainer.disc.load_state_dict(discrim_state_dict)

        # discrim_state_dict = torch.load(disc_model_path + "/" + disc_model_candidate + "_AIRL_discrim_optim.pkl")
        # self.AIRL_trainer.optim_disc.load_state_dict(discrim_state_dict)

        # Load start state:
        qpos_dir = os.path.abspath("learning/config_files/C1_start_state/qpos_start_state.pkl")
        qvel_dir = os.path.abspath("learning/config_files/C1_start_state/qvel_start_state.pkl")

        qpos_file = open(qpos_dir, "rb")
        qvel_file = open(qvel_dir, "rb")

        self.Q_list = pickle.load(qpos_file)
        self.Q_vel_list = pickle.load(qvel_file)







        # Load expert data [AIRL]
        path_expert_data_1 = os.path.abspath("learning/airl_UR/expert_demo/C1_16_25-05-08_13-30-39URSim_SKRL_env_PPO/expert_memory.pkl")

        expert_data_file_1 = open(path_expert_data_1, "rb")
        self.expert_memory_1 = pickle.load(expert_data_file_1)
        

        # DEBUG; This is for getting the observation space size.. 
        # But mujoco must initialize before its possible to run it
        self.reset()
        # print("Observation test: ", self._get_obs())
        assert len(self._get_obs()) == observation_space_size, f"Observation space size is not correct. Expected {observation_space_size}, got {len(self._get_obs())}"

    def define_cloth_joints(self):
        return ['c1_free_joint_1', 'c1_free_joint_2', 'c1_free_joint_3', 'c1_free_joint_4', 'c1_free_joint_5', 'c1_free_joint_6', 'c1_free_joint_7', 'c1_free_joint_8', 'c1_free_joint_9', 'unnamed_composite_0J0_1_0', 'unnamed_composite_0J1_1_0', 'unnamed_composite_0J2_1_0', 'unnamed_composite_0J0_1_1', 'unnamed_composite_0J1_1_1', 'unnamed_composite_0J2_1_1', 'unnamed_composite_0J0_1_2', 'unnamed_composite_0J1_1_2', 'unnamed_composite_0J2_1_2', 'unnamed_composite_0J0_1_3', 'unnamed_composite_0J1_1_3', 'unnamed_composite_0J2_1_3', 'unnamed_composite_0J0_1_4', 'unnamed_composite_0J1_1_4', 'unnamed_composite_0J2_1_4', 'unnamed_composite_0J0_1_5', 'unnamed_composite_0J1_1_5', 'unnamed_composite_0J2_1_5', 'unnamed_composite_0J0_1_6', 'unnamed_composite_0J1_1_6', 'unnamed_composite_0J2_1_6', 'unnamed_composite_0J0_1_7', 'unnamed_composite_0J1_1_7', 'unnamed_composite_0J2_1_7', 'unnamed_composite_0J0_1_8', 'unnamed_composite_0J1_1_8', 'unnamed_composite_0J2_1_8', 'unnamed_composite_0J0_2_0', 'unnamed_composite_0J1_2_0', 'unnamed_composite_0J2_2_0', 'unnamed_composite_0J0_2_1', 'unnamed_composite_0J1_2_1', 'unnamed_composite_0J2_2_1', 'unnamed_composite_0J0_2_2', 'unnamed_composite_0J1_2_2', 'unnamed_composite_0J2_2_2', 'unnamed_composite_0J0_2_3', 'unnamed_composite_0J1_2_3', 'unnamed_composite_0J2_2_3', 'unnamed_composite_0J0_2_4', 'unnamed_composite_0J1_2_4', 'unnamed_composite_0J2_2_4', 'unnamed_composite_0J0_2_5', 'unnamed_composite_0J1_2_5', 'unnamed_composite_0J2_2_5', 'unnamed_composite_0J0_2_6', 'unnamed_composite_0J1_2_6', 'unnamed_composite_0J2_2_6', 'unnamed_composite_0J0_2_7', 'unnamed_composite_0J1_2_7', 'unnamed_composite_0J2_2_7', 'unnamed_composite_0J0_2_8', 'unnamed_composite_0J1_2_8', 'unnamed_composite_0J2_2_8', 'unnamed_composite_0J0_3_0', 'unnamed_composite_0J1_3_0', 'unnamed_composite_0J2_3_0', 'unnamed_composite_0J0_3_1', 'unnamed_composite_0J1_3_1', 'unnamed_composite_0J2_3_1', 'unnamed_composite_0J0_3_2', 'unnamed_composite_0J1_3_2', 'unnamed_composite_0J2_3_2', 'unnamed_composite_0J0_3_3', 'unnamed_composite_0J1_3_3', 'unnamed_composite_0J2_3_3', 'unnamed_composite_0J0_3_4', 'unnamed_composite_0J1_3_4', 'unnamed_composite_0J2_3_4', 'unnamed_composite_0J0_3_5', 'unnamed_composite_0J1_3_5', 'unnamed_composite_0J2_3_5', 'unnamed_composite_0J0_3_6', 'unnamed_composite_0J1_3_6', 'unnamed_composite_0J2_3_6', 'unnamed_composite_0J0_3_7', 'unnamed_composite_0J1_3_7', 'unnamed_composite_0J2_3_7', 'unnamed_composite_0J0_3_8', 'unnamed_composite_0J1_3_8', 'unnamed_composite_0J2_3_8', 'unnamed_composite_0J0_4_0', 'unnamed_composite_0J1_4_0', 'unnamed_composite_0J2_4_0', 'unnamed_composite_0J0_4_1', 'unnamed_composite_0J1_4_1', 'unnamed_composite_0J2_4_1', 'unnamed_composite_0J0_4_2', 'unnamed_composite_0J1_4_2', 'unnamed_composite_0J2_4_2', 'unnamed_composite_0J0_4_3', 'unnamed_composite_0J1_4_3', 'unnamed_composite_0J2_4_3', 'unnamed_composite_0J0_4_4', 'unnamed_composite_0J1_4_4', 'unnamed_composite_0J2_4_4', 'unnamed_composite_0J0_4_5', 'unnamed_composite_0J1_4_5', 'unnamed_composite_0J2_4_5', 'unnamed_composite_0J0_4_6', 'unnamed_composite_0J1_4_6', 'unnamed_composite_0J2_4_6', 'unnamed_composite_0J0_4_7', 'unnamed_composite_0J1_4_7', 'unnamed_composite_0J2_4_7', 'unnamed_composite_0J0_4_8', 'unnamed_composite_0J1_4_8', 'unnamed_composite_0J2_4_8', 'unnamed_composite_0J0_5_0', 'unnamed_composite_0J1_5_0', 'unnamed_composite_0J2_5_0', 'unnamed_composite_0J0_5_1', 'unnamed_composite_0J1_5_1', 'unnamed_composite_0J2_5_1', 'unnamed_composite_0J0_5_2', 'unnamed_composite_0J1_5_2', 'unnamed_composite_0J2_5_2', 'unnamed_composite_0J0_5_3', 'unnamed_composite_0J1_5_3', 'unnamed_composite_0J2_5_3', 'unnamed_composite_0J0_5_4', 'unnamed_composite_0J1_5_4', 'unnamed_composite_0J2_5_4', 'unnamed_composite_0J0_5_5', 'unnamed_composite_0J1_5_5', 'unnamed_composite_0J2_5_5', 'unnamed_composite_0J0_5_6', 'unnamed_composite_0J1_5_6', 'unnamed_composite_0J2_5_6', 'unnamed_composite_0J0_5_7', 'unnamed_composite_0J1_5_7', 'unnamed_composite_0J2_5_7', 'unnamed_composite_0J0_5_8', 'unnamed_composite_0J1_5_8', 'unnamed_composite_0J2_5_8', 'unnamed_composite_0J0_6_0', 'unnamed_composite_0J1_6_0', 'unnamed_composite_0J2_6_0', 'unnamed_composite_0J0_6_1', 'unnamed_composite_0J1_6_1', 'unnamed_composite_0J2_6_1', 'unnamed_composite_0J0_6_2', 'unnamed_composite_0J1_6_2', 'unnamed_composite_0J2_6_2', 'unnamed_composite_0J0_6_3', 'unnamed_composite_0J1_6_3', 'unnamed_composite_0J2_6_3', 'unnamed_composite_0J0_6_4', 'unnamed_composite_0J1_6_4', 'unnamed_composite_0J2_6_4', 'unnamed_composite_0J0_6_5', 'unnamed_composite_0J1_6_5', 'unnamed_composite_0J2_6_5', 'unnamed_composite_0J0_6_6', 'unnamed_composite_0J1_6_6', 'unnamed_composite_0J2_6_6', 'unnamed_composite_0J0_6_7', 'unnamed_composite_0J1_6_7', 'unnamed_composite_0J2_6_7', 'unnamed_composite_0J0_6_8', 'unnamed_composite_0J1_6_8', 'unnamed_composite_0J2_6_8', 'unnamed_composite_0J0_7_0', 'unnamed_composite_0J1_7_0', 'unnamed_composite_0J2_7_0', 'unnamed_composite_0J0_7_1', 'unnamed_composite_0J1_7_1', 'unnamed_composite_0J2_7_1', 'unnamed_composite_0J0_7_2', 'unnamed_composite_0J1_7_2', 'unnamed_composite_0J2_7_2', 'unnamed_composite_0J0_7_3', 'unnamed_composite_0J1_7_3', 'unnamed_composite_0J2_7_3', 'unnamed_composite_0J0_7_4', 'unnamed_composite_0J1_7_4', 'unnamed_composite_0J2_7_4', 'unnamed_composite_0J0_7_5', 'unnamed_composite_0J1_7_5', 'unnamed_composite_0J2_7_5', 'unnamed_composite_0J0_7_6', 'unnamed_composite_0J1_7_6', 'unnamed_composite_0J2_7_6', 'unnamed_composite_0J0_7_7', 'unnamed_composite_0J1_7_7', 'unnamed_composite_0J2_7_7', 'unnamed_composite_0J0_7_8', 'unnamed_composite_0J1_7_8', 'unnamed_composite_0J2_7_8', 'unnamed_composite_0J0_8_0', 'unnamed_composite_0J1_8_0', 'unnamed_composite_0J2_8_0', 'unnamed_composite_0J0_8_1', 'unnamed_composite_0J1_8_1', 'unnamed_composite_0J2_8_1', 'unnamed_composite_0J0_8_2', 'unnamed_composite_0J1_8_2', 'unnamed_composite_0J2_8_2', 'unnamed_composite_0J0_8_3', 'unnamed_composite_0J1_8_3', 'unnamed_composite_0J2_8_3', 'unnamed_composite_0J0_8_4', 'unnamed_composite_0J1_8_4', 'unnamed_composite_0J2_8_4', 'unnamed_composite_0J0_8_5', 'unnamed_composite_0J1_8_5', 'unnamed_composite_0J2_8_5', 'unnamed_composite_0J0_8_6', 'unnamed_composite_0J1_8_6', 'unnamed_composite_0J2_8_6', 'unnamed_composite_0J0_8_7', 'unnamed_composite_0J1_8_7', 'unnamed_composite_0J2_8_7', 'unnamed_composite_0J0_8_8', 'unnamed_composite_0J1_8_8', 'unnamed_composite_0J2_8_8']
            




            
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

        edge_indices = [[0,0], [0,8], [4,8], [8,8], [8,0], [4,0]]
        
        positions = []

        z_position_error = 0
        desired_z_position = 0.1768

        
        for val in edge_indices:
            cloth_body_name = "unnamed_composite_0B" + str(val[0]) + "_" + str(val[1])

            cloth_pose = get_body_pose(self.data, self.model, cloth_body_name)

            positions.append([cloth_pose.t[0],
                            cloth_pose.t[1]])
            z_position_error += abs(cloth_pose.t[2] - desired_z_position)



        cloth_Polygon = Polygon(positions)
        area = cloth_Polygon.area # [meters]


        # normalize the range [0 - 0.136], to [0 - 1] (full area is nearly impossible to achieve)
        area_normalized = min(area / 0.110, 1)

        #normalize error between [0 and 1] from [0, 3]

        z_position_error_average  = z_position_error/3

        performance_metric = area_normalized - z_position_error_average


        reward = performance_metric



        return reward





    def _d(self) -> bool:

        

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


        # Add cartesian position of the end effector
        ee_pose = self.ur5e.get_ee_pose()

        cartesian_ee_pos = ee_pose.t
        cartesian_ee_rot = r2q(ee_pose.R)

        observation = np.append(observation, cartesian_ee_pos)
        observation = np.append(observation, cartesian_ee_rot)

        # Cartesian space coordinates for each point on the cloth.



        #get the vertices of the cloth to get the area. put both the vertices and the area into the observation state.


        edge_indices = [[0,0], [0,8], [4,8], [8,8], [8,0], [4,0]]
        


        
        for val in edge_indices:
            cloth_body_name = "unnamed_composite_0B" + str(val[0]) + "_" + str(val[1])

            cloth_pose = get_body_pose(self.data, self.model, cloth_body_name)

    
            observation = np.append(observation, cloth_pose.t[0])
            observation = np.append(observation, cloth_pose.t[1])
            observation = np.append(observation, cloth_pose.t[2])




        for val in edge_indices:
            cloth_body_name = "unnamed_composite_0B" + str(val[0]) + "_" + str(val[1])

            cloth_vel = get_body_vel(self.data, self.model, cloth_body_name)

                            
            observation = np.append(observation, cloth_vel.t[0])
            observation = np.append(observation, cloth_vel.t[1])
            observation = np.append(observation, cloth_vel.t[2])



        # Add simple reward function to the observation state

        # reward = self._r()
        # observation = np.append(observation, reward)


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


        Kp = 30


        # Calculate the desired velocity based on a p controller



        q_tilde = np.array(input_val) - np.array(self.get_robot_q())


        u = Kp*(q_tilde)



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


        # a = self.expert_memory_1[self.step_number+150*self.episode_count][1]

        # saturate action values 
        for i, val in enumerate(a):
            a[i] = min(max(a[i], -5), 5)




        self.do_simulation(a, self.frame_skip)

        self.step_number += 1


        reward = self._r()
        done = self._d()
        obs = self._get_obs()   




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



        self.episode_count += 1
        self.step_number = 0


        self.set_state(self.Q_list, self.Q_vel_list)
        self.memory_single_episode = deque()


        self.record_mode = False

        return  self._get_obs()



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