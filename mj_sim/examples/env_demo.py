

import queue
import time
from threading import Lock
import argparse

import glfw
import mujoco as mj
import mujoco.viewer
import spatialmath as sm

from robots.ur_robot import URRobot
from robots.twof85 import Twof85
from sensors import Camera
from utils.mj import attach, detach, get_mj_data, get_mj_model, body_name2id
from utils.helpers import Bool
from utils.rtb import make_tf

import numpy as np

class PickAndPlaceSim:
    def __init__(self, args) -> None:
        """
        Initialize the MjSim class with the given arguments.

        This constructor sets up the MuJoCo simulation environment, including loading the model,
        creating the robot interface, and initializing the camera. It also registers the control
        loop callback and attaches the robot's tool to the flange.

        Parameters
        ----------
        args : argparse.Namespace
            The arguments used to configure the simulation, typically passed from a command-line interface.

        Returns
        ----------
        None
        """
        self._args = args
        self._scene_path = self._args.scene_path
        self._model = get_mj_model(scene_path=self._scene_path)
        self._data = get_mj_data(self._model)
        self.frequency = 500.0
        self.dt = 1 / self.frequency
        self._model.opt.timestep = self.dt

        # lock to set state
        self._lock = Lock()

        # UR5e robot interface
        self.ur5e = URRobot(args, self._data, self._model, robot_type=URRobot.Type.UR5e)

        # UR5e robot interface
        # self.gripper = Twof85(args, self._data, self._model)

        # camera in scene
        # self.cam = Camera(self._args, self._model, self._data, "cam")


        # register control loop to trigger at each timestep
        mj.set_mjcb_control(self.controller_callback)



        # attach(
        #     self._data,
        #     self._model,
        #     "cloth_to_robot",
        #     "unnamed_composite_0J0_0_1",
        #     self.ur5e.T_world_base @ self.ur5e.get_ee_pose()
        # )


        print("To run the demo, press the SPACE button!")



    def get_box_position(self):
            # Get robot base position
            self.ur5e_base_cartesian = [0.22331901, 0.37537452, 0.08791326]
            self.ur5e_rot_quat = [ -0.19858483999999996, -0.00311175, 0.0012299899999999998, 0.98007799]
            self.ur5e_base_SE3 = make_tf(self.ur5e_base_cartesian, self.ur5e_rot_quat)
        

            block_id = body_name2id(self._model, "block_1")
            
            # Get block base position
            block_pos = self._data.xpos[block_id]
            block_rot = self._data.xquat[block_id]
            block_pos = np.append(block_pos, 1)


            # position above the block to pick it up in box frame
            box_shift_translation_box_space = [-0.05,   0.3,   -0.075,  1.]

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

            return point_translated



    def keyboard_callback(self, key) -> None:
        """
        Handle keyboard input during the simulation.

        This method is triggered by keyboard events during the MuJoCo simulation.
        Depending on the key pressed, it can move the robot, print force/torque measurements,
        or capture an image from the camera.

        Parameters
        ----------
        key : int
            The key code representing the keyboard input.

        Returns
        ----------
        None
        """

        # Get position of foldable block and set it as target



        if key == glfw.KEY_SPACE:


            detach(
                self._data,
                self._model,
                "attach"
            )




            # Get the current end-effector pose of your robot (an SE3 object)
            current_ee_pose = self.ur5e.get_ee_pose()  # assuming this returns an SE3 object
            # Extract the rotation matrix from the current pose
        

            joint_vals = self.ur5e.ik(current_ee_pose)

            print(joint_vals)

            print("Hello, MuJoCo engineer!")
            #self.ur5e.move_j([0, 0, 1.0, -0.5708, -0.5708, 0])



            self.ur5e.move_l(current_ee_pose)
            #self.ur5e.move_l(self.ur5e.get_ee_pose() @ sm.SE3.Tz(0.2))





        elif key == glfw.KEY_PERIOD:
            print("UR5e force/torque measurement: ", self.ur5e.w)

        elif key == glfw.KEY_COMMA:
            print("Shoot! I just took a picture")
            self.cam.shoot()



    def controller_callback(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """
        Control callback function for MuJoCo's control loop.

        This method is called at every simulation timestep to execute the robot's control logic.
        If manual control is not enabled, the robot's control step is executed.

        Parameters
        ----------
        model : mj.MjModel
            The MuJoCo model object representing the simulation model.
        data : mj.MjData
            The MuJoCo data object containing the current state of the simulation.

        Returns
        ----------
        None
        """

        if not self._args.manual_ctrl:
            self.ur5e.step()

    def run(self) -> None:
        """
        Run the main simulation loop with the MuJoCo viewer.

        This method starts the MuJoCo viewer, manages keyboard events, synchronizes the viewer
        with the simulation, and controls the timing of each simulation step. It also renders
        the camera view if the camera is enabled.

        Returns
        ----------
        None
        """
        # in order to enable camera rendering in main thread, queue the key events
        key_queue = queue.Queue()

        with mj.viewer.launch_passive(
            model=self._model,
            data=self._data,
            key_callback=lambda key: key_queue.put(key),
        ) as viewer:
            if hasattr(self, "cam"):
                self.cam._renderer.render()

            # toggle site frame visualization.
            if self._args.show_site_frames:
                viewer.opt.frame = mj.mjtFrame.mjFRAME_SITE

            while viewer.is_running():
                step_start = time.time()

                while not key_queue.empty():
                    self.keyboard_callback(key_queue.get())

                mj.mj_step(self._model, self._data)
                viewer.sync()

                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo simulation of manipulators and controllers."
    )

    parser.add_argument(
        "--scene_path",
        type=str,
        default="learning/scenes/RL_task_flick_f3.xml",
        help="Path to the XML file defining the simulation scene.",
    )
    parser.add_argument(
        "--show_site_frames",
        type=Bool,
        default=True,
        help="Boolean flag to display the site frames in the simulation.",
    )
    parser.add_argument(
        "--gravity_comp",
        type=Bool,
        default=True,
        help="Boolean flag to enable gravity compensation.",
    )
    parser.add_argument(
        "--manual_ctrl",
        type=Bool,
        default=False,
        help="Boolean flag to enable manual control in the simulation.",
    )
    parser.add_argument(
        "--render_size",
        type=float,
        default=0.1,
        help="Size of the rendered site axes in the scene.",
    )
    parser.add_argument(
        "--state0", type=str, default="home", help="Initial state of the simulation."
    )
    parser.add_argument(
        "--cam_width",
        type=int,
        default=640,
        help="Width of the camera view in the simulation.",
    )
    parser.add_argument(
        "--cam_height",
        type=int,
        default=480,
        help="Height of the camera view in the simulation.",
    )

    args, _ = parser.parse_known_args()

    sim = PickAndPlaceSim(args)

    sim.run()

if __name__ == "__main__":
    main()
