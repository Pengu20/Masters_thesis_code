from collections import deque
from typing import Union

import mujoco as mj
import mujoco.viewer
import numpy as np
import spatialmath as sm

from robots import BaseRobot
from utils.mj import (
    get_site_pose,
    set_actuator_ctrl,
    RobotInfo
)


class Twof85(BaseRobot):
    def __init__(self, args, data, model) -> None:
        self._args = args
        self._data = data
        self._model = model
        self.dt = self._model.opt.timestep

        self._info = RobotInfo(self._data, self._model, self.name)

        self.tcp_id = model.site("tcp").id

        self._task_queue = deque()

    def step(self) -> None:
        """
        Perform a step in the controller.

        This method advances the controller by one step. It dequeues the next control target from the
        task queue, if available, and sets it as the current control target using the `set_ctrl()` method.
        This is typically used in a control loop to iteratively apply control signals to the robot.

        Returns
        -------
        None
        """
        if self._task_queue:
            ctrl_target = self._task_queue.popleft()
            self.set_ctrl(ctrl_target)


    @property
    def info(self) -> RobotInfo:
        """
        Get detailed information about the robot.

        This property returns an instance of the `RobotInfo` class, which provides comprehensive
        details about the robot's structure and components. This includes information on the robot's
        bodies, joints, actuators, and geometries, among other attributes. The `RobotInfo` instance
        can be used to access various properties such as the number of joints, actuator limits, joint
        limits, and more.

        Returns
        -------
        RobotInfo
            An object containing detailed information about the robot's configuration and components.
        """
        return self._info


    @property
    def args(self):
        """
        Get the arguments for the simulation.

        Returns
        -------
        Any
            Arguments for the simulation.
        """
        return self._args


    @property
    def data(self) -> mj.MjData:
        """
        Get the MuJoCo data object.

        This property returns the MuJoCo data object associated with the robot. The `mj.MjData`
        object contains the dynamic state of the simulation, including positions, velocities, forces,
        and other simulation-specific data for the robot.

        Returns
        -------
        mj.MjData
            The MuJoCo data object containing the dynamic state of the simulation.
        """
        return self._data


    @property
    def model(self) -> mj.MjModel:
        """
        Get the MuJoCo model object.

        This property returns the MuJoCo model object associated with the robot. The `mj.MjModel`
        object represents the static model of the simulation, including the robot's physical
        structure, joint configurations, and other model-specific parameters.

        Returns
        -------
        mj.MjModel
            The MuJoCo model object representing the robot's static configuration.
        """
        return self._model


    @property
    def name(self) -> str:
        """
        Get the name of the gripper.

        This property returns the name assigned to the gripper robot within the system. The name is
        typically used for identification purposes within the control and simulation environment.

        Returns
        -------
        str
            The name of the gripper robot.
        """
        return "2f85"


    def get_ee_pose(self) -> sm.SE3:
        """
        Get the end-effector pose.

        This method retrieves the pose of the robot's end-effector, specifically the TCP (Tool Center
        Point). The pose is returned as an instance of `sm.SE3`, representing the position and
        orientation of the end-effector in 3D space.

        Returns
        -------
        sm.SE3
            The pose of the robot's end-effector (TCP) in 3D space.
        """
        return get_site_pose(self._data, "tcp")


    def set_ctrl(self, x: Union[list, np.ndarray]) -> None:
        """
        Set the control signal for the robot.

        This method applies the specified control signal to the robot's actuators. The control
        signal can be provided as a list or a NumPy array. The first actuator in the `actuator_names`
        list of the `RobotInfo` instance is targeted, and the control signal is applied to it.

        Parameters
        ----------
        x : Union[list, np.ndarray]
            The control signal to apply, specified as a list or NumPy array.

        Returns
        -------
        None
        """
        set_actuator_ctrl(self._data, self.model, self.info.actuator_names[0], x[0])
