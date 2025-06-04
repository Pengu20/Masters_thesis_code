from typing import Union

import mujoco as mj
import mujoco.viewer
import numpy as np

from robots.base_robot import BaseRobot
from utils.mj import RobotInfo, get_actuator_ctrl, site_name2id, actuator_name2id


class InvPenMotor(BaseRobot):
    def __init__(self, args, data, model) -> None:
        self._args = args
        self._data = data
        self._model = model
        self.dt = self._model.opt.timestep
        self._actuator_ids = [actuator_name2id(self._model, "motor")]

        self._info = RobotInfo(self._data, self._model, self.name)

    def step(self):
        """
        Perform a step in the controller.

        This method calls the `step()` method of the controller object.
        """
        self.set_ctrl()

    @property
    def info(self) -> RobotInfo:
        return self._info

    @property
    def args(self):
        return self._args

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def name(self) -> np.ndarray:
        return "inverted_pendulum"

    @property
    def ctrl(self) -> np.ndarray:
        return [get_actuator_ctrl(self._data, "motor")]

    def set_ctrl(self, x: Union[list, np.ndarray]) -> None:
        self._data.ctrl[self._actuator_ids] = x[self._actuator_ids]


# J, Jo, Jp, Mq, Mx, c, ctrl, ddq, dq, name