import spatialmath as sm

from ctrl.base_ctrl import BaseController
from robots import BaseRobot
from utils.mj import get_mocap_pose, set_mocap_pose
from utils.rtb import make_tf


class Mocap(BaseController):
    """
    A controller class for handling motion capture (Mocap) data and robot transformations.
    """

    def __init__(self, args: dict, robot: BaseRobot, mocap_name: str = "mocap") -> None:
        """
        Initialize the Mocap controller.

        Parameters
        ----------
        args : Namespace
            Arguments for the controller.
        robot : Robot
            Robot instance with model and data attributes.
        """
        self._args = args
        self._data = robot._data
        self._model = robot._model
        self.robot = robot
        self._name = mocap_name

        self.tcp_id = self.robot.tcp_id
        self.T_target = robot.get_ee_pose()

        self._T_mocap_tcp = self._T_world_mocap.inv() @ self._T_world_tcp

    @property
    def name(self) -> str:
        return self._name

    def step(self) -> None:
        """
        Perform a single step of the Mocap controller, updating the robot pose based on mocap data.
        """
        T_mocap = self.T_target @ self._T_mocap_tcp.inv()
        set_mocap_pose(self._data, self._model, self._name, T_mocap)

    @property
    def _T_world_tcp(self) -> sm.SE3:
        """
        Get the transformation matrix from the world frame to the TCP (Tool Center Point).

        Returns
        -------
        sm.SE3
            Transformation matrix from world to TCP.
        """
        return make_tf(pos=self._data.site("tcp").xpos, ori=self._data.site("tcp").xmat)

    @property
    def _T_world_mocap(self) -> sm.SE3:
        """
        Get the transformation matrix from the world frame to the mocap system.

        Returns
        -------
        sm.SE3
            Transformation matrix from world to mocap system.
        """
        return get_mocap_pose(self._data, self._model, "mocap")
