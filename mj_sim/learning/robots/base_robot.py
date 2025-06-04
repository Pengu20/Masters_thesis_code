from abc import ABC, abstractmethod
from typing import List, Union

import mujoco as mj
import numpy as np
import spatialmath as sm

from utils.mj import (
    RobotInfo,
    get_actuator_ctrl,
    get_body_pose,
    get_joint_ddq,
    get_joint_dq,
    get_joint_q,
    get_number_of_dof,
    site_name2id,
)


class BaseRobot(ABC):
    pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the robot.
        """
        raise NotImplementedError("property 'name' must be implemented in robot.")

    @property
    @abstractmethod
    def info(self) -> RobotInfo:
        raise NotImplementedError(
            "property 'info' of type RobotInfo must be implemented in robot."
        )

    @property
    def T_world_base(self) -> sm.SE3:
        return get_body_pose(self._data, self._model, self.info._base_body_name)

    @property
    def ctrl(self) -> List[float]:
        """
        The control signal sent to the robot's actuator(s).
        """
        return np.array(
            [get_actuator_ctrl(self._data, self._model, an) for an in self.info._actuator_names]
        )

    @property
    def Jp(self) -> np.ndarray:
        """
        Get the position Jacobian in base frame.

        Returns
        ----------
                Position Jacobian as a numpy array.
        """
        sys_J = np.zeros((6, get_number_of_dof(self._model)))

        mj.mj_jacSite(
            self._model,
            self._data,
            sys_J[:3],
            sys_J[3:],
            site_name2id(self._model, "tcp"),
        )
        self._Jp = self.T_world_base.R @ sys_J[:3, self.info._dof_indxs]
        return self._Jp

    @property
    def Jo(self) -> np.ndarray:
        """
        Get the orientation Jacobian in base frame.

        Returns
        ----------
                Orientation Jacobian as a numpy array.
        """
        # Jacobian.
        sys_J = np.zeros((6, get_number_of_dof(self._model)))

        mj.mj_jacSite(
            self._model,
            self._data,
            sys_J[:3],
            sys_J[3:],
            site_name2id(self._model, "tcp"),
        )
        self._Jo = self.T_world_base.R @ sys_J[3:]
        return self._Jo[:, self.info._dof_indxs]

    @property
    def J(self) -> np.ndarray:
        """
        Get the full Jacobian in base frame.

        Returns
        ----------
                Full Jacobian as a numpy array.
        """
        return np.vstack((self.Jp, self.Jo))

    @property
    def c(self) -> np.ndarray:
        """
        bias force: Coriolis, centrifugal, gravitational
        """
        return self._data.qfrc_bias[self.info._dof_indxs]

    @property
    def Mq(self) -> np.ndarray:
        """
        Getter property for the inertia matrix M(q) in joint space.

        Returns
        ----------
        - numpy.ndarray: Symmetric inertia matrix in joint space.
        """
        sys_Mq_inv = np.zeros(
            (get_number_of_dof(self._model), get_number_of_dof(self._model))
        )

        mj.mj_solveM(
            self._model, self._data, sys_Mq_inv, np.eye(get_number_of_dof(self._model))
        )
        Mq_inv = sys_Mq_inv[np.ix_(self.info._dof_indxs, self.info._dof_indxs)]

        if abs(np.linalg.det(Mq_inv)) >= 1e-2:
            self._Mq = np.linalg.inv(Mq_inv)
        else:
            self._Mq = np.linalg.pinv(Mq_inv, rcond=1e-2)
        return self._Mq

    @property
    def Mx(self) -> np.ndarray:
        """
        Getter property for the inertia matrix M(q) in task space.

        Returns
        ----------
        - numpy.ndarray: Symmetric inertia matrix in task space.
        """
        Mx_inv = self.J @ np.linalg.inv(self.Mq) @ self.J.T

        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            self._Mx = np.linalg.inv(Mx_inv)
        else:
            self._Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        return self._Mx

    @abstractmethod
    def set_ctrl(self, x: Union[list, np.ndarray]) -> None:
        """
        This function sends the control signal to the simulated robot.

        Args
        ----------
                x (Union[list, np.ndarray]): control signal
        """
        raise NotImplementedError("method 'set_ctrl' must be implemented in robot.")

    @abstractmethod
    def step(self) -> None:
        """
        Perform a step in the controller.

        This method calls the `step()` method of the controller object and
        before doing so it checks if there are any tasks to be performed in
        the robot task queue
        """
        raise NotImplementedError("method 'step' must be implemented in robot.")

    @property
    def q(self) -> np.ndarray:
        """
        Get the joint positions.

        Returns
        ----------
                Joint positions as a numpy array.
        """
        # return [get_joint_q(self._data, self._model, jn) for jn in self._joint_ids]
        q = np.array(
            [get_joint_q(self._data, self._model, jn) for jn in self.info._joint_ids]
        ).flatten()
        return q

    @property
    def dq(self) -> np.ndarray:
        """
        Get the joint velocities.

        Returns
        ----------
                Joint velocities as a numpy array.
        """
        # return [get_joint_dq(self._data, self._model, jn) for jn in self._joint_ids]
        dq = np.array(
            [get_joint_dq(self._data, self._model, jn) for jn in self.info._joint_ids]
        ).flatten()
        return dq

    @property
    def ddq(self) -> np.ndarray:
        """
        Get the joint accelerations.

        Returns
        ----------
                Joint accelerations as a numpy array.
        """
        # return [get_joint_ddq(self._data, self._model, jn) for jn in self._joint_ids]
        ddq = np.array(
            [get_joint_ddq(self._data, self._model, jn) for jn in self.info._joint_ids]
        ).flatten()
        return ddq
