import mujoco as mj
import numpy as np
import spatialmath as sm
import spatialmath.base as smb

from ctrl.base_ctrl import BaseController
from robots.base_robot import BaseRobot


class OpSpace(BaseController):
    def __init__(self, args, robot: BaseRobot) -> None:
        self._args = args
        self._data = robot._data
        self._model = robot._model

        # Cartesian impedance control gains.
        self.impedance_pos = np.asarray([10000.0, 10000.0, 10000.0])  # [N/m]
        # self.impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
        self.impedance_ori = np.asarray([5000.0, 5000.0, 5000.0])  # [Nm/rad]
        # self.impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

        # Joint impedance control gains.
        self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0])

        # Damping ratio for both Cartesian and joint impedance control.
        self.damping_ratio = 1.0

        # Gains for the twist computation. These should be between 0 and 1. 0 means no
        # movement, 1 means move the end-effector to the target in one integration step.
        self.Kpos: float = 0.95

        # Gain for the orientation component of the twist computation. This should be
        # between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
        # orientation in one integration step.
        self.Kori: float = 0.95

        # Integration timestep in seconds.
        self.integration_dt: float = 1.0

        # Compute damping and stiffness matrices.
        self.damping_pos = self.damping_ratio * 2 * np.sqrt(self.impedance_pos)
        self.damping_ori = self.damping_ratio * 2 * np.sqrt(self.impedance_ori)
        self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori], axis=0)
        self.Kd = np.concatenate([self.damping_pos, self.damping_ori], axis=0)
        self.Kd_null = self.damping_ratio * 2 * np.sqrt(self.Kp_null)

        self.jac = np.zeros((6, self._model.nv))
        self.M_inv = np.zeros((self._model.nv, self._model.nv))
        self.Mx = np.zeros((6, 6))

        # Pre-allocate numpy arrays.
        self.twist = np.zeros(6)
        self.error_quat = np.zeros(4)

        self.robot = robot
        self.T_target: sm.SE3 = robot.fk(self.robot.q)
        self.q0 = self.robot.q

        # self.site_quat = smb.r2q(self.T_target.R)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)

    def step(self) -> None:
        """
        Perform a control step to compute the control signal.

        This method computes the control signal based on the current state of the
        robot and the target pose, and applies the control signal to the robot.

        Returns:
            np.ndarray: Control signal (torques) to be applied to the robot's joints.
        """

        # get the .tcp pose in base frame
        T_base_tcp: sm.SE3 = self.robot.fk(self.robot.q)

        # spatial velocity (aka twist).
        dx = self.T_target.t - T_base_tcp.t
        self.twist[:3] = self.Kpos * dx / self.integration_dt

        mj.mju_mat2Quat(self.site_quat, T_base_tcp.R.flatten())
        mj.mju_negQuat(self.site_quat_conj, self.site_quat)
        mj.mju_mulQuat(self.error_quat, smb.r2q(self.T_target.R), self.site_quat_conj)
        mj.mju_quat2Vel(self.twist[3:], self.error_quat, self.integration_dt)
        self.twist[3:] *= self.Kori / self.integration_dt

        # Compute generalized forces.
        tau = (
            self.robot.J.T
            @ self.robot.Mx
            @ (self.Kp * self.twist - self.Kd * (self.robot.J @ self.robot.dq))
        )

        # Add joint task in nullspace for overactuated manipulators.
        if self.robot.info.n_joints > self.robot.info.n_actuators:
            Jbar = np.linalg.inv(self.robot.Mq) @ self.robot.J.T @ self.robot.Mx
            ddq = self.Kp_null * (self.q0 - self.robot.q) - self.Kd_null * self.robot.dq
            tau += (np.eye(self.robot.info.n_joints) - self.robot.J.T @ Jbar.T) @ ddq

        # Add gravity compensation.
        if self._args.gravity_comp:
            tau += self.robot.c

        # Set the control signal and step the simulation.
        tau = np.clip(tau, *self.robot.info.actuator_limits)

        return tau
