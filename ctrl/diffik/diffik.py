import mujoco as mj
import numpy as np
import spatialmath as sm
import spatialmath.base as smb

from ctrl.base_ctrl import BaseController
from robots import BaseRobot
from utils.mj import get_joint_dim, site_name2id


class DiffIk(BaseController):
    def __init__(self, args, robot: BaseRobot) -> None:
        self._args = args
        self._data = robot._data
        self._model = robot._model

        # Integration timestep in seconds. This corresponds to the amount of time the joint
        # velocities will be integrated for to obtain the desired joint positions.
        self.integration_dt: float = 1.0

        # Damping term for the pseudoinverse. This is used to prevent joint velocities from
        # becoming too large when the Jacobian is close to singular.
        self.damping: float = 1e-4

        # Whether to enable gravity compensation.
        if self._args.gravity_comp:
            body_ids = robot.info._body_ids
            self._model.body_gravcomp[body_ids] = 1.0

        # Maximum allowable joint velocity in rad/s. Set to 0 to disable.
        self.max_angvel = 0.0

        # preallocate arrays in memory
        self.diag = self.damping * np.eye(6)
        self.error = np.zeros(6)
        self.error_pos = self.error[:3]
        self.error_ori = self.error[3:]
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

        self.tcp_id = site_name2id(self._model, "tcp")
        self.robot = robot

        self.T_target = robot.get_ee_pose()

        self.i = 0

    def step(self) -> None:
        """
        Perform a differential inverse kinematics control step.

        This method computes the joint positions required to minimize the error between
        the current end-effector pose and the target pose using differential inverse
        kinematics. The computed joint positions are then used to control the robot.

        Returns
        ----------
            np.ndarray: Joint positions to be applied to the robot.
        """

        # get the .tcp pose in base frame
        T_base_tcp: sm.SE3 = self.robot.get_ee_pose()

        # compute position error: p0 - p1
        self.error_pos[:] = self.T_target.t - T_base_tcp.t

        # compute orientation error:
        # 	compute Q_tcp
        mj.mju_mat2Quat(self.site_quat, T_base_tcp.R.flatten())
        # 	compute Q_tcp_conj
        mj.mju_negQuat(self.site_quat_conj, self.site_quat)
        # 	compute Q_error = Q_target * Q_tcp_conj (quaternion product!)
        mj.mju_mulQuat(self.error_quat, smb.r2q(self.T_target.R), self.site_quat_conj)
        # 	Convert Q_error (corresponding to orientation difference) to 3D velocity
        mj.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)

        # Solve system of equations: J @ dq = error.
        dq = self.robot.J.T @ np.linalg.solve(
            self.robot.J @ self.robot.J.T + self.diag, self.error
        )

        # Scale down joint velocities if they exceed maximum.
        if self.max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.max_angvel:
                dq *= self.max_angvel / dq_abs_max

        # Integrate joint velocities to obtain joint positions.
        q = self.robot.q

        qpos = np.zeros(shape=len(self._data.qpos))

        indexes = self.robot.info._joint_indxs

        qpos_dims = [
            get_joint_dim(self._data, self._model, jid)
            for jid in self.robot.info._joint_ids
        ]

        for i, indx in enumerate(indexes):
            s = slice(indx, indx + qpos_dims[i])
            qpos[s] = q[i]

        qvel = self._data.qvel

        qvel[self.robot.info._dof_indxs] = dq

        mj.mj_integratePos(self._model, qpos, qvel, self.integration_dt)

        q = qpos[indexes]

        # clip control q values to joint limits
        q = np.clip(q, *self.robot.info.actuator_limits)

        # Set the control signal.
        return q
