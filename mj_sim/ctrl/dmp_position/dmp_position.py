from __future__ import division, print_function

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import spatialmath as sm

from ctrl.dmp_position.canonical_system import CanonicalSystem
from utils.math import calculate_rotation_between_vectors, normalize_vector


class DMPPosition:
    def __init__(
        self,
        n_bfs=100,
        alpha: float = 100,
        beta: float = None,
        cs_alpha=None,
        cs=None,
        roto_dilatation=False,
    ):
        """
        Initialize the DMPPosition controller.

        Parameters
        ----------
        - n_bfs (int): Number of basis functions.
        - alpha (float): Scaling factor for the spring-damper system.
        - beta (float): Damping factor.
        - cs_alpha: Alpha parameter for the canonical system.
        - cs: Canonical system object.
        - roto_dilatation (bool): Flag to enable rotation dilatation.
        """
        self.n_bfs = n_bfs
        self.alpha = alpha
        self.beta = beta if beta is not None else self.alpha / 4
        self.cs: CanonicalSystem = (
            cs
            if cs is not None
            else CanonicalSystem(
                alpha=cs_alpha if cs_alpha is not None else self.alpha / 2
            )
        )

        # Centres of the Gaussian basis functions
        self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, self.n_bfs))

        # Variance of the Gaussian basis functions
        self.h = 1.0 / np.gradient(self.c) ** 2
        # self.h = 0.0025
        # phi_c = 0.5
        # x_min = 0.01
        # k = - 4 * np.log(phi_c) / (np.log(x_min) ** 2)
        # self.h = k * n_bfs ** 2 / (self.c ** 2)

        # Scaling factor
        self.Dp = np.identity(3)

        # Initially weights are zero (no forcing term)
        self.w = np.zeros((3, self.n_bfs))

        # Initial- and goal positions
        self._p0 = np.zeros(3)
        self._gp = np.zeros(3)

        self._p0_train = np.zeros(3)
        self._gp_train = np.zeros(3)

        self._R_fx = np.identity(3)

        # Reset
        self.p = self._p0.copy()
        self.dp = np.zeros(3)
        self.ddp = np.zeros(3)
        self.train_p = None
        self.train_d_p = None
        self.train_dd_p = None
        self.dt = None
        self.tau = None
        self.ts = None

        self._roto_dilatation = roto_dilatation

    def step(
        self,
        x: float,
        dt: float,
        tau: float,
        force_disturbance: np.ndarray = np.array([0, 0, 0]),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a single step of the DMP controller.

        Parameters
        ----------
        - x: Phase variable.
        - dt: Time step size.
        - tau: Movement duration scaling factor.
        - force_disturbance: External force disturbance.

        Returns
        ----------
        - Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing position, velocity, and acceleration.
        """

        def fp(xj):
            psi = np.exp(-self.h * (xj - self.c) ** 2)
            return self.Dp.dot(self.w.dot(psi) / psi.sum() * xj)

        # DMP system acceleration
        self.ddp = (
            self.alpha * (self.beta * (self._gp - self.p) - tau * self.dp)
            + self._R_fx @ fp(x)
            + force_disturbance
        ) / tau**2

        # Integrate acceleration to obtain velocity
        self.dp += self.ddp * dt

        # Integrate velocity to obtain position
        self.p += self.dp * dt

        return self.p, self.dp, self.ddp

    def rollout(
        self, ts: float, tau: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a rollout of the DMP trajectory.

        Parameters
        ----------
        - ts: Time vector.
        - tau: Movement duration scaling factor.

        Returns
        ----------
        - Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing positions, velocities, and accelerations.
        """
        self.reset()

        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts)  # Differential time vector

        n_steps = len(ts)
        p = np.empty((n_steps, 3))
        dp = np.empty((n_steps, 3))
        ddp = np.empty((n_steps, 3))

        for i in range(n_steps):
            p[i], dp[i], ddp[i] = self.step(x[i], dt[i], tau[i])

        return p, dp, ddp

    def reset(self) -> None:
        """
        Reset the DMP to the initial state.
        """
        self.p = self._p0.copy()
        self.dp = np.zeros(3)
        self.ddp = np.zeros(3)

    def train(self, positions: np.ndarray, ts: float, tau: float) -> None:
        """
        Train the DMP with a given trajectory.

        Parameters
        ----------
        - positions (np.ndarray): Positions of the trajectory.
        - ts (np.ndarray): Time vector.
        - tau (float): Movement duration scaling factor.
        """
        p = positions

        # Sanity-check input
        if len(p) != len(ts):
            raise RuntimeError(f"len(p) != len(ts) | {len(p)} != {len(ts)}")

        # Initial- and goal positions
        self._p0 = p[0]
        self._gp = p[-1]

        self._p0_train = p[0]
        self._gp_train = p[-1]

        # Differential time vector
        dt = np.gradient(ts)[:, np.newaxis]

        # Scaling factor
        # self.Dp = np.diag(self.gp - self.p0)
        # Dp_inv = np.linalg.inv(self.Dp)
        Dp_inv = np.identity(3)

        # Desired velocities and accelerations
        d_p = np.gradient(p, axis=0) / dt
        dd_p = np.gradient(d_p, axis=0) / dt

        # Integrate canonical system
        x = self.cs.rollout(ts, tau)

        # Set up system of equations to solve for weights
        def features(xj):
            psi = np.exp(-self.h * (xj - self.c) ** 2)
            return xj * psi / psi.sum()

        def forcing(j):
            return Dp_inv.dot(
                tau**2 * dd_p[j]
                - self.alpha * (self.beta * (self._gp - p[j]) - tau * d_p[j])
            )

        A = np.stack([features(xj) for xj in x])
        f = np.stack([forcing(j) for j in range(len(ts))])

        # Least squares solution for Aw = f (for each column of f)
        self.w = np.linalg.lstsq(A, f, rcond=None)[0].T

        # Cache variables for later inspection
        self.train_p = p
        self.train_d_p = d_p
        self.train_dd_p = dd_p

    def set_trained(
        self, w: np.ndarray, c: np.ndarray, h: np.ndarray, y0: np.ndarray, g: np.ndarray
    ) -> None:
        """
        Set the trained parameters of the DMP.

        Parameters
        ----------
        - w (np.ndarray): Weights for the forcing term.
        - c (np.ndarray): Centres of the Gaussian basis functions.
        - h (np.ndarray): Variance of the Gaussian basis functions.
        - y0 (np.ndarray): Initial position.
        - g (np.ndarray): Goal position.
        """
        self.w = w
        self.c = c
        self.h = h
        self._p0 = y0
        self._gp = g

        # Scaling factor
        self.Dp = np.diag(self._gp - self._p0)

    def _update_goal_change_parameters(self):
        """
        Update parameters when the goal position changes.
        """
        # print("Updating goal rotation parameters")
        self._sg = np.linalg.norm(self._gp_train - self._p0_train) / np.linalg.norm(
            self._gp - self._p0
        )

        v_new = np.array(self._gp) - np.array(self._p0)
        v_new = normalize_vector(v_new)
        v_old = np.array(self._gp_train) - np.array(self._p0_train)
        v_old = normalize_vector(v_old)
        self._R_fx = calculate_rotation_between_vectors(v_old, v_new)
        # print("Position fx rotation: ", self._R_fx)

    @property
    def gp(self) -> np.ndarray:
        """
        Get the goal position.
        """
        return self._gp

    @gp.setter
    def gp(self, new_goal_position: np.ndarray) -> None:
        """
        Set the goal position and update parameters if rotation dilatation is enabled.

        Parameters
        ----------
        - value (np.ndarray): New goal position.
        """
        self._gp = new_goal_position
        if self._roto_dilatation:
            self._update_goal_change_parameters()

    @property
    def p0(self) -> np.ndarray:
        """
        Get the initial position.
        """
        return self._p0

    @p0.setter
    def p0(self, new_start_position: np.ndarray) -> None:
        """
        Set the initial position and update parameters if rotation dilatation is enabled.

        Parameters
        ----------
        - value (np.ndarray): New initial position.
        """
        self._p0 = new_start_position
        if self._roto_dilatation:
            self._update_goal_change_parameters()

    def load(
        self,
        Traj: Union[str, sm.SE3],
        dt: float = 0.05,
        csv_headers: List[str] = [
            "target_TCP_pose_0",
            "target_TCP_pose_1",
            "target_TCP_pose_2",
        ],
    ) -> None:
        """
        Load and train the DMP with a trajectory.

        Parameters
        ---------------
         - Traj (sm.SE3): Trajectory as an SE3 object.
         - dt (float): Time step size.
        """

        if isinstance(Traj, str):
            df = pd.read_csv(Traj)
            positions = df[csv_headers].to_numpy()
        else:
            positions = np.array([pose.t for pose in Traj])

        self.dt = dt
        self.tau = (len(positions)) * self.dt
        self.ts = np.arange(0, self.tau, self.dt)

        # in case of rounding error for arange positions and ts might be off my one. So here we correct it
        if len(positions) != len(self.ts):
            positions = (
                positions[: len(self.ts)]
                if len(positions) > len(self.ts)
                else positions
            )
            self.ts = (
                self.ts[: len(positions)] if len(self.ts) > len(positions) else self.ts
            )

        self.train(positions, self.ts, self.tau)
        self.reset()
        self.cs.reset()

    def is_trained(self) -> bool:
        """
        Check if the DMP has been trained.

        Returns
        ----------
        - bool: True if the DMP has been trained, False otherwise.
        """
        if self.dt is None and self.tau is None and self.ts is None:
            return False
        else:
            return True
