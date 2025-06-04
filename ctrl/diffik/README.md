# DiffIk Controller Documentation

The `DiffIk` controller is a differential inverse kinematics controller used for controlling the end-effector position and orientation of a robot arm. This README explains the mathematics behind the controller's operation.

## Controller Overview

The controller aims to compute joint velocity commands (`dq`) based on the desired end-effector pose (`T_target`) and the current robot state. These joint velocity commands are then integrated over time to obtain the desired joint positions (`q`). The controller utilizes the robot's Jacobian matrix (`J`) to map end-effector errors to joint velocity commands.

## Mathematical Formulation

### Forward Kinematics

The forward kinematics function `fk` computes the pose (`T_base_tcp`) of the end-effector (TCP) in the base frame of the robot given the current joint positions (`q`). This is achieved by:

$$
\mathbf{T}^{base}_{tcp} = \text{fk}(\mathbf{q})
$$

where $\mathbf{T}^{base}_{tcp}$ is the end-effector pose in the base frame.

### Error Computation

The controller computes the position and orientation errors between the desired end-effector pose (`T_target`) and the current end-effector pose (`T_base_tcp`). The position error ($\tilde{\mathbf{t}}$) is computed as the difference in position vectors:

$$
\tilde{\mathbf{t}} = \mathbf{t}^{base}_{target} - t^{base}_{tcp}
$$

The orientation error $\tilde{\mathbf{Q}}$ is computed using quaternion algebra:

1. Compute the quaternion representation of the current end-effector orientation ($\mathbf{Q}_{tcp}$).
2. Compute the quaternion conjugate ($\mathbf{Q}_{tcp,conj}$).
3. Compute the quaternion error ($\tilde{\mathbf{Q}}$) by multiplying the desired orientation quaternion by the conjugate of the current orientation quaternion.
4. Convert the quaternion error to a 3D angular velocity.

### Jacobian-based Control

The controller solves the equation $\mathbf{J} \cdot d\mathbf{q} = d\tilde{\mathbf{x}}$, where $\mathbf{J}$ is the robot's Jacobian matrix, to compute joint velocity commands ($d\mathbf{q}$). The pseudoinverse of the Jacobian ($\mathbf{J^+}$) is computed as:

$$
\mathbf{J}^+ = (\mathbf{J} \cdot \mathbf{J}^T + \text{diag})^{-1} \cdot \mathbf{J}
$$

where $\text{diag}$ is a damping term to prevent singularities.

### Joint Velocity Clipping

Optionally, joint velocity commands are scaled down if they exceed a maximum allowable joint velocity (`max_angvel`). This prevents joint velocities from becoming too large.

### Integration

The joint velocity commands ($d\mathbf{q}$) are integrated over time using the Euler integration method to obtain the desired joint positions ($\mathbf{q}$). These joint positions are then clipped to the joint limits of the robot.

### Control Signal

Finally, the computed joint positions ($\mathbf{q}$) are set as the control signal (`_data.ctrl`) to be applied to the robot.

## Usage

To use the `DiffIk` controller, follow these steps:

1. Instantiate the controller with appropriate arguments and a reference to the robot.
2. Call the `step()` method of the controller at each control loop iteration to compute and apply the control signal.

### Acknowledgements

This work is based on the work of [Kevin Zakka](https://github.com/kevinzakka)'s implementation of [Introduction to Inverse Kinematics with
Jacobian Transpose, Pseudoinverse and Damped
Least Squares methods](https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf).
