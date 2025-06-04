# Mocap Controller Documentation

The `Mocap` controller facilitates controlling a robotic system within a Mujoco simulation using motion capture data. This README elucidates the functionality and operation of the controller.

## Controller Overview

The controller's primary objective is to update the position and orientation of the robot's end-effector based on motion capture (mocap) data. It continuously receives pose information from the mocap system and applies it to the robot model within the simulation environment.

## Operational Flow

1. **Initialization**: Upon instantiation, the controller initializes necessary parameters and retrieves relevant data from the robot.

2. **Mocap Pose Retrieval**: The controller retrieves the current pose (position and orientation) of the end-effector from the motion capture system using the `get_mocap_pose()` function.

3. **Control Step**: During each control step, the controller checks if there are available mocap, poses in the buffer. If mocap poses exist, it updates the robot's end-effector pose using the `set_mocap_pose()` function.

4. **Simulation Update**: The updated end-effector pose affects the robot's state within the Mujoco simulation, influencing its behavior and interaction with the environment.

## Mathematical Basis

The controller operates based on the following principles:

- **Pose Representation**: Poses are represented using spatialmath library's `SE3` class, encapsulating both position and orientation information.

## Usage

To utilize the `Mocap` controller effectively, adhere to the following steps:

1. **Initialization**: Instantiate the controller with appropriate arguments, including references to the robot and any necessary configuration parameters.

2. **Integration**: Incorporate the controller into your control loop structure, ensuring that it is called at each iteration to synchronize the robot's state with the motion capture data.

3. **Simulation Execution**: Execute the Mujoco simulation with the integrated controller, observing how the robot's motion aligns with the captured poses.

## Conclusion

The `Mocap` controller serves as a bridge between motion capture data and robot simulation, enabling accurate and dynamic control of robotic systems within Mujoco environments. Its seamless integration facilitates various research and development tasks in robotics and related fields.
