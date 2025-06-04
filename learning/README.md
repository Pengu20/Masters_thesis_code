# Learning Framework

This framework implements a modular deep reinforcement learning (DRL) setup, designed to easily integrate new algorithms, tasks, and environments. It supports learning agents that interact with customizable robotic environments, defined using MJCF files.

## Requirements 

To run this submodule, the following dependencies are required:
- **Python** (version >= 3.8)
- **MuJoCo** for simulating robotic environments (compatible with MJCF files).
- **PyTorch** for deep learning.
- Additional Python packages (install using `pip install -r requirements.txt`).

Ensure MuJoCo is installed and properly configured in your system (see MuJoCo's documentation for installation guidelines).

## Infrastructure

- [`algos`](learning/algos) contains various learning agents. Currently supported algorithms include:
  - **[TD3](learning/algos/td3/README.md)** (Twin Delayed DDPG), based on the [TD3 paper](https://arxiv.org/pdf/1802.09477).
  - More algorithms can be added by inheriting from the base algorithm class.
  
- [`robots`](learning/robots) contains robotic models for training. These are MJCF (MuJoCo XML) files representing physical robots. Custom robots can be added to this folder and referenced in the learning loop. [Inverted pendulum example](learning/robots/inv_pen_motor.py).
  
- [`scenes`](learning/scenes) contains MJCF files for describing the environments (e.g., terrains, obstacles) in which the agents will operate. These scenes provide the context for the learning tasks and can be modified or extended. [Inverted pendulum example](learning/scenes/inverted_pendulum.xml)

## Running a Demo

To run a demo of the DRL framework using the existing configuration:
```bash
python -m learning.demo
```
This will launch a predefined task using a selected learning algorithm and environment. Modify the demo file to customize the task, agent, or environment.

## Setting up New Learning

To set up a new learning process, follow these steps:

### 1. Adding a New Learning Algorithm
Create a new learning algorithm by inheriting from the base algorithm class for compatibility with the learning loop. This ensures your new agent fits into the existing infrastructure seamlessly.

### 2. Defining a New Learning Task
If you want to define a new task, implement the following base functions:
- `step`: This function should define how the agent interacts with the environment during each timestep.
- `done`: This function should check whether the task is complete (e.g., reaching a goal, exhausting timesteps).
- `reward`: This function should define the reward structure based on the agent's performance in the environment.
- `reset`: This function should reset the environment and agent's state to the initial condition for a new episode.

Make sure to integrate these new tasks into the existing framework and modify configuration files accordingly.

For examples on these see [helpers.py](learning/helpers.py)