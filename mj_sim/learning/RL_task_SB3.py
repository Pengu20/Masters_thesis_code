import argparse
import os


from learning.Utility_RL_task import Bool, Callable, d, o, r, reset, step
from skrl.agents.torch.q_learning import Q_LEARNING, Q_LEARNING_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG

import time
from skrl.envs.wrappers.torch import wrap_env
import numpy as np
from skrl.models.torch import Model
from skrl.models.torch import TabularMixin
import torch


from skrl.trainers.torch import SequentialTrainer

from gymnasium.spaces import Discrete, Box

import matplotlib.pyplot as plt


from skrl.memories.torch import RandomMemory

# import the skrl components to build the RL system
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils import set_seed



# ----------- Import peter stuff -----------
# from learning.SKRL_UR_env_Q_learning import URSim_SKRL_env
from learning.Environments.SKRL_UR_env_PPO import URSim_SKRL_env
from learning.models.Generate_models import get_PPO_agent_SKRL, get_TD3_agent, make_TD3_models, Policy, Value

from stable_baselines3 import PPO
import gymnasium as gym

# -------------------------------------------


def main():
    debug = False


    max_episodes = int(10000)
    max_time_per_episode = int(100)
    max_timesteps = max_episodes * max_time_per_episode




    parser = argparse.ArgumentParser(
        description="MuJoCo simulation of manipulators and controllers."
    )
    parser.add_argument(
        "--scene_path",
        type=str,
        default="learning/scenes/RL_task.xml",
        help="Path to the XML file defining the simulation scene.",
    )

    parser.add_argument(
        "--load_model",
        type=str,
        default="",
        help="loads model from relative path",
    )

    parser.add_argument(
        "--plot_data",
        type=bool,
        default=False,
        help="Path to the XML file defining the simulation scene.",
    )

    parser.add_argument(
        "--show_site_frames",
        type=Bool,
        default=False,
        help="Flag to display the site frames in the simulation visualization.",
    )
    parser.add_argument(
        "--gravity_comp",
        type=Bool,
        default=True,
        help="Enable or disable gravity compensation in the controller.",
    )
    parser.add_argument(
        "--manual",
        type=Bool,
        default=False,
        help="Enable or disable manual control of the simulation.",
    )
    parser.add_argument(
        "--render_size",
        type=float,
        default=0.1,
        help="Size of the rendered frames in the simulation visualization.",
    )

    parser.add_argument(
        "--eval_episodes",
        default=1 if debug else 1,
        type=int,
        help="Number of episodes for evaluation during training.",
    )
    parser.add_argument(
        "--eval_freq",
        default=10 if debug else int(max_episodes/50),
        type=int,
        help="Frequency (in timesteps) of evaluations during training.",
    )
    parser.add_argument(
        "--seed_eval",
        type=int,
        default=110,
        help="Random seed for evaluation to ensure reproducibility.",
    )
    parser.add_argument(
        "--render",
        default=False,
        type=Bool,
        help="Enable or disable rendering during the simulation.",
    )
    parser.add_argument(
        "--episode_timeout",
        default=1 if debug else int(max_time_per_episode),
        type=int,
        help="Maximum duration (in time steps) for each episode before timeout.",
    )
    parser.add_argument(
        "--expl_noise",
        type=float,
        default=0.1,
        help="Percentage of exploration noise to add to the actor's actions.",
    )
    parser.add_argument(
        "--seed_train",
        type=int,
        default=100,
        help="Random seed for training to ensure reproducibility.",
    )

    parser.add_argument(
        "--t", type=int, default=0, help="Time step for the simulation."
    )

    parser.add_argument(
        "--reward_function",
        type=Callable,
        default=r,
        help="Callable function to compute the reward.",
    )
    parser.add_argument(
        "--step_function",
        type=Callable,
        default=step,
        help="Callable function to perform a simulation step.",
    )
    parser.add_argument(
        "--done_function",
        type=Callable,
        default=d,
        help="Callable function to check if the episode is done.",
    )
    parser.add_argument(
        "--reset_function",
        type=Callable,
        default=reset,
        help="Callable function to reset the environment.",
    )
    parser.add_argument(
        "--observation_function",
        type=Callable,
        default=o,
        help="Callable function to get the current observation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10 if debug else 20,
        help="Batch size for training the reinforcement learning algorithm.",
    )
    parser.add_argument(
        "--start_timesteps",
        type=int,
        # default=10 if debug else int(0.01 * max_timesteps),
        default=10 if debug else max_episodes*0.01,
        help="Number of initial timesteps for exploration before training starts.",
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=100 if debug else max_timesteps,
        help="Maximum number of timesteps for the entire training process.",
    )
    parser.add_argument(
        "--exit_on_done",
        type=Bool,
        default=True,
        help="Flag to exit the simulation when the training is done.",
    )

    parser.add_argument(
        "--state0",
        type=str,
        default="default",
        help="init state.",
    
    )

    parser.add_argument(
        "--training_name",
        type=str,
        default="",
        help="name of the training session that will be made"
    )


    Training_model = "PPO"

    args, _ = parser.parse_known_args()

    print(" > Loaded simulation configs:")
    for key, value in vars(args).items():
        print(f"\t{key:30}{value}")


    # instantiate the agent
    # (assuming a defined environment <env>)
    environment_name = time.strftime("%y-%m-%d_%H-%M-%S") + "URSim_SKRL_env_" + Training_model
    env = URSim_SKRL_env(args = args, name=environment_name, render_mode="rgb_array")
    # it will check your custom environment and output additional warnings if needed




    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_UR_ARM")

    # evaluate the agent(s)
    # trainer.eval()

    # sim = DRLSim_UR(args)
    # sim.run()


if __name__ == "__main__":
    main()
