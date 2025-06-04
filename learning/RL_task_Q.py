import argparse


from skrl.agents.torch.q_learning import Q_LEARNING, Q_LEARNING_DEFAULT_CONFIG
import time
from learning.Environments.SKRL_UR_env_Q_learning import URSim_SKRL_env
from skrl.envs.wrappers.torch import wrap_env
import numpy as np
from skrl.models.torch import Model
from skrl.models.torch import TabularMixin
import torch
from skrl.trainers.torch import SequentialTrainer

from gymnasium.spaces import Discrete, Box

import matplotlib.pyplot as plt




# define the model
class EpilonGreedyPolicy(TabularMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs=1, epsilon=0.1):

        Model.__init__(self, observation_space, action_space, device)
        TabularMixin.__init__(self, num_envs)

        # Handle both Discrete and Box action spaces
        if isinstance(action_space, Discrete):
            self.num_actions = action_space.n
        elif isinstance(action_space, Box):
            self.num_actions = int(action_space.high[0] - action_space.low[0] + 1)
        else:
            raise ValueError("Unsupported action space type")

        self.epsilon = epsilon
        self.q_table = torch.ones((num_envs, self.num_observations, self.num_actions), dtype=torch.float32)

    def compute(self, inputs, role):
        
        states = inputs["states"].type(torch. int64) 
        actions = torch.argmax(self.q_table[torch.arange(self.num_envs).view(-1, 1), states],
                               dim=-1, keepdim=True).view(-1,1)


        if role == "policy":
            indexes = (torch.rand(states.shape[0], device=self.device) < self.epsilon).nonzero().view(-1)
            if indexes.numel():
                actions[indexes] = torch.randint(self.num_actions, (indexes.numel(), 1), device=self.device)


        return actions, {}



def main():
    debug = False


    max_episodes = int(100)
    max_time_per_episode = int(300)
    max_timesteps = max_episodes * max_time_per_episode



    parser = argparse.ArgumentParser(
        description="MuJoCo simulation of manipulators and controllers."
    )

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


    args, _ = parser.parse_known_args()

    print(" > Loaded simulation configs:")
    for key, value in vars(args).items():
        print(f"\t{key:30}{value}")


    # instantiate the agent
    # (assuming a defined environment <env>)
    env = URSim_SKRL_env(args = args, render_mode="rgb_array")
    # it will check your custom environment and output additional warnings if needed
    env = wrap_env(env)


    # Create the model table in the required dictionary format
    models = {
        "policy": EpilonGreedyPolicy(env.observation_space, 
                                     env.action_space,
                                     env.device)
    }



    # adjust some configuration if necessary
    cfg_agent = Q_LEARNING_DEFAULT_CONFIG.copy()

    # instantiate the agent
    # (assuming a defined environment <env>)
    agent = Q_LEARNING(models=models,
                    memory=None,
                    cfg=cfg_agent,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=env.device)

    # assuming there is an environment called 'env'
    # and an agent or a list of agents called 'agents'

    # create a sequential trainer
    cfg = {"timesteps": args.max_timesteps, 
           "headless": args.render,
           #"environment_info": "robot_joints",
           }
    trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg)

    # train the agent(s)
    # trainer.train()
    # trainer.train()
    trainer.train()
    # evaluate the agent(s)
    # trainer.eval()

    # sim = DRLSim_UR(args)
    # sim.run()


if __name__ == "__main__":
    main()