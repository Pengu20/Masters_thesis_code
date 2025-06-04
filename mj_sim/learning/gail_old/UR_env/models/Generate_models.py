


from skrl.agents.torch.q_learning import Q_LEARNING, Q_LEARNING_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG


from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG

# import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

# import the skrl components to build the RL system
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from torch.optim import adamw

from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from skrl.resources.noises.torch import GaussianNoise

import numpy as np



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



# --------------------- MODELS FOR TD3 ---------------------

class MLP_policy(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.actuated_joints = self.action_space.shape[0]


        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.actuated_joints))


    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"]], dim=1)), {}


class MLP_critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}
    

    # For TD3

def make_TD3_models(observation_space, action_space, device):

    models = {
        "policy": MLP_policy(observation_space, action_space, device),
        "target_policy": MLP_policy(observation_space, action_space, device),
        "critic_1": MLP_critic(observation_space, action_space, device),
        "critic_2": MLP_critic(observation_space, action_space, device),
        "target_critic_1": MLP_critic(observation_space, action_space, device),
        "target_critic_2": MLP_critic(observation_space, action_space, device),
    }


        # initialize models' parameters (weights and biases)
    for item in models.items():
        models[item[0]].init_parameters(method_name="normal_", mean=0.0, std=0.1)

    return models
# --------------------- MODELS FOR TD3 ---------------------




# --------------------- MODELS FOR PPO ---------------------

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)


        self.actuated_joints = self.action_space.shape[0]
        self.max_mean = np.pi*2


        self.all_joints_net = nn.Sequential(
                                    nn.Linear(self.num_observations, 32),
                                    nn.LeakyReLU(),
                                    nn.Linear(32, 32),
                                    nn.LeakyReLU(),
                                    nn.Linear(32, self.actuated_joints),
                                    nn.Tanh()
                                    )
        
        

        

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))




    def compute(self, inputs, role):
        

        output_mean = self.all_joints_net(inputs["states"]) * self.max_mean



        return output_mean, self.log_std_parameter, {}


        #return self.max_mean*output_all_joints_mean, self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)



        self.all_joints_net = nn.Sequential(
                                    nn.Linear(self.num_observations, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 1))






    def compute(self, inputs, role):

        output_all_joints = self.all_joints_net(inputs["states"])


        return output_all_joints, {}





# --------------------- MODELS FOR PPO ---------------------





def get_PPO_agent_SKRL(env, models, write_interval = 1000):
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 4096  # memory_size
    cfg["learning_epochs"] = 128
    cfg["mini_batches"] = 8
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-6
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}


    cfg["grad_norm_clip"] = 0.5
    cfg["ratio_clip"] = 0.3
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = False
    cfg["entropy_loss_scale"] = 0.001
    cfg["value_loss_scale"] = 0.3
    cfg["kl_threshold"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": env.device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}


    cfg["rewards_shaper"] = None

    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = write_interval
    cfg["experiment"]["checkpoint_interval"] = 10000
    cfg["experiment"]["directory"] = "runs/torch/UR_robot_PPO"



    memory =  RandomMemory(memory_size=4096, num_envs=env.num_envs, device=env.device)
    agent = PPO(models=models,
                    memory=memory,
                    cfg=cfg,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=env.device)


    return agent



def get_TD3_agent(env, models, write_interval = 100):
    cfg = TD3_DEFAULT_CONFIG.copy()

    cfg["gradient_steps"] = 1
    cfg["batch_size"] = 256
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.005
    cfg["actor_learning_rate"] = 1e-6
    cfg["critic_learning_rate"] = 1e-6
    # cfg["learning_rate_scheduler"] = adamw
    # cfg["learning_rate_scheduler_kwargs"] = {}

    cfg["state_preprocessor"] = None
    cfg["state_preprocessor_kwargs"] = {}

    cfg["random_timesteps"] = 1000
    cfg["learning_starts"] = 1000

    cfg["grad_norm_clip"] = 0

    cfg["exploration"] = {
        "noise": GaussianNoise(0, 0.1, device=env.device),
        "initial_scale": 1.0,
        "final_scale": 1e-3,
        "timesteps": None,
    }
    cfg = TD3_DEFAULT_CONFIG.copy()
        

    cfg["policy_delay"] = 2
    cfg["smooth_regularization_noise"] = GaussianNoise(0, 0.2, device=env.device)
    cfg["smooth_regularization_clip"] = 0.5

    cfg["rewards_shaper"] = None

    cfg["experiment"]["write_interval"] = write_interval
    cfg["experiment"]["checkpoint_interval"] = 100000
    cfg["experiment"]["directory"] = "runs/torch/UR_robot_TD3"


    memory =  RandomMemory(memory_size=20000, num_envs=env.num_envs, device=env.device)
    agent = TD3(models=models,
                    memory=memory,
                    cfg=cfg,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=env.device)


    return agent