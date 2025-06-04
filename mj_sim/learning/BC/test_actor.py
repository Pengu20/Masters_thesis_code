import torch
import numpy as np
from learning.airl_UR.utils.utils import get_entropy, log_prob_density
from torch.distributions.categorical import Categorical
import os
import gym
import pickle
import argparse
import numpy as np
from collections import deque

import torch
torch.cuda.empty_cache()
import torch.optim as optim
from tensorboardX import SummaryWriter 

from learning.airl_UR.utils.utils import *
from learning.airl_UR.utils.zfilter import ZFilter
from learning.airl_UR.model import Actor, Critic, Discriminator
from learning.airl_UR.airl import AIRL
from learning.airl_UR.train_model import train_actor_critic, train_discrim

import matplotlib.pyplot as plt
from torch import nn
from torch.nn import GaussianNLLLoss
# ----------- Import peter stuff -----------
# from learning.SKRL_UR_env_Q_learning import URSim_SKRL_env


#from learning.environments.SKRL_UR_env_PPO import URSim_SKRL_env

#Flick task environment
from learning.environments.UR_env_Flick_TASK_C1_Expert_demonstration  import URSim_SKRL_env

from learning.models.Generate_models import get_PPO_agent_SKRL, get_TD3_agent, make_TD3_models, Policy, Value
from learning.config_files.generate_config_file import generate_config_file


import time
from skrl.envs.wrappers.torch import wrap_env

# Calculating the log_probability (log_pis), of the actions taken using the GAIL repo implementation
from learning.airl_UR.utils.utils import get_entropy, log_prob_density

import copy

import scipy

parser = argparse.ArgumentParser(description='PyTorch GAIL')
parser.add_argument('--env_name', type=str, default="Hopper-v2", 
                    help='name of the environment to run')
parser.add_argument('--load_model', type=str, default=None, 
                    help='path to load the saved model')
parser.add_argument('--render', action="store_true", default=False, 
                    help='if you dont want to render, set this to False')
parser.add_argument('--gamma', type=float, default=0.99, 
                    help='discounted factor (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.95, 
                    help='GAE hyper-parameter (default: 0.98)')
parser.add_argument('--hidden_size', type=int, default=512, 
                    help='hidden unit size of actor, critic and discrim networks (default: 100)')
parser.add_argument('--learning_rate', type=float, default=8e-5, 
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3, 
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--entropy_gain_PPO', type=float, default=1e-3, 
                    help='gain for entropy of PPO (default: 1e-3)')
parser.add_argument('--critic_gain_PPO', type=float, default=0.5, 
                    help='critic gain for PPO (default: 1e-3)')
parser.add_argument('--clip_param', type=float, default=0.2, 
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--discrim_update_num', type=int, default=10, 
                    help='update number of discriminator (default: 2)')
parser.add_argument('--actor_critic_update_num', type=int, default=10, 
                    help='update number of actor-critic (default: 10)')
parser.add_argument('--total_sample_size', type=int, default=8, 
                    help='Mini batch size')
parser.add_argument('--batch_size', type=int, default=128, 
                    help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.80,
                    help='accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.80,
                    help='accuracy for suspending discriminator about generated data (default: 0.8)')
parser.add_argument('--max_iter_num', type=int, default=1000000,
                    help='maximal number of main iterations (default: 4000)')
parser.add_argument('--seed', type=int, default=np.random.randint(1e6),
                    help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


def split_dataset_index(n, train_percent, test_percent):

    data = np.arange(n)

    np.random.shuffle(data)

    # Calcola le dimensioni
    train_size = int(train_percent * n)
    test_size = int(test_percent * n)
    val_size = n - train_size - test_size  # Resto per validation

    # Split dei dati
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    val_data = data[train_size + test_size:]

    return train_data, test_data, val_data


def train_actor_BC(actor, actor_optim, states, actions):

    # Make data
    train_percent = 0.7
    test_percent = 0.2


    train_index, test_index, val_index = split_dataset_index(n=len(states), train_percent=train_percent, test_percent=test_percent)






    training_input = states[train_index]
    training_label = actions[train_index]


    test_input = states[test_index]
    test_label = actions[test_index]



    training_epochs = 20
    training_loops = 2000

    training_loss = []
    validation_loss = []
    test_loss = []

    loss_fn = nn.MSELoss()  # Usare MSELoss per dati continui
    print("Training time started")

    for loop_time in range(training_loops):
        print("Training time: ", loop_time + 1 , " / ", training_loops)

        # Train on training set
        actor.train()
        average_loss_training = 0

        for _ in range(training_epochs):

            mu, std = actor(training_input)
            #a_pred = torch.normal(mu, std)
            # a_pred = torch.normal(mu, std)  # <-- USA SOLO MU, non torch.normal!


            loss = -log_prob_density(training_label, mu, std).mean()
        

            #loss = loss_fn(a_pred, training_label)

            average_loss_training += loss.item()


            actor_optim.zero_grad()

            loss.backward()

            actor_optim.step() 





        # Test set
        actor.eval()

        with torch.no_grad():
            mu, std = actor(test_input)
                
            #a_pred = torch.normal(mu, std)
            #a_pred = torch.normal(mu, std)  # <-- USA SOLO MU, non torch.normal!
            
            
            loss = -log_prob_density(test_label, mu, std).mean()
            #loss = loss_fn(a_pred, test_label)



        training_loss.append(average_loss_training / training_epochs)
        test_loss.append(loss.item())






    # -------- Plot dei 3 loss insieme --------

    os.makedirs("learning/BC", exist_ok=True)


    dir_name = time.strftime("%y-%m-%d_%H-%M-%S") 
    directory = "learning/BC/pretrained_actors/" + dir_name + "/"

    if not os.path.exists(directory):
        os.makedirs(directory) 


    torch.save(actor.state_dict(), directory + 'actor_BC.pkl')
    torch.save(actor_optim.state_dict(), directory + 'best_actor_optim.pkl')


    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel("Training Loops")
    plt.ylabel("Loss")
    plt.title("Behavior Cloning Loss (Training/Validation/Test)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig( directory + "training_validation_test_loss.png")
    plt.close()


torch.cuda.set_device(0)


# -------------- SET CONFIG OF THIS ENVIRONMENT ----------------:

# ------- Agent settings -------

args.hidden_size = 256
args.learning_rate = 8e-6
args.entropy_gain_PPO = 0.0
args.l2_rate = 3e-1






BC_model_path = "learning/BC/pretrained_actors/25-05-06_14-36-23"
num_inputs = 55
num_actions = 6
episode_length = 250



# Initiating Actor and critic
task_device = "cuda"
actor_BC = Actor(num_inputs, num_actions, args).to(task_device)
actor_optim_BC = optim.Adam(actor_BC.parameters(), lr=args.learning_rate, weight_decay=args.l2_rate)



actor_state_dict = torch.load(BC_model_path + "/" + "actor_BC.pkl")
actor_optim_state_dict = torch.load(BC_model_path + "/" + "best_actor_optim.pkl")

actor_BC.load_state_dict(actor_state_dict)
actor_optim_BC.load_state_dict(actor_optim_state_dict)



# Load expert data [AIRL]
path_expert_data = os.path.abspath("learning/airl_UR/expert_demo/C1_easy_25-04-28_11-54-27URSim_SKRL_env_PPO/expert_memory_modified_no_time_no_performance_metric.pkl")

expert_data_file = open(path_expert_data, "rb")

expert_memory = pickle.load(expert_data_file)
states_exp =    torch.tensor(np.array([entry[0] for entry in expert_memory])).cuda().to(torch.float32)  # Stack states vertically
actions_exp =   torch.tensor(np.array([entry[1] for entry in expert_memory])).cuda().to(torch.float32)  # Convert actions to array




# Test mu std output of model

loss_fn = GaussianNLLLoss()


mu, std = actor_BC(states_exp)

loss = loss_fn(mu, actions_exp, std**2)

print("loss: ", loss)


print(" -------- testing BC actor -------- ")
episode = 0


joint_index = 0
for joint_index in range(6):


    action_all_episodes = []
    mu_list_all_episodes = []
    std_list_all_episodes = []

    for episode in range(10):
        action = []
        mu_list = []
        std_list = []

        for i in range(episode_length*episode, episode_length + episode_length*episode):
            mu, std = actor_BC(states_exp[i])

            action.append(actions_exp[i][joint_index].item())
            mu_list.append(mu[joint_index].item())
            std_list.append(std[joint_index].item())


        action_all_episodes.append(action)
        mu_list_all_episodes.append(mu_list)
        std_list_all_episodes.append(std_list)


    # Convert to numpy arrays of shape (num_episodes, steps_per_episode)
    action_all_episodes = np.array(action_all_episodes)
    mu_list_all_episodes = np.array(mu_list_all_episodes)
    std_list_all_episodes = np.array(std_list_all_episodes)

    # Compute mean and std over episodes
    mu_mean = mu_list_all_episodes.mean(axis=0)

    std_mean = std_list_all_episodes.mean(axis=0)
    action_mean = action_all_episodes.mean(axis=0)
    confidence_val = 0.95

    action_95_confidence = mean_confidence_interval(action_all_episodes, confidence=confidence_val)
    # action_std = action_all_episodes.std(axis=0)


    # Plot for a single joint (joint_index), over all samples
    plt.plot(action_mean, label='Expert mean action', color='blue', linewidth=1.3)
    plt.plot(mu_mean, label='Policy mean action', color='purple', linewidth=1.3)
    plt.fill_between(
        np.arange(len(mu_mean)),
        mu_mean - std_mean,
        mu_mean + std_mean,
        alpha=0.3,
        color='orange',
        label='Policy mean action ± Std'
    )
    plt.fill_between(
        np.arange(len(mu_list)),
        action_mean - action_95_confidence,
        action_mean + action_95_confidence,
        alpha=0.2,
        color='red',
        label=f'Expert mean action ± {confidence_val}% confidence interval'
    )
    plt.title(f'Behavior Cloning Actor Output vs Expert Action (Joint {joint_index})',  fontsize=13)
    plt.xlabel('Environment timestep', fontsize=13)
    plt.ylabel(f'Action Value joint {joint_index}',  fontsize=13)
    plt.legend(prop={'size': 10}, facecolor='white')
    plt.grid(True)
    plt.tight_layout()

    ax = plt.gca()
    ax.set_facecolor('white')
    ax = plt.gca()  # Get current axes
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)   # Set thicker border
        spine.set_edgecolor('black')  # Set border color

    plt.savefig( BC_model_path + "/" + f"joint-{joint_index}_policy_analysis.png")
    plt.close()



