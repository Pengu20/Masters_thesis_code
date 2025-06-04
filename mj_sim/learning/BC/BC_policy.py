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


def log_prob_density(x, mu, std):
    # Gail implementation

    log_prob_density = -(x - mu).pow(2) / (2 * std.pow(2)) \
                     - 0.5 * torch.log(torch.tensor(2 * math.pi))
    return log_prob_density.sum(1, keepdim=True)





def split_dataset_index(n, train_percent, test_percent):

    data = np.arange(n)

    np.random.shuffle(data)

    # Calcola le dimensioni
    train_size = int(train_percent * n)
    test_size = int(test_percent * n)

    # Split dei dati
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]

    return train_data, test_data


def train_actor_BC(actor, actor_optim, states, actions, training_epochs = 20, training_loops = 5000):

    # Make data
    train_percent = 0.8
    test_percent = 0.2


    train_index, test_index = split_dataset_index(n=len(states), train_percent=train_percent, test_percent=test_percent)




    training_input = states[train_index]
    training_label = actions[train_index]


    test_input = states[test_index]
    test_label = actions[test_index]


    training_loss = []
    test_loss = []

    training_Log_probability_imitation_score = []

    test_Log_probability_imitation_score = []




    loss_fn = GaussianNLLLoss()  # Usare MSELoss per dati continui
    print("Training time started")

    for loop_time in range(training_loops):
        print("Training time: ", loop_time + 1 , " / ", training_loops)

        # Train on training set
        actor.train()
        average_loss_training = 0
        average_log_prob = 0

        for _ in range(training_epochs):

            mu, std = actor(training_input)
            #a_pred = torch.normal(mu, std)
            # a_pred = torch.normal(mu, std)  # <-- USA SOLO MU, non torch.normal!

            Log_probability = log_prob_density(training_label, mu, std)
            average_log_prob = torch.mean(Log_probability).item()



            loss = loss_fn(training_label, mu, std**2)

            #loss = loss_fn(a_pred, training_label)

            average_loss_training += loss.item()


            actor_optim.zero_grad()

            loss.backward()

            actor_optim.step() 




        # Test set
        actor.eval()

        with torch.no_grad():
            mu, std = actor(test_input)
                
            loss = loss_fn(test_label, mu, std**2)
            #loss = loss_fn(a_pred, test_label)


        Log_probability = log_prob_density(test_label, mu, std)
        

        
        training_Log_probability_imitation_score.append(average_log_prob / training_epochs)
        training_loss.append(average_loss_training / training_epochs)


        test_Log_probability_imitation_score.append(torch.mean(Log_probability).item())
        test_loss.append(loss.item())





    os.makedirs("learning/BC", exist_ok=True)


    dir_name = time.strftime("%y-%m-%d_%H-%M-%S") 
    directory = "learning/BC/pretrained_actors/" + dir_name + "/"

    if not os.path.exists(directory):
        os.makedirs(directory) 


    torch.save(actor.state_dict(), directory + 'actor_BC.pkl')
    torch.save(actor_optim.state_dict(), directory + 'best_actor_optim.pkl')


    plt.plot(training_loss, label="Training Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel("Training Loops")
    plt.ylabel("Loss")
    plt.title("Behavior Cloning Loss (Training/Validation)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig( directory + "training_validation_loss.png")
    plt.close()


    plt.plot(training_Log_probability_imitation_score, label="Training log prob")
    plt.plot(test_Log_probability_imitation_score, label="Test log prob")
    plt.xlabel("Training Loops")
    plt.ylabel("log probability density")
    plt.title("probability of taking expert action with policy")
    plt.ylim(-100, 0.2)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig( directory + "training_validation_log_prob.png")
    plt.close()



torch.cuda.set_device(0)


# -------------- SET CONFIG OF THIS ENVIRONMENT ----------------:

# ------- Agent settings -------

args.hidden_size = 256
args.learning_rate = 8e-6
args.entropy_gain_PPO = 0.0
args.l2_rate = 1e-1


training_loops = 300
training_epochs = 30


num_inputs = 20
num_actions = 6




# Initiating Actor and critic
task_device = "cuda"
actor_BC = Actor(num_inputs, num_actions, args).to(task_device)
actor_optim_BC = optim.Adam(actor_BC.parameters(), lr=args.learning_rate, weight_decay=args.l2_rate)





# Load expert data [AIRL]
path_expert_data = os.path.abspath("learning/airl_UR/expert_demo/F1_17_25-05-08_18-33-53URSim_SKRL_env_PPO/expert_memory.pkl")

expert_data_file = open(path_expert_data, "rb")

expert_memory = pickle.load(expert_data_file)
states_exp =    torch.tensor(np.array([entry[0] for entry in expert_memory])).cuda().to(torch.float32)  # Stack states vertically
actions_exp =   torch.tensor(np.array([entry[1] for entry in expert_memory])).cuda().to(torch.float32)  # Convert actions to array


loss_fn = GaussianNLLLoss()  # Usare MSELoss per dati continui

mu, std = actor_BC(states_exp)
loss_before = loss_fn(mu, actions_exp, std**2)




train_actor_BC(actor=actor_BC, actor_optim=actor_optim_BC, states=states_exp, actions=actions_exp, training_epochs=training_epochs, training_loops=training_loops)


mu, std = actor_BC(states_exp)
loss_after = loss_fn(mu, actions_exp, std**2)

print("loss before training: ", loss_before)
print("loss after training: ", loss_after)

