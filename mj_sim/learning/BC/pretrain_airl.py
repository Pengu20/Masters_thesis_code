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

# ----------- Import peter stuff -----------
# from learning.SKRL_UR_env_Q_learning import URSim_SKRL_env


#from learning.environments.SKRL_UR_env_PPO import URSim_SKRL_env

#Flick task environment
from learning.environments.UR_env_Flick_TASK_C1_easy import URSim_SKRL_env


from learning.models.Generate_models import get_PPO_agent_SKRL, get_TD3_agent, make_TD3_models, Policy, Value
from learning.config_files.generate_config_file import generate_config_file


import time
from skrl.envs.wrappers.torch import wrap_env

# Calculating the log_probability (log_pis), of the actions taken using the GAIL repo implementation
from learning.airl_UR.utils.utils import get_entropy, log_prob_density

import matplotlib.pyplot as plt



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
parser.add_argument('--total_sample_size', type=int, default=1, 
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

def save_parameters(params, directory):
    """
    SUMMARY: 
        This file is for saving a specific params file that is given in the main code

    ARGS:
        params: The param file that will be saved
        directory: The directory at which it will be saved in.
    """


    # Save parameters to file
    params_file = os.path.join(directory, "params.txt")
    with open(params_file, "w") as f:
        for key, value in params.items():

            if key == "hidden_size":
                f.write(" ------- RL agent settings -------\n")
            elif key == "airl_nodes":
                f.write("\n")
                f.write(" ------- AIRL agent settings -------\n")
            elif key == "Train_With_AIRL":
                f.write("\n")
                f.write(" ------- General training settings -------\n")
            
            f.write(f"{key}: {value}\n")

    print(f"Parameters saved to {params_file}")



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





def pretrain_airl(training_loops = 10, training_epochs = 10):

    
    train_percent = 0.6
    test_percent = 0.4

    num_inputs = 55
    num_actions = 6


    # Run script in background:
    # $ nohup python -u -m learning.airl_UR.main_airl > IRL_learning_test_1.log 3>&1 &

    torch.cuda.set_device(0)

    # -------------- SET CONFIG OF THIS ENVIRONMENT ----------------:
    # ------- Agent settings -------

    args.hidden_size = 256
    args.learning_rate = 8e-6

    args.l2_rate = 3e-4

    # ------- AIRL settings -------
    airl_nodes = 256
    airl_epocs = 10
    airl_learning_rate = 1e-5
    AIRL_L2_weight_decay = 1e-1



    BC_model_path = "learning/BC/pretrained_actors/25-05-06_14-36-23"


    expert_address = "learning/airl_UR/expert_demo/C1_easy_25-04-28_11-54-27URSim_SKRL_env_PPO/expert_memory_modified_no_time_no_performance_metric.pkl"
    # --------------------------------- SET CONFIG END ---------------------------------


    # Initiating Actor and critic
    task_device = "cuda"
    actor = Actor(num_inputs, num_actions, args).to(task_device)


    # Initializing the AIRL discriminator
    AIRL_trainer = AIRL(
        state_shape=(num_inputs,),
        action_shape=(num_actions,),
        device=torch.device(task_device), #  "cuda" if args.cuda else "cpu"
        seed=args.seed,
        units_disc_r=(airl_nodes,airl_nodes),
        units_disc_v=(airl_nodes,airl_nodes),
        lr_disc=airl_learning_rate,
        epoch_disc=airl_epocs,
        weight_decay_L2 = AIRL_L2_weight_decay,
        value_shaping=False
        #rollout_length=args.rollout_length
    )

    
    # reward_func = torch.load("learning/BC/pretrained_discriminator/25-05-06_16-44-21/AIRL_disc_BC_reward_func.pkl")
    # AIRL_trainer.upload_BC_reward_func(reward_func)


    # BC_reward_state_dict = torch.load("learning/BC/pretrained_discriminator/25-05-06_16-51-38/AIRL_disc_BC_reward_func.pkl")
    # AIRL_trainer.upload_BC_reward_func(BC_reward_state_dict)



    # Setup BC actor


    actor_state_dict = torch.load(BC_model_path + "/" + "actor_BC.pkl")
    actor.load_state_dict(actor_state_dict)


    # Load expert data [AIRL]
    path_expert_data = os.path.abspath(expert_address)


    expert_data_file = open(path_expert_data, "rb")
    
    expert_memory = pickle.load(expert_data_file)
    states_exp =    torch.tensor(np.array([entry[0] for entry in expert_memory])).cuda().to(torch.float32)  # Stack states vertically
    actions_exp =   torch.tensor(np.array([entry[1] for entry in expert_memory])).cuda().to(torch.float32)  # Convert actions to array



    train_index, test_index = split_dataset_index(n=2500, train_percent=train_percent, test_percent=test_percent)


    training_input = states_exp[train_index]
    training_label = actions_exp[train_index]


    test_input = states_exp[test_index]
    test_label = actions_exp[test_index]


    training_expert = [
                    training_input, # states
                    [], # actions
                    [], # Rewards exp that is not used
                    [], # mask
                    [], # new log pis has not been calculated yet
                    [], # next state
                    ]
    
    test_expert = [
                    test_input, # states
                    [], # actions
                    [], # Rewards exp that is not used
                    [], # mask
                    [], # new log pis has not been calculated yet
                    [], # next state
                    ]


    expert_accuracies_training = []
    policy_accuracies_training = []

    expert_accuracies_test = []
    policy_accuracies_test = []

    for loop_time in range(training_loops):
        print("Training time: ", loop_time + 1 , " / ", training_loops)


        expert_accuracies = []
        policy_accuracies = []

        for _ in range(training_epochs):
            memory = deque()
            # Calculate log_pis of the expert data based on the current policy
            with torch.no_grad():
                mu, std = actor(training_input)

                action = get_action(mu, std)

                log_pis_actor = log_prob_density(torch.tensor(action).cuda(), mu, std)
                log_pis_expert = torch.zeros_like(training_input)


            for i in range(len(training_input)):
                memory.append([np.array(training_input[i].cpu()), action[i], 0, 0, log_pis_actor[i], training_input[i]])


            training_expert[4] = log_pis_expert



            learner_acc, expert_acc = AIRL_trainer.update(memory=memory, expert_data=training_expert)

            expert_accuracies.append(expert_acc)
            policy_accuracies.append(learner_acc)


        expert_accuracies = np.array(expert_accuracies)
        policy_accuracies = np.array(policy_accuracies)

        expert_accuracies_training.append(expert_accuracies.mean())
        policy_accuracies_training.append(policy_accuracies.mean())
            


        # Test of dataset
        expert_accuracies = []
        policy_accuracies = []
        memory = deque()
        # Calculate log_pis of the expert data based on the current policy
        with torch.no_grad():
            mu, std = actor(test_input)

            action = get_action(mu, std)

            log_pis_actor = log_prob_density(torch.tensor(action).cuda(), mu, std)
            log_pis_expert = torch.zeros_like(test_input)

            for i in range(len(test_input)):
                memory.append([np.array(test_input[i].cpu()), action[i], 0, 0, log_pis_actor[i], test_input[i]])

            test_expert[4] = log_pis_expert



            learner_acc, expert_acc = AIRL_trainer.evaluate_disc(memory=memory, expert_data=test_expert)

            expert_accuracies.append(expert_acc)
            policy_accuracies.append(learner_acc)

        
        expert_accuracies = np.array(expert_accuracies)
        policy_accuracies = np.array(policy_accuracies)

        expert_accuracies_test.append(expert_accuracies.mean())
        policy_accuracies_test.append(policy_accuracies.mean())
            

            
    os.makedirs("learning/BC", exist_ok=True)


    dir_name = time.strftime("%y-%m-%d_%H-%M-%S") 
    directory = "learning/BC/pretrained_discriminator/" + dir_name + "/"

    if not os.path.exists(directory):
        os.makedirs(directory) 

    torch.save(AIRL_trainer.disc.g.state_dict(),directory + 'AIRL_disc_BC_reward_func.pkl' )
    torch.save(AIRL_trainer.disc.state_dict(), directory + 'AIRL_disc_BC.pkl')
    torch.save(AIRL_trainer.optim_disc.state_dict(), directory + 'AIRL_disc_BC_optim.pkl')


    plt.plot(expert_accuracies_training, label="Expert accuracy training")
    plt.plot(expert_accuracies_test, label="Expert accuracy test")

    plt.plot(policy_accuracies_training, label="Policy accuracy training")
    plt.plot(policy_accuracies_test, label="Policy accuracy test")

    plt.xlabel("Training Loops")
    plt.ylabel("discriminator accuracy")
    plt.title("Discriminator training (Training/Test)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig( directory + "training_test_accuracies.png")
    plt.close()

    


if __name__=="__main__":
    pretrain_airl(training_loops=10, training_epochs=5)