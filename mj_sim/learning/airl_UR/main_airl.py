import os
import pickle
import argparse
import numpy as np
from collections import deque

import torch

import torch.optim as optim
from tensorboardX import SummaryWriter 

from learning.airl_UR.utils.utils import *
from learning.airl_UR.utils.zfilter import ZFilter
from learning.airl_UR.model import Actor, Critic, Discriminator
from learning.airl_UR.airl import AIRL
from learning.airl_UR.train_model import train_actor_critic, train_discrim, train_actor_critic_process, train_actor_REINFORCE

# ----------- Import peter stuff -----------
# from learning.SKRL_UR_env_Q_learning import URSim_SKRL_env


#from learning.environments.SKRL_UR_env_PPO import URSim_SKRL_env

#Flick task environment
from learning.environments.UR_env_Flick_TASK_1 import URSim_SKRL_env


from learning.models.Generate_models import get_PPO_agent_SKRL, get_TD3_agent, make_TD3_models, Policy, Value
from learning.config_files.generate_config_file import generate_config_file
from skrl.envs.wrappers.torch import wrap_env

import time

# Calculating the log_probability (log_pis), of the actions taken using the GAIL repo implementation
from learning.airl_UR.utils.utils import get_entropy, log_prob_density
from learning.airl_UR.memory_process import process_mem, Generate_memory_from_expert_data

from torch import multiprocessing
from torch.multiprocessing import Process
from torch.multiprocessing import Lock
torch.set_default_dtype(torch.float32)

import copy



class RunningNorm:
    def __init__(self, dim, alpha=0.001, eps=1e-6):

        self.mean = np.zeros(dim, dtype=float)
        self.var  = np.ones(dim, dtype=float)
        self.alpha = alpha
        self.eps = eps


    def update(self, x: np.ndarray):

        delta = x - self.mean
        self.mean += self.alpha * delta
        self.var  = (1 - self.alpha) * self.var + self.alpha * (delta**2)
        # now normalized x
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)





def make_args():

    parser = argparse.ArgumentParser(description='PyTorch GAIL')
    parser.add_argument('--env_name', type=str, default="Hopper-v2", 
                        help='name of the environment to run')
    parser.add_argument('--load_model', type=str, default=None, 
                        help='path to load the saved model')
    parser.add_argument('--render', action="store_true", default=False, 
                        help='if you dont want to render, set this to False')
    parser.add_argument('--gamma', type=float, default=0.70, 
                        help='discounted factor (default: 0.99)')
    parser.add_argument('--lamda', type=float, default=0.65, 
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
    parser.add_argument('--clip_param_actor', type=float, default=0.2, 
                        help='clipping parameter for PPO (default: 0.2)')
    parser.add_argument('--clip_param_critic', type=float, default= 10, 
                        help='clipping parameter for PPO (default: 0.2)')
    parser.add_argument('--discrim_update_num', type=int, default=10, 
                        help='update number of discriminator (default: 2)')
    parser.add_argument('--actor_critic_update_num', type=int, default=10, 
                        help='update number of actor-critic (default: 10)')
    parser.add_argument('--total_sample_size', type=int, default=1, 
                        help='Mini batch size')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size to update (default: 64)')
    parser.add_argument('--suspend_accu_exp', type=float, default=0.90,
                        help='accuracy for suspending discriminator about expert data (default: 0.8)')
    parser.add_argument('--suspend_accu_gen', type=float, default=0.90,
                        help='accuracy for suspending discriminator about generated data (default: 0.8)')
    parser.add_argument('--max_iter_num', type=int, default=1000000,
                        help='maximal number of main iterations (default: 4000)')
    parser.add_argument('--seed', type=int, default=np.random.randint(1e6),
                        help='random seed (default: 500)')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='tensorboardx logs directory')
    parser.add_argument('--max_grad_norm', type=float, default=10,
                        help='gradient clipping of PPO')
    args = parser.parse_args()

    return args



def mahalanobis_distance(action, mean, std):
    """
    This function calculates the mahalanobis distance between the action point, and the gaussian variables.
    """

    mahalanobis_distance_val = torch.sqrt((action - mean).pow(2) / std.pow(2))

    return mahalanobis_distance_val

def euclidean_distance(action, action2):
    """
    This function calculates the mahalanobis distance between the action point, and the gaussian variables.
    """

    euclidean_val = torch.sqrt((action - action2).pow(2))

    return euclidean_val


def get_IRL_reward(memory, AIRL_trainer, normalization_val = None):
    torch.set_num_threads(1)
    


    with torch.no_grad():

        states, actions, rewards, masks, log_pis, next_states = process_mem(memory)

        observation_space = len(states[0])
        next_state = torch.tensor(np.array([entry[5] for entry in memory])).type(dtype=torch.float32).view(-1, observation_space)    # Convert masks to array


        



        AIRL_trainer.disc.eval()
        irl_reward, D = AIRL_trainer.get_reward(states, actions, masks, log_pis, next_state)

        debug = 0
        for i, val in enumerate(memory):
            if normalization_val != None:
                val[2] = irl_reward[i] / normalization_val # Normalize reward based on expert reward
            else:
                val[2] = irl_reward[i]
            
            debug += irl_reward[i]

            memory[i] = val


    return memory

def check_nan_in_model(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            return True
        else:
            return False

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



def run_rollout(process_ID, episodes_per_mini_batch, memory_list, actor_list, env_args, args,  Lock):
    torch.set_num_threads(1)
    torch.manual_seed(time.time() + process_ID*42)
    actor, actor_optim = actor_list[process_ID]


    actor.eval(),
    env = URSim_SKRL_env(args = env_args, name=f"sub_process: {process_ID}", render_mode="rgb_array")

    memory_process = deque()

    memory_process = run_rollout_loop(actor=actor, episodes_per_mini_batch=episodes_per_mini_batch, args=args, memory=memory_process, env=env)

    with Lock:
        memory_list[process_ID] = memory_process




def run_rollout_loop(actor, episodes_per_mini_batch, memory, env, args, debug_expert_memory=None, state_list=None):
    torch.set_num_threads(1)
    
    # Expert works but its just an average i think, check on main computer

    episodes_per_mini_batch = args.total_sample_size
    episodes_count = 0
    


    while episodes_count < episodes_per_mini_batch: 
        state, _ = env.reset()
        steps = 0


        for _ in range(10000): 

            steps += 1


            with torch.no_grad():
                
                state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)

                mu, std = actor(state_tensor)
            
                action = get_action(mu, std)[0]

                # Deploy with expert action

                # NOTE: Will break if RL is active!
                # action = debug_expert_memory[steps - 1 + 200*episodes_count][1]


                action_tensor = torch.tensor(action, dtype=torch.float32).view(1, -1)

                log_pis = log_prob_density(action_tensor, mu, std).item()


            next_state, reward, done, _, _ = env.step(action)




            if done:    
                mask = 1
            else:
                mask = 0

            # if steps == 1:
            #     print("states: ", state)
            #     print("actions: ", action)
            #     input()


            memory.append([state, action_tensor.detach(), reward, mask, log_pis, next_state])
            


            state = next_state



            if done:
                break
        





        episodes_count += 1




    return memory, state_list



def run_rollout_loop_get_expert_observations(episodes_per_mini_batch, memory, env, args, debug_expert_memory):
    torch.set_num_threads(1)
    
    # Expert works but its just an average i think, check on main computer

    episodes_per_mini_batch = args.total_sample_size
    episodes_count = 0
    


    while episodes_count < episodes_per_mini_batch: 
        state, _ = env.reset()
        steps = 0


        for _ in range(10000): 

            steps += 1


            with torch.no_grad():
                
                action = debug_expert_memory[steps - 1 + 200*episodes_count][1]

                action_tensor = torch.tensor(action, dtype=torch.float32).view(1, -1)

                log_pis = 0


            next_state, reward, done, _, _ = env.step(action)




            if done:    
                mask = 1
            else:
                mask = 0

            # if steps == 1:
            #     print("states: ", state)
            #     print("actions: ", action)
            #     input()


            memory.append([state, action, reward, mask, log_pis, next_state])
            


            state = next_state



            if done:
                break
        





        episodes_count += 1




    return memory




def main():
    args = make_args()
    torch.set_num_threads(1)
    # Run script in background:
    # $ nohup python -u -m learning.airl_UR.main_airl > IRL_learning_test_1.log 3>&1 &

    # env = gym.make(args.env_name, render_mode="rgb_array") # human, rgb_array
    env_args=generate_config_file()
    Training_model = "PPO"

    # -------------- SET CONFIG OF THIS ENVIRONMENT ----------------:

    # F2 IRL RL OPT with AIRL increased weight decay

    # 10 agents, weight decay 1e-4
    # ------- Agent settings -------
    args.gamma = 0.99
    args.lamda = 0.85
    args.actor_critic_update_num = 20
    args.batch_size = 256
    args.hidden_size = 128
    args.learning_rate = 1e-5
    args.max_grad_norm = 10
    agents = 0


    args.total_sample_size = 17
    args.entropy_gain_PPO = 1e-3
    args.critic_gain_PPO = 1
    args.l2_rate = 1e-3
    off_policy_data_size = 200*args.total_sample_size*(agents + 1)


    # ------- AIRL settings -------
    hidden_layer_nodes_r=(256,256,256)
    hidden_layer_nodes_v=(256,256,256)
    airl_gamma = args.gamma
    airl_learning_rate = 3e-4
    airl_epocs = 5
    AIRL_L2_weight_decay = 1e-1
    Expert_mini_batch = args.batch_size



    # ------- General training settings -------
    env_args.agent_disabled = False
    env_args.expert_demonstration = False


    discriminator_delay_value = 20
    Train_With_AIRL = True # Set to false to enable GAIL


    training_RL = True
    training_discriminator = True

    env_render_mode = "human"

    # NOTE: It is a mistake if both BC actor and pretrained actor are both True!
    use_BC_actor = False
    use_BC_disc = False


    BC_model_path = "learning/BC/pretrained_actors/25-05-06_14-36-23"
    BC_disc_path = "learning/airl_UR/Saved_models/AIRL_UR_FLICK_3_TASK_25-05-20_06-00-50"


    use_pretrained_RL  = False
    use_pretrained_IRL = False





    # Load RL model
    model_path = "learning/airl_UR/Saved_models/AIRL_UR_FLICK_3_TASK_25-05-31_12-36-51"
    

    model_candidate = "latest"




    # Load AIRL model
    disc_model_path = "learning/airl_UR/Saved_models/AIRL_UR_FLICK_3_TASK_25-05-31_12-36-51"
   
   
    disc_model_candidate = "latest"





    expert_address = "learning/airl_UR/expert_demo/F1_17_25-05-08_18-33-53URSim_SKRL_env_PPO/expert_memory.pkl"


    # --------------------------------- SET CONFIG END ---------------------------------




    # Make sure that the training computer is one of the computers that match the computers that the program is setup to run in
    
    assert os.path.isdir("learning/airl_UR"), "The directory 'learning/airl_UR' does not exist! Code is run from an illegal folder"


    environment_name = time.strftime("%y-%m-%d_%H-%M-%S") + "URSim_SKRL_env_" + Training_model

    env = URSim_SKRL_env(args = env_args, name=environment_name, render_mode=env_render_mode)


    #env.seed(args.seed)
    args.seed = np.random.randint(1e6)
    torch.manual_seed(args.seed)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    # running_state = ZFilter((num_inputs,), clip=5) # Maybe this has some features in the future that is worth exploring


    # Initiating Actor and critic
    task_device = "cpu"


    actor = copy.deepcopy(Actor(num_inputs, num_actions, args).to(task_device))
    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate, weight_decay=args.l2_rate)

    critic = copy.deepcopy(Critic(num_inputs, args).to(task_device))
    critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate, weight_decay=args.l2_rate) 





    
    agents_list = multiprocessing.Manager().list([None]*agents)
    critic_list = multiprocessing.Manager().list([None]*agents)

    actor_loss_process = multiprocessing.Manager().list([None]*agents)
    critic_loss_process = multiprocessing.Manager().list([None]*agents)
    entropy_loss_process = multiprocessing.Manager().list([None]*agents)
    ratio_list_process = multiprocessing.Manager().list([None]*agents)





    for i in range(agents):
        actor_process = copy.deepcopy(Actor(num_inputs, num_actions, args))
        actor_optim_proces = optim.Adam(actor_process.parameters(), lr=args.learning_rate, weight_decay=args.l2_rate)

        critic_proces = copy.deepcopy(Critic(num_inputs, args))
        critic_optim_proces = optim.Adam(critic_proces.parameters(), lr=args.learning_rate, weight_decay=args.l2_rate)


        if use_pretrained_RL:
            actor_state_dict = copy.deepcopy(torch.load(model_path + "/" + model_candidate + "_actor.pkl"))
            actor_process.load_state_dict(actor_state_dict)

            actor_optim_state_dict = copy.deepcopy(torch.load(model_path + "/" + model_candidate + "_actor_optim.pkl"))
            actor_optim_proces.load_state_dict(actor_optim_state_dict)


            critic_state_dict = copy.deepcopy(torch.load(model_path + "/" + model_candidate + "_critic.pkl"))
            critic_proces.load_state_dict(critic_state_dict)

            critic_optim_state_dict = copy.deepcopy(torch.load(model_path + "/" + model_candidate + "_critic_optim.pkl"))
            critic_optim_proces.load_state_dict(critic_optim_state_dict)
        else:
            model_path = None



        agents_list[i] = [actor_process, actor_optim_proces]
        critic_list[i] = [critic_proces, critic_optim_proces]





    # Initiating GAIL discriminator
    discrim = Discriminator(num_inputs + num_actions, args)
    discrim_optim = optim.Adam(discrim.parameters(), lr=args.learning_rate)


    # Initializing the AIRL discriminator
    AIRL_trainer = AIRL(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device(task_device), #  "cuda" if args.cuda else "cpu"
        seed=args.seed,
        units_disc_r=hidden_layer_nodes_r,
        units_disc_v=hidden_layer_nodes_v,
        lr_disc=airl_learning_rate,
        epoch_disc=airl_epocs,
        weight_decay_L2 = AIRL_L2_weight_decay,
        gamma=airl_gamma,
        state_only=False,
        value_shaping=True,
        #rollout_length=args.rollout_length
    )






    # Setup BC actor
    if use_BC_actor:
        actor_state_dict = torch.load(BC_model_path + "/" + "actor_BC.pkl", map_location=torch.device('cpu'))
        actor.load_state_dict(actor_state_dict)

        actor_optim_state_dict = torch.load(BC_model_path + "/" + "best_actor_optim.pkl", map_location=torch.device('cpu'))
        actor_optim.load_state_dict(actor_optim_state_dict)
    else:
        BC_model_path = None



    if use_BC_disc:
        reward_state_dict = torch.load(BC_disc_path + "/" + "AIRL_disc_BC_reward_func.pkl", map_location=torch.device('cpu'))
        AIRL_trainer.upload_BC_reward_func(reward_state_dict)
    else:
        BC_disc_path = None




    if use_pretrained_RL:
        actor_state_dict = torch.load(model_path + "/" + model_candidate + "_actor.pkl")
        actor.load_state_dict(actor_state_dict)

        actor_optim_state_dict = torch.load(model_path + "/" + model_candidate + "_actor_optim.pkl")
        actor_optim.load_state_dict(actor_optim_state_dict)


        critic_state_dict = torch.load(model_path + "/" + model_candidate + "_critic.pkl")
        critic.load_state_dict(critic_state_dict)

        critic_optim_state_dict = torch.load(model_path + "/" + model_candidate + "_critic_optim.pkl")
        critic_optim.load_state_dict(critic_optim_state_dict)
    else:
        model_path = None



    if use_pretrained_IRL:
        # AIRL load pretrained
        discrim_state_dict = torch.load(disc_model_path + "/" + disc_model_candidate + "_AIRL_discrim.pkl")
        AIRL_trainer.disc.load_state_dict(discrim_state_dict)

        discrim_state_dict = torch.load(disc_model_path + "/" + disc_model_candidate + "_AIRL_discrim_optim.pkl")
        AIRL_trainer.optim_disc.load_state_dict(discrim_state_dict)
    else:
        disc_model_path = None



    for g in actor_optim.param_groups:
        g['lr'] = args.learning_rate
        

    for g in critic_optim.param_groups:
        g['lr'] = args.learning_rate


    for g in AIRL_trainer.optim_disc.param_groups:
        g['weight_decay'] = AIRL_L2_weight_decay
    




    # Expert dataset observational space joint and velocity: [GAIL expert data]

    # path_expert_data_joint_velocity_numpy = "/home/peter/OneDrive/Skrivebord/Uni stuff/Masters/masters_code/mj_sim/learning/airl_UR/expert_demo/Go_to_object_expert_data/all_state_actions.npy"
    # expert_demo = np.load(path_expert_data_joint_velocity_numpy)


    # # With controllable gripper, remove that controllability for now.rgb_array
    # demonstrations = np.array(expert_demo)




    # Load expert data [AIRL]
    path_expert_data = os.path.abspath(expert_address)


    expert_data_file = open(path_expert_data, "rb")
    
    expert_memory = pickle.load(expert_data_file)
    states_exp =    torch.tensor(np.array([entry[0] for entry in expert_memory])).to(torch.float)  # Stack states vertically
    actions_exp =   torch.tensor(np.array([entry[1] for entry in expert_memory])).to(torch.float)  # Convert actions to array
    dones_exp =     torch.tensor([entry[3] for entry in expert_memory]).to(torch.int8)  # Convert actions to array
    log_pis_exp =   torch.tensor([entry[4] for entry in expert_memory]) # Convert actions to array
    next_states_exp = torch.tensor(np.array([entry[5] for entry in expert_memory])).to(torch.float)  # Convert actions to array


    expert_data = [
                    states_exp,
                    actions_exp,
                    [], # Rewards exp that is not used
                    dones_exp,
                    [], # new log pis has not been calculated yet
                    next_states_exp,
                    ]



    path = os.path.abspath("learning/airl_UR/Saved_models/")
    name = "AIRL_UR_FLICK_3_TASK_" + time.strftime("%y-%m-%d_%H-%M-%S") 


    # Make directories for training
    if training_RL == True or training_discriminator == True:

        log_address = "learning/airl_UR/logs/" + name
        writer = SummaryWriter(log_address)


        directory = path+ "/" + name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory) 

        # Save the parameters file
        params = {
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "entropy_gain_PPO": args.entropy_gain_PPO,
            "critic_gain_PPO": args.critic_gain_PPO,
            "actor_critic_update_num": args.actor_critic_update_num,
            "PPO max_grad_norm": args.max_grad_norm,
            "PPO_batch size": args.batch_size,
            "PPO gamma": args.gamma,
            "PPO lambda": args.lamda,
            "PPO weight decay": args.l2_rate,
            "off_policy_data_size": off_policy_data_size,
            "agent_disabled": env_args.agent_disabled,
            "expert_demonstration": env_args.expert_demonstration,
            "hidden_layer_nodes_r": hidden_layer_nodes_r,
            "hidden_layer_nodes_v": hidden_layer_nodes_v,
            "airl_learning_rate": airl_learning_rate,
            "airl_epocs": airl_epocs,
            "airl gamma": airl_gamma,
            "airl reward NN: ": hidden_layer_nodes_r,
            "AIRL_L2_weight_decay": AIRL_L2_weight_decay,
            "Expert_mini_batch": Expert_mini_batch,
            "discriminator_delay_value": discriminator_delay_value,
            "Train_With_AIRL": Train_With_AIRL,
            "training_discriminator": training_discriminator,
            "training_RL": training_RL,
            "env_render_mode": env_render_mode,
            "model_path": model_path,
            "model_candidate": model_candidate,
            "disc_model_path": disc_model_path,
            "disc_model_candidat": disc_model_candidate,
            "BC model_path: ": BC_model_path,
            "BC_disc_path: ": BC_disc_path,
        }


        save_parameters(params=params, directory=directory)




    episodes = 0
    train_discrim_flag = True
    discriminator_delay = discriminator_delay_value
    training_start_time = time.time()
    best_model_score = -np.inf

    memory_off_policy = deque(maxlen=off_policy_data_size)
    
    #env = wrap_env(env)

    
    state_list = RunningNorm(dim=env.observation_space.shape[0])

    episodes_per_mini_batch = args.total_sample_size


    for iter in range(args.max_iter_num):
        memory_list_process = multiprocessing.Manager().list([None]*agents)
        memory = deque()

        AIRL_trainer.disc.eval()
        actor.eval(), critic.eval()





        # Start multi processed agents
        jobs = []
        Lock = multiprocessing.Lock()
        for i in range(agents):
            p = multiprocessing.Process(target=run_rollout, args=(i, episodes_per_mini_batch, memory_list_process, agents_list, env_args,args, Lock))
            jobs.append(p)
            p.start()



        memory, state_list = run_rollout_loop(actor, episodes_per_mini_batch, memory, env,args, debug_expert_memory=expert_memory, state_list=state_list)


        # memory.append([state, action_tensor, reward, mask, log_pis, next_state])
        # Get reward average of policy, and append transitions to off policy memory
        reward_epoch_end = 0
        reward_sum = 0
        mem_epoch_count = 0

        for transitions in memory:
            if transitions[3] == True:
                reward_epoch_end += transitions[2] # Add all rewards
                mem_epoch_count += 1



            reward_sum += transitions[2]
            memory_off_policy.append(transitions)


        rewards_average_end = reward_epoch_end / mem_epoch_count
        reward_average = reward_sum / len(memory)

        # Get reward average of policy in parallel process, and append transitions to off policy memory
        for proc in jobs:
            proc.join()

        for mem in memory_list_process:

            for transition in mem:
                memory_off_policy.append(transition)





        # Save the best model so far before training the models
        if reward_average > best_model_score: # first begin save models after 15 minutes
            best_model_score = reward_average

            if training_RL:
                best_actor = copy.deepcopy(actor.state_dict())
                best_critic = copy.deepcopy(critic.state_dict())
                best_actor_optim = copy.deepcopy(actor_optim.state_dict())
                best_critic_optim = copy.deepcopy(critic_optim.state_dict())

                torch.save(best_actor, directory+'best_actor.pkl')
                torch.save(best_critic, directory+'best_critic.pkl')

                torch.save(best_actor_optim, directory+'best_actor_optim.pkl')
                torch.save(best_critic_optim, directory+'best_critic_optim.pkl')
            


            if training_discriminator:
                best_airl_dis = copy.deepcopy(AIRL_trainer.disc.state_dict())
                best_airl_dis_optim = copy.deepcopy(AIRL_trainer.optim_disc.state_dict())


                torch.save(best_airl_dis, directory+'best_AIRL_discrim.pkl')
                torch.save(best_airl_dis_optim, directory+'best_AIRL_discrim_optim.pkl')
            




        # Save the latest models so far

        if training_RL:
            torch.save(actor.state_dict(), directory+'latest_actor.pkl')
            torch.save(critic.state_dict(), directory+'latest_critic.pkl')

            torch.save(actor_optim.state_dict(), directory+'latest_actor_optim.pkl')
            torch.save(critic_optim.state_dict(), directory+'latest_critic_optim.pkl')


        if training_discriminator:
            torch.save(AIRL_trainer.disc.state_dict(), directory+'latest_AIRL_discrim.pkl')
            torch.save(AIRL_trainer.optim_disc.state_dict(), directory+'latest_AIRL_discrim_optim.pkl')

            


        # Training the discriminator    
        if train_discrim_flag and training_discriminator: # disc training is for enabling and disabling the discriminator

            if discriminator_delay == discriminator_delay_value:
                if Train_With_AIRL: # Train with AIRL discriminator

                    AIRL_trainer.disc.eval()
                    learner_acc, expert_acc  = AIRL_trainer.evaluate_disc(memory=memory_off_policy, expert_data=expert_data, actor=actor)
                    
                    AIRL_trainer.disc.train()




                    pi_score_after_training, exp_score_after_training, loss_pi, loss_exp  = AIRL_trainer.update(memory=memory, expert_data=expert_data, Expert_mini_batch=Expert_mini_batch, actor=actor)
                    

                    Policy_change = pi_score_after_training - learner_acc

                    Expert_change = exp_score_after_training - expert_acc


                    # print(f"AIRL - Expert: %.2f%% | Learner: %.2f%%," % (, learner_acc * 100))

                    print(f"AIRL - Expert: {expert_acc*100:.2f}% | Learner: {learner_acc*100:.2f}% - Expert change: {Expert_change*100:.2f}% | Learner: {Policy_change*100:.2f}%")



                    Disc_acc_pause_criterium = (expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen)

                    if Disc_acc_pause_criterium and iter>10:
                        train_discrim_flag = False
                        discriminator_delay = 0



                        if training_discriminator:
                            torch.save(AIRL_trainer.disc.state_dict(), directory+'latest_AIRL_discrim.pkl')
                            torch.save(AIRL_trainer.optim_disc.state_dict(), directory+'latest_AIRL_discrim_optim.pkl')



                else:
                    # GAIL training
                    expert_acc, learner_acc = train_discrim(discrim, memory, discrim_optim, demonstrations, args)
                    print("GAIL - Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
                    if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
                        train_discrim_flag = False
                        discriminator_delay = 0


        # When this is commented out, then the IRL reward will never come back after being disabled
        if discriminator_delay < discriminator_delay_value:

            discriminator_delay += 1



        if discriminator_delay == discriminator_delay_value:
            train_discrim_flag = True




        # Training the actor and critic
        if training_RL:
            # writer.add_scalar('environment/score_IRL_sum', float(score_irl_sum_value), iter)
            # writer.add_scalar('environment/score_IRL_avg', float(score_IRL_avg_value), iter)




            # writer.add_scalar('environment/score_IRL_variance', float(score_IRL_var_value), iter)
            # writer.add_scalar('environment/Discriminator_value_avg', float(Discriminator_list_value), iter)


            if training_discriminator:
                writer.add_scalar('IRL/expert_accuracy', float(expert_acc), iter)
                writer.add_scalar('IRL/trainer_accuracy', float(learner_acc), iter)





            jobs = []

            for i in range(agents):

                mem_process = memory_list_process[i]


                actor_process, actor_optim_process = agents_list[i]
                critic_process, critic_optim_proces = critic_list[i]

                
                reward_epoch_end_process = 0
                reward_sum_process = 0
                mem_epoch_count_process = 0

                for transitions in mem_process:
                    if transitions[3] == True:
                        reward_epoch_end_process += transitions[2] # Add all rewards
                        mem_epoch_count_process += 1


                    reward_sum_process += transitions[2]



                rewards_average_end_process = reward_epoch_end_process / mem_epoch_count_process
                reward_average_process = reward_sum_process / len(mem_process)


                # Activate IRL reward
                mem_process_IRL = get_IRL_reward(memory=mem_process, AIRL_trainer=AIRL_trainer)


                #sum of IRL rewards
                rewards_sum_IRL = 0
                # Check environment reward
                for transition in mem_process:
                    reward = copy.deepcopy(transition[2])
                    rewards_sum_IRL += reward

                    

                rewards_IRL_average_process = rewards_sum_IRL / len(mem_process)

                print(f"Process ID {i} :: env avg: {reward_average_process} -- env END: {rewards_average_end_process:.2f} -- IRL avg: {rewards_IRL_average_process:.2f}" )


                writer.add_scalar(f'Sub process reward avg/process ID {i}', reward_average_process, iter)

                writer.add_scalar(f'Sub process reward end/process ID {i}', rewards_average_end_process, iter)

                writer.add_scalar(f'Sub process IRL reward/process ID {i}', rewards_IRL_average_process, iter)



                writer.add_scalars(f'reward/avg', {
                    f'process ID {i}': reward_average_process,
                }, iter)

                writer.add_scalars(f'reward/end', {
                    f'process ID {i}': rewards_average_end_process,
                }, iter)

                writer.add_scalars(f'reward/IRL', {
                    f'process ID {i}': rewards_IRL_average_process,
                }, iter)



                actor_process.train(), critic_process.train()


                p = multiprocessing.Process(target=train_actor_critic_process, args=(i, agents_list, critic_list, mem_process_IRL, args, actor_loss_process, critic_loss_process, entropy_loss_process, ratio_list_process))
                jobs.append(p)
                p.start()


                

            actor.train(), critic.train()

            # for transistions in memory:
            #     print("rewards: ", transistions[2])
            #     input()


            # Get IRL reward 


            memory_irl_reward = get_IRL_reward(memory=memory, AIRL_trainer=AIRL_trainer)



            rewards_sum = 0
            for transition in memory:
                reward = copy.deepcopy(transition[2])
                rewards_sum += reward

            IRL_rewards_average = rewards_sum / len(memory)
              
            
            writer.add_scalar('environment/reward_env_end', float(rewards_average_end), iter)
            writer.add_scalar('environment/reward_env_avg', float(reward_average), iter)
            writer.add_scalar('environment/reward IRL avg', float(IRL_rewards_average), iter)

            print(f"{iter}:: {episodes}, env avg: {reward_average:.2f} -- env END: {rewards_average_end:.2f} -- IRL avg: {IRL_rewards_average:.2f}" )
            print("\n")


            actor_loss, critic_loss, entropy, ratio = train_actor_critic(actor, critic, memory_irl_reward, actor_optim, critic_optim, args, expert_data)



            with torch.no_grad():
                mu, std = actor(states_exp)
                log_pis_exp = log_prob_density(actions_exp, mu, std)
                print("Mean log probability value: ", log_pis_exp.mean())

            




            writer.add_scalar('IRL/log_prob_expert_actions', log_pis_exp.mean(), iter)
            writer.add_scalar('RL/actor_loss', actor_loss, iter)
            writer.add_scalar('RL/critic_loss', critic_loss, iter)
            writer.add_scalar('RL/PPO_entropy', entropy, iter)
            writer.add_scalar('RL/ratio: ', ratio, iter)


        # print(f"{iter}:: {episodes}, environment reward {scores_env_done_reward_avg} - IRL reward {IRL_rewards_average}")




        

        #Measure Immitation performance compared to expert demonstration
        with torch.no_grad():

    
            mu_exp, std_exp  = actor(torch.tensor(np.array(states_exp)))

    

            mahala_val = mahalanobis_distance(actions_exp, mu_exp, std_exp).mean().item()



            action_policy = get_action(mu_exp, std_exp)


            euclidean_sampled_distance = euclidean_distance(action=actions_exp, action2=action_policy).mean().item()

            euclidean_distance_mean_based = euclidean_distance(action=actions_exp, action2=mu_exp).mean().item()

        
        if training_RL or training_discriminator:
            writer.add_scalar('IRL/mahalanobis_distance', mahala_val, iter)
            writer.add_scalar('IRL/euclidean_sampled_distance', euclidean_sampled_distance, iter)
            writer.add_scalar('IRL/euclidean_mean_distance', euclidean_distance_mean_based, iter)

        if training_RL:
            for proc in jobs:
                proc.join()


            for iterable_process_ID in range(agents):

                a = actor_loss_process[iterable_process_ID]
                c = critic_loss_process[iterable_process_ID]
                e = entropy_loss_process[iterable_process_ID]
                r = ratio_list_process[iterable_process_ID]

                writer.add_scalars(f'loss/PPO_loss_values', {
                    f'Loss advantage process ID {iterable_process_ID}': a,
                }, iter)

                writer.add_scalars(f'loss/critic', {
                    f'Loss critic process ID {iterable_process_ID}': c,
                }, iter)

                writer.add_scalars(f'loss/entropy', {
                    f'Entropy regularizer process ID {iterable_process_ID}': e,
                }, iter)


                writer.add_scalars(f'loss/ratio', {
                    f'ratio val process ID {iterable_process_ID}': r,
                }, iter)

                # Clear the list of all memory
                memory_list_process[:] = []


        time_trained = time.time() - training_start_time
        if time_trained > 3600*42: # 12 hours of training
            print("Training time limit reached")
            print("Current time is: ", time.strftime("%y-%m-%d_%H-%M-%S"))
            break

    





    # Ensure the directory exists before saving the .npy file
    directory = path+ "/" + name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist


    if training_RL:
        torch.save(actor.state_dict(), directory+'latest_actor.pkl')
        torch.save(critic.state_dict(), directory+'latest_critic.pkl')

        torch.save(actor_optim.state_dict(), directory+'latest_actor_optim.pkl')
        torch.save(critic_optim.state_dict(), directory+'latest_critic_optim.pkl')


        torch.save(best_actor, directory+'best_actor.pkl')
        torch.save(best_critic, directory+'best_critic.pkl')

        torch.save(best_actor_optim, directory+'latest_actor_optim.pkl')
        torch.save(best_critic_optim, directory+'latest_critic_optim.pkl')



    if training_discriminator:
        torch.save(AIRL_trainer.disc.state_dict(), directory+'latest_AIRL_discrim.pkl')
        torch.save(AIRL_trainer.optim_disc.state_dict(), directory+'latest_AIRL_discrim_optim.pkl')



        torch.save(best_airl_dis, directory+'best_AIRL_discrim.pkl')
        torch.save(best_airl_dis_optim, directory+'best_AIRL_discrim_optim.pkl')


    
    
    


if __name__=="__main__":

    main()