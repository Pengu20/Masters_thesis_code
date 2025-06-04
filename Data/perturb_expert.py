import os
import gym
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
from learning.airl_UR.train_model import train_actor_critic, train_discrim, train_actor_critic_process

# ----------- Import peter stuff -----------
# from learning.SKRL_UR_env_Q_learning import URSim_SKRL_env


#from learning.environments.SKRL_UR_env_PPO import URSim_SKRL_env

#Flick task environment
from learning.environments.UR_env_Flick_TASK_C1_easy import URSim_SKRL_env


from learning.models.Generate_models import get_PPO_agent_SKRL, get_TD3_agent, make_TD3_models, Policy, Value
from learning.config_files.generate_config_file import generate_config_file
from skrl.envs.wrappers.torch import wrap_env

import time

# Calculating the log_probability (log_pis), of the actions taken using the GAIL repo implementation
from learning.airl_UR.utils.utils import get_entropy, log_prob_density


from torch import multiprocessing
from torch.multiprocessing import Process
from torch.multiprocessing import Lock
torch.set_default_dtype(torch.float32)

import copy
import scipy

from shapely import Polygon





def get_manual_reward(next_states):

            
    '''
    Summary: This is the reward function for the RL mujoco simulation task.
                The reward is based on the proximity to 0

    ARGS:
        args: The arguments that is passed from the main script
        robot: The robot model that also contains all mujoco data and model information

    RETURNS:
        punishment: Quantitative value that represents the performance of the robot, based on intended task specified in this function.
    '''

    start_index = 20-1

    positions_list = []

    for i in range(6):
        vals = next_states[start_index + 3*i:start_index + 3*(i+1)]
        positions_list.append(vals)



    z_position_error = 0
    desired_z_position = 0.1768

    xy_positions = []
    
    for positions in positions_list:
        xy_positions.append([positions[0],
                        positions[1]])
        
        z_position_error += abs(positions[2] - desired_z_position)



    cloth_Polygon = Polygon(xy_positions)
    area = cloth_Polygon.area # [meters]


    # normalize the range [0 - 0.136], to [0 - 1] (full area is nearly impossible to achieve)
    area_normalized = min(area / 0.110, 1)

    #normalize error between [0 and 1] from [0, 3]

    z_position_error_average  = z_position_error/3

    performance_metric = area_normalized - z_position_error_average


    reward = performance_metric



    return reward


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., 10-1)
    return h


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
    parser.add_argument('--clip_param_critic', type=float, default=0.2, 
                        help='clipping parameter for PPO (default: 0.2)')
    parser.add_argument('--discrim_update_num', type=int, default=10, 
                        help='update number of discriminator (default: 2)')
    parser.add_argument('--actor_critic_update_num', type=int, default=10, 
                        help='update number of actor-critic (default: 10)')
    parser.add_argument('--total_sample_size', type=int, default=1, 
                        help='Mini batch size')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size to update (default: 64)')
    parser.add_argument('--suspend_accu_exp', type=float, default=0.9999,
                        help='accuracy for suspending discriminator about expert data (default: 0.8)')
    parser.add_argument('--suspend_accu_gen', type=float, default=0.9999,
                        help='accuracy for suspending discriminator about generated data (default: 0.8)')
    parser.add_argument('--max_iter_num', type=int, default=1000000,
                        help='maximal number of main iterations (default: 4000)')
    parser.add_argument('--seed', type=int, default=np.random.randint(1e6),
                        help='random seed (default: 500)')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='tensorboardx logs directory')
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


def get_IRL_reward(memory, AIRL_trainer):
    torch.set_num_threads(1)

    with torch.no_grad():
        observation_space = len(memory[0][0])

        states = torch.tensor(np.array([entry[0] for entry in memory])).type(dtype=torch.float32).view(-1, observation_space) # Stack states vertically
        actions = torch.vstack([entry[1] for entry in memory])  # Convert actions to array
        masks = torch.tensor([entry[3] for entry in memory])    # Convert masks to array	  
        log_pis = torch.tensor([entry[4] for entry in memory])    # Convert masks to array
        next_state = torch.tensor(np.array([entry[5] for entry in memory])).type(dtype=torch.float32).view(-1, observation_space)    # Convert masks to array




        AIRL_trainer.disc.eval()
        irl_reward, D = AIRL_trainer.get_reward(states, actions, masks, log_pis, next_state)


        for i, val in enumerate(memory):
            val[2] = irl_reward[i]


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





def run_rollout_loop(actor, episodes_per_mini_batch, memory, env, args, debug_expert_memory=None):
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
                action = debug_expert_memory[steps - 1 + 250*episodes_count][1]


                action_tensor = torch.tensor(action, dtype=torch.float32).view(1, -1)

                log_pis = log_prob_density(action_tensor, mu, std).item()


            val = env.step(action)
            next_state, reward, done, _, _ = val



            if done:    
                mask = 1
            else:
                mask = 0

            # if steps == 1:
            #     print("states: ", state)
            #     print("actions: ", action)
            #     input()


            memory.append([state, action_tensor, reward, mask, log_pis, next_state])
            


            state = next_state



            if done:
                break
        





        episodes_count += 1




    return memory



def Generate_memory_from_expert(actor, episodes_per_mini_batch, memory, env, args, debug_expert_memory=None):
    torch.set_num_threads(1)
    

    episodes_per_mini_batch = args.total_sample_size
    episodes_count = 0

    while episodes_count < episodes_per_mini_batch: 
        state, _ = env.reset()
        steps = 0


        for _ in range(10000): 

            steps += 1


            state = reward = debug_expert_memory[steps - 1 + 250*episodes_count][0]
            action = torch.tensor(debug_expert_memory[steps - 1 + 250*episodes_count][1], dtype=torch.float32).view(1, -1)
            reward = debug_expert_memory[steps - 1 + 250*episodes_count][2]
            done = debug_expert_memory[steps - 1 + 250*episodes_count][3]
            log_pis = debug_expert_memory[steps - 1 + 250*episodes_count][4]

            next_state = debug_expert_memory[steps - 1 + 250*episodes_count][5]



            if done:    
                mask = 1
            else:
                mask = 0


            memory.append([state, action, reward, mask, log_pis, next_state])
            



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



    # IRL training
    # ------- Agent settings -------
    args.gamma = 0.99
    args.lamda = 0.90
    args.actor_critic_update_num = 20
    args.batch_size = 64
    args.hidden_size = 128
    args.learning_rate = 2e-4
    agents = 0


    args.total_sample_size = 10
    args.entropy_gain_PPO = 1e-4
    args.critic_gain_PPO = 0.1
    args.l2_rate = 5e-4
    off_policy_data_size = 250*args.total_sample_size*(agents + 1)


    # ------- AIRL settings -------


    hidden_layer_nodes_r=(256,256)
    hidden_layer_nodes_v=(256,256)
    airl_gamma = args.gamma
    airl_learning_rate = 3e-6
    airl_epocs = 10
    AIRL_L2_weight_decay = 5e-4
    Expert_mini_batch = args.batch_size



    # ------- General training settings -------
    env_args.agent_disabled = False
    env_args.expert_demonstration = False



    env_render_mode = "rgb_array"

    # NOTE: It is a mistake if both BC actor and pretrained actor are both True!
    use_BC_actor = False
    use_BC_disc = False


    BC_model_path = "learning/BC/pretrained_actors/25-05-06_14-36-23"
    BC_disc_path = "learning/airl_UR/Saved_models/AIRL_UR_FLICK_3_TASK_25-05-20_06-00-50"


    use_pretrained_RL  = False
    use_pretrained_IRL = True





    # Load RL model
    model_path = "learning/airl_UR/saved_models_best/2025_05_03/Saved_models/AIRL_UR_FLICK_3_TASK_25-05-02_14-25-06"
    

    model_candidate = "latest"




    # Load AIRL model
    disc_model_path = "Data/C1/C1_state_action/C1_data_IRL_state_action/saved_models/AIRL_UR_FLICK_3_TASK_25-05-26_16-16-57"


    disc_model_candidate = "latest"





    expert_address = "learning/airl_UR/expert_demo/C1_easy_25-04-28_11-54-27URSim_SKRL_env_PPO/expert_memory_modified_no_time_no_performance_metric.pkl"
    
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





    memory_off_policy = deque(maxlen=off_policy_data_size)
    
    #env = wrap_env(env)




    # memory = run_rollout_loop(actor, episodes_per_mini_batch, memory, env,args, debug_expert_memory=expert_memory)


    episodes_per_mini_batch = args.total_sample_size



    perturbation_mean_list= []

    perturbation_interval_list = []


    for i in range(20):
        print("iteration :: ", i)
        # perturbation_list = [0.0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]
        perturbation_list = np.arange(0,3.14, 0.1)
        perturbation_mean = []
        perburbation_interval = []

        for perturbation_val in perturbation_list:
            memory = deque()
            # memory = run_rollout_loop(actor, episodes_per_mini_batch, memory, env,args, debug_expert_memory=expert_memory)


            memory = Generate_memory_from_expert(actor, episodes_per_mini_batch, memory, env,args, debug_expert_memory=expert_memory)
        
            for transition in memory:
                reward = get_manual_reward(copy.deepcopy(transition[5]))

        
            # Make data from expert


            for perturb_index, transition in enumerate(memory):
                if transition[3] == True:
                    transition_copy = copy.deepcopy(transition)

                    obs_noise = np.random.normal(0, perturbation_val, size=len(transition[0]))  # small Gaussian noise
                    action_noise = np.random.normal(0, perturbation_val, size=len(transition[1]))  # small Gaussian noise
                    actions_exp_np = transition_copy[1].cpu().numpy()

                    transition_copy[0] += obs_noise  # pertubate states

                    transition_copy[1] = torch.tensor(actions_exp_np + action_noise, dtype=transition_copy[1].dtype, device=transition_copy[1].device)  # pertubate states


                    transition_copy[5] += obs_noise  # Add all rewards


                    memory[perturb_index] = copy.deepcopy(transition_copy)



            memory = get_IRL_reward(memory=memory, AIRL_trainer=AIRL_trainer)
            # memory.append([state, action_tensor, reward, mask, log_pis, next_state])
            # Get reward average of policy, and append transitions to off policy memory
            reward_epoch_end = []
            reward_sum = 0
            mem_epoch_count = 0

            for transitions in memory:
                if transitions[3] == True:
                    # transitions[2] = get_manual_reward(copy.deepcopy(transitions[5]))

                    reward_epoch_end.append(transitions[2]) # Add all rewards

                    mem_epoch_count += 1

                reward_sum += transitions[2]
                memory_off_policy.append(transitions)

            reward_epoch_end = np.array(reward_epoch_end)

            rewards_average_end = reward_epoch_end.mean()
            rewards_average_end_interval_95 = mean_confidence_interval(reward_epoch_end, confidence=0.95)

            reward_average = reward_sum / len(memory)
            perturbation_mean.append(rewards_average_end)
            perburbation_interval.append(rewards_average_end_interval_95)
        

        perturbation_mean_list.append(perturbation_mean)
        perturbation_interval_list.append(perburbation_interval)



    arr1 = np.array(perturbation_mean_list)
    arr2 = np.array(perturbation_interval_list)


    mean1 = np.mean(arr1, axis=0)
    mean2 = np.mean(arr2, axis=0)

    print("mean reward:")
    print("mean1 =", ", ".join([f"{x:.8f}" for x in mean1]))

    print("confidence interval reward:")

    print("mean2 =", ", ".join([f"{x:.8f}" for x in mean2]))


    input()

    
    
    


if __name__=="__main__":

    main()