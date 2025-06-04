import os
import gym
import pickle
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 

from learning.gail.utils.utils import *
from learning.gail.utils.zfilter import ZFilter
from learning.gail.model import Actor, Critic
from learning.gail.airl import AIRL
from learning.gail.train_model import train_actor_critic, train_discrim

# ----------- Import peter stuff -----------
# from learning.SKRL_UR_env_Q_learning import URSim_SKRL_env
from learning.environments.SKRL_UR_env_PPO import URSim_SKRL_env
from learning.models.Generate_models import get_PPO_agent_SKRL, get_TD3_agent, make_TD3_models, Policy, Value
from learning.config_files.generate_config_file import generate_config_file


import time
from skrl.envs.wrappers.torch import wrap_env


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
parser.add_argument('--learning_rate', type=float, default=3e-4, 
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-2, 
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--clip_param', type=float, default=0.2, 
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--discrim_update_num', type=int, default=5, 
                    help='update number of discriminator (default: 2)')
parser.add_argument('--actor_critic_update_num', type=int, default=10, 
                    help='update number of actor-critic (default: 10)')
parser.add_argument('--total_sample_size', type=int, default=4096, 
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--batch_size', type=int, default=1024, 
                    help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.90,
                    help='accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.90,
                    help='accuracy for suspending discriminator about generated data (default: 0.8)')
parser.add_argument('--max_iter_num', type=int, default=1000,
                    help='maximal number of main iterations (default: 4000)')
parser.add_argument('--seed', type=int, default=np.random.randint(1e6),
                    help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()


def main():
    # env = gym.make(args.env_name, render_mode="rgb_array") # human, rgb_array

    task_device = "cuda"

    actor_critic_learning_rate = 3e-6
    env_args=generate_config_file()
    Training_model = "PPO"

    # instantiate the agent
    # (assuming a defined environment <env>)
    environment_name = time.strftime("%y-%m-%d_%H-%M-%S") + "URSim_SKRL_env_" + Training_model
    env = URSim_SKRL_env(args = env_args, name=environment_name, render_mode="rgb_array")
    # it will check your custom environment and output additional warnings if needed
    env = wrap_env(env)
    
    print("environment device: ", env.device)
    #env.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Env observaton shape: ", env.observation_space.shape[0])
    print("Env action shape: ", env.action_space.shape[0])
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    running_state = ZFilter((num_inputs,), clip=5)

    actor = Actor(num_inputs, num_actions, args).to(task_device)
    critic = Critic(num_inputs, args).to(task_device)

    AIRL_trainer = AIRL(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device(task_device), #  "cuda" if args.cuda else "cpu"
        seed=args.seed,
        #rollout_length=args.rollout_length
    )



    actor_optim = optim.Adam(actor.parameters(), lr=actor_critic_learning_rate)
    

    critic_optim = optim.Adam(critic.parameters(), lr=actor_critic_learning_rate, 
                              weight_decay=args.l2_rate) 
    

    #Load UR expert data
    # expert_demo = np.load("/home/peter/OneDrive/Skrivebord/Uni stuff/Masters/masters_code/mj_sim/learning/gail/expert_demo/Expert_data_constant_joint/all_state_actions.npy")

    # # With controllable gripper, remove that controllability for now.
    # demonstrations = np.array(expert_demo)



    
    log_address = "learning/gail/" + args.logdir + '/' + str(time.time()) + '_' + args.env_name + '_' + 'GAIL'

    writer = SummaryWriter(log_address)


    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        # Load with a new discrimininator
        #discrim.load_state_dict(ckpt['discrim'])
        

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    
    # Generate expert memory from file
    expert_memory = pickle.load(open("/home/peter/OneDrive/Skrivebord/Uni stuff/Masters/masters_code/mj_sim/state_action_data/25-01-11_12-06-55URSim_SKRL_env_PPO/expert_memory.pkl", "rb"))
    states_exp =    torch.tensor(np.array([entry[0] for entry in expert_memory])).cuda().to(torch.float32)  # Stack states vertically
    actions_exp =   torch.tensor(np.array([entry[1] for entry in expert_memory])).cuda().to(torch.float32)  # Convert actions to array
    dones_exp =     torch.tensor([entry[3] for entry in expert_memory]).cuda().to(torch.int8)  # Convert actions to array
    log_pis_exp =   torch.tensor([entry[4] for entry in expert_memory]).cuda()  # Convert actions to array
    next_states_exp = torch.tensor(np.array([entry[5] for entry in expert_memory])).cuda().to(torch.float32)  # Convert actions to array

    expert_data = [
                    states_exp,
                    actions_exp,
                    [], # Rewards exp that is not used
                    dones_exp,
                    [], # new log pis has not been calculated yet
                    next_states_exp,
                    ]


    episodes = 0
    train_discrim_flag = True

    for iter in range(args.max_iter_num):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores_IRL = []

        while steps < args.total_sample_size: 
            state, _ = env.reset() # Needs seed argument for standard envs, but not for the UR env

            score_IRL = 0
            #state = running_state(np.array(state.cpu()[0]))
            state = np.array(state.cpu()[0]) # transform to np.array

            for _ in range(10000): 
                
                if args.render:
                    env.render()

                steps += 1
                state_type_transformed = torch.tensor(state).unsqueeze(0).cuda()
                mu, std = actor(state_type_transformed)
                
                #action = get_action(mu, std)[0]
                action, log_pis = actor.reparameterize(mu, torch.log(std))
                action = action.detach()

                next_state, reward, done, _, _ = env.step(action)

                irl_reward = AIRL_trainer.get_reward(state_type_transformed, action, done, log_pis, next_state)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, irl_reward, mask, log_pis, next_state])

                #next_state = running_state(np.array(next_state.cpu()[0]))
                state = np.array(next_state.cpu()[0])

                # Might be needed that we log the actual task reward for the imitator
                # score += reward.cpu().numpy()[0][0]

                score_IRL += irl_reward.cpu().numpy()[0][0]

                if done:
                    break
            

            episodes += 1
            scores_IRL.append(score_IRL)

        
        score_IRL_avg = np.mean(scores_IRL)
        #print('{}:: {} episode score is {:.2f}'.format(iter, episodes, score_avg))
        print('{}:: {} IRL reward is {:.2f}'.format(iter, episodes, score_IRL_avg))
        writer.add_scalar('log/score_IRL', float(score_IRL_avg), iter)

        actor.train(), critic.train()

        if train_discrim_flag:
            states_exp = torch.tensor(np.array([entry[0] for entry in expert_memory])).cuda().to(torch.float32)
            new_log_pis = torch.flatten(actor.generate_log_probs(states_exp))

            expert_data[4] = new_log_pis

            # for i, (state, action, reward, done, log_pis, next_state) in enumerate(expert_memory):
            #      expert_memory[i] = [state, action, reward, done, new_log_pis[i], next_state]

            expert_acc, learner_acc = AIRL_trainer.update(memory=memory, expert_data=expert_data)

            print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))

            if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
                train_discrim_flag = False
            else:
                train_discrim_flag = True


        if not(expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen):
            train_discrim_flag = True


        train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args)

        if iter % 10:
            # score_avg = int(score_avg)

            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_IRL_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'z_filter_n':running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': args,
                # 'score': score_avg
            }, filename=ckpt_path)

if __name__=="__main__":
    main()