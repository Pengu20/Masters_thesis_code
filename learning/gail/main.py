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
from learning.gail.model import Actor, Critic, Discriminator
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
parser.add_argument('--hidden_size', type=int, default=256, 
                    help='hidden unit size of actor, critic and discrim networks (default: 100)')
parser.add_argument('--learning_rate', type=float, default=3e-3, 
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3, 
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--clip_param', type=float, default=0.2, 
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--discrim_update_num', type=int, default=5, 
                    help='update number of discriminator (default: 2)')
parser.add_argument('--actor_critic_update_num', type=int, default=10, 
                    help='update number of actor-critic (default: 10)')
parser.add_argument('--total_sample_size', type=int, default=4096, 
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--batch_size', type=int, default=128, 
                    help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.99,
                    help='accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.99,
                    help='accuracy for suspending discriminator about generated data (default: 0.8)')
parser.add_argument('--max_iter_num', type=int, default=4000,
                    help='maximal number of main iterations (default: 4000)')
parser.add_argument('--seed', type=int, default=np.random.randint(1e6),
                    help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()


def main():
    # env = gym.make(args.env_name, render_mode="rgb_array") # human, rgb_array
    actor_critic_learning_rate = 3e-5
    env_args=generate_config_file()
    Training_model = "PPO"

    # instantiate the agent
    # (assuming a defined environment <env>)
    environment_name = time.strftime("%y-%m-%d_%H-%M-%S") + "URSim_SKRL_env_" + Training_model
    env = URSim_SKRL_env(args = env_args, name=environment_name, render_mode="rgb_array")
    # it will check your custom environment and output additional warnings if needed
    env = wrap_env(env)
    
    
    #env.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Env observaton shape: ", env.observation_space.shape[0])
    print("Env action shape: ", env.action_space.shape[0])
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    running_state = ZFilter((num_inputs,), clip=5)

    actor = Actor(num_inputs, num_actions, args)
    critic = Critic(num_inputs, args)
    discrim = Discriminator(num_inputs + num_actions, args)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_critic_learning_rate)
    
    critic_optim = optim.Adam(critic.parameters(), lr=actor_critic_learning_rate, 
                              weight_decay=args.l2_rate) 
    
    discrim_optim = optim.Adam(discrim.parameters(), lr=args.learning_rate)
    


    # load demonstrations
    # expert_demo, _ = pickle.load(open('./expert_demo/expert_demo.p', "rb"))
    # expert_demo = pickle.load(open('./expert_demo/expert_demonstrations.p', "rb"))

    #Load UR expert data
    expert_demo = np.load("/home/peter/OneDrive/Skrivebord/Uni stuff/Masters/masters_code/mj_sim/learning/gail/expert_demo/Expert_data_constant_joint/all_state_actions.npy")

    # With controllable gripper, remove that controllability for now.
    demonstrations = np.array(expert_demo)
    
    log_address = "learning/gail/" + args.logdir + '/' + str(time.time()) + '_' + args.env_name + '_' + 'GAIL'

    writer = SummaryWriter(log_address)


    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        discrim.load_state_dict(ckpt['discrim'])

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    
    episodes = 0
    train_discrim_flag = True

    for iter in range(args.max_iter_num):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []
        scores_IRL = []

        while steps < args.total_sample_size: 
            state, _ = env.reset() # Needs seed argument for standard envs, but not for the UR env

            score = 0
            score_IRL = 0
            #state = running_state(np.array(state.cpu()[0]))
            state = np.array(state.cpu()[0]) # transform to np.array

            for _ in range(10000): 
                
                if args.render:
                    env.render()

                steps += 1

                mu, std = actor(torch.Tensor(state).unsqueeze(0))


                action = get_action(mu, std)[0]
                next_state, reward, done, _, _ = env.step(torch.Tensor(action))
                irl_reward = get_reward(discrim, state, action)

                if done:
                    mask = 0
                else:
                    mask = 1


                memory.append([state, action, irl_reward, mask])

                #next_state = running_state(np.array(next_state.cpu()[0]))
                state = np.array(next_state.cpu()[0])

                # print("reward: ", reward.cpu().numpy()[0][0])
                score += reward.cpu().numpy()[0][0]
                score_IRL += irl_reward

                if done:
                    break
            
            episodes += 1
            scores.append(score)
            scores_IRL.append(score_IRL)

        
        score_avg = np.mean(scores)
        score_IRL_avg = np.mean(scores_IRL)
        #print('{}:: {} episode score is {:.2f}'.format(iter, episodes, score_avg))
        print('{}:: {} IRL reward is {:.2f}'.format(iter, episodes, score_IRL_avg))
        writer.add_scalar('log/score', float(score_avg), iter)
        

        actor.train(), critic.train(), discrim.train()
        if train_discrim_flag:
            expert_acc, learner_acc = train_discrim(discrim, memory, discrim_optim, demonstrations, args)
            print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
            if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
                train_discrim_flag = False

        train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args)

        if iter % 100:
            score_avg = int(score_avg)

            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'discrim': discrim.state_dict(),
                'z_filter_n':running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': args,
                'score': score_avg
            }, filename=ckpt_path)

if __name__=="__main__":
    main()