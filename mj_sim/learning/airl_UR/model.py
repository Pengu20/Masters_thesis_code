import torch
import torch.nn as nn
import numpy as np

from learning.airl.gail_airl_ppo.network.utils import build_mlp, reparameterize, evaluate_lop_pi, calculate_log_pi


# class Actor(nn.Module):
#     def __init__(self, num_inputs, num_outputs, args):
#         super(Actor, self).__init__()

#         #hidden_size = int(args.hidden_size/8)
#         hidden_size = args.hidden_size
#         self.fa1 = nn.Linear(num_inputs, hidden_size)
#         self.fa2 = nn.Linear(hidden_size, hidden_size)
#         #self.fc3 = nn.Linear(hidden_size, num_outputs)
#         self.fa3 = nn.Linear(hidden_size, num_outputs)
#         # one single log-std parameter per action dim:
#         init_log_std = torch.log(torch.tensor([1.0]))   # start with σ=1.0
#         self.log_std = nn.Parameter(torch.ones(num_outputs) * init_log_std)

#         #self.fc3.weight.data.mul_(0.1)
#         # self.fc3.bias.data.mul_(0.0)

#         #self.log_std = nn.Parameter(torch.zeros(1, num_outputs))
     
#         # For actor training
#         self.entropy_gain = args.entropy_gain_PPO
#         self.critic_gain = args.critic_gain_PPO
    

#     def forward(self, x):
#         x = torch.relu(self.fa1(x))
#         x = torch.relu(self.fa2(x))
#         mu = self.fa3(x)
#         std = torch.exp(self.log_std)         # constant across states!
#         return mu, std


#     def generate_log_probs(self, states):

#         mu, std = self.forward(states)
#         _, log_prob = reparameterize(mu, torch.log(std))
#         return log_prob

#     def reparameterize(self, mu, log_std):
#         return reparameterize(mu, log_std)

#     # def evaluate_log_pi(self, states, actions):
#     #     mu, std = self.forward(states)
#     #     return evaluate_lop_pi(mu, self.logstd, actions) # The tanh, is a quick fix, not sure it works on the long run

#     def evaluate_log_pi(self, states, actions):
#         mu, std = self.forward(states)
#         return evaluate_lop_pi(mu, torch.log(std), actions)




class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Actor, self).__init__()

        #hidden_size = int(args.hidden_size/8)
        hidden_size = args.hidden_size
        self.fa1 = nn.Linear(num_inputs, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        #self.fc3 = nn.Linear(hidden_size, num_outputs)
        self.fa3 = nn.Linear(hidden_size, num_outputs * 2)
        with torch.no_grad():
            self.fa3.bias[:num_outputs].fill_(0.0)
            self.fa3.bias[num_outputs:].fill_(2.0)
        # self.log_std_layer = nn.Linear(num_outputs, num_outputs)


        #self.fc3.weight.data.mul_(0.1)
        # self.fc3.bias.data.mul_(0.0)

        #self.log_std = nn.Parameter(torch.zeros(1, num_outputs))
     
        # For actor training
        self.entropy_gain = args.entropy_gain_PPO
        self.critic_gain = args.critic_gain_PPO
    


    def forward(self, x):
        x = torch.relu(self.fa1(x))
        
        x = torch.relu(self.fa2(x))


        output = self.fa3(x)
        mu, std_val = output.chunk(2, dim=-1)
        #mu = self.fc3(x)

        std = torch.relu(std_val)
        std = torch.clamp(std, min=0.01, max=20)

        return mu, std
    

    def generate_log_probs(self, states):

        mu, std = self.forward(states)
        _, log_prob = reparameterize(mu, torch.log(std))
        return log_prob

    def reparameterize(self, mu, log_std):
        return reparameterize(mu, log_std)

    # def evaluate_log_pi(self, states, actions):
    #     mu, std = self.forward(states)
    #     return evaluate_lop_pi(mu, self.logstd, actions) # The tanh, is a quick fix, not sure it works on the long run

    def evaluate_log_pi(self, states, actions):
        mu, std = self.forward(states)
        return evaluate_lop_pi(mu, torch.log(std), actions)



class Critic(nn.Module):
    def __init__(self, num_inputs, args):
        super(Critic, self).__init__()

        hidden_size = args.hidden_size
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v




class model_dynamics(nn.Module):
    def __init__(self, input_obs, input_action, args):
        super(model_dynamics, self).__init__()

        #hidden_size = int(args.hidden_size/8)
        hidden_size = args.hidden_size

        self.fc1_obs = nn.Linear(input_obs, hidden_size)
        self.fc1_action = nn.Linear(input_action, hidden_size)


        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        #self.fc3 = nn.Linear(hidden_size, num_outputs)
        self.fc3 = nn.Linear(hidden_size, input_obs)
    
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # 0.01 is default



    def forward(self, obs, action):
        """
        observation and action with leaky relu into residual connection with observation again
        
        """
        obs_layer = self.leaky_relu(self.fc1_obs(obs))
        action_layer = self.leaky_relu(self.fc1_action(action))
        # now we can reshape `c` and `f` to 2D and concat them

        combined = torch.cat((obs_layer.view(obs_layer.size(0), -1),
                                action_layer.view(action_layer.size(0), -1)), dim=1)
        
        x = self.leaky_relu(self.fc2(combined))

        next_obs = obs + self.fc3(x)


        return next_obs








class Discriminator(nn.Module):
    def __init__(self, num_inputs, args):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        prob = torch.sigmoid(self.fc3(x))
        return prob