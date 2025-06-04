import torch
import torch.nn as nn
import numpy as np

from learning.airl.gail_airl_ppo.network.utils import build_mlp, reparameterize, evaluate_lop_pi, calculate_log_pi


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Actor, self).__init__()

        #hidden_size = int(args.hidden_size/8)
        hidden_size = args.hidden_size
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

        self.logstd = nn.Parameter(torch.zeros(1, num_outputs))
        #self.logstd = torch.zeros_like(mu)


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        
        std = torch.exp(self.logstd)

        mu = mu
        return mu, std
    
    def reparameterize(self, means, log_stds):
        noises = torch.randn_like(means)
        us = means + noises * log_stds.exp()
        actions = torch.tanh(us).detach()
        return actions, calculate_log_pi(log_stds, noises, actions)


    # def evaluate_log_pi(self, states, actions):
    #     mu, std = self.forward(states)
    #     return evaluate_lop_pi(mu, self.logstd, actions) # The tanh, is a quick fix, not sure it works on the long run





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