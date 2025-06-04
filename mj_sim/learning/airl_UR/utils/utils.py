import math
import torch
from torch.distributions import Normal
import numpy as np


def get_action(mu, std):
    action = torch.normal(mu, std).cpu()
    action = action.data.numpy()
    return action

def get_entropy(mu, std):
    torch.set_num_threads(1)
    dist = Normal(0, std)
    entropy = dist.entropy().mean()
    return entropy


def log_prob_density(x, mu, std):
    # Gail implementation

    torch.set_num_threads(1)


    dist = Normal(mu, std)
    log_prob = dist.log_prob(x).sum(dim=1)
    

    return log_prob

def gaussian_probability_density(x, mu,std):

    # f = 1 / np.sqrt(2*np.pi*np.power(std, 2)) * np.exp( -1 * np.power((x-mu), 2) / (2 * np.power(std, 2)))

    f = 1 / torch.sqrt(2*torch.pi*torch.pow(std, 2)) * torch.exp( -1 * torch.pow(x-mu, 2) / (2 * torch.pow(std, 2)))

    # var = std ** 2
    # denom = torch.sqrt(2 * torch.pi * var)
    # num = torch.exp(-0.5 * ((x - mu) ** 2) / var)



    return f




# def log_prob_density(x, mu, std):


#     dist = Normal(mu, std)
#     d = dist.log_prob(x).sum(1, keepdim=True)
#     return d



def get_reward(discrim, state, action):
    state = torch.Tensor(state)
    action = torch.Tensor(action)
    state_action = torch.cat([state, action])


    with torch.no_grad():
        return -math.log(discrim(state_action)[0].item())

def save_checkpoint(state, filename):
    torch.save(state, filename)