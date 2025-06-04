import numpy as np
import random
import torch
from torch.distributions import Normal

def log_prob_density(x, mu, std):
    # Gail implementation

    torch.set_num_threads(1)


    dist = Normal(mu, std)
    log_prob = dist.log_prob(x).sum(dim=1)
    

    return log_prob





mu = torch.tensor([1.0])
std = torch.tensor([0.001])


for i in np.arange(0.9, 1.1, 0.01):

    dist = Normal(mu, std)
    log_prob = dist.log_prob(torch.tensor([i]))

    print("log_prob: ",log_prob )
    