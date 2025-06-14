import math
import torch
from torch import nn
import numpy as np


def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.LeakyReLU(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        if hidden_activation != None:
            layers.append(hidden_activation)

        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    return gaussian_log_probs - torch.log(
        1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(means, log_stds):
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)


def atanh(x):
    
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    print("means: ", means)
    print("log_stds: ", log_stds)
    print("actions: ", actions)
    print("arctan: ", np.arctanh(actions.cpu()))
    
    noises = (atanh(actions) - means) / (torch.exp(log_stds) + 1e-8)
    print("noises: ", noises)


    val = calculate_log_pi(log_stds, noises, actions)

    input()
    return val
