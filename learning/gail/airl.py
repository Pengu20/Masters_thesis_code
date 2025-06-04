import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np

from learning.airl.gail_airl_ppo.network.utils import build_mlp
import time

class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, gamma,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):

        rs = torch.flatten(self.g(states))
        vs = torch.flatten(self.h(states))


        next_vs = torch.flatten(self.h(next_states))
        dones = torch.flatten(dones.to(dtype=torch.int))
        # dones = dones.cpu().to(dtype=torch.int).cuda()
        val = rs + self.gamma*(1 - dones)*next_vs - vs

        return val
    
    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        val1 = self.f(states, dones, next_states) 
        val2 = log_pis
        return val1 - val2

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)



class AIRL:

    def __init__(self, state_shape, action_shape, device, seed,
                 gamma=0.995,
                 batch_size=64, lr_disc=3e-4,
                 units_disc_r=(100, 100), units_disc_v=(100, 100), epoch_disc=10):
        print("AIRL device: ", device)

        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc    
        self.learning_steps = 0

    def update(self, memory, expert_data):
        """
        SUMMARY: this method updates the discriminator
        INPUT: memory: the memory buffer of the agent
               expert_data: the expert data in list format for quicker access
        OUTPUT: acc_pi: the accuracy of the discriminator on the agent data
                acc_exp: the accuracy of the discriminator on the expert data
        """
        self.learning_steps += 1

        states =        torch.tensor(np.array([entry[0] for entry in memory])).cuda().to(torch.float32)  # Stack states vertically
        dones =         torch.tensor([entry[3] for entry in memory]).cuda().to(torch.int8)  # Convert actions to array
        log_pis =       torch.tensor(np.array([entry[4].detach().cpu().numpy() for entry in memory])).cuda()  # Convert actions to array
        next_states =   torch.tensor(np.array([entry[5].detach().cpu().numpy() for entry in memory])).cuda().to(torch.float32)  # Convert actions to array

        states_exp = expert_data[0]
        dones_exp = expert_data[3]
        log_pis_exp = expert_data[4]
        next_states_exp = expert_data[5]


        # states_exp =    torch.tensor(np.array([entry[0] for entry in expert_memory])).cuda().to(torch.float32)  # Stack states vertically
        # dones_exp =     torch.tensor(np.array([entry[3] for entry in expert_memory])).cuda().to(torch.int8)  # Convert actions to array
        # log_pis_exp =   torch.tensor(np.array([entry[4].detach().cpu().numpy() for entry in expert_memory])).cuda()  # Convert actions to array
        # next_states_exp = torch.tensor(np.array([entry[5] for entry in expert_memory])).cuda().to(torch.float32)  # Convert actions to array


        for i in range(self.epoch_disc):
            self.learning_steps_disc += 1
            # Update discriminator.

            if i == self.epoch_disc - 1:
                last_run = True
            else:
                last_run = False

            acc_pi, acc_exp =     self.update_disc(
                                        states, dones, log_pis, next_states, states_exp,
                                        dones_exp, log_pis_exp, next_states_exp, last_run
                                    )

        return acc_pi, acc_exp
    


    def get_reward(self, states, actions, dones, log_pis, next_states):
        return self.disc.calculate_reward(states, dones, log_pis, next_states)


    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, last_run):
        # Output of discriminator is (-inf, inf), not [0, 1].


        logits_pi = self.disc(states, dones, log_pis, next_states)

        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()

        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()

        if last_run:
            loss_disc.backward()
        else:
            loss_disc.backward(retain_graph=True)

        self.optim_disc.step()

        
        # Discriminator's accuracies.
        with torch.no_grad():
            acc_pi = (logits_pi < 0).float().mean().item()
            acc_exp = (logits_exp > 0).float().mean().item()


        return acc_pi, acc_exp