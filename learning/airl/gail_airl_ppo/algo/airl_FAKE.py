import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from learning.airl.gail_airl_ppo.network import AIRLDiscrim

import numpy as np






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
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)



class AIRL:

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=10000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc_r=(100, 100), units_disc_v=(100, 100),
                 epoch_ppo=50, epoch_disc=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

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

    def update(self, log_pis_exp, memory, expert_memory):
        self.learning_steps += 1


        states = np.array([entry[0] for entry in memory])  # Stack states vertically
        dones = np.array([entry[3] for entry in memory])  # Convert actions to array
        log_pis = np.array([entry[4] for entry in memory])  # Convert actions to array
        next_states = np.array([entry[5] for entry in memory])  # Convert actions to array

        states_exp = np.array([entry[0] for entry in expert_memory])  # Stack states vertically
        actions_exp = np.array([entry[1] for entry in expert_memory])  # Stack states vertically
        dones_exp = np.array([entry[3] for entry in expert_memory])  # Convert actions to array
        log_pis_exp = np.array([entry[4] for entry in expert_memory])  # Convert actions to array
        next_states_exp = np.array([entry[5] for entry in expert_memory])  # Convert actions to array



        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # # Samples from current policy's trajectories.
            # states, _, _, dones, log_pis, next_states = \
            #     self.buffer.sample(self.batch_size)
            # # Samples from expert's demonstrations.
            # states_exp, actions_exp, _, dones_exp, next_states_exp = \
            #     self.buffer_exp.sample(self.batch_size)
            
            # # Calculate log probabilities of expert actions.
            # with torch.no_grad():
            #     log_pis_exp = self.actor.evaluate_log_pi(
            #         states_exp, actions_exp)
                


            # Update discriminator.
        acc_pi, acc_exp =     self.update_disc(
                                    states, dones, log_pis, next_states, states_exp,
                                    dones_exp, log_pis_exp, next_states_exp
                                )

        # # We don't use reward signals here,
        # states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # # Calculate rewards.
        # rewards = self.disc.calculate_reward(
        #     states, dones, log_pis, next_states)

        # # Update PPO using estimated rewards.
        # self.update_ppo(
        #     states, actions, rewards, dones, log_pis, next_states, writer)

        return acc_pi, acc_exp
    





    def get_reward(self, states, actions, dones, log_pis, next_states):
        return self.disc.calculate_reward(states, dones, log_pis, next_states)

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)


        return acc_pi, acc_exp