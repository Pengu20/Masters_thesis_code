import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from robots.base_robot import BaseRobot
from utils.learning import MLP, ReplayMemory

# from timer.timer import T
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3:
    def __init__(
        self,
        sim,
        robot: BaseRobot,
        args,
        o_dim: int,
        a_dim: int,
        a_max: float,
        replay_buffer: ReplayMemory = ReplayMemory(int(10e6)),
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: float = 2,
        learning_rate: float = 3e-4,
        test_env=None,
        seed: int = 0,
    ):
        self.robot = robot
        self._data = robot._data
        self._model = robot._model
        self._args = args
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.a_max = (a_max * torch.ones(self.a_dim, dtype=torch.float32)).to(device)




        
        self.actor = MLP(self.o_dim, self.a_dim, self.a_max).to(device)
        self.Q1 = MLP(
            self.o_dim + self.a_dim, 1, output_activation_func=lambda x: x
        ).to(device)
        self.Q2 = MLP(
            self.o_dim + self.a_dim, 1, output_activation_func=lambda x: x
        ).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )

        self.Q1_target = copy.deepcopy(self.Q1)
        self.Q2_target = copy.deepcopy(self.Q2)
        self.Q1_optimizer = torch.optim.Adam(self.Q1.parameters(), lr=learning_rate)
        self.Q2_optimizer = torch.optim.Adam(self.Q2.parameters(), lr=learning_rate)

        self._replay_buffer = replay_buffer






        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.test_env = test_env
        self.train_seed = seed
        self.eval_seed = self.train_seed + 100

        self.total_it = 0
        if not os.path.exists("./data"):
            os.makedirs("./data")

        self.t = 0

    @property
    def name(self) -> str:
        return "td3"

    @property
    def replay_buffer(self) -> ReplayMemory:
        return self._replay_buffer

    @replay_buffer.setter
    def replay_buffer(self, new_replay_buffer: ReplayMemory) -> None:
        self._replay_buffer = new_replay_buffer

    def select_action(self, o: np.ndarray) -> torch.FloatTensor:
        o = torch.FloatTensor(o.reshape(1, -1)).to(device)
        a = self.actor.forward(o).flatten().to(device)
        return a

    def optimize(self, batch_size: int, tensorboard_writer: SummaryWriter = None):
        self.total_it += 1

        # Sample replay buffer
        batch = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                (torch.randn_like(batch.action) * self.policy_noise)
                .clamp(-self.noise_clip, self.noise_clip)
                .to(device)
            )

            next_action = (self.actor_target(batch.next_state) + noise).clamp(
                -self.a_max, self.a_max
            )

            next_action = next_action.type(torch.FloatTensor).to(device)
            # Compute the target Q value
            target_Q1, target_Q2 = (
                self.Q1_target(torch.cat([batch.next_state, next_action], 1)),
                self.Q2_target(torch.cat([batch.next_state, next_action], 1)),
            )
            target_Q = torch.min(target_Q1, target_Q2)

            # target_Q1, target_Q2 = self.Q1_target(torch.cat([batch.next_state.to(device), next_action], 1)), self.Q2_target(
            #     torch.cat([batch.next_state, next_action], 1))
            # target_Q = torch.min(target_Q1, target_Q2)

            target_Q = batch.reward + (1 - batch.done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = (
            self.Q1(torch.cat([batch.state, batch.action], 1)),
            self.Q2(torch.cat([batch.state, batch.action], 1)),
        )

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.Q1_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        critic_loss.backward()
        self.Q1_optimizer.step()
        self.Q2_optimizer.step()



        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.Q1(
                torch.cat([batch.state, self.actor(batch.state)], 1)
            ).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.Q1.parameters(), self.Q1_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.Q2.parameters(), self.Q2_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss/actor", actor_loss, self.t)

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("Loss/critic", critic_loss, self.t)

    def save(self, name):
        torch.save(self.Q1.state_dict(), name + "Q1")
        torch.save(self.Q2.state_dict(), name + "Q2")
        torch.save(self.Q1_optimizer.state_dict(), name + "Q1_optimizer")
        torch.save(self.Q2_optimizer.state_dict(), name + "Q2_optimizer")
        torch.save(self.actor.state_dict(), name + "actor")
        torch.save(self.actor_optimizer.state_dict(), name + "actor_optimizer")

    def load(self, name):
        if torch.cuda.is_available():
            self.Q1.load_state_dict(torch.load(name + "Q1"))
            self.Q1_target = copy.deepcopy(self.Q1)
            self.Q2.load_state_dict(torch.load(name + "Q2"))
            self.Q2_target = copy.deepcopy(self.Q2)
            self.Q1_optimizer.load_state_dict(torch.load(name + "Q1_optimizer"))
            self.Q2_optimizer.load_state_dict(torch.load(name + "Q2_optimizer"))
            self.actor.load_state_dict(torch.load(name + "actor"))
            self.actor_optimizer.load_state_dict(torch.load(name + "actor_optimizer"))
        else:
            self.Q1.load_state_dict(
                torch.load(name + "Q1", map_location=torch.device("cpu"))
            )
            self.Q1_target = copy.deepcopy(self.Q1)
            self.Q2.load_state_dict(
                torch.load(name + "Q2", map_location=torch.device("cpu"))
            )
            self.Q2_target = copy.deepcopy(self.Q2)
            self.Q1_optimizer.load_state_dict(
                torch.load(name + "Q1_optimizer", map_location=torch.device("cpu"))
            )
            self.Q2_optimizer.load_state_dict(
                torch.load(name + "Q2_optimizer", map_location=torch.device("cpu"))
            )
            self.actor.load_state_dict(
                torch.load(name + "actor", map_location=torch.device("cpu"))
            )
            self.actor_optimizer.load_state_dict(
                torch.load(name + "actor_optimizer", map_location=torch.device("cpu"))
            )

    def a_noisy(self, o: np.ndarray) -> np.ndarray:
        self.a_max = self.a_max.cpu() if self.a_max.is_cuda else self.a_max

        a_noisy = (
            self.select_action(np.array(o)).cpu().detach().numpy()
            + np.random.normal(0, self.a_max * self._args.expl_noise, size=self.a_dim)
        ).clip(-self.a_max, self.a_max)

        self.a_max = self.a_max.to(device) if not self.a_max.is_cuda else self.a_max

        return np.array(a_noisy, dtype=np.float32)
