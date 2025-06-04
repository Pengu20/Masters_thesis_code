import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from robots.base_robot import BaseRobot
from utils.learning import MLP, ReplayMemory

import itertools

import matplotlib.pyplot as plt

# from timer.timer import T
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Q_learning:
    def __init__(
        self,
        sim,
        robot: BaseRobot,
        args,
        o_dim: int,
        a_dim: int,
        a_max: float,
        replay_buffer: ReplayMemory = ReplayMemory(int(10e6)),

        # RL specific variables. 
        discount: float = 0.75,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        policy_freq: float = 100,
        learning_rate: float = 3e-2,
        test_env=None,
        seed: int = 0,
    ):
        
        
        # Initializing the variables regarding the RL environment
        self.robot = robot
        self._data = robot._data
        self._model = robot._model
        self._args = args
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.a_max = (a_max * torch.ones(self.a_dim, dtype=torch.float32)).to(device)

        self.using_NN = False

        self.learning_rate = learning_rate

        # Possible actions (Move joint positive, or move joint negative) 
        self.robot_actions = [0, 1]

        self.Observational_space_resolution = 100


        self.action_space = np.power(len(self.robot_actions), self.a_dim)
        self.observation_space = self.get_observational_space_size(o_dim)


        # Creating the Q-table, non NN
        self.Q = np.ones((self.observation_space, self.action_space)).astype(float)*10


        # Initializing the RL setup (Input is state and action, output is list of all possible actions)
        # self.Q = MLP(input_dim=self.o_dim + self.a_dim, output_dim=1).to(device)
        # self.Q_target = copy.deepcopy(self.Q)

        # self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)
        # self.Q_target_optimizer = torch.optim.Adam(self.Q_target.parameters(), lr=learning_rate)

        self.initialize_all_actions()


        self._replay_buffer = replay_buffer



        # Initializing the RL parameters 
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.test_env = test_env
        self.train_seed = seed
        self.eval_seed = self.train_seed + 100

        self.total_it = 0
        if not os.path.exists("./data"):
            os.makedirs("./data")

        self.t = 0


        self.Q_table_data = np.array([])
        self.Q_table_mean_data = np.array([])
        self.Q_table_data_var = np.array([])


    @property
    def name(self) -> str:
        return "Q_learning"

    @property
    def replay_buffer(self) -> ReplayMemory:
        return self._replay_buffer

    @replay_buffer.setter
    def replay_buffer(self, new_replay_buffer: ReplayMemory) -> None:
        self._replay_buffer = new_replay_buffer

    def select_action(self, o: np.ndarray, greedy = False) -> torch.FloatTensor:
        # There is always the same 128 possible actions at all times 

        # Select based on Epsilon greedy policy

        epsilon = 0.01

        if (np.random.rand() < epsilon) and (greedy == False): # Return random action with a 10 percent probability
            rand_action = np.random.choice(range(len(self.robot_actions)**self.a_dim))

            return torch.FloatTensor(self.all_actions[rand_action])
        
        else: # Return best action.
            # o = torch.FloatTensor(o.reshape(1, -1)).to(device)


            # all_actions = torch.FloatTensor(np.array(self.all_actions)).to(device)
            # o_repeated = o.repeat(len(self.all_actions), 1)

            # Q_values_list = self.Q.forwar            

            # return torch.FloatTensor(self.all_actions[torch.argmax(Q_values_list).item()])d(torch.cat([o_repeated, all_actions], 1)).to(device)


            # Non neural network version

            o_vec = self.vectorize_observation(o)

            Q_values_list = self.Q[o_vec]

            best_action = np.argmax(Q_values_list)

            # Save the Q-table data
            self.Q_table_data = np.append(self.Q_table_data, np.max(Q_values_list))
            # if not(Q_values_list[0] == 50) and not(Q_values_list[1] == 50):
            #     print("o: ", o)
            #     print("o_vec: ", o_vec)
            #     print("Q_values_list: ", Q_values_list)
            #     print("best_action: ", best_action)
            #     input("Q learning interrupt...")


            return torch.FloatTensor(self.all_actions[best_action])


    def get_observational_space_size(self, o_dim):
        "This function returns the size of the observational space, based on the observation function"

        observations_size = self.Observational_space_resolution**o_dim

        return observations_size
    
    def vectorize_observation(self, observation):
        "This function vectorizes the observation"


        observation = np.array(observation)

        res = 0
        for i, obs in enumerate(observation):
            res += obs * self.Observational_space_resolution**i
        

        return res

    def vectorize_action(self, action):
        "This function vectorizes the action"

        action = np.array(action)

        res = 0
        for i, act in enumerate(action):
            res += act * len(self.robot_actions)**i

        return res
    def get_action_space_size(self):    
        "This function returns the size of the action space, based on the action function"

        return self.action_space

    def initialize_all_actions(self):
        """
        SUMMARY: 
            This function creates a list of all possible actions, since Q-learning 
            is a value based method with a discrete action space.
        """
        # Define the two possible values (0 and 1)


        # Generate all combinations for a 7-sized list with 2 possible values at each index
        self.all_actions = np.array(list(itertools.product(self.robot_actions, repeat=self.a_dim)))


    def optimize(self, batch_size: int, tensorboard_writer: SummaryWriter = None):
        """
        SUMMARY: 
            This function updates the Q_table
        
        ARGS:
            batch_size [int]: The size of the batch to be used for training the Q_table
            tensorboard_writer [SummaryWriter]: The tensorboard writer to log the data.
        """

        self.total_it += 1

        # Sample replay buffer
        # batch = self.replay_buffer.sample(batch_size)

        # Insert latest memory into optimizer also
        batch = self.replay_buffer.get_latest(batch_size)





        # Get latest memory
        # batch = self.replay_buffer.get_latest(1)
        
        # Optimize the Q-table
        if not self.using_NN:
            self.optimize_table(batch_size, batch)
        else:
            self.optimize_NN(batch_size, batch)

        # Optimize the NN
        # self.optimize_NN(batch_size, batch)

    def optimize_table(self, batch_size, batch, verbose=False):
        """
        SUMMARY:
            This function updates the Q-table using the Bellman equation.
        """

        # Get the Q value of doing the action
        states = np.array(batch.state.cpu())
        actions = np.array(batch.action.cpu())
        episode = np.array(batch.episode.cpu())


        states_vectorized = [int(self.vectorize_observation(state)) for state in states]
        actions_vectorized = [int(self.vectorize_action(action)) for action in actions]


        # Debugging
        # for state in states_vectorized:
        #     if state == 93:
        #         verbose = True
        #         break
        # Do n-step Q learning


        Q_value = self.Q[states_vectorized, actions_vectorized]
        
        if verbose:
            print("Episode: ", episode)
            print("states: ", states)
            print("actions: ", actions)
            print("old Q value: ", Q_value)


        # Get the Q value of the next state based on the best possible action
        Q_value_next_best_list = np.array([0.0]*batch_size).astype(float)


        next_states = np.array(batch.next_state.cpu())

        if verbose:
            print("next_states: ", next_states)

        next_states = [int(self.vectorize_observation(next_state)) for next_state in next_states]

        if verbose:
            print("next_states vectorized: ", next_states)

        for i, sample in enumerate(next_states):
            Q_value_next_best_list[i] = np.max(self.Q[sample]).astype(float)    
            
            if verbose:
                print("\n")
                print("\n")
                print("i: ", i)
                print("sample: ", sample)
                print("Q samples: ", self.Q[sample])
                print("Choosen max value: ",  np.max(self.Q[sample]))
                print("Q_value_next_best_list: ", Q_value_next_best_list)


        # Update the Q value based on the Bellman equation
        batch_reward = np.ndarray.flatten(np.array(batch.reward.cpu()))
        if verbose:
            print("batch_reward: ", batch_reward)

        target_Q = np.array(Q_value + self.learning_rate*(batch_reward + self.discount * Q_value_next_best_list  - Q_value))

        if verbose:
            print("target_Q: ", target_Q)
            print("learning step: ", self.learning_rate*(batch_reward + self.discount * Q_value_next_best_list  - Q_value))



        self.Q[states_vectorized, actions_vectorized] = target_Q

        if verbose:
            print("Q_value = ", Q_value)
            print("new Q value: ", self.Q[states_vectorized, actions_vectorized])
            input("Q learning interrupt...")




    def optimize_NN(self, batch_size, batch):
        """
        SUMMARY:
            This function updates the Q-table using a neural network.
            NOTE: Not sure if this works
        """
        with torch.no_grad():

            # Get the Q value of doing the action
            Q_value = self.Q(torch.cat([batch.state, batch.action], 1)).reshape(-1, 1).to(device)


            # Get the Q value of the next state based on the best possible action
            Q_value_next_best_list = [0]*batch_size



            for i, sample in enumerate(batch.next_state):
                sample_repeated = sample.repeat(len(self.all_actions), 1)

                all_actions = torch.FloatTensor(np.array(self.all_actions)).to(device)
                Q_value_next = self.Q_target(torch.cat([sample_repeated, all_actions], 1)).to(device)

                Q_value_next_best_list[i] = max(Q_value_next)



            # Update the Q value based on the Bellman equation
            Q_value_next_best_list = torch.FloatTensor(Q_value_next_best_list).reshape(-1, 1).to(device)

            #target_Q = Q_value + self.learning_rate*(batch.reward + self.discount * Q_value_next_best_list  - Q_value)
            
            # Alpha = 1
            target_Q = batch.reward + self.discount * Q_value_next_best_list

        # Get current Q estimates. 
        current_Q = self.Q(torch.cat([batch.state, batch.action], 1))


        # No idea how this works found it on this guide: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
       
        self.Q_optimizer.zero_grad()
        loss = F.mse_loss(current_Q, target_Q)
        loss.backward()
        self.Q_optimizer.step()

        
        # Update the target Q table
        if self.total_it % self.policy_freq == 0:

            current_Q_target = self.Q_target(torch.cat([batch.state, batch.action], 1))
            self.Q_target_optimizer.zero_grad()
            loss = F.mse_loss(current_Q_target, target_Q)
            loss.backward()
            self.Q_target_optimizer.step()


    def save(self, name):
        if self.using_NN:
            torch.save(self.Q.state_dict(), name + "Q")
            torch.save(self.Q_target.state_dict(), name + "Q_target")
        else:
            np.save(name + "Q", self.Q)


        mean_val = self.Q_table_data.mean()
        var_val = self.Q_table_data.var()

        self.Q_table_mean_data = np.append(self.Q_table_mean_data, mean_val)
        self.Q_table_data_var = np.append(self.Q_table_data_var, var_val)

        plt.clf()
        plt.plot(range(len(self.Q_table_mean_data)), self.Q_table_mean_data)
        plt.savefig(name + "Q_table_performance_mean.png")

        plt.clf()
        plt.plot(range(len(self.Q_table_data_var)), self.Q_table_data_var)
        plt.savefig(name + "Q_table_performance_var.png")



    def load(self, name):
        # At one point i forgot the parenthesis when saving, so it might need the parenthesis when loading

        # Name of model with forgotten parenthesis:
        # - 

        if self.using_NN:
            if torch.cuda.is_available():
                self.Q.load_state_dict(torch.load(name + "Q"))
                self.Q_target.load_state_dict(torch.load(name + "Q_target"))

            else:
                self.Q.load_state_dict(
                    torch.load(name + "Q", map_location=torch.device("cpu"))()
                )
                self.Q_target.load_state_dict(
                    torch.load(name + "Q_target", map_location=torch.device("cpu"))()
                )
        else:
            self.Q = np.load(name + "Q.npy")