


import torch
import numpy as np

from collections import deque

def process_mem(memory):
        
    observation_space = len(memory[0][0])
    states = torch.tensor(np.array([entry[0] for entry in memory])).type(dtype=torch.float32).view(-1, observation_space) # Stack states vertically
    # states = torch.vstack([entry[0] for entry in memory])
    actions = torch.vstack([entry[1] for entry in memory])  # Convert actions to array
    
    rewards = torch.tensor([entry[2] for entry in memory])  # Convert rewards to array

    masks = torch.tensor([entry[3] for entry in memory])    # Convert masks to array	  
    log_pis = torch.tensor([entry[4] for entry in memory])    # Convert masks to array

    next_states = torch.tensor(np.array([entry[5] for entry in memory])).type(dtype=torch.float32).view(-1, observation_space)    # Convert masks to array


    return states, actions, rewards, masks, log_pis, next_states





def Generate_memory_from_expert_data(expert_data):
    torch.set_num_threads(1)
    


    memory_output = deque()


    for steps in range(len(expert_data)):

        #state = torch.tensor(memory[steps][0], dtype=torch.float32).view(1, -1)
        state = expert_data[steps][0]
        action = torch.tensor(expert_data[steps][1], dtype=torch.float32).view(1, -1)
        reward = expert_data[steps][2]
        done = expert_data[steps][3]
        log_pis = 0
        next_state = expert_data[steps][5]



        if done:    
            mask = 1
        else:
            mask = 0


        memory_output.append([state, action, reward, mask, log_pis, next_state])



    return memory_output

