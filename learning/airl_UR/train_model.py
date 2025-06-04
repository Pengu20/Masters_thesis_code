import torch
import numpy as np
from learning.airl_UR.utils.utils import get_entropy, log_prob_density
import time
import copy
import torch.nn
from torchviz import make_dot

from learning.airl_UR.memory_process import process_mem



def kl_divergence(mu1, mu2, sigma_1, sigma_2):

    sigma_diag_1 = sigma_1
    sigma_diag_2 = sigma_2


    sigma_diag_2_inv = torch.div(torch.ones_like(sigma_diag_2), sigma_diag_2)
    print("sigma variables: ", sigma_2)
    print("sigma_diag_2_inv: ", sigma_diag_2_inv)
    print("size sigma_diag_2_inv: ", sigma_diag_2_inv.size())
    print("torch.prod(sigma_diag_2) / torch.prod(sigma_diag_2)): ", torch.prod(sigma_diag_2) / torch.prod(sigma_diag_1))
    print("torch.sum(torch.mul(sigma_diag_2_inv, sigma_diag_1)): ", torch.sum(torch.mul(sigma_diag_2_inv, sigma_diag_1)))
    print("torch.sum((mu2 - mu1) * (mu2 - mu1) * sigma_diag_2_inv): ", torch.sum((mu2 - mu1) * (mu2 - mu1) * sigma_diag_2_inv))


    kl = 0.5 * (torch.log(torch.prod(sigma_diag_2) / torch.prod(sigma_diag_2))
                - mu1.shape[0] + torch.sum(torch.mul(sigma_diag_2_inv, sigma_diag_1))
                + torch.sum((mu2 - mu1) * (mu2 - mu1) * sigma_diag_2_inv)
                )

    return kl




def check_nan_in_model(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            return True
        else:
            return False

def split_dataset_index(n, train_percent, test_percent):

    data = np.arange(n)

    np.random.shuffle(data)


    train_size = int(train_percent * n)
    test_size = int(test_percent * n)


    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]

    return train_data, test_data




def train_discrim(discrim, memory, discrim_optim, demonstrations, args):
    states = np.array([entry[0] for entry in memory])  # Stack states vertically
    actions = np.array([entry[1] for entry in memory])  # Convert actions to array
    

    states = torch.Tensor(states)
    actions = torch.Tensor(actions)
    
    criterion = torch.nn.BCELoss()

    for _ in range(args.discrim_update_num):


        learner = discrim(torch.cat([states, actions], dim=1))
        demonstrations = torch.Tensor(demonstrations)
        expert = discrim(demonstrations)
        learner_observations = torch.cat([states, actions], dim=1)



        discrim_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
                        criterion(expert, torch.zeros((demonstrations.shape[0], 1)))
                
        discrim_optim.zero_grad()
        discrim_loss.backward()
        discrim_optim.step()

    expert_acc = ((discrim(demonstrations) < 0.2).float()).mean()

    learner_acc = ((discrim(torch.cat([states, actions], dim=1)) > 0.8).float()).mean()

    return expert_acc, learner_acc






def train_actor_critic_process(process_ID, actor_list, critic_list, memory, args, actor_loss_process, critic_loss_process, entropy_loss_process, ratio_list_process):
    torch.set_num_threads(1)


    actor, actor_optim = actor_list[process_ID]
    critic, critic_optim = critic_list[process_ID]



    observation_space = len(memory[0][0])



    states = torch.tensor(np.array([entry[0] for entry in memory])).type(dtype=torch.float32).view(-1, observation_space) # Stack states vertically
    # actions = torch.tensor(np.array([entry[1] for entry in memory])).type(dtype=torch.float32).view(-1, action_space)  # Convert actions to array
    
    # states = torch.vstack([entry[0] for entry in memory])
    actions = torch.vstack([entry[1] for entry in memory])
    
    rewards = torch.tensor([entry[2] for entry in memory])
    masks = torch.tensor([entry[3] for entry in memory]) 
    old_policy = torch.tensor([entry[4] for entry in memory]) 





    criterion = torch.nn.MSELoss(reduction='none')
    n = len(states)
    arr = np.arange(n)



    actor_loss_list = []
    critic_loss_list = []
    entropy_loss_list = []
    ratio_list = []



    for _ in range(args.actor_critic_update_num):
        np.random.shuffle(arr)


        old_values = critic(states)
        returns, advants = get_gae(rewards, masks, old_values, args)
        

        for i in range(n // args.batch_size): 


            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)

            inputs = states[batch_index]
            actions_samples = actions[batch_index]

            returns_samples = returns[batch_index]


            advants_samples = advants[batch_index]
            oldvalue_samples = old_values[batch_index].squeeze(1).detach()
            old_policy_samples = old_policy[batch_index]


            objective_function, ratio, entropy_regularizer = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy=old_policy_samples, actions=actions_samples,
                                         batch_index=batch_index)
            



            values = critic(inputs).squeeze(1)
    

            critic_gradient = values - oldvalue_samples

            clipped_values = oldvalue_samples + torch.clamp(critic_gradient, -critic_gradient * args.clip_param_critic,  critic_gradient * args.clip_param_critic)

            critic_loss1 = criterion(clipped_values, returns_samples)

            critic_loss2 = criterion(values, returns_samples)

            critic_loss = torch.max(critic_loss1, critic_loss2).mean()
            




            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param_actor,
                                        1.0 + args.clip_param_actor)
            

            clipped_objective_function = clipped_ratio * advants_samples
    
            # actor_loss = -advants_samples.mean()
            actor_loss = -torch.min(objective_function, clipped_objective_function).mean()





            loss = actor_loss + critic_loss - entropy_regularizer


            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)
            entropy_loss_list.append(entropy_regularizer)
            ratio_list.append(ratio.mean().item())



            critic_optim.zero_grad()
            actor_optim.zero_grad()

            loss.backward() 

            critic_optim.step()
            actor_optim.step() 


    actor_loss_mean = torch.tensor(actor_loss_list).mean().item()
    critic_loss_mean = torch.tensor(critic_loss_list).mean().item()
    entropy_loss_mean = torch.tensor(entropy_loss_list).mean().item()
    ratio_list_mean = torch.tensor(ratio_list).mean().item()
            



    actor_loss_process[process_ID] = actor_loss_mean
    critic_loss_process[process_ID] = critic_loss_mean
    entropy_loss_process[process_ID] = entropy_loss_mean
    ratio_list_process[process_ID] = ratio_list_mean

    # Return the actor, critic and entropy loss for review
    # return actor_loss, actor.critic_gain * critic_loss, actor.entropy_gain * entropy

    actor_list[process_ID] = [actor, actor_optim] 
    critic_list[process_ID] = [critic, critic_optim]





def train_actor_REINFORCE(actor, memory, actor_optim, args, expert_data):
    torch.set_num_threads(1)
    states, actions, rewards, masks, log_pis, next_states = process_mem(memory)




    n = len(states)
    arr = np.arange(n)
    
    actor_loss_list = []

    returns = get_returns(rewards, masks, args)
            


    for _ in range(args.actor_critic_update_num):
        np.random.shuffle(arr)


        for i in range(n // args.batch_size): 


            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)

            inputs = states[batch_index]
            actions_samples = actions[batch_index]



            returns_samples = returns[batch_index]


            mu, std = actor(inputs)
            log_prob_actions = log_prob_density(actions_samples, mu, std).squeeze()

            # print("log_prob_actions: ", log_prob_actions)
            # print("returns: ", returns_samples)
            # print("log_prob_actions: ", log_prob_actions.size())
            # print("returns_samples: ", returns_samples.size())
            # print("log_prob_actions*returns_samples: ", (log_prob_actions*returns_samples).size())
            # input()

            actor_loss = -(log_prob_actions*returns_samples).mean()


            mu, std = actor(expert_data[0])
            log_pis_exp = -log_prob_density(expert_data[1], mu, std)

            COL_loss = log_pis_exp.mean()

            loss = COL_loss + actor_loss

            actor_loss_list.append(loss.item())


            actor_optim.zero_grad()

            loss.backward()  # Only one backward pass now

            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            actor_optim.step()  # No second loss.backward()


            if check_nan_in_model(actor):
                print("Is nan guy!!")

    actor_loss_list = torch.tensor(actor_loss_list).mean().item()
    critic_loss = 0
    entropy = 0
    ratio = 0

    return actor_loss_list, critic_loss, entropy, ratio



def train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args, expert_data):
    torch.set_num_threads(1)
    states, actions, rewards, masks, old_policy, next_states = process_mem(memory)


    criterion = torch.nn.MSELoss(reduction='none')
    n = len(states)
    arr = np.arange(n)
    

        # Experimental alternative calculation of Generalized Advantage estimates.
        # old_values_next_state = critic(next_states)
        # returns, advants = calculate_gae(values=old_values, rewards=rewards, dones=masks, next_values=old_values_next_state, gamma=args.gamma, lambd=args.lamda)


    actor_loss_list = []
    critic_loss_list = []
    entropy_loss_list = []
    ratio_list = []

    for _ in range(args.actor_critic_update_num):
        np.random.shuffle(arr)

        with torch.no_grad():
            old_values = critic(states)
            returns, advants = get_gae(rewards, masks, old_values, args)
            

        for i in range(n // args.batch_size): 


            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)

            inputs = states[batch_index]
            actions_samples = actions[batch_index]

            returns_samples = returns[batch_index]


            advants_samples = advants[batch_index]
            oldvalue_samples = old_values[batch_index].squeeze(1).detach()
            old_policy_samples = old_policy[batch_index]



                             
            objective_function, ratio, entropy_regularizer = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy=old_policy_samples, actions=actions_samples,
                                         batch_index=batch_index)
            


            values = critic(inputs).squeeze(1)
    

            critic_gradient = values - oldvalue_samples

            clipped_values = oldvalue_samples + \
                            torch.clamp(critic_gradient, 
                            -args.clip_param_critic,  
                            args.clip_param_critic)
            

            

            critic_loss1 = criterion(clipped_values, returns_samples)

            critic_loss2 = criterion(values, returns_samples)

            critic_loss = torch.max(critic_loss1, critic_loss2).mean()
            


            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param_actor,
                                        1.0 + args.clip_param_actor)
            

            clipped_objective_function = clipped_ratio * advants_samples
    
            # actor_loss = -advants_samples.mean()
            actor_loss = -torch.min(objective_function, clipped_objective_function).mean()


            # print("critic max size: ", )
            # print("oldvalue_samples: ", oldvalue_samples.size())
            # print("clipped_values: ", clipped_values.size())
            # print("returns_samples: ", returns_samples.size())
            # print("critic_loss1: ", critic_loss1.size())
            # print("critic_loss2: ", critic_loss2.size())

            # print("advants size: ", advants_samples.size())
            # print("max ratio: ", torch.max(ratio) )
            # print("min ratio: ", torch.min(ratio) )
            # print("average ratio: ", ratio.mean())
            
            # print("ratio size: ", ratio.size())
            # print("ratio size clipped: ", clipped_ratio.size())
            # print("size advants_samples: ", advants_samples.size())
            # print("size loss: ", loss.size())


            # print("advants samples: ", advants_samples)
            # print("size clipped advantage: ", clipped_loss)
            # print("Type advantage: ", ratio)

            # print("ratio: ", ratio.mean())


            # print("loss: ", loss)
            # print("actor loss mean: ", -torch.min(loss, clipped_loss).mean())
            # print("actor loss: ", torch.min(loss, clipped_loss))
            # print("critic loss: ", torch.max(critic_loss1, critic_loss2))
            # print("entropy: ", actor.entropy_gain * entropy)
            # input()


            # These comments are for zeroing out the effects of each loss for debugging
            #critic_loss = torch.tensor(0).type(torch.float) #  actor.critic_gain * critic_loss
            #entropy_regularizer = torch.tensor(0).type(torch.float)  # actor.entropy_gain * entropy
            #actor_loss = torch.tensor(0).type(torch.float) # actor_loss





            loss = actor_loss + critic_loss - entropy_regularizer 


            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)
            entropy_loss_list.append(entropy_regularizer)
            ratio_list.append(ratio.mean().item())




            critic_optim.zero_grad()
            actor_optim.zero_grad()

            loss.backward()  # Only one backward pass now
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)


            critic_optim.step()
            actor_optim.step()  # No second loss.backward()


            if check_nan_in_model(actor):
                print("Is nan guy!!")

    actor_loss_list = torch.tensor(actor_loss_list).mean().item()
    critic_loss_list = torch.tensor(critic_loss_list).mean().item()
    entropy_loss_list = torch.tensor(entropy_loss_list).mean().item()
    ratio_list = torch.tensor(ratio_list).mean().item()

    print("Ratio: ", ratio_list)
    # input()
    # Return the actor, critic and entropy loss for review
    return actor_loss_list, critic_loss_list, entropy_loss_list, ratio_list


def get_gae(rewards, masks, values, args):
    rewards = rewards.type(torch.float)
    masks = masks.type(torch.float)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):

        running_returns = rewards[t].item() + (args.gamma * running_returns * (1 - masks[t].item()))
        returns[t] = running_returns

        running_delta = rewards[t].item() + (args.gamma * previous_value * (1 - masks[t].item())) - values.data[t].item()                           
        previous_value = values.data[t].item()

        running_advants = running_delta + (args.gamma * args.lamda * running_advants * (1 - masks[t].item()))
        advants[t] = running_advants


    advants = (advants - advants.mean()) / (advants.std() )


    return returns, advants



def get_returns(rewards, masks, args):
    rewards = rewards.type(torch.float)
    masks = masks.type(torch.float)  # should be 1 for non-terminal, 0 for terminal
    returns = torch.zeros_like(rewards)

    running_return = 0.0

    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + args.gamma * running_return * masks[t]
        returns[t] = running_return

    return returns




def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values



    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)



def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):


    mu, std = actor(states)



    # ---- GAIL implementation ----



    new_policy = log_prob_density(actions, mu, std).squeeze()



    log_ratio = new_policy - old_policy

    ratio = torch.exp(log_ratio)


    if torch.isnan(ratio).any():
        print(" ----------------- RATIO is a NAN guy! ----------------- ")




    surrogate_loss = ratio * advants
    # print("ratio: ", ratio)
    entropy = get_entropy(mu, std)

    return surrogate_loss, ratio, entropy


