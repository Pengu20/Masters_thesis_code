import torch
import numpy as np
from learning.gail.utils.utils import get_entropy, log_prob_density

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

        # print("learner: ", learner_observations[0:10])
        # print("expert: ", demonstrations[0:10])
        # print("learner: ", learner[0:10])
        # print("expert: ", expert[0:10])
        # input()

        discrim_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
                        criterion(expert, torch.zeros((demonstrations.shape[0], 1)))
                
        discrim_optim.zero_grad()
        discrim_loss.backward()
        discrim_optim.step()

    expert_acc = ((discrim(demonstrations) < 0.2).float()).mean()

    learner_acc = ((discrim(torch.cat([states, actions], dim=1)) > 0.8).float()).mean()

    return expert_acc, learner_acc




def train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args):
    states = torch.tensor([entry[0] for entry in memory]).cuda().to(torch.float32)  # Stack states vertically
    actions = torch.tensor([entry[1].detach().cpu().numpy() for entry in memory]).cuda()  # Convert actions to array
    rewards = torch.tensor([entry[2].detach().cpu().numpy() for entry in memory]).cuda()  # Convert rewards to array
    masks = torch.tensor([entry[3] for entry in memory]).cuda()    # Convert masks to array	    	


    old_values = critic(states)
    returns, advants = get_gae(rewards, masks, old_values, args)
    mu, std = actor(states)

    old_policy = log_prob_density(actions, mu, std)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for _ in range(args.actor_critic_update_num):
        np.random.shuffle(arr)

        for i in range(n // args.batch_size): 
            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            
            inputs = torch.Tensor(states)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()
            
            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -args.clip_param, 
                                         args.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)
            

            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy


            critic_optim.zero_grad()
            actor_optim.zero_grad()

            loss.backward()  # Only one backward pass now

            critic_optim.step()
            actor_optim.step()  # No second loss.backward()


def get_gae(rewards, masks, values, args):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (args.gamma * running_returns * masks[t])
        returns[t] = running_returns

        running_delta = rewards[t] + (args.gamma * previous_value * masks[t]) - \
                                        values.data[t]
        previous_value = values.data[t]
        
        running_advants = running_delta + (args.gamma * args.lamda * \
                                            running_advants * masks[t])
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    mu, std = actor(states)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    entropy = get_entropy(mu, std)



    return surrogate_loss, ratio, entropy