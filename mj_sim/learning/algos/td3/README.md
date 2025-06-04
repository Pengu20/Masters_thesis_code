# TD3 (Twin Delayed Deep Deterministic Policy Gradient) Implementation

This implementation provides an agent based on the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm. TD3 is an off-policy algorithm used for continuous control tasks. Below is a detailed explanation of the mathematical concepts involved in the TD3 class, along with the corresponding code.

## Initialization

The TD3 class is initialized with various parameters and networks:

- **Actor Network**: Generates actions given states.
- **Critic Networks (Q1 and Q2)**: Estimate the Q-values for state-action pairs.
- **Target Networks**: Used for stable training.

```python
self.actor = MLP(self.o_dim, self.a_dim, self.a_max).to(device)
self.Q1 = MLP(self.o_dim + self.a_dim, 1, output_activation_func=lambda x: x).to(device)
self.Q2 = MLP(self.o_dim + self.a_dim, 1, output_activation_func=lambda x: x).to(device)

self.actor_target = copy.deepcopy(self.actor)
self.Q1_target = copy.deepcop

def o(args, robot) -> np.ndarray:


    right_driver_joint = robot._gripper.q[0]
    left_driver_joint = robot._gripper.q[4]

    gripper_q = [right_driver_joint, left_driver_joint]
    return np.append(np.array(robot.q), np.array(gripper_q))y(self.Q1)
self.Q2_target = copy.deepcopy(self.Q2)
```

## Action Selection

To select an action, the actor network processes the given state:

$$ a = \pi_\theta(o) $$

where \( \pi_\theta \) is the actor network parameterized by \(\theta\).

```python
def select_action(self, o: np.ndarray) -> torch.FloatTensor:
    o = torch.FloatTensor(o.reshape(1, -1)).to(device)
    a = self.actor.forward(o).flatten().to(device)
    return a
```

## Optimization

### Sampling from Replay Buffer

A batch of experiences is sampled from the replay buffer for training:

$$ \{(s, a, r, s', d)\} $$

### Target Action with Noise

To reduce overestimation bias, noise is added to the target actions:

$$ a' \leftarrow \pi_{\theta'}(s') + \epsilon $$
$$ \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c) $$

where \(\sigma\) is the policy noise and \(c\) is the noise clip value.

```python
noise = (
    (torch.randn_like(batch.action) * self.policy_noise)
    .clamp(-self.noise_clip, self.noise_clip)
    .to(device)
)
next_action = (self.actor_target(batch.next_state) + noise).clamp(-self.a_max, self.a_max)
```

### Target Q-Value

The target Q-value is computed using the minimum of the two target critic networks to mitigate overestimation bias:

$$ y = r + \gamma (1 - d) \min(Q_{\theta'_1}(s', a'), Q_{\theta'_2}(s', a')) $$

```python
target_Q1, target_Q2 = (
    self.Q1_target(torch.cat([batch.next_state, next_action], 1)),
    self.Q2_target(torch.cat([batch.next_state, next_action], 1)),
)
target_Q = torch.min(target_Q1, target_Q2)
target_Q = batch.reward + (1 - batch.done) * self.discount * target_Q
```

### Critic Loss

The critic networks are updated by minimizing the Mean Squared Error (MSE) between the current Q-values and the target Q-values:

$$ L_{\text{critic}} = \frac{1}{N} \sum_{i=1}^N \left( Q_{\theta_1}(s_i, a_i) - y_i \right)^2 + \left( Q_{\theta_2}(s_i, a_i) - y_i \right)^2 $$

```python
current_Q1, current_Q2 = (
    self.Q1(torch.cat([batch.state, batch.action], 1)),
    self.Q2(torch.cat([batch.state, batch.action], 1)),
)
critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
```

### Actor Loss

The actor network is updated less frequently by maximizing the Q-value predicted by the first critic network:

$$ L_{\text{actor}} = -\frac{1}{N} \sum_{i=1}^N Q_{\theta_1}(s_i, \pi_{\theta}(s_i)) $$

```python
if self.total_it % self.policy_freq == 0:
    actor_loss = -self.Q1(torch.cat([batch.state, self.actor(batch.state)], 1)).mean()
```

### Target Networks Update

The target networks are updated using Polyak averaging:

$$ \theta' \leftarrow \tau \theta + (1 - \tau) \theta' $$

```python
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
```

## Save and Load

The models and optimizers' states can be saved and loaded for checkpointing and resuming training:

```python
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
```

This README provides an overview of the TD3 class, focusing on the mathematical formulations and their implementation in code. TD3 leverages noise addition, dual critic networks, and delayed policy updates to achieve stable and efficient learning in continuous action spaces.
