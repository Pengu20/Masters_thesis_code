# Adversarial Inverse Reinforcement learning for Cloth manipulation

This is a cleaned version of the code used for my masters thesis project on Adversarial Inverse Reinforcement Learning within the field of cloth manipulation

## Guide:

### setup virtual environment (optional)
$ python -m venv venv
$ source venv/bin/activate


### install requirements
$ pip install -r requirements.txt


### Run AIRL with PPO 
$ python -m learning.airl_UR.main_airl

_Modify learning.airl_UR.main_AIRL.py to try alterations of of the code_



## Repository contains:
- AIRL implementation
- PPO implementation

Along with the methods documented in the thesis, other experimental methods in this thesis also include
- GAIL implementation (Not sure if works anymore)
- Imitation Initalized Learning
- Circle of Learning loss within AIRL (based on Negative Gaussian Log Likelihood)

The experimental methods are all disabled in the current version of the github repository.

All method implementations are used within the file:
- mj_sim/learning/airl_UR/main_airl.py



This repository also consists of expert demonstrations and MuJoCo environments for:

1 DOF - underactuated object manipulation

2 DOF - underactuated object manipulation

3 DOF - underactuated object manipulation

4 DOF - underactuated object manipulation

## The expert demonstrations consist of a set of state action transitions as:
- state
- action
- reward
- mask
- action log probability (empty index, must be calculated online)
- next state

1 DOF - underactuated manipulation expert demonstration
[download video](https://raw.githubusercontent.com/Pengu20/Masters_thesis_code/main/mj_sim/Expert_demonstrations/mp4/F1_expert.mp4)

2 DOF - underactuated manipulation expert demonstration
[download video](https://raw.githubusercontent.com/Pengu20/Masters_thesis_code/main/mj_sim/Expert_demonstrations/mp4/F2_expert.mp4)

4 DOF - underactuated manipulation expert demonstration
[download video](https://raw.githubusercontent.com/Pengu20/Masters_thesis_code/main/mj_sim/Expert_demonstrations/mp4/F3_expert.mp4)

cloth manipulation expert demonstration
[download video](https://raw.githubusercontent.com/Pengu20/Masters_thesis_code/main/mj_sim/Expert_demonstrations/mp4/cloth_expert.mp4)


Reward-injected state-only AIRL policy
[download video](https://raw.githubusercontent.com/Pengu20/Masters_thesis_code/main/mj_sim/learned_policies/mp4/C1_reward_injected_AIRL_policy.mp4)


# Code sources
- Mujoco environment design
https://gitlab.sdu.dk/sdurobotics/teaching/mj_sim


- PPO inspired implementation 
https://github.com/reinforcement-learning-kr/lets-do-irl


- AIRL repo
https://github.com/toshikwa/gail-airl-ppo.pytorch



