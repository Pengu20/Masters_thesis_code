import argparse
import os




import time
from skrl.envs.wrappers.torch import wrap_env


from skrl.trainers.torch import SequentialTrainer


# ----------- Import peter stuff -----------
# from learning.SKRL_UR_env_Q_learning import URSim_SKRL_env
from learning.environments.UR_env_Flick_TASK_3_Expert_demonstration import URSim_SKRL_env
from learning.models.Generate_models import get_PPO_agent_SKRL, get_TD3_agent, make_TD3_models, Policy, Value
from learning.config_files.generate_config_file import generate_config_file

# -------------------------------------------


def main():
    args=generate_config_file()
    Training_model = "PPO"


    print(" > Loaded simulation configs:")
    for key, value in vars(args).items():
        print(f"\t{key:30}{value}")

    # instantiate the agent
    # (assuming a defined environment <env>)

    args.expert_demonstration = True
    args.save_state_actions = True

    environment_name = time.strftime("%y-%m-%d_%H-%M-%S") + "URSim_SKRL_env_" + Training_model
    env = URSim_SKRL_env(args = args, name=environment_name, render_mode="human")
    # it will check your custom environment and output additional warnings if needed
    env = wrap_env(env)

    # create a sequential trainer
    cfg = {"timesteps": args.max_timesteps, 
           "headless": True,
           #"environment_info": "robot_joints",
           }
    
    # PPO
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, env.device, clip_actions=True)
    models["value"] = Value(env.observation_space, env.action_space, env.device)

    agent = get_PPO_agent_SKRL(env, models, write_interval = 1000)



    trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg)

    # train the agent(s)
    # Have not tried the eval mode. Maybe it works
    trainer.eval()


if __name__ == "__main__":
    main()
