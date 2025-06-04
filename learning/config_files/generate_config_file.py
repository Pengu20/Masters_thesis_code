import argparse



def generate_config_file():
    max_episodes = int(400)
    max_time_per_episode = int(200)
    max_timesteps = max_episodes * max_time_per_episode

    debug = False

    parser = argparse.ArgumentParser(
            description="MuJoCo simulation of manipulators and controllers."
        )
        
    parser.add_argument(
        "--scene_path",
        type=str,
        default="learning/scenes/RL_task_flick_f1.xml",
        help="Path to the XML file defining the simulation scene.",
    )

    parser.add_argument(
        "--load_model",
        type=str,
        default="",
        help="loads model from relative path",
    )

    parser.add_argument(
        "--expert_demonstration",
        type=bool,
        default=False,
        help="if environment should use expert demonstration or not",
    )

    parser.add_argument(
        "--agent_disabled",
        type=bool,
        default=False,
        help="This boolean enables a transfer learning variant of the task where the agent is disabled",
    )



    parser.add_argument(
        "--plot_data",
        type=bool,
        default=False,
        help="Path to the XML file defining the simulation scene.",
    )
    
    parser.add_argument(
        "--save_state_actions",
        type=bool,
        default=False,
        help="Save the state and actions of the simulation.",
    )

    parser.add_argument(
        "--show_site_frames",
        type=bool,
        default=False,
        help="Flag to display the site frames in the simulation visualization.",
    )
    parser.add_argument(
        "--gravity_comp",
        type=bool,
        default=True,
        help="Enable or disable gravity compensation in the controller.",
    )
    parser.add_argument(
        "--manual",
        type=bool,
        default=False,
        help="Enable or disable manual control of the simulation.",
    )
    parser.add_argument(
        "--render_size",
        type=float,
        default=0.1,
        help="Size of the rendered frames in the simulation visualization.",
    )

    parser.add_argument(
        "--eval_episodes",
        default=1 if debug else 1,
        type=int,
        help="Number of episodes for evaluation during training.",
    )
    parser.add_argument(
        "--eval_freq",
        default=10 if debug else int(max_episodes/50),
        type=int,
        help="Frequency (in timesteps) of evaluations during training.",
    )
    parser.add_argument(
        "--seed_eval",
        type=int,
        default=110,
        help="Random seed for evaluation to ensure reproducibility.",
    )
    parser.add_argument(
        "--render",
        default=False,
        type=bool,
        help="Enable or disable rendering during the simulation.",
    )
    parser.add_argument(
        "--episode_timeout",
        default=1 if debug else int(max_time_per_episode),
        type=int,
        help="Maximum duration (in time steps) for each episode before timeout.",
    )
    parser.add_argument(
        "--expl_noise",
        type=float,
        default=0.1,
        help="Percentage of exploration noise to add to the actor's actions.",
    )
    parser.add_argument(
        "--seed_train",
        type=int,
        default=100,
        help="Random seed for training to ensure reproducibility.",
    )

    parser.add_argument(
        "--t", type=int, default=0, help="Time step for the simulation."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10 if debug else 20,
        help="Batch size for training the reinforcement learning algorithm.",
    )
    parser.add_argument(
        "--start_timesteps",
        type=int,
        # default=10 if debug else int(0.01 * max_timesteps),
        default=10 if debug else max_episodes*0.01,
        help="Number of initial timesteps for exploration before training starts.",
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=100 if debug else max_timesteps,
        help="Maximum number of timesteps for the entire training process.",
    )
    parser.add_argument(
        "--exit_on_done",
        type=bool,
        default=True,
        help="Flag to exit the simulation when the training is done.",
    )

    parser.add_argument(
        "--state0",
        type=str,
        default="default",
        help="init state.",
    
    )

    parser.add_argument(
        "--training_name",
        type=str,
        default="",
        help="name of the training session that will be made"
    )

    args, _ = parser.parse_known_args()


    return args