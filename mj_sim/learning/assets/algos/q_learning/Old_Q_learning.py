import time
import numpy as np




class Deep_SARSA:
    def __init__(self, player_name) -> None:
        self.rf = Dominion_reward()
        self.initialize_NN()
        self.game_state_history = []
        self.action_history = []


        self.player_name = player_name
        self.file_address = f"reward_history/{self.player_name}/{self.player_name}_reward_history.txt"

        self.file_average_expected_rewards = f"reward_history/{self.player_name}/{self.player_name}_average_expected_rewards.txt"
        self.file_variance_expected_rewards = f"reward_history/{self.player_name}/{self.player_name}_variance_expected_rewards.txt"

        self.file_average_returns = f"reward_history/{self.player_name}/{self.player_name}_average_returns.txt"
        self.file_variance_returns = f"reward_history/{self.player_name}/{self.player_name}_variance_returns.txt"

        self.file_variance_NN_error = f"reward_history/{self.player_name}/{self.player_name}_variance_NN_error.txt"
        self.file_variance_NN_error = f"reward_history/{self.player_name}/{self.player_name}_variance_NN_error.txt"

        self.file_victory_points = f"reward_history/{self.player_name}/{self.player_name}_victory_points.txt"
        self.file_games_won = f"reward_history/{self.player_name}/{self.player_name}_games_won.txt"
        self.file_game_length = f"reward_history/{self.player_name}/{self.player_name}_game_length.txt"
        self.file_Average_NN_error = f"reward_history/{self.player_name}/{self.player_name}_average_NN_error.txt"
        self.file_variance_NN_error = f"reward_history/{self.player_name}/{self.player_name}_variance_NN_error.txt"


        self.delete_all_previous_history()


        self.SARSA_update_time = []
        self.convert_state2list_time = []
        self.NN_predict_time = []
        self.NN_training_time = []
        self.take_action_time = []



        self.NN_error = []
        ## Experimental design, where neural network is first updates at the end of the game using SARSA


        self.games_played = 0
        self.turns_in_game = 0


        self.all_expected_returns = [] # This is used to keep track of the sum of expected returns gained by the players
        self.all_returns = []


        # DEBUG, so i can see the latest reward

        self.latest_reward = None
        self.latest_action = None
        self.latest_action_type = None
        self.latest_updated_expected_return = None
        self.latest_desired_expected_return = None

        self.greedy_mode = False
        

        self.only_terminate_action = True

        # This variables is used to log the data from the previous games
        self.input_data_past_game_states = []
        self.input_data_past_actions = []
        self.output_label_past_games = []

        # self.set_new_epsilon_value(min_val=0.8, max_val=1)





    def load_NN_from_file(self, path):
        self.model = keras.models.load_model(path)

    def delete_all_previous_history(self):
        '''
        This funtions opens all the file paths for overwrite, to delete all previous data
        '''


        open_file = open(self.file_address, "w")
        open_file.close()

        open_file = open(self.file_average_expected_rewards, "w")
        open_file.close()

        open_file = open(self.file_variance_expected_rewards, "w")
        open_file.close()

        open_file = open(self.file_victory_points, "w")
        open_file.close()

        open_file = open(self.file_games_won, "w")
        open_file.close()

        open_file = open(self.file_game_length, "w")
        open_file.close()

        open_file = open(self.file_Average_NN_error, "w")
        open_file.close()

        open_file = open(self.file_variance_NN_error, "w")
        open_file.close()

        open_file = open(self.file_average_returns, "w")
        open_file.close()

        open_file = open(self.file_variance_returns, "w")
        open_file.close()

        

    def initialize_NN(self):
        
        input_1 = keras.Input(shape=(110,))
        input_2 = keras.Input(shape=(8,))

        weight_mean = 0.04
        weight_dev = 0.01

        # action layer handling
        action_layer = Dense(8, activation='relu', kernel_initializer=initializers.RandomNormal(mean=weight_mean, stddev=weight_dev))(input_2)



        Hidden_layer = layers.concatenate([input_1, action_layer], axis=1)

        Hidden_layer = Dense(80, activation='relu', kernel_initializer=initializers.RandomNormal(mean=weight_mean, stddev=weight_dev))(Hidden_layer)

        Hidden_layer = Dense(64, activation='relu', kernel_initializer=initializers.RandomNormal(mean=weight_mean, stddev=weight_dev))(Hidden_layer)

        #action handling layers
        Concatenated_layer = layers.concatenate([Hidden_layer, action_layer], axis=1)

        Hidden_layer = Dense(32, activation='relu', kernel_initializer=initializers.RandomNormal(mean=weight_mean, stddev=weight_dev))(Concatenated_layer)

        linear_layer = Dense(12,activation='linear', kernel_initializer=initializers.RandomNormal(mean=weight_mean, stddev=weight_dev))(Hidden_layer)

        output = Dense(1,activation='linear', kernel_initializer=initializers.RandomNormal(mean=weight_mean, stddev=weight_dev))(linear_layer)



        self.model = Model(inputs=[input_1, input_2], outputs=output)


        self.model.compile( optimizer='adam',
                            loss='huber',
                            metrics='accuracy',
                            loss_weights=None,
                            weighted_metrics=None,
                            run_eagerly=None,
                            steps_per_execution=None,
                            jit_compile=None,
                            )
        
        self.model.summary()


    def update_NN(self, game_state, action, expected_return_updated):
        '''
        This function is used to update the neural network with the new values
        '''


        NN_input_state, NN_input_action = self.game_state2list_NN_input(game_state, [action])
        self.model.fit((NN_input_state, NN_input_action), np.array([[expected_return_updated]]), epochs=6, verbose=0)

    
    def game_state_list2NN_input(self, game_state_list, action_list):
        '''
        This function maps the input of the game state to the input of the neural network
        It is assumed that the size of the gamestate value is 9000
        '''

        input_state_matrix = np.zeros((len(game_state_list),110))
        input_action_matrix = np.zeros((len(action_list),8)) # Number of bits used to represent the action value

        for i in range(len(game_state_list)):
            NN_input_state, NN_input_action = self.game_state2list_NN_input(game_state_list[i], [action_list[i]])
            input_state_matrix[i,:] = NN_input_state

            input_action_matrix[i,:] = NN_input_action


        return input_state_matrix, input_action_matrix
    

    def expected_return_list2NN_output(self, expected_return_updated_list):
        '''
        This function is used to convert the expected return list to the output of the neural network
        '''
        output_label = np.zeros([len(expected_return_updated_list),1])

        for i in range(len(expected_return_updated_list)):
            output_label[i,:] = expected_return_updated_list[i]

        return output_label



    def update_NN_np_mat(self, input_matrix, output_matrix, epochs=1, verybose=0, batch_size=16):
        '''
        This function is used to update the neural network using a list of all the values used in the game
        '''
        time_start = time.time()

        self.model.fit(input_matrix, output_matrix, epochs=epochs, verbose=0, batch_size=16)

        self.NN_training_time.append(time.time() - time_start)



    def decompose_gamestate2_NN_input(self, game_state, actions_count):

        '''
        This function decomposes the gametate into an input that a neural network is capable of reading.
        '''


        NN_inputs_state = np.zeros((110, actions_count))


        # Process game data to neural netowrk input
        i = 0
        max_size = 110

        str_values = [] # Only occupied by the state "unique actions type"
        for data_bin in game_state:
            if i >= max_size:
                break


            if isinstance(game_state[data_bin], int):
                NN_inputs_state[i] = game_state[data_bin]
                i += 1
            elif isinstance(game_state[data_bin], np.ndarray):

                for val in game_state[data_bin]:
                    if i >= max_size:
                        break

                    if isinstance(val, np.ndarray): # If this is the case, then the value was a card ["name", "ID", "cost"]
                    
                        NN_inputs_state[i] = val[1]
                    else:
                        NN_inputs_state[i] = val


                    i += 1
            elif isinstance(game_state[data_bin], str):
                string2bytes = bytes(game_state[data_bin], 'ascii')
                str_values.append(string2bytes)



        for strings_in_bytes in str_values:  


            for byte in strings_in_bytes:
                if i >= max_size:
                    break

                NN_inputs_state[i] = byte/255 # normalize the value
                i += 1
        
        return NN_inputs_state

    def game_state2list_NN_input(self, game_state, action_list):
        '''
        This function is used to convert the game state to a 
        list that can be used as input for the neural network
        '''

        start_time = time.time()

        NN_inputs_state = self.decompose_gamestate2_NN_input(game_state=game_state, actions_count=len(action_list))
        NN_inputs_actions = np.zeros((8, len(action_list))) # 8 is the bit number representation of the action
        i = 0
        for action in action_list:
            NN_inputs_actions[0,i] = action

            # Action binarisation
            binarised_action = np.binary_repr(action.astype(int), width=8)
            for bin in range(8):
                NN_inputs_actions[bin,i] = int(binarised_action[bin])



            i += 1

        self.convert_state2list_time.append(time.time() - start_time)




        return NN_inputs_state.T, NN_inputs_actions.T  # Apparently keras needs the matrix transposed


    def NN_get_expected_return(self, game_state, actions_list):
        '''
        This function gives the value from the neural network to the state action pair
        '''

        NN_input_state, NN_input_action = self.game_state2list_NN_input(game_state, actions_list)


        expected_return = self.model([NN_input_state, NN_input_action])


        return expected_return


    def SARSA_update(self, game_state, action, game_ended=False):
        '''
        This function is used to update the previous timestep with the new reward
        '''

        start_time = time.time()
        self.alpha = 0.1 # Learning rate
        gamma = 0.45 # Discount factor
        self.gamma = gamma


        # SA -> State action

        if game_ended:
            expected_return = 0
        else:
            expected_return = self.NN_get_expected_return(game_state, [action])[0]

        old_expected_return = self.NN_get_expected_return(self.game_state_history[-1], [self.action_history[-1]])[0]

        reward_list = self.rf.get_reward_from_state(game_state, self.game_state_history[-1])
        reward = np.sum(reward_list)
        self.latest_reward = reward_list

        NN_error = (reward + gamma*expected_return - old_expected_return)**2
        self.NN_error.append(NN_error)
        self.all_returns.append(reward)

        # Defining learning step - Is 0 if the only action available is the terminate action
        learning_step = float(self.alpha * (reward + gamma*expected_return - old_expected_return))


        if not(not(self.greedy_mode) or game_ended):
            learning_step = 0

        # Q_learning update
        old_expected_return_updated = old_expected_return + learning_step
        self.all_expected_returns.append(old_expected_return_updated[0])


        old_expected_return_updated = np.array(old_expected_return_updated).reshape((1,1))
        # Train the neural network with the new values


        NN_error = (reward + gamma*expected_return - old_expected_return)**2
        self.NN_error.append(NN_error)

        old_expected_return_updated = np.array(old_expected_return_updated).reshape((1,1))
        # Train the neural network with the new values


        # Printing the reward update step.
        self.latest_action = self.action_history[-1]
        self.latest_updated_expected_return = learning_step
        self.latest_action_type = self.game_state_history[-1]["Unique_actions"]
        self.latest_desired_expected_return = self.all_expected_returns[-1]


        self.turns_in_game += 1
        

        if self.greedy_mode == False and not game_ended:
            # self.update_NN(self.game_state_history[-1], self.action_history[-1], old_expected_return_updated)

            self.batch_size = 16

            # Every batch_size turns we will update the neural network with the batch_size new datasets
            if self.turns_in_game % self.batch_size == 0:

                input_matrix = self.game_state_list2NN_input(self.game_state_history[-self.batch_size:], self.action_history[-self.batch_size:])
                output_matrix = self.expected_return_list2NN_output(self.all_expected_returns[-self.batch_size:])
                

                self.update_NN_np_mat(input_matrix, output_matrix, batch_size=self.batch_size, epochs=6)


        # Game end update
        if game_ended:
            self.batch_size = 16

            input_matrix = self.game_state_list2NN_input(self.game_state_history[-self.batch_size:], self.action_history[-self.batch_size:])
            output_matrix = self.expected_return_list2NN_output(self.all_expected_returns[-self.batch_size:])
            self.update_NN_np_mat(input_matrix, output_matrix, batch_size=self.batch_size, epochs=6)



        self.SARSA_update_time.append(time.time() - start_time)


        self.SARSA_update_time.append(time.time() - start_time)


    def greedy_choice(self, list_of_actions, game_state):
        '''
        Until a neural network can give us the best state action rewards, we will use this function to give us the rewards
        '''
        time_start = time.time()
        expected_return = self.NN_get_expected_return(game_state, list_of_actions)

        self.NN_predict_time.append(time.time() - time_start)

        return list_of_actions[np.argmax(expected_return)]


    def epsilon_greedy_policy(self, list_of_actions, game_state, epsilon):
        '''
        This function is used to get the action from the epsilon greedy policy
        '''


        if np.random.rand() < epsilon:


            choice = np.random.choice(list_of_actions)


            # If random choice is choosen, then reduce the probability of choosing action -> -1.
            # Reroll random choice, if the choice was -1
            # Disabled if luck is 0
            if len(list_of_actions) != 1:
                choose_terminate_luck_score = 0
                for i in range(choose_terminate_luck_score):
                    if choice == -1:
                        choice = np.random.choice(list_of_actions)
                    else:
                        break
                

            return choice
        else:
            return self.greedy_choice(list_of_actions, game_state)


    def choose_action(self, list_of_actions, game_state):

        if self.game_state_history == []:
            self.game_state_history.append(copy.deepcopy(game_state))
            self.action_history.append(np.random.choice(list_of_actions))
            return self.action_history[-1]
        else:
            
            if self.greedy_mode:
                action = self.greedy_choice(list_of_actions, game_state)
            else:
                action = self.epsilon_greedy_policy(list_of_actions, copy.deepcopy(game_state), epsilon=0.2)
            
            self.SARSA_update(copy.deepcopy(game_state), action)


            self.game_state_history.append(copy.deepcopy(game_state))
            self.action_history.append(copy.deepcopy(action))




            #Remove the previous old values of game state and action history
            return action



    def write_state_reward_to_file(self, game_state):
        '''
        This function is used to write the current player reward into the reward file
        '''
        reward = self.rf.get_reward_from_state(game_state, self.game_state_history[-1])

        open_file = open(self.file_address, "a")
        open_file.write(f"{np.sum(reward)}  - {reward}\n")
        open_file.close()


        open_file = open(self.file_average_expected_rewards, "a")
        open_file.write(f"{np.mean(self.all_expected_returns)}\n")
        open_file.close()


        open_file = open(self.file_variance_expected_rewards, "a")
        open_file.write(f"{np.var(self.all_expected_returns)}\n")
        open_file.close()

        open_file = open(self.file_victory_points, "a")
        victory_points = np.sum(game_state["Victory_points"])
        open_file.write(f"{victory_points}\n")
        open_file.close()


        # Log the accuracy of the neural network
        open_file = open(self.file_Average_NN_error, "a")
        open_file.write(f"{np.mean(self.NN_error)}\n")
        open_file.close()

        open_file = open(self.file_variance_NN_error, "a")
        open_file.write(f"{np.var(self.NN_error)}\n")
        open_file.close()

        self.NN_error = []



        # sum of discounted returns of from the game


        discounted_returns = self.get_discounted_returns()
        
        open_file = open(self.file_average_returns, "a")
        open_file.write(f"{np.mean(discounted_returns)}\n")
        open_file.close()

        open_file = open(self.file_variance_returns, "a")
        open_file.write(f"{np.var(self.all_returns)}\n")
        open_file.close()


        open_file = open(self.file_games_won, "a")
        if game_state["main_Player_won"] == 1:
            open_file.write("1\n")
        else:
            open_file.write("0\n")
        open_file.close()


        open_file = open(self.file_game_length, "a")
        open_file.write(f"{np.sum(self.turns_in_game)}\n")
        open_file.close()



        sarsa_time = np.array(self.SARSA_update_time)
        convert2list_time = np.array(self.convert_state2list_time)
        NN_predict_time = np.array(self.NN_predict_time)
        # NN_training_time = np.array(self.NN_training_time)



        # print(f"NN predict: {np.mean(NN_predict_time)} - RUN {len(NN_predict_time)}")
        # print(f"NN training: {np.mean(NN_training_time)} - RUN {len(NN_training_time)}")



        self.take_action_time = []
        self.SARSA_update_time = []
        self.convert_state2list_time = []
        self.NN_predict_time = []
        # self.NN_training_time = []

        if self.games_played % 50 == 0:
            # Save model every 50 games
            self.model.save(f"NN_models/{self.player_name}_model.keras")



    def get_discounted_returns(self):
        '''
        This function is used to get the discounted returns from the reward list
        '''
        discounted_returns = 0
        for i in range(len(self.all_returns)):
            discounted_returns += self.all_returns[i] * self.gamma**i

        
        return discounted_returns

    def game_end_update(self, game_state):
        '''
        This function is used to update the neural network with the new values
        '''

        self.SARSA_update(game_state, None, game_ended=True)


        # At game end, train the neural network with all the new values of the 10 past games.
        input_matrix_gamestate, action_matrix = self.game_state_list2NN_input(self.game_state_history, self.action_history)
        output_matrix = self.expected_return_list2NN_output(self.all_expected_returns)


        self.input_data_past_game_states.append(input_matrix_gamestate)
        self.input_data_past_actions.append(action_matrix)


        self.output_label_past_games.append(output_matrix)
        all_game_states = np.concatenate(self.input_data_past_game_states, axis=0)
        all_actions = np.concatenate(self.input_data_past_actions, axis=0)
        all_output = np.concatenate(self.output_label_past_games, axis=0)

        self.update_NN_np_mat((all_game_states, all_actions), all_output, epochs=6, verybose=0, batch_size=32)



        if len(self.input_data_past_game_states) >= 10:
            self.input_data_past_game_states = self.input_data_past_game_states[1:]
            self.input_data_past_actions = self.input_data_past_actions[1:]

            self.output_label_past_games = self.output_label_past_games[1:]






    def notify_game_end(self, game_state):
        ''' [summary]
            This function is used to notify the player that the game has ended
        '''


        if self.greedy_mode:
            self.write_state_reward_to_file(game_state)

        # Deep sarsa will update its neural network with the new values
        self.game_end_update(game_state)


        # Saving the game data

        discounted_returns = self.get_discounted_returns()
        expected_returns = np.sum(self.all_expected_returns) # Already discounted


        ''' Removed these so that we can see the last reward
        self.latest_reward = None
        self.latest_action = None
        self.latest_action_type = None
        self.latest_updated_expected_return = None
        self.latest_desired_expected_return = None
        '''

        self.game_state_history = []
        self.action_history = []
        self.all_expected_returns = []
        self.turns_in_game = 0

        self.games_played += 1


        self.all_returns = []

        return discounted_returns, expected_returns



class Deep_Q_learning(Deep_SARSA):

    def __init__(self, player_name) -> None:
        super().__init__(player_name)
        self.initialize_target_NN()
        # Set epsilon randomly, such that the player sometimes learns using the known knowledge, and sometimes completely explores.
        # self.set_new_epsilon_value(min_val=0.4, max_val=1)
        self.epsilon_value = 0.95



    def Q_learning_update(self, game_state, list_of_actions, game_ended=False):
        '''
        This function is used to update the neural network based on the Q_learning_algorithm
        '''

        start_time = time.time()
        alpha = 0.04 # Learning rate
        gamma = 0.45 # Discount factor
        self.gamma = gamma


        # SA -> State action

        if game_ended:
            expected_return = 0
        else:

            ## Take the next step based on a greedy policy
            action = self.greedy_choice_target_NN(list_of_actions, game_state)
            expected_return = self.target_NN_get_expected_return(game_state, [action])[0]

        old_expected_return = self.NN_get_expected_return(self.game_state_history[-1], [self.action_history[-1]])[0]

        reward_list = self.rf.get_reward_from_state(game_state, self.game_state_history[-1])
        reward = np.sum(reward_list)
        self.latest_reward = reward_list


        NN_error = (reward + gamma*expected_return - old_expected_return)**2
        self.NN_error.append(NN_error)
        self.all_returns.append(reward)


        # Defining learning step - Is 0 if the only action available is the terminate action
        learning_step = float(alpha * (reward + gamma*expected_return - old_expected_return))


        if not(not(self.greedy_mode) or game_ended):
            learning_step = 0

        # Q_learning update
        old_expected_return_updated = old_expected_return + learning_step
        self.all_expected_returns.append(old_expected_return_updated[0])


        old_expected_return_updated = np.array(old_expected_return_updated).reshape((1,1))
        # Train the neural network with the new values


        # Printing the reward update step.
        self.latest_action = self.action_history[-1]
        self.latest_updated_expected_return = learning_step
        self.latest_action_type = self.game_state_history[-1]["Unique_actions"]
        self.latest_desired_expected_return = self.all_expected_returns[-1]



        self.turns_in_game += 1
        if self.greedy_mode == False and not game_ended:
            # self.update_NN(self.game_state_history[-1], self.action_history[-1], old_expected_return_updated)

            self.batch_size = 16

            # Every batch_size turns we will update the neural network with the batch_size new datasets
            if self.turns_in_game % self.batch_size == 0:

                input_matrix = self.game_state_list2NN_input(self.game_state_history[-self.batch_size:], self.action_history[-self.batch_size:])
                output_matrix = self.expected_return_list2NN_output(self.all_expected_returns[-self.batch_size:])
                

                self.update_NN_np_mat(input_matrix, output_matrix, batch_size=self.batch_size, epochs=6)


        # Game end update
        if game_ended:


            self.batch_size = 16

            input_matrix = self.game_state_list2NN_input(self.game_state_history[-self.batch_size:], self.action_history[-self.batch_size:])
            output_matrix = self.expected_return_list2NN_output(self.all_expected_returns[-self.batch_size:])
            self.update_NN_np_mat(input_matrix, output_matrix, batch_size=self.batch_size, epochs=6)



        self.SARSA_update_time.append(time.time() - start_time)


    def initialize_target_NN(self):
        '''
        To avoid maximation bias, a target neural network is formed, which is updated every 5 games.
        '''
        self.target_model = keras.models.clone_model(self.model)

        self.target_model.compile( optimizer='adam',
                            loss='huber',
                            metrics='accuracy',
                            loss_weights=None,
                            weighted_metrics=None,
                            run_eagerly=None,
                            steps_per_execution=None,
                            jit_compile=None,
                            )
        

        self.target_model.summary()



    def target_NN_get_expected_return(self, game_state, actions_list):
        '''
        This function gives the value from the target neural network to the state action pair
        '''

        NN_input_state, NN_input_action = self.game_state2list_NN_input(game_state, actions_list)
        expected_return = self.target_model([NN_input_state, NN_input_action])


        return expected_return

    def greedy_choice_target_NN(self, list_of_actions, game_state):
        '''
        Until a neural network can give us the best state action rewards, we will use this function to give us the rewards
        '''
        time_start = time.time()
        expected_return = self.target_NN_get_expected_return(game_state, list_of_actions)

        self.NN_predict_time.append(time.time() - time_start)

        return list_of_actions[np.argmax(expected_return)]

    def update_target_NN_np_mat(self, input_matrix, output_matrix, epochs=6, verbose=0, batch_size=16):
        '''
        This function is used to update the neural network using a list of all the values used in the game
        '''
        self.target_model.fit(input_matrix, output_matrix, epochs=6, verbose=0, batch_size=16)



    def choose_action(self, list_of_actions, game_state):

        if self.game_state_history == []:
            self.game_state_history.append(copy.deepcopy(game_state))
            self.action_history.append(np.random.choice(list_of_actions))
            return self.action_history[-1]
        else:
            action_time = time.time()

            if self.greedy_mode:
                action = self.greedy_choice(list_of_actions, game_state)
            else:
                action = self.epsilon_greedy_policy(list_of_actions, game_state, epsilon=self.epsilon_value )


            self.Q_learning_update(game_state, list_of_actions, game_ended=False)



            # Set boolean so the reward function is constricted for the next state
            if len(list_of_actions) == 1 and action == -1:
                self.only_terminate_action = True
            else:
                self.only_terminate_action = False


            self.game_state_history.append(copy.deepcopy(game_state))
            self.action_history.append(copy.deepcopy(action))

            #Remove the previous old values of game state and action history
            self.take_action_time.append(time.time() - action_time)
            return action


    def game_end_update(self, game_state):
        '''
        This function is used to update the neural network with the new values
        '''

        self.Q_learning_update(game_state, None, game_ended=True)

        # At game end, train the neural network with all the new values of the 10 past games.
        input_matrix_gamestate, action_matrix = self.game_state_list2NN_input(self.game_state_history, self.action_history)
        output_matrix = self.expected_return_list2NN_output(self.all_expected_returns)


        self.input_data_past_game_states.append(input_matrix_gamestate)
        self.input_data_past_actions.append(action_matrix)


        self.output_label_past_games.append(output_matrix)
        all_game_states = np.concatenate(self.input_data_past_game_states, axis=0)
        all_actions = np.concatenate(self.input_data_past_actions, axis=0)
        all_output = np.concatenate(self.output_label_past_games, axis=0)

        self.update_NN_np_mat((all_game_states, all_actions), all_output, epochs=6, verybose=0, batch_size=32)



        if len(self.input_data_past_game_states) >= 30:
            self.input_data_past_game_states = self.input_data_past_game_states[1:]
            self.input_data_past_actions = self.input_data_past_actions[1:]

            self.output_label_past_games = self.output_label_past_games[1:]

        
        # If n games has passed, then update the target neural network
        if self.games_played % 15 == 0:
            self.update_target_NN_np_mat((all_game_states, all_actions), all_output, epochs=6, verbose=0, batch_size=32)

