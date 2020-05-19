import random 
import math
import tensorflow as tf 
import gym 
import numpy as np 
from collections import deque
import Blackjack
import argparse
import matplotlib.pyplot as plt

class DQN_Solver:

    def __init__(self, env_config=None):
        if env_config is None:
            self.env = gym.make('Blackjack-v1')#, env_config={"card_values": [2] * 52})
            self._init()
        else:
            self.env = gym.make('Blackjack-v1', env_config=env_config)#, env_config={"card_values": [2] * 52})
            self._init(**env_config)
    

    def _init(self, one_card_dealer=False, card_values=None, n_episodes=20, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.003, alpha_decay=0.01, batch_size=32):
        self.memory = deque(maxlen=100000)
        #self.num_states = len(self.env.observation_space.sample())
        self.num_states = 57 #3
        self.num_actions = self.env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.learning_rate = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.rewards = []

        # Blackjack specific
        if card_values is None:
            self._card_values = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4)
        else:
            self._card_values = np.asarray(card_values)
        

        hidden_units=[64,32]#[24,48]

        ## training network
        self.train_model = tf.keras.Sequential()
        self.train_model.add(tf.keras.layers.InputLayer(input_shape=(self.num_states,)))
        for i in hidden_units:
            self.train_model.add(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'
            ))
        self.train_model.add(tf.keras.layers.Dense(
            self.num_actions, activation='linear', kernel_initializer='RandomNormal'
        ))
        self.train_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, decay=alpha_decay))


        ## target network
        self.target_model = tf.keras.Sequential()
        self.target_model.add(tf.keras.layers.InputLayer(input_shape=(self.num_states,)))
        for i in hidden_units:
            self.target_model.add(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'
            ))
        self.target_model.add(tf.keras.layers.Dense(
            self.num_actions, activation='linear', kernel_initializer='RandomNormal'
        ))
        self.target_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, decay=alpha_decay))

        print(f'one card dealer: {one_card_dealer}')
        print(f'card values: {self._card_values}')

    def get_dealer_value(self, card_index):
        if verbose: 
            print(f'dealer card: {card_index}, dealer value: {self._card_values[card_index]}')
        return self._card_values[card_index]

    def get_player_value(self, deck_state):
        card_indices = np.argwhere(deck_state==True).flatten()
        #print(f'player card indices: {card_indices}')
        n_cards = len(card_indices)
        card_values = self._card_values[card_indices]

        #print(f'card values: {card_values}')
        
        card_values = np.sum(card_values)
        if verbose: 
            print(f'player summed score: {card_values}')
        
        return card_values, int(1 in card_indices), n_cards
        

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.train_model.predict(state))
    
    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def get_card_deck(self, state):
        
        player_cards = ~state[0]
        
        dealer_card = state[1]
        player_cards[dealer_card] = False
        

        deck = [val if b else 0 for (val,b) in zip(self._card_values, player_cards)]
        return np.array(deck)

    def preprocess_state(self, state):
        feature_vec = self.get_card_deck(state)
        player_value, has_ace, n_cards = self.get_player_value(state[0])
        dealer_value = self.get_dealer_value(state[1])
        player_missing = 21-player_value
        dealer_missing = 21-dealer_value
        
        #state = np.array([player_value, has_ace, dealer_value])
        state = np.concatenate([np.array([player_value, player_missing, n_cards, dealer_value, dealer_missing]), feature_vec])
        #print(f'preprocessed state: {state}')
        #print(f'length: {len(state)}')
        return np.reshape(state, [1,57])

    def replay(self, batch_size):
        #print(f'replay')
        x_batch, y_batch = [],[]
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        minibatch = np.array(minibatch)
        
        for state, action, reward, next_state, done in minibatch:  

            #print(f'state while replaying: {state}')   

            y_target = self.train_model.predict(state)
            
            ## ACTION IS AN INDEX!
            y_target[0][action] = reward if done else reward + self.gamma*np.max(self.target_model.predict(next_state)[0])
            
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.train_model.fit(np.array(x_batch), np.array(y_batch), verbose=0)#,batch_size=len(x_batch))
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def copy_weights(self):
        self.target_model.set_weights(self.train_model.get_weights())


    def run(self):

        scores = deque(maxlen=100)

        for e in range(self.n_episodes):

            
            unprocessed_state = self.env.reset()
            #print(f'unprocessed: {unprocessed_state}')
            state = self.preprocess_state(unprocessed_state)
            
            if verbose:     
                print(f'new episode')
                print(f'preprocessed state: {state}')
            #print(f'type of state: {type(state)}')
            done = False


            # perform actions until pole falls down
            while not done: 

                #self.env.render()

                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                
                if verbose: 
                    print(f'action chosen: {action}')
                    print(f'preprocessed state: {state}')
                    print(f'reward: {reward}')

                self.memorize(state, action, reward, next_state, done)
                state = next_state
            
            # as soon as the pole falls down, we record the achieved score
            score = reward
            scores.append(score)
            mean_score = np.mean(scores)
            self.rewards.append(mean_score)

            # we want consistent performance over a certain amount of episodes
            if mean_score > self.n_win_ticks and e >= 100:
                print(f'Ran {e} episodes, solved after {e-100} trials.')

            if e % 10 == 0:
                print(f'Episode {e}, mean score over the last 100 episodes: {mean_score}!')

            

            # after every episode, we train the DQN by replaying past experiences
            self.replay(self.batch_size)
            

            if e % 1 == 0:
                self.copy_weights()

        titles = {
            1: "Basic",
            2: "Regular",
            3: "All the 2's",
            4: "Random"
        }
        plt.plot(self.rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Mean Reward")
        plt.title(f'{titles[MODE]}')
        plt.ylim((0,1))
        plt.savefig(f'./plots/{MODE}.png')
        #plt.show()


if __name__ == "__main__":

    # select Blackjack mode
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', type=int, help='Blackjack mode')
    args = parser.parse_args()

    MODE = args.mode

    if MODE == 1:
        env_config = {"one_card_dealer": True}

    elif MODE == 2:
        env_config = {}

    elif MODE == 3:
        env_config = {"card_values": [2] * 52}

    elif MODE == 4:
        env_config = {"card_values": [3,  1,  3,  9,  6,  0,  7, -2,  2,  6,  8,  1,  3,
                                               4, -1,  4,  3,  9, -1,  4,  0,  4,  7, -2, -1,  5,
                                               2,  6, -3, -1,  2,  2, -1,  7,  1,  0,  7,  8,  4,
                                               5,  3, -1,  0,  3, -1,  3,  0,  6, -2,  4, -3,  4]}

    else: 
        print(f'Mode has to be an integer between 1 and 4')
        exit(0)

    verbose = 0
    agent = DQN_Solver(env_config=env_config)
    agent.run()
