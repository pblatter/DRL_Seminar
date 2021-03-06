import random 
import math
import tensorflow as tf 
import gym 
import numpy as np 
from collections import deque
import Blackjack

class DQN_Solver:

    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=32, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('Blackjack-v1')#, env_config={"card_values": [2] * 52})
        #self.num_states = len(self.env.observation_space.sample())
        self.num_states = 5 #3
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

        # Blackjack specific
        self._card_values = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4)
        #self._card_values = np.asarray([2] * 52)

        hidden_units=[64,32,24]#[24,48]

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

    def get_dealer_value(self, card_index):
        if verbose: 
            print(f'dealer card: {card_index}, dealer value: {self._card_values[card_index]}')
        return self._card_values[card_index]

    def get_player_value(self, deck_state):
        card_indices = np.argwhere(deck_state==True).flatten()
        #print(f'player card indices: {card_indices}')
        n_cards = len(card_indices)
        card_values = self._card_values[card_indices]
        
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

    def preprocess_state(self, state):
        player_value, has_ace, n_cards = self.get_player_value(state[0])
        dealer_value = self.get_dealer_value(state[1])
        player_missing = 21-player_value
        dealer_missing = 21-dealer_value
        
        #state = np.array([player_value, has_ace, dealer_value])
        state = np.array([player_value, player_missing, n_cards, dealer_value, dealer_missing])
        #print(f'preprocessed state: {state}')
        return np.reshape(state, [1,5])

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

            
            
            state = self.preprocess_state(self.env.reset())
            # [player_value, has_ace, dealer_value]
            
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

            # we want consistent performance over a certain amount of episodes
            if mean_score > self.n_win_ticks and e >= 100:
                print(f'Ran {e} episodes, solved after {e-100} trials.')

            if e % 10 == 0:
                print(f'Episode {e}, mean score over the last 100 episodes: {mean_score}!')

            

            # after every episode, we train the DQN by replaying past experiences
            self.replay(self.batch_size)
            

            if e % 15 == 0:
                self.copy_weights()

if __name__ == "__main__":
    verbose = 0
    agent = DQN_Solver()
    agent.run()
