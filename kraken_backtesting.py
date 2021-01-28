import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import sys

np.set_printoptions(threshold=sys.maxsize, linewidth=200, suppress=True)

model = tf.keras.models.load_model("modeldeepqkraken")

class Trading:

    training_sets = []
    close_prices = []
    #Defines long, cash, short position
    pos = 0
    #Index when this was last changed
    pos_i = 0
    i = 0
    depth = 20

    def __init__(self):
        self.training_sets, self.close_prices = self.preprocess()


    def preprocess(self):
        training_sets = np.load('Kraken_Trading_History/XBTUSD_15min_train_sets.npy').astype('float32')
        candle_sticks = np.load('Kraken_Trading_History/XBTUSD_15min_candles.npy').astype('float32')
        #print(training_sets[22796].reshape(13, 13))
        #print(np.where(training_sets == np.min(training_sets))) #, np.max(training_sets)
        #print(np.min(candle_sticks), np.max(candle_sticks))
        return training_sets, candle_sticks[300:, 3]

    def reset(self):
        self.pos = 0.
        self.i = random.randint(0,200000) + self.depth
        self.pos_i = self.i
        return np.column_stack(self.training_sets[self.i - self.depth:self.i]).reshape(13, 13, self.depth)

    def pnl(self):
        return self.close_prices[self.i] - self.close_prices[self.pos_i]

    def step(self, value):
        self.i += 1
        if self.i >= self.training_sets.shape[0] - 1:
            done = True
        else:
            done = False

        previous_pos = self.pos
        if value == 0:
            #short fully
            self.pos = -1
        elif value == 1:
            #cash
            self.pos = 0
        elif value == 2:
            #long fully
            self.pos = 1

        reward = 0
        if previous_pos != self.pos:
            self.pos_i = self.i
            #Assuming 1% trading costs and slippage
            reward = previous_pos * self.pnl() - (np.abs(previous_pos) + np.abs(self.pos)) * 0.0056

        nn_input = np.column_stack(self.training_sets[self.i-self.depth:self.i]).reshape(13, 13, self.depth)

        return nn_input, reward, done

env = Trading()

state = np.array(env.reset())
episode_reward = 0

for timestep in range(1, 1000000):
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    # Take best action
    action = tf.argmax(action_probs[0]).numpy()

    # Apply the sampled action in our environment
    state_next, reward, done = env.step(action)
    state_next = np.array(state_next)

    episode_reward += reward

    state = state_next

    if timestep % 10000 == 0:
        print(episode_reward)
        print(episode_reward / timestep)

    if done:
        print(episode_reward)
        print(episode_reward / timestep)
        break