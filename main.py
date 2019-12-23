from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf


import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
# from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import ta

from ddqn import DuelingDQN



def process_data(env):
    df = ta.utils.dropna(env.df)
    df = ta.add_all_ta_features(env.df, open="Open", high="High", low="Low", close="Close", volume="Volume")

    prices = env.df.loc[:, 'Close'].to_numpy()

    index = env.frame_bound[0] - env.window_size
    prices[index]  # validate index (TODO: Improve validation)
    prices = prices[index : env.frame_bound[1]]

    # for col in df.columns:
    #     print(col)
    # diff = np.insert(np.diff(prices), 0, 0)

    
    # signal_features = np.column_stack((prices, diff))
    # print(signal_features)
    signal_features = df.loc[:, df.columns].to_numpy()
    print(signal_features)

    return prices, signal_features

class WithTechnicalIndicators(ForexEnv):
    _process_data = process_data



def render_all(self, mode='human'):
    window_ticks = np.arange(len(self.prices))
    # plt.plot(self.prices)

    short_ticks = []
    long_ticks = []
    for i, tick in enumerate(window_ticks):
        if i >= len(self._position_history):
            break
        if self._position_history[i] == Positions.Short:
            short_ticks.append(tick)
        elif self._position_history[i] == Positions.Long:
            long_ticks.append(tick)

    # plt.plot(short_ticks, self.prices[short_ticks], 'ro')
    # plt.plot(long_ticks, self.prices[long_ticks], 'go')

    # plt.suptitle(
    print(
        "Total Reward: %.6f" % self._total_reward + ' ~ ' +
        "Total Profit: %.6f" % self._total_profit
    )



env = WithTechnicalIndicators(df=(pd.read_csv('testdata.csv', index_col='Timestamp')), frame_bound=(50, 60000), window_size=1)
# env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)

MEMORY_SIZE = 3000
ACTION_SPACE = 2

sess = tf.Session()

RL = DuelingDQN(n_actions=ACTION_SPACE, n_features=72, memory_size=MEMORY_SIZE, e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)

sess.run(tf.global_variables_initializer())

while True:
    accumulated_reward = [0]
    total_steps = 0
    observation = env.reset()
    for i in tqdm(range(0, 50000+MEMORY_SIZE)): #while True:
        # action = env.action_space.sample()

        action = RL.choose_action(observation[0])

        observation_, reward, done, info = env.step(action)

        accumulated_reward.append(reward + accumulated_reward[-1])

        RL.store_transition(observation[0], action, reward, observation_[0])


        if total_steps > MEMORY_SIZE:
            RL.learn()

        if total_steps - MEMORY_SIZE > 50000:
            break

        observation = observation_
        total_steps += 1
        # env.render()
        if done:
            print("info:", info)
            

    # plt.cla()
    # env.render_all()
    render_all(env)
    # plt.show()

    if env._total_reward > 50000000:
        break
