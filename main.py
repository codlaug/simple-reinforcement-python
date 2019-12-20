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

from ddqn import DuelingDQN

# data = json.load(open('testdata.json', 'r'))




def render_all(self, mode='human'):
    window_ticks = np.arange(len(self.prices))
    plt.plot(self.prices)

    short_ticks = []
    long_ticks = []
    for i, tick in enumerate(window_ticks):
        if i >= len(self._position_history):
            break
        if self._position_history[i] == Positions.Short:
            short_ticks.append(tick)
        elif self._position_history[i] == Positions.Long:
            long_ticks.append(tick)

    plt.plot(short_ticks, self.prices[short_ticks], 'ro')
    plt.plot(long_ticks, self.prices[long_ticks], 'go')

    plt.suptitle(
        "Total Reward: %.6f" % self._total_reward + ' ~ ' +
        "Total Profit: %.6f" % self._total_profit
    )



env = gym.make('forex-v0', df=(pd.read_csv('testdata.csv', index_col='Time')), frame_bound=(50, 60000), window_size=10)
# env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)

MEMORY_SIZE = 3000
ACTION_SPACE = 2

sess = tf.Session()

RL = DuelingDQN(n_actions=ACTION_SPACE, n_features=2, memory_size=MEMORY_SIZE, e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)

sess.run(tf.global_variables_initializer())

accumulated_reward = [0]
total_steps = 0
observation = env.reset()
while True:
    # action = env.action_space.sample()

    action = RL.choose_action(observation[0])

    # TODO: multiple episodes - do env.reset() inside a loop so it can try over and over again


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
        

plt.cla()
# env.render_all()
render_all(env)
plt.show()
