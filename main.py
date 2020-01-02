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
from sklearn.preprocessing import MinMaxScaler

from policies.dqn import DqnPolicy


import os
import time
from configs.manager import ConfigManager

from ti_env import WithTechnicalIndicators


# import tensorflow as tf
# sess=tf.Session() 
# signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
# input_key = 'x_input'
# output_key = 'y_output'

# export_path =  './savedmodel'
# meta_graph_def = tf.saved_model.loader.load(
#            sess,
#           [tf.saved_model.tag_constants.SERVING],
#           export_path)
# signature = meta_graph_def.signature_def

# x_tensor_name = signature[signature_key].inputs[input_key].name
# y_tensor_name = signature[signature_key].outputs[output_key].name

# x = sess.graph.get_tensor_by_name(x_tensor_name)
# y = sess.graph.get_tensor_by_name(y_tensor_name)

# y_out = sess.run(y, {x: 3.0})


def run(config_name, model_name=None):
    cfg = ConfigManager.load(config_name)
    # cfg.add_env(env)

    if model_name is None:
        model_name = '-'.join([
            cfg.env_name.lower(),
            cfg.policy_name.replace('_', '-'),
            os.path.splitext(os.path.basename(config_name))[0] if config_name else 'default',
            str(int(time.time()))
        ])

    model_name = model_name.lower()
    cfg.start_training(model_name)




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



# env = WithTechnicalIndicators(df=(pd.read_csv('testdata.csv', index_col='Timestamp')), frame_bound=(50, 60000), window_size=1)
# env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)

from gym.envs.registration import register


register(
    id='technical-v0',
    entry_point=WithTechnicalIndicators,
    kwargs={
        'df': pd.read_csv('testdata.csv', index_col='Timestamp'),
        'window_size': 1,
        'frame_bound': (50, 60000)
    }
)

# env = gym.make('technical-v0')

MEMORY_SIZE = 3000
ACTION_SPACE = 2

# sess = tf.Session()

# RL = DuelingDQN(n_actions=ACTION_SPACE, n_features=71, memory_size=MEMORY_SIZE, e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)
# RL = DqnPolicy(env, 'trading')
# RL.build()
# config = RL.TrainConfig(lr = 0.001, epsilon = 1.0, epsilon_final = 0.02)

        # "warmup_episodes": 450,
        # "log_every_episode": 10,
        # "n_episodes": 500,
        # "target_update_every_step": 10
# RL.train(config)

run('configs/data/ppo.json')


# sess.run(tf.global_variables_initializer())

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
