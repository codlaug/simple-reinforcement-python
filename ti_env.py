from sklearn.preprocessing import MinMaxScaler
import ta
import pandas as pd
import numpy as np
import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 

def process_data(env):
    # df = env.df[env.frame_bound[0] : env.frame_bound[1]]
    df = ta.utils.dropna(env.df)
    df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    
    # print(df)

    
    # min_max_scaler.fit(df.loc[:, df.columns].values)

    columns = df.columns.values.tolist()
    columns.remove('momentum_kama') # was all nans
    print(columns)

    signals = np.nan_to_num(df.loc[:, columns].to_numpy())

    
    prices = df.loc[:, 'Close'].to_numpy()


    index = env.frame_bound[0] - env.window_size
    prices[index]  # validate index (TODO: Improve validation)
    prices = prices[index : env.frame_bound[1]]

    # signal_features = []

    # for col in columns:
    #     print(col)
    #     print(df.loc[:, col].to_numpy())
    # diff = np.insert(np.diff(prices), 0, 0)

    
    # signal_features = np.column_stack((prices, diff))
    # print(signal_features)
    
    # print(signal_features)

    return prices, signals

def get_observation(self):
    if(self._current_tick <= 28):
        signals = self.signal_features[self._current_tick : self._current_tick+28]
    else:
        signals = self.signal_features[(self._current_tick-28) : self._current_tick]
    # print(lookback, self._current_tick)

    min_max_scaler = MinMaxScaler()

    signal_features = min_max_scaler.fit_transform(signals)
    print(signal_features[self._current_tick])

    if self._current_tick < 28:
        return signal_features[self._current_tick]
    else:
        return signal_features[-1]


class WithTechnicalIndicators(ForexEnv):
    _process_data = process_data

    _get_observation = get_observation

