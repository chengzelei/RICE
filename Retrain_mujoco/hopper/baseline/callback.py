from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from typing import Any, Callable, Dict, List, Optional, Union

import os
import gym
import numpy as np
import warnings
from collections import deque
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization




class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=0):
        super(CustomCallback, self).__init__(verbose)

        self.check_freq = check_freq
        self.log_dir = log_dir


    def _init_callback(self) -> None:
        # Create folder if needed
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              self.logger.record("train/mean_reward", float(mean_reward))
        
        
        return True