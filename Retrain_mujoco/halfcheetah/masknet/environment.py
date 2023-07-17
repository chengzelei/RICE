import gym
from gym import Wrapper
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
import pickle
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
from stable_baselines3.common.running_mean_std import RunningMeanStd

class Multi2SingleEnv(Wrapper):

    def __init__(self, env, base_model, vec_normalize):

        """ from multi-agent environment to single-agent environment.
        :param: env: two-agent environment.
        :param: agent: victim agent.
        :param: agent_idx: victim agent index.
        :param: shaping_params: shaping parameters.
        :param: scheduler: anneal scheduler.
        :param: norm: normalize agent or not.
        :param: retrain_victim: retrain victim agent or not.
        :param: clip_obs: observation clip value.
        :param: clip_rewards: reward clip value.
        :param: gamma: discount factor.
        :param: epsilon: additive coefficient.
        """
        Wrapper.__init__(self, env)
        self.env = env
        self.base_model = base_model
        self.obs = None
        self.done = False
        self.vec_normalize = vec_normalize



    def step(self, action):
        base_action, _states = self.base_model.predict(self.obs, deterministic=True)

        if action == 0:
            real_action = self.env.action_space.sample()
        else:
            real_action = base_action

        real_action = base_action
            
        # obtain needed information from the environment.
        obs, reward, done, info = self.env.step(real_action)
        obs = np.clip((obs - self.vec_normalize.obs_rms.mean) / np.sqrt(self.vec_normalize.obs_rms.var + self.vec_normalize.epsilon), -self.vec_normalize.clip_obs, self.vec_normalize.clip_obs).astype(np.float32)
        self.obs = obs
        return obs, reward, done, info



    def reset(self):
        obs = self.env.reset()

        obs = np.clip((obs - self.vec_normalize.obs_rms.mean) / np.sqrt(self.vec_normalize.obs_rms.var + self.vec_normalize.epsilon), -self.vec_normalize.clip_obs, self.vec_normalize.clip_obs).astype(np.float32)
        self.obs = obs
        return obs





def make_multi2single_env(env_name):

    vec_norm_path = "/home/zck7060/Retrain_mujoco/halfcheetah/baseline/weak_models/best_model/vec_normalize.pkl"
    with open(vec_norm_path, "rb") as file_handler:
        vec_normalize = pickle.load(file_handler)

    base_model = PPO.load("/home/zck7060/Retrain_mujoco/halfcheetah/baseline/weak_models/best_model/best_model")

    env = DummyVecEnv([lambda: Multi2SingleEnv(gym.make(env_name), base_model, vec_normalize)])
    env = VecMonitor(env, "weak_models/")
    
    return env