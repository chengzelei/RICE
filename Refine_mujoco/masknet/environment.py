import gym
from gym import Wrapper
from stable_baselines3 import PPO
import numpy as np

class Multi2SingleEnv(Wrapper):

    def __init__(self, env, base_model, train, alpha = 0):

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
        self.train = train
        self.alpha = alpha



    def step(self, action):
        base_action, _states = self.base_model.predict(self.obs, deterministic=True)
        
        if action == 0:
            real_action = self.env.action_space.sample()
        else:
            real_action = base_action
            
        # obtain needed information from the environment.
        obs, reward, done, info = self.env.step(real_action)

        if self.train:
            reward += self.alpha * (1 - action)

        self.obs = obs
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.obs = obs
        return obs

def make_multi2single_env(env_name, train, alpha=0):
    env = gym.make(env_name)
    base_model = PPO.load("mujoco/hopper/near_optimal/model/best_model")
    return Multi2SingleEnv(env, base_model, train, alpha)
