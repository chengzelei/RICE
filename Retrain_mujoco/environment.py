import gym
from gym import Wrapper
from stable_baselines3 import PPO
import numpy as np
import torch
import torch.nn as nn
import random
from utils import gen_one_traj


class RetrainEnv(Wrapper):

    def __init__(self, env, go_prob, agent_path, masknet_path, rand_sampling=False, vec_norm_path=None):
        Wrapper.__init__(self, env)
        self.env = env
        self.seed = 0
        self.random_sampling = rand_sampling
        self.p = go_prob
        self.init_reward = 0
        self.flag = False
        self.agent_path = agent_path
        self.masknet_path = masknet_path
        self.vec_norm_path = vec_norm_path

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not self.flag:
            info["true_reward"] = self.init_reward + reward
            self.flag = True
        else:
            info["true_reward"] = reward
        return obs, reward, done, info


    def reset(self):
        self.flag = False
        self.seed += 1
        traj = gen_one_traj(self.env, self.seed, self.agent_path, self.masknet_path, self.vec_norm_path)

        if np.random.rand() > self.p:
            if self.random_sampling:
                start_idx = np.random.choice(traj.eps_len)
            else:
                start_idx = np.argmax(traj.mask_probs)
        else:
            start_idx = 0
        self.init_reward = traj.reward_seq[start_idx]
        self.env.sim.set_state(traj.state_seq[start_idx])
        position = self.env.sim.data.qpos.flat.copy()[1:]
        velocity = self.env.sim.data.qvel.flat.copy()
        obs = np.concatenate((position, velocity)).ravel()
        return obs



def make_retrain_env(env_name, go_prob, agent_path, masknet_path, rand_sampling=False, vec_norm_path=None):
    env = gym.make(env_name)
    return RetrainEnv(env, go_prob, agent_path, masknet_path, rand_sampling, vec_norm_path)

