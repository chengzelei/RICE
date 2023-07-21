import gym
from gym import Wrapper
from stable_baselines3 import PPO
import numpy as np
import torch
import torch.nn as nn
import random
from utils import gen_one_traj


class RetrainEnv(Wrapper):

    def __init__(self, env, rand_sampling=False):
        Wrapper.__init__(self, env)
        self.env = env
        self.seed = 0
        self.random_sampling = rand_sampling
        self.p = 1
        self.init_reward = 0
        self.flag = False

    def step(self, action):
        # obtain needed information from the environment.
        obs, reward, done, info = self.env.step(action)
        if not self.flag:
            info["true_reward"] = self.init_reward + reward
            self.flag = True
        info["true_reward"] = reward
        return obs, reward, done, info


    def reset(self):
        self.flag = False
        self.seed += 1
        traj = gen_one_traj(self.env, self.seed)

        if np.random.rand() > self.p:
            if self.random_sampling:
                start_idx = np.random.choice(traj.eps_len)
            else:
                start_idx = np.argmax(traj.mask_probs)
        else:
            start_idx = 0
        self.init_reward = traj.reward_seq[start_idx]
        print("start idx: ", start_idx)
        print("initial reward: ", self.init_reward)
        self.env.sim.set_state(traj.state_seq[start_idx])
        position = self.env.sim.data.qpos.flat.copy()[1:]
        velocity = self.env.sim.data.qvel.flat.copy()
        obs = np.concatenate((position, velocity)).ravel()
        # for i in range(self.env.num_envs):
        #     self.env.envs[i].sim.set_state(traj.state_seq[start_idx])
        #     position = self.env.envs[i].sim.data.qpos.flat.copy()[1:]
        #     velocity = self.env.envs[i].sim.data.qvel.flat.copy()
        #     obs = np.concatenate((position, velocity)).ravel()
        #     self.env._save_obs(i, obs)
        # return self.env._obs_from_buf()
        return obs



def make_retrain_env(env_name, rand_sampling=False):
    env = gym.make(env_name)
    return RetrainEnv(env, rand_sampling=False)

