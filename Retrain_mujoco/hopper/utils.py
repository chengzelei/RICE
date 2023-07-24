import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor


class Traj:
    def __init__(self):
        self.eps_len = 0
        self.act_seq = []
        self.state_seq = []
        self.reward_seq = []
        self.mask_probs = []
        self.reward = 0
    
    def set(self, eps_len, act_seq, state_seq, reward_seq, mask_probs, reward):
        self.eps_len = eps_len
        self.act_seq = act_seq
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        self.mask_probs = mask_probs
        self.reward = reward


def gen_one_traj(env, seed):
    traj = Traj()
    model = PPO.load("/home/zck7060/Retrain_mujoco/hopper/masknet/weak_models/best_model/best_model")

    # eval_env = VecNormalize(env, norm_obs=True, norm_reward=False,
    #                    clip_obs=10.)

    base_model = PPO.load("/home/zck7060/Retrain_mujoco/hopper/baseline/weak_tmp/best_model/best_model")

    reward = 0
    mask_num = 0
    count = 0
    action_seq = []
    state_seq = []
    reward_seq = []
    mask_probs = []
    env.seed(seed)
    obs = env.reset()

    while True:

        action, _states = model.predict(obs, deterministic=True)
        base_action, _states = base_model.predict(obs, deterministic=True)
        obs, vectorized_env = model.policy.obs_to_tensor(obs)
        mask_dist = model.policy.get_distribution(obs)
        mask_prob = np.exp(mask_dist.log_prob(torch.Tensor([1]).cuda()).detach().cpu().numpy()[0])
        state_seq.append(env.sim.get_state())
        mask_probs.append(mask_prob)
        action_seq.append(base_action)
            
        if action == 0:
            mask_num += 1
        obs, rewards, dones, info = env.step(base_action)
        
        reward += rewards
        reward_seq.append(reward)
        count += 1
        if dones:
            traj.set(count, action_seq, state_seq, reward_seq, mask_probs, reward)
            break

    return traj

from typing import Tuple

import numpy as np


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: int = 0):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count