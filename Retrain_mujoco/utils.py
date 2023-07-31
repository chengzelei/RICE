import gym
import numpy as np
import torch
from stable_baselines3 import PPO
import pickle

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


def gen_one_traj(env, seed, agent_path, masknet_path, vec_norm_path):
    traj = Traj()
    model = PPO.load(masknet_path)
    base_model = PPO.load(agent_path)

    if vec_norm_path != None:
        with open(vec_norm_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)

    reward = 0
    mask_num = 0
    count = 0
    action_seq = []
    state_seq = []
    reward_seq = []
    mask_probs = []
    # env.seed(seed)
    obs = env.reset()

    while True:
        if vec_norm_path != None:
            obs = np.clip((obs - vec_normalize.obs_rms.mean) / np.sqrt(vec_normalize.obs_rms.var + vec_normalize.epsilon), - vec_normalize.clip_obs, vec_normalize.clip_obs).astype(np.float32)
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


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: int = 0):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
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