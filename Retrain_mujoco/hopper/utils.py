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
        self.mask_probs = []
        self.reward = 0
    
    def set(self, eps_len, act_seq, state_seq, mask_probs, reward):
        self.eps_len = eps_len
        self.act_seq = act_seq
        self.state_seq = state_seq
        self.mask_probs = mask_probs
        self.reward = reward


def gen_one_traj(env, seed):
    traj = Traj()
    model = PPO.load("/home/zck7060/Retrain_mujoco/hopper/masknet/models/best_model/best_model")

    # eval_env = VecNormalize(env, norm_obs=True, norm_reward=False,
    #                    clip_obs=10.)
    #reward_vec, length_vec = evaluate_policy(model, env,  n_eval_episodes=500, return_episode_rewards=True)

    base_model = PPO.load("/home/zck7060/Retrain_mujoco/hopper/baseline/tmp/best_model/best_model")

    reward = 0
    mask_num = 0
    count = 0
    action_seq = []
    state_seq = []
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

        count += 1
        if dones:
            traj.set(count, action_seq, state_seq, mask_probs, reward)
            break



    return traj