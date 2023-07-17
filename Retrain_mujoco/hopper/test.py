import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO.load("/home/zck7060/Retrain_mujoco/hopper/masknet/models/best_model/best_model")

env = DummyVecEnv([lambda: gym.make("Hopper-v3")])
env = VecMonitor(env, "models/")
# eval_env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)
#reward_vec, length_vec = evaluate_policy(model, env,  n_eval_episodes=500, return_episode_rewards=True)

base_model = PPO.load("/home/zck7060/Retrain_mujoco/hopper/baseline/models/best_model/best_model")

reward_vec = []

for i in range(6,7):
    reward = 0
    mask_num = 0
    count = 0
    action_seq = []
    mask_probs = []
    env.seed(i)
    env.action_space.seed(0)
    obs = env.reset()

    while count < 1:
        
        action, _states = model.predict(obs, deterministic=True)
        base_action, _states = base_model.predict(obs, deterministic=True)
        obs, vectorized_env = model.policy.obs_to_tensor(obs)
        mask_dist = model.policy.get_distribution(obs)
        mask_prob = np.exp(mask_dist.log_prob(torch.Tensor(action).cuda()).detach().cpu().numpy()[0])
        #print(mask_prob)
        mask_probs.append(mask_prob)
        
        if action == 0:
            mask_num += 1
        
        print(obs)
        print(base_action)
        obs, rewards, dones, info = env.step(base_action)

        action_seq.append(base_action)

        reward += rewards[0]

        count += 1
        if dones:
            reward_vec.append(reward)
            break

    print("Test ", i)
    print("reward: ", reward)


