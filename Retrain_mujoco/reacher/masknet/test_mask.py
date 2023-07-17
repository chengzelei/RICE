import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from environment import make_multi2single_env
model = PPO.load("/home/zck7060/Retrain_mujoco/hopper/masknet/weak_models/best_model/best_model")

env = DummyVecEnv([lambda: make_multi2single_env("Hopper-v3")])
env = VecMonitor(env, "weak_models/")
# eval_env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)
#reward_vec, length_vec = evaluate_policy(model, env,  n_eval_episodes=500, return_episode_rewards=True)


reward_vec = []
mask_vec = []

for i in range(500):
    reward = 0
    mask_num = 0
    count = 0
    obs = env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=True)
        if action == 0:
            mask_num += 1
        obs, rewards, dones, info = env.step(action)
        reward += rewards
        count += 1
        if dones:
            reward_vec.append(reward)
            break

    print("Test ", i)
    print("reward: ", reward)
    print("Mask ratio: ", mask_num/count)
    mask_vec.append(mask_num/count)

print("Average reward: ", np.mean(reward_vec))
print("Reward std: ", np.std(reward_vec))

print("Average mask ratio: ", np.mean(mask_vec))

#print("Average length: ", np.mean(length_vec))