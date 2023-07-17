import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

env = DummyVecEnv([lambda: gym.make("HalfCheetah-v3")])
env = VecMonitor(env, "models/")
env = VecNormalize.load("models/best_model/vec_normalize.pkl", env)
env.training = False
env.norm_reward = False

model = PPO.load("models/best_model/best_model", env=env)

reward_vec, length_vec = evaluate_policy(model, env,  n_eval_episodes=10, return_episode_rewards=True)


print("Average reward: ", np.mean(reward_vec))
print("Reward std: ", np.std(reward_vec))

print("Average length: ", np.mean(length_vec))