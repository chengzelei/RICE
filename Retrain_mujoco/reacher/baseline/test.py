import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
model = PPO.load("models/best_model/best_model")

env = DummyVecEnv([lambda: gym.make("Hopper-v3")])
env = VecMonitor(env, "models/")
# eval_env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)
# reward_vec, length_vec = evaluate_policy(model, env,  n_eval_episodes=500, return_episode_rewards=True)


reward_vec = []
for i in range(1):
    reward = 0
    obs = env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        reward += rewards
        print(rewards)
        if dones:
            if "episode" in info.keys():
                reward_vec.append(info["episode"]["r"])
            else:
                reward_vec.append(reward)
            break

    print("Test ", i)
    print("reward: ", reward)

# print("Average reward: ", np.mean(reward_vec))
# print("Reward std: ", np.std(reward_vec))

# print("Average length: ", np.mean(length_vec))
    
