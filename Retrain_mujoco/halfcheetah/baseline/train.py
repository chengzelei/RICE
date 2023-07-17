import gym
import os
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3 import PPO
from callback import EvalCallback

env = DummyVecEnv([lambda: gym.make("HalfCheetah-v3")])
env = VecMonitor(env, "weak_tmp/")
# Automatically normalize the input features
env = VecNormalize(env, norm_obs=True, norm_reward=False,
                   clip_obs=10.)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='weak_models/',
                                         name_prefix='halfcheetah')

eval_callback = EvalCallback(env, best_model_save_path='weak_models/best_model',
                             log_path='weak_models/results', eval_freq=1000)
callback = CallbackList([checkpoint_callback, eval_callback])

model = PPO("MlpPolicy", env, tensorboard_log="weak_tmp/")
model.learn(total_timesteps=1e5, callback=callback)
#near-optimal model train 1e7

# Don't forget to save the running average when saving the agent
log_dir = "weak_tmp/"
model.save(log_dir + "ppo_halfcheetah")

