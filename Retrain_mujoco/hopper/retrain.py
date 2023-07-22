import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from PPO import PPO
from environment import make_retrain_env
from callback import CustomCallback
#from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from monitor import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

env = DummyVecEnv([lambda: make_retrain_env("Hopper-v3", rand_sampling=False) for _ in range(1)])
env = VecMonitor(env, "retrain_log/")

eval_env = gym.make("Hopper-v3")
eval_callback = EvalCallback(eval_env, best_model_save_path='retrain_models/best_model',
                             log_path='retrain_models/results', eval_freq=5)
custom_callback = CustomCallback(check_freq=5, log_dir="retrain_log/")
callback = CallbackList([custom_callback, eval_callback])
# Automatically normalize the input features
# env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)
log_dir = "retrain_models/"

custom_objects = { 'learning_rate': 1e-5}
model = PPO.load("/home/zck7060/Retrain_mujoco/hopper/baseline/tmp/best_model/best_model", env = env, custom_objects=custom_objects, tensorboard_log="weak_retrain_log/")

model.learn(total_timesteps=1e6, callback=callback)

