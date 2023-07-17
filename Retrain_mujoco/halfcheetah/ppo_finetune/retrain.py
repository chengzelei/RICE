import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from callback import CustomCallback, EvalCallback
from stable_baselines3.common.callbacks import CallbackList

env = DummyVecEnv([lambda: gym.make("HalfCheetah-v3")])
env = VecMonitor(env, "weak_retrain_log/")
env = VecNormalize.load("/home/zck7060/Retrain_mujoco/halfcheetah/baseline/weak_models/best_model/vec_normalize.pkl", env)
env.training = False
env.norm_reward = False

eval_callback = EvalCallback(env, best_model_save_path='weak_retrain_models/best_model',
                             log_path='weak_retrain_log/results', eval_freq=10000)
custom_callback = CustomCallback(check_freq=10000, log_dir="weak_retrain_log/")
callback = CallbackList([custom_callback, eval_callback])

custom_objects = { 'learning_rate': 1e-4}
model = PPO.load("/home/zck7060/Retrain_mujoco/halfcheetah/baseline/weak_models/best_model/best_model", env=env, custom_objects=custom_objects, tensorboard_log="weak_tmp/")
model.learn(total_timesteps=1e7, callback=callback)