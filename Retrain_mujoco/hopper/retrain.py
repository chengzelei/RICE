import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from environment import make_retrain_env, MLP_Net
from callback import EvalCallback, CustomCallback
from monitor import VecMonitor
from stable_baselines3.common.callbacks import CallbackList
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# Load RND_NET and INV_COV
cov = np.load('inv_cov.npz')['cov']
input_dim = 11 + 3
random_net = MLP_Net(input_dim, [500, 500])
RND_PATH = 'RND'
RND_checkpoint = torch.load(RND_PATH)
random_net.load_state_dict(RND_checkpoint)
bonus_scale = 1e-5

env = DummyVecEnv([lambda: make_retrain_env("Hopper-v3", random_net, bonus_scale, cov)])
env = VecMonitor(env, "weak_retrain_log/")

eval_callback = EvalCallback(env, best_model_save_path='weak_retrain_models/best_model',
                             log_path='weal_retrain_models/results', eval_freq=2000)
custom_callback = CustomCallback(check_freq=2000, log_dir="weak_retrain_log/")
callback = CallbackList([custom_callback, eval_callback])

# Automatically normalize the input features
# env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)
log_dir = "weak_retrain_models/"

custom_objects = { 'learning_rate': 1e-4}
model = PPO.load("/home/zck7060/Retrain_mujoco/hopper/baseline/weak_tmp/best_model/best_model", env = env, custom_objects=custom_objects, tensorboard_log="weak_retrain_log/")
#model = PPO("MlpPolicy", env, tensorboard_log="retrain_log/")

model.learn(total_timesteps=1e6, callback=callback)

