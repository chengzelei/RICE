import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from environment import make_retrain_env, MLP_Net
from callback import EvalCallback, CustomCallback
from monitor import VecMonitor
from stable_baselines3.common.callbacks import CallbackList

# Load RND_NET and INV_COV
inv_cov = np.load('inv_cov.npz')['inv_cov']
input_dim = 11 + 3
random_net = MLP_Net(input_dim, [500, 500])
RND_PATH = 'RND'
RND_checkpoint = torch.load(RND_PATH)
random_net.load_state_dict(RND_checkpoint)
bonus_scale = 0.1

env = DummyVecEnv([lambda: make_retrain_env("Hopper-v3", random_net, bonus_scale, inv_cov)])
env = VecMonitor(env, "tmp/")

eval_callback = EvalCallback(env, best_model_save_path='models/best_model',
                             log_path='models/results', eval_freq=2000)
custom_callback = CustomCallback(check_freq=2000, log_dir="tmp/")
callback = CallbackList([custom_callback, eval_callback])

# Automatically normalize the input features
# env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)
log_dir = "models/"

model = PPO("MlpPolicy", env, tensorboard_log="tmp/")

best_reward = 3559.44

i=1


model.learn(total_timesteps=1e6, callback=callback)