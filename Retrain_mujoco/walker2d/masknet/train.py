import gym

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from cus_PPO import PPO
from environment import make_multi2single_env

env = make_multi2single_env("Walker2d-v3")

checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='weak_models/',
                                         name_prefix='walker')

eval_callback = EvalCallback(env, best_model_save_path='weak_models/best_model',
                             log_path='weak_models/results', eval_freq=2000)
callback = CallbackList([checkpoint_callback, eval_callback])

model = PPO("MlpPolicy", env, tensorboard_log="weak_tmp/")
model.learn(total_timesteps=1e5, callback=callback)

# Don't forget to save the running average when saving the agent
log_dir = "weak_tmp/"
model.save(log_dir + "mask_walker")