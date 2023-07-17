import gym

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3 import PPO
from callback import CustomCallback, EvalCallback

env = DummyVecEnv([lambda: gym.make("Hopper-v3")])
env = VecMonitor(env, "weak_retrain_log/")
# Automatically normalize the input features
# env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)


eval_callback = EvalCallback(env, best_model_save_path='weak_retrain_models/best_model',
                             log_path='weak_retrain_log/results', eval_freq=2000)
custom_callback = CustomCallback(check_freq=2000, log_dir="weak_retrain_log/")
callback = CallbackList([custom_callback, eval_callback])

custom_objects = { 'learning_rate': 1e-4, 'n_steps': 32}
model =  PPO.load("/home/zck7060/Retrain_mujoco/hopper/baseline/weak_tmp/best_model/best_model", env = env, custom_objects=custom_objects, tensorboard_log="weak_tmp/")
model.learn(total_timesteps=1e6, callback=callback)

# Don't forget to save the running average when saving the agent
log_dir = "weak_retrain_models/"
model.save(log_dir + "ppo_hopper")