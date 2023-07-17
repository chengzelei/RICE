import gym

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3 import PPO
from callback import CustomCallback

env = DummyVecEnv([lambda: gym.make("Reacher-v2")])
env = VecMonitor(env, "weak_tmp/")
# Automatically normalize the input features
# env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)


eval_callback = EvalCallback(env, best_model_save_path='weak_tmp/best_model',
                             log_path='weak_tmp/results', eval_freq=1000)
custom_callback = CustomCallback(check_freq=1000, log_dir="weak_tmp/")
callback = CallbackList([custom_callback, eval_callback])

model = PPO("MlpPolicy", env, tensorboard_log="weak_tmp/")
model.learn(total_timesteps=1e5, callback=callback)

# Don't forget to save the running average when saving the agent
log_dir = "weak_tmp/"
model.save(log_dir + "ppo_reacher")

