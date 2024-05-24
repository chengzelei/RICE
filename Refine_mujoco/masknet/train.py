import gym
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from cus_PPO import PPO
from environment import make_multi2single_env
from stable_baselines3.ppo import MlpPolicy

alpha = 0.01

env = DummyVecEnv([lambda: make_multi2single_env("Hopper-v3", train=True, alpha=alpha)])
env = VecMonitor(env, "tmp/")
# Automatically normalize the input features
# env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='models/',
                                         name_prefix='hopper')
eval_env = DummyVecEnv([lambda: make_multi2single_env("Hopper-v3", train=False)])
eval_env = VecMonitor(env, "models/")
# eval_env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)
eval_callback = EvalCallback(eval_env, best_model_save_path='models/best_model',
                             log_path='models/results', eval_freq=1000)
callback = CallbackList([checkpoint_callback, eval_callback])

model = PPO(MlpPolicy, env, tensorboard_log="tmp/")
model.learn(total_timesteps=3e5, callback=callback)

# Don't forget to save the running average when saving the agent
log_dir = "tmp/"
model.save(log_dir + "mask_hopper")
