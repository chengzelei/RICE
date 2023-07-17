from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from callback import CustomCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = make_atari_env("MsPacman-v0", n_envs=32, monitor_dir='weak_tmp/')

# Frame-stacking with 4 frames
#vec_env = VecFrameStack(vec_env, n_stack=4)

eval_callback = EvalCallback(vec_env, best_model_save_path='weak_tmp/best_model',
                             log_path='weak_tmp/results', eval_freq=1000)
custom_callback = CustomCallback(check_freq=1000, log_dir="weak_tmp/")
callback = CallbackList([custom_callback, eval_callback])
custom_objects = { 'learning_rate': 1e-4, 'n_steps': 32}

model = PPO.load('/home/zck7060/Retrain_atari/mspacman/baseline/weak_mspacman.zip', env = vec_env, custom_objects=custom_objects, tensorboard_log="weak_tmp/") 
model.learn(total_timesteps=1e7, callback=callback)
model.save("weak_mspacman.zip")