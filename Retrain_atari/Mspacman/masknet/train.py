import gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor, VecFrameStack
from ppo import PPO

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = make_atari_env("MsPacman-v0", n_envs=32)



eval_callback = EvalCallback(vec_env, best_model_save_path='weak_tmp/best_model',
                             log_path='weak_tmp/results', eval_freq=2000)

model = PPO("CnnPolicy", 
            vec_env,             
            batch_size = 256,
            clip_range = 0.1,
            ent_coef = 0.01,
            gae_lambda = 0.9,
            gamma = 0.99,
            learning_rate = 2.5e-4,
            max_grad_norm = 0.5,
            n_epochs = 4,
            n_steps = 128,
            vf_coef = 0.5,
            tensorboard_log="weak_tmp/",
            verbose=1,)
model.learn(total_timesteps=5e5, callback=eval_callback)
model.save("weak_masknet-mspacman.zip")

