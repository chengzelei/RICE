import os
import gym
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from PPO import PPO
from environment import make_retrain_env
from callback import CustomCallback
from monitor import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


os.environ['CUDA_VISIBLE_DEVICES'] = "1"

##################
# Hyper-parameters
##################
parser = argparse.ArgumentParser()
# game env
parser.add_argument("--env", type=str, default="Hopper-v3")
# number of game environments
parser.add_argument("--n_envs", type=int, default=20)
# log path
parser.add_argument("--log_path", type=str, default="weak_retrain_log/")
# target agent path
parser.add_argument("--agent_path", type=str, default="/home/zck7060/Retrain_mujoco/hopper/baseline/weak_tmp/best_model/best_model")
# masknet path
parser.add_argument("--masknet_path", type=str, default="/home/zck7060/Retrain_mujoco/hopper/masknet/weak_models/best_model/best_model")
# tensorboard path
parser.add_argument("--tensorboard_path", type=str, default="tensorboards/")
# bonus scheme
parser.add_argument("--bonus", type=str, default='e3b')
# bonus scale
parser.add_argument("--bonus_scale", type=float, default=1)
# probability of go-explore
parser.add_argument("--go_prob", type=float, default=0.5)
# whether sample based on explanation or not
parser.add_argument("--random_sampling", type=bool, default=False)
# training steps
parser.add_argument("--total_steps", type=int, default=1e6)
# evaluation frequency
parser.add_argument("--eval_freq", type=int, default=200)
# check frequency
parser.add_argument("--check_freq", type=int, default=100)
args = parser.parse_args()

env = DummyVecEnv([lambda: make_retrain_env(env_name=args.env, go_prob=args.go_prob, \
                                            agent_path=args.agent_path, masknet_path=args.masknet_path,\
                                            rand_sampling=args.random_sampling) for _ in range(args.n_envs)])
env = VecMonitor(env, args.log_path)

eval_env = gym.make(args.env)
eval_callback = EvalCallback(eval_env, best_model_save_path=args.log_path + 'best_model',
                             log_path=args.log_path +'results', eval_freq=args.eval_freq)
custom_callback = CustomCallback(check_freq=args.check_freq, log_dir=args.log_path)
callback = CallbackList([custom_callback, eval_callback])

model = PPO.load(path=args.agent_path, tensorboard_log=args.tensorboard_path,\
                  env=env, bonus=args.bonus, bonus_scale=args.bonus_scale)

model.learn(total_timesteps=args.total_steps, callback=callback)

