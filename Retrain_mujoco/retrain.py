import os
import gym
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from PPO import PPO
from environment import make_retrain_env
from callback import CustomCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

##################
# Hyper-parameters
##################
parser = argparse.ArgumentParser()
# game env
parser.add_argument("--env", type=str, default="Hopper-v3")
# number of game environments
parser.add_argument("--n_envs", type=int, default=20)
# seed
parser.add_argument("--seed", type=int, default=0)
# log path
parser.add_argument("--log_path", type=str, default="retrain_log/")
# optimal or not
parser.add_argument("--optimal", type=str, default="sub_optimal")
# target agent path
parser.add_argument("--agent_path", type=str, default="/home/zck7060/Retrain_mujoco/hopper/baseline/weak_tmp/best_model/best_model")
# masknet path
parser.add_argument("--masknet_path", type=str, default="/home/zck7060/Retrain_mujoco/hopper/masknet/weak_models/best_model/best_model")
# vector normalization path
parser.add_argument("--vec_norm_path")
# tensorboard path
parser.add_argument("--tensorboard_path", type=str, default="tensorboards/")
# bonus scheme
parser.add_argument("--bonus", type=str, default='None')
# bonus scale
parser.add_argument("--bonus_scale", type=float, default=1e-2)
# probability of go-explore
parser.add_argument("--go_prob", type=float, default=0.5)
# whether sample based on explanation or not
parser.add_argument("--random_sampling", type=bool, default=False)
# training steps
parser.add_argument("--total_steps", type=int, default=1e6)
# evaluation frequency
parser.add_argument("--eval_freq", type=int, default=100)
# check frequency
parser.add_argument("--check_freq", type=int, default=100)
args = parser.parse_args()

log_path = args.log_path + args.env + '/' + args.optimal + '/' + args.bonus + '/' + 'bonus_scale_' + str(args.bonus_scale) + '/' + 'p_' + str(args.go_prob) + '/seed_' + str(args.seed) + '/'
tensorboard_path = args.tensorboard_path + args.env + '/' + args.optimal + '/' + args.bonus + '/' + 'bonus_scale_' + str(args.bonus_scale) + '/' + 'p_' + str(args.go_prob) + '/seed_' + str(args.seed) + '/'

if not os.path.exists(log_path):
    os.makedirs(log_path)

if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)



norm_games = ["HalfCheetah-v3", "Walker2d-v3"]
if args.env not in norm_games:
    env = DummyVecEnv([lambda: make_retrain_env(env_name=args.env, go_prob=args.go_prob, \
                                                agent_path=args.agent_path, masknet_path=args.masknet_path,\
                                                rand_sampling=args.random_sampling) for _ in range(args.n_envs)])
    env = VecMonitor(env, log_path)
    eval_env = DummyVecEnv([lambda:gym.make(args.env)])
    eval_env = VecMonitor(eval_env, log_path)
else:
    env = DummyVecEnv([lambda: make_retrain_env(env_name=args.env, go_prob=args.go_prob, \
                                                agent_path=args.agent_path, masknet_path=args.masknet_path,\
                                                rand_sampling=args.random_sampling, vec_norm_path=args.vec_norm_path) for _ in range(args.n_envs)])    
    env = VecMonitor(env, log_path)
    env = VecNormalize.load(args.vec_norm_path, env)
    eval_env = DummyVecEnv([lambda:gym.make(args.env)])
    eval_env = VecNormalize.load(args.vec_norm_path, eval_env)
    eval_env = VecMonitor(eval_env, log_path)

eval_callback = EvalCallback(eval_env, best_model_save_path=log_path + 'best_model',
                             log_path=log_path +'results', eval_freq=args.eval_freq)
custom_callback = CustomCallback(check_freq=args.check_freq, log_dir=log_path)
callback = CallbackList([custom_callback, eval_callback])

#custom_objects = { 'learning_rate': 1e-5}
model = PPO.load(path=args.agent_path, tensorboard_log=tensorboard_path,\
#                 custom_objects=custom_objects, 
                 env=env, bonus=args.bonus,\
                 bonus_scale=args.bonus_scale, seed=args.seed)

model.learn(total_timesteps=args.total_steps, callback=callback)

