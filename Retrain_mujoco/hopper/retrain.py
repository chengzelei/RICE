import os
import gym
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from PPO import PPO
from environment import make_retrain_env
from callback import CustomCallback
from monitor import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
# target agent path
parser.add_argument("--agent_path", type=str, default="/home/zck7060/Retrain_mujoco/hopper/baseline/weak_tmp/best_model/best_model")
# masknet path
parser.add_argument("--masknet_path", type=str, default="/home/zck7060/Retrain_mujoco/hopper/masknet/weak_models/best_model/best_model")
# tensorboard path
parser.add_argument("--tensorboard_path", type=str, default="tensorboards/")
# bonus scheme
parser.add_argument("--bonus", type=str, default='rnd')
# bonus scale
parser.add_argument("--bonus_scale", type=float, default=1e-2)
# probability of go-explore
parser.add_argument("--go_prob", type=float, default=0.5)
# whether sample based on explanation or not
parser.add_argument("--random_sampling", type=bool, default=False)
# training steps
parser.add_argument("--total_steps", type=int, default=5e4)
# evaluation frequency
parser.add_argument("--eval_freq", type=int, default=100)
# check frequency
parser.add_argument("--check_freq", type=int, default=100)
args = parser.parse_args()

log_path = args.log_path + args.env + '/' + 'sub_optimal' + '/' + args.bonus + '/' + 'bonus_scale_' + str(args.bonus_scale) + '/' + 'p_' + str(args.go_prob) + '/seed' + str(args.seed) + '/'
tensorboard_path = args.tensorboard_path + args.env + '/' + 'sub_optimal' + '/' + args.bonus + '/' + 'bonus_scale_' + str(args.bonus_scale) + '/' + 'p_' + str(args.go_prob) + '/seed_' + str(args.seed) + '/'

if not os.path.exists(log_path):
    os.makedirs(log_path)

if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)


env = DummyVecEnv([lambda: make_retrain_env(env_name=args.env, go_prob=args.go_prob, \
                                            agent_path=args.agent_path, masknet_path=args.masknet_path,\
                                            rand_sampling=args.random_sampling) for _ in range(args.n_envs)])
env = VecMonitor(env, log_path)

eval_env = gym.make(args.env)
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

