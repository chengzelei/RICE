import argparse
import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
# from model import Net
from tianshou.utils.net.discrete import Actor, Critic
import sys
sys.path.append("../baseline")
from reinforcement_learning import *
from blockchain_mdps import *
from tianshou.data import Batch
from reinforcement_learning.base.blockchain_simulator.mdp_blockchain_simulator_dqn import MDPBlockchainSimulatorDQN
from utils import load_dqn_model, load_mask_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='blockchainfee')
    parser.add_argument('--alpha', help='miner size', default=0.35, type=float)
    parser.add_argument('--gamma_mine', help='rushing factor', default=0.5, type=float)
    parser.add_argument('--max_fork', help='maximal fork size', default=10, type=int)
    parser.add_argument('--fee', help='transaction fee', default=100, type=float)
    parser.add_argument('--block-length', help='block length', default=100, type=int)
    parser.add_argument('--delta', help='chance for a transaction', default=0.01, type=float)
    parser.add_argument('--reward_threshold', type=float, default=100000)
    parser.add_argument('--seed', type=int, default=500)
    parser.add_argument('--buffer-size', type=int, default=2000000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=10000)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--training-num', type=int, default=50)
    parser.add_argument('--test-num', type=int, default=20)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    args = parser.parse_known_args()[0]
    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_env(args):
    alpha = args.alpha
    gamma = args.gamma_mine
    max_fork = args.max_fork
    fee = args.fee
    transaction_chance = args.delta
    mdp = BitcoinFeeModel(alpha=alpha, gamma=gamma, max_fork=max_fork, fee=fee, transaction_chance=transaction_chance,max_pool=max_fork)
    # mdp = BitcoinModel(alpha=0.35, gamma=0.5, max_fork=10)
    simulator = MDPBlockchainSimulatorDQN(mdp, args.block_length)
    return simulator

def sample_traj(env,num,args,path):
    dqn_model = load_dqn_model()
    mask_model = load_mask_model(args)
    reward_log = []
    action_log = []
    mask_prob_log = []
    Rewards = 0
    for i in range(num):
        mask_record = []
        action_record = []
        set_seed(i)
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            batch = Batch(obs=obs.reshape((1,-1)), info=None)
            action = dqn_model(batch).act[0]

            mask_prob = mask_model(batch).logits[0][0] #the prob of not doing the mask-> state importance
            mask_record.append(mask_prob.detach().cpu())
            action_record.append(action)

            obs, reward, done, _ = env.step(action)
            total_reward += reward
        print("Episode {}, Reward {}".format(i, total_reward))
        Rewards += total_reward
        reward_log.append(total_reward)
        mask_prob_log.append(mask_record)
        action_log.append(action_record)
    print("Average Reward {}".format(Rewards/num))
    reward_log = np.array(reward_log)
    mask_prob_log = np.array(mask_prob_log)
    action_log = np.array(action_log, dtype=np.int32)
    np.savetxt(path + "reward_log.txt", reward_log)
    np.savetxt(path + "mask_prob_log.txt", mask_prob_log)
    np.savetxt(path + "action_log.txt", action_log)

'''
used for comparing the performance of the baseline model and the model trained by tianshou
'''

if __name__ == '__main__':
    args = get_args()
    env = create_env(args)
    path = 'recordings/'
    sample_traj(env,500,args,path)