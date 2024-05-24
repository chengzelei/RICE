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
from tianshou.data import Batch
from sample_mask_score import get_args, create_env
from utils import load_dqn_model, load_mask_model

import sys
sys.path.append("../baseline")
from reinforcement_learning import *
from blockchain_mdps import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_traj(env,num,path):
    dqn_model = load_dqn_model()
    reward_log = []
    action_log = []
    value_highlihgts_log = []
    value_trust_log = []
    value_max_log = []
    Rewards = 0
    for i in range(num):
        value_highlights_record = []
        value_trust_record = []
        value_max_record = []
        action_record = []
        set_seed(i)
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            batch = Batch(obs=obs.reshape((1,-1)), info=None)
            output_info = dqn_model(batch)
            action = output_info.act[0]

            values = output_info.logits[0].detach().cpu().numpy()   #the raw output of DQN is the value of each action

            value_trust_record.append(np.max(values) - np.mean(values))
            value_highlights_record.append(np.max(values) - np.min(values))
            value_max_record.append(np.max(values))
            action_record.append(action)

            obs, reward, done, _ = env.step(action)
            total_reward += reward
        print("Episode {}, Reward {}".format(i, total_reward))
        Rewards += total_reward
        reward_log.append(total_reward)
        value_highlihgts_log.append(value_highlights_record)
        value_trust_log.append(value_trust_record)
        value_max_log.append(value_max_record)
        action_log.append(action_record)
    print("Average Reward {}".format(Rewards/num))
    reward_log = np.array(reward_log)
    value_highlihgts_log = np.array(value_highlihgts_log)
    value_trust_log = np.array(value_trust_log)
    action_log = np.array(action_log, dtype=np.int32)
    np.savetxt(path + "reward_log.txt", reward_log)
    np.savetxt(path + "highlights_prob_log.txt", value_highlihgts_log)
    np.savetxt(path + "trust_prob_log.txt", value_trust_log)
    np.savetxt(path + "max_prob_log.txt", value_max_log)
    np.savetxt(path + "action_log.txt", action_log)

'''
used for comparing the performance of the baseline model and the model trained by tianshou
'''

if __name__ == '__main__':
    args = get_args()
    env = create_env(args)
    path = 'recordings/'
    sample_traj(env,500,path)