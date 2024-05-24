import os
from tianshou.data import Batch
from sample_mask_score import get_args, create_env
from utils import load_mask_model, load_dqn_model
from sample_mask_score import set_seed
import numpy as np
import argparse
import torch

import sys
sys.path.append("../baseline")
from reinforcement_learning import *
from blockchain_mdps import *

def select_critical(score,width=0.1):
    frame = score.shape[0]
    sub_length = int(frame * width)
    best_score = 0
    best_score_index = 0
    for i in range(frame):
        if i + sub_length > frame:
            break
        sub_score = score[i:i+sub_length]
        if np.sum(sub_score) > best_score:
            best_score = np.sum(sub_score)
            best_score_index = i
    return best_score_index,sub_length


def replay(env, num, path, method, width=0.1, test_num=1):
    dqn_model = load_dqn_model()
    reward_log = []
    for i in range(num):
        
        action_set = np.loadtxt(path + "action_log.txt", dtype=np.int32)[i]
        if method == "random":
            sub_length = int(action_set.shape[0] * width)
            best_score_index = np.random.randint(0,action_set.shape[0]-sub_length)
        else:
            mask_prob_set = np.loadtxt(path + str(method) + "_prob_log.txt")[i]
            best_score_index,sub_length = select_critical(mask_prob_set,width)
        episode_reward = 0
        for j in range(test_num):
            set_seed(i)
            obs = env.reset()
            done = False
            total_reward = 0
            step = 0
            while not done:
                if step < best_score_index:
                    action = action_set[step]
                elif step < best_score_index + sub_length:
                    set_seed(2021+i+j)
                    legal_action = env.get_state_legal_actions_tensor(env._current_state)
                    action = np.random.choice(np.where(legal_action == 1)[0])
                else:
                    batch = Batch(obs=obs.reshape((1,-1)), info=None)
                    action = dqn_model(batch).act[0]
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                step+=1
            print("Episode {}, Reward {}".format(i, total_reward))
            episode_reward += total_reward
        reward_log.append(episode_reward/test_num)
    reward_log = np.array(reward_log)
    np.savetxt(path + str(method) + "_replay_reward_log_" + str(int(width * 100)) + ".txt" , reward_log)



if __name__ == '__main__':
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
    parser.add_argument('--method', type=str, default="mask")
    args = parser.parse_known_args()[0]
    # methods = ["mask"]
    widths = [0.02,0.04,0.06,0.08]
    env = create_env(args)
    test_num = 5
    path = 'recordings/'
    for width in widths:
        replay(env, 500, path, args.method, width, test_num)
    print("Done")