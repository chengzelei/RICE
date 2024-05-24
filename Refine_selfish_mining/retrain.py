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
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
# from model import Net
from tianshou.utils.net.discrete import Actor, Critic
from rnd_model import RNDModel
import sys
sys.path.append("baseline/")
from reinforcement_learning import *
from blockchain_mdps import *
from baseline.reinforcement_learning.base.blockchain_simulator.mdp_blockchain_simulator_dqn import MDPBlockchainSimulatorDQN
from baseline.reinforcement_learning.base.blockchain_simulator.mdp_blockchain_simulator_retrain import MDPBlockchainSimulatorRetrain
import os
from rnd_policy import RNDPolicy


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='blockchainfee')
    parser.add_argument('--alpha_mine', help='miner size', default=0.35, type=float)
    parser.add_argument('--gamma_mine', help='rushing factor', default=0.5, type=float)
    parser.add_argument('--max_fork', help='maximal fork size', default=10, type=int)
    parser.add_argument('--fee', help='transaction fee', default=10, type=float)
    parser.add_argument('--block-length', help='block length', default=100, type=int)
    parser.add_argument('--delta', help='chance for a transaction', default=0.01, type=float)
    parser.add_argument('--reward_threshold', type=float, default=1000000)
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.0)
    parser.add_argument('--eps-train', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=2)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=10000)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--training-num', type=int, default=50)
    parser.add_argument('--test-num', type=int, default=20)
    parser.add_argument('--logdir', type=str, default='retrain_model')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--prioritized-replay', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)

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

    parser.add_argument('--rnd-lr', type=float, default=1e-5)
    parser.add_argument('--rnd-bonus-scale', type=float, default=1e-2)
    parser.add_argument('--rnd-loss-weight', type=float, default=1)
    parser.add_argument('--go-prob', type=float, default=0.5)
    parser.add_argument('--wo_exp', default=False)
    args = parser.parse_known_args()[0]
    return args

def create_retrain_env(args):
    alpha = args.alpha_mine
    gamma = args.gamma_mine
    max_fork = args.max_fork
    fee = args.fee
    transaction_chance = args.delta
    mdp = BitcoinFeeModel(alpha=alpha, gamma=gamma, max_fork=max_fork, fee=fee, transaction_chance=transaction_chance,max_pool=max_fork)
    # mdp = BitcoinModel(alpha=0.35, gamma=0.5, max_fork=10)
    simulator = MDPBlockchainSimulatorRetrain(args=args, blockchain_model=mdp, expected_horizon=args.block_length)
    return simulator

def create_env(args):
    alpha = args.alpha_mine
    gamma = args.gamma_mine
    max_fork = args.max_fork
    fee = args.fee
    transaction_chance = args.delta
    mdp = BitcoinFeeModel(alpha=alpha, gamma=gamma, max_fork=max_fork, fee=fee, transaction_chance=transaction_chance,max_pool=max_fork)
    # mdp = BitcoinModel(alpha=0.35, gamma=0.5, max_fork=10)
    simulator = MDPBlockchainSimulatorDQN(mdp, args.block_length)
    return simulator


def train_dqn(args=get_args()):
    env = create_env(args)
    args.state_shape = env.state_space_dim
    args.action_shape = env.num_of_actions

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)
    # model
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        # dueling=(Q_param, V_param),
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
    )
    ckpt_path = '/home/zck7060/xrl4security/selfish_mining/baseline/model/checkpoint.pth'
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    policy.load_state_dict(checkpoint["model"])

    rnd_model = RNDModel(
        args.state_shape,
        64,
        device=args.device
    ).to(args.device)
    rnd_optim = torch.optim.Adam(rnd_model.parameters(), lr=args.rnd_lr)
    policy = RNDPolicy(
            policy, rnd_model, rnd_optim, args.rnd_bonus_scale, args.rnd_loss_weight
        ).to(args.device)

    # train_envs = gym.make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: create_retrain_env(args) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: create_env(args) for _ in range(args.test_num)]
    )


    # buffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    param_str = 'go_prob_' + str(args.go_prob) + '_bonus_scale_' + str(args.rnd_bonus_scale)
    if args.wo_exp:
        param_str += '_no_exp'
    else:
        param_str += '_exp'
    log_path = os.path.join(args.logdir, args.task, param_str, str(args.seed))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(                
            {
                    "model": policy.state_dict(),
                    "optim": optim.state_dict(),
                }, os.path.join(log_path, 'best_policy.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        if epoch % 5 == 0:
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optim": optim.state_dict(),
                }, ckpt_path
            )
        return ckpt_path

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        save_checkpoint_fn=save_checkpoint_fn,
    )
    
    assert stop_fn(result['best_reward'])



if __name__ == '__main__':
    train_dqn(get_args())
