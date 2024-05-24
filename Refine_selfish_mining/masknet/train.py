import argparse
import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import AsyncCollector, VectorReplayBuffer, Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
# from model import Net
from tianshou.utils.net.discrete import Actor, Critic

import sys
sys.path.append("../baseline")
from reinforcement_learning import *
from blockchain_mdps import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import baseline_model
from reinforcement_learning.base.blockchain_simulator.mdp_blockchain_simulator_masknet import MDPBlockchainSimulatorMaskNet
policy = None
# trainer = baseline_model()
# policy = trainer.orchestrator.agent
# policy.approximator.eval()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='blockchainfee')
    parser.add_argument('--alpha', help='miner size', default=0.35, type=float)
    parser.add_argument('--gamma_mine', help='rushing factor', default=0.5, type=float)
    parser.add_argument('--max_fork', help='maximal fork size', default=10, type=int)
    parser.add_argument('--fee', help='transaction fee', default=10, type=float)
    parser.add_argument('--block-length', help='block length', default=100, type=int)
    parser.add_argument('--delta', help='chance for a transaction', default=0.01, type=float)
    parser.add_argument('--reward_threshold', type=float, default=100000)
    parser.add_argument('--seed', type=int, default=500)
    parser.add_argument('--buffer-size', type=int, default=2000000)
    parser.add_argument('--lr', type=float, default=5e-5)
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

def create_env(args):
    alpha = args.alpha
    gamma = args.gamma_mine
    max_fork = args.max_fork
    fee = args.fee
    transaction_chance = args.delta
    mdp = BitcoinFeeModel(alpha=alpha, gamma=gamma, max_fork=max_fork, fee=fee, transaction_chance=transaction_chance,max_pool=max_fork)
    # mdp = BitcoinModel(alpha=0.35, gamma=0.5, max_fork=10)
    simulator = MDPBlockchainSimulatorMaskNet(mdp, 100, policy=policy)
    return simulator


def test_ppo(args=get_args()):
    env = create_env(args)
    args.state_shape = env.state_space_dim
    args.action_shape = 2

    # train_envs = gym.make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: create_env(args) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: create_env(args) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)
    # model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    # net = Net().to(args.device)
    if torch.cuda.is_available():
        actor = DataParallelNet(
            Actor(net, args.action_shape, device=None).to(args.device)
        )
        critic = DataParallelNet(Critic(net, device=None).to(args.device))
    else:
        actor = Actor(net, args.action_shape, device=args.device).to(args.device)
        critic = Critic(net, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space,
        deterministic_eval=True,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv
    )
    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
    )
    test_collector = Collector(policy, test_envs)
    # test_collector = None
    # log
    log_path = os.path.join(args.logdir, args.task, 'ppo')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'best_policy.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            }, ckpt_path
        )
        return ckpt_path

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num * 5,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        save_checkpoint_fn=save_checkpoint_fn,
    )
    


if __name__ == '__main__':
    test_ppo()