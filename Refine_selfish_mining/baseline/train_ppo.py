import argparse
import os
import pprint
from torch.optim.lr_scheduler import LambdaLR
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
# from model import Net
from tianshou.utils.net.discrete import Actor, Critic
from reinforcement_learning import *
from reinforcement_learning.base.blockchain_simulator.mdp_blockchain_simulator_dqn import MDPBlockchainSimulatorDQN
from blockchain_mdps import *
import os
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
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--lr-decay", type=int, default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=100)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=10000)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--training-num', type=int, default=80)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.1)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=0)
    parser.add_argument("--resume-id", type=str, default=None)
    args = parser.parse_known_args()[0]
    return args

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


def train_ppo(args=get_args()):
    env = create_env(args)
    args.state_shape = env.state_space_dim
    args.action_shape = env.num_of_actions

    # train_envs = gym.make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: create_env(args) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: create_env(args) for _ in range(args.test_num)]
    )

    # model
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        # dueling=(Q_param, V_param),
    ).to(args.device)

    actor = Actor(net, args.action_shape, device=args.device, softmax_output=False)
    critic = Critic(net, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)


    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    def dist(p):
        return torch.distributions.Categorical(logits=p)
    
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    ).to(args.device)

    buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, args.task, 'dqn')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'best_policy.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        if epoch % 10 == 0:
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
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    )
    
    assert stop_fn(result['best_reward'])



if __name__ == '__main__':
    train_ppo(get_args())
