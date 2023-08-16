import argparse
import signal
import sys
from pathlib import Path
from typing import Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import sys
sys.path.append("../baseline")
from blockchain_mdps import *
from reinforcement_learning import *
# noinspection PyUnusedLocal
from reinforcement_learning.base.training.callbacks.bva_callback import BVACallback
from tianshou.data import Batch
from tianshou.policy import DQNPolicy, PPOPolicy
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic

def solve_mdp_exactly(mdp: BlockchainModel) -> Tuple[float, BlockchainModel.Policy]:
    expected_horizon = int(1e4)
    solver = PTOSolver(mdp, expected_horizon=expected_horizon)
    p, r, _, _ = solver.calc_opt_policy(epsilon=1e-7, max_iter=int(1e10))
    sys.stdout.flush()
    return np.float32(solver.mdp.calc_policy_revenue(p)), p

def baseline_model():
    alpha = 0.35
    gamma = 0.5
    max_fork = 10
    fee = 10
    transaction_chance = 0.01
    simple_mdp = BitcoinModel(alpha=alpha, gamma=gamma, max_fork=max_fork)
    rev, _ = solve_mdp_exactly(simple_mdp)
    mdp = BitcoinFeeModel(alpha=alpha, gamma=gamma, max_fork=max_fork, fee=fee, transaction_chance=transaction_chance,
                          max_pool=max_fork)
    # mdp = BitcoinModel(alpha=alpha, gamma=gamma, max_fork=max_fork)
    smart_init = rev * (1 + fee * transaction_chance)
    # smart_init = None
    print(f'{mdp.state_space.size:,}')
    trainer = MCTSTrainer(mdp, orchestrator_type='synced_multi_process', build_info=None,
                          output_root=None, output_profile=False, output_memory_snapshots=False,
                          random_seed=0, expected_horizon=10_000, depth=5, batch_size=100, dropout=0,
                          length_factor=10, starting_epsilon=0.05, epsilon_step=0, bva_smart_init=smart_init,
                          prune_tree_rate=250, num_of_episodes_for_average=1000, learning_rate=2e-4,
                          nn_factor=0.0001, mc_simulations=25, num_of_epochs=5001, epoch_shuffles=2, save_rate=100,
                          use_base_approximation=True, ground_initial_state=False,
                          train_episode_length=100, evaluate_episode_length=100, lr_decay_epoch=1000,
                          number_of_training_agents=1, number_of_evaluation_agents=1,
                          lower_priority=False, bind_all=True, load_experiment='/home/jys3649/projects/xai_masknet/pto-selfish-mining/logs/BitcoinFeeModel(0.35, 0.5, 10, 10, 0.01, 10)_20221115-035251',
                          output_value_heatmap=False, normalize_target_values=True, use_cached_values=False)

    return trainer

def load_dqn_model():
    state_shape = 46
    action_shape = 44
    hidden_sizes = [128, 128, 128, 128]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        device=device,
        # dueling=(Q_param, V_param),
    ).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=5e-5)
    gamma = 0.99
    n_step = 2
    target_update_freq = 320
    policy = DQNPolicy(
        net,
        optim,
        gamma,
        n_step,
        target_update_freq=target_update_freq,
    )
    ckpt_path = '/home/zck7060/xrl4security/selfish_mining/baseline/model/checkpoint.pth'
    checkpoint = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(checkpoint["model"])
    # policy.load_state_dict(checkpoint)
    policy.set_eps(0.0)
    return policy

def load_mask_model(args):
    args.state_shape = 46
    args.action_shape = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        deterministic_eval=True,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv
    )
    ckpt_path = '/home/zck7060/xrl4security/selfish_mining/masknet/model/ckpt1.pth'
    checkpoint = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(checkpoint["model"])
    # policy.load_state_dict(checkpoint)
    return policy