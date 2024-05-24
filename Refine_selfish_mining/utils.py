import torch
from tianshou.policy import DQNPolicy, PPOPolicy
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic

class Traj:
    def __init__(self):
        self.eps_len = 0
        self.state_seq = []
        self.reward_seq = []
        self.mask_probs = []

    
    def set(self, eps_len, state_seq, reward_seq, mask_probs):
        self.eps_len = eps_len
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        self.mask_probs = mask_probs

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