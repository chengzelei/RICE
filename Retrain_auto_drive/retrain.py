import metadrive
import gym
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter
import torch
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import PPOPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator
from retrain_env import CustomWrapper
from rnd_reward_model import RndRewardModel
from learner import BaseLearner
from core.envs import DriveEnvWrapper
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--go_prob', type=float, default=0.5)
parser.add_argument('--bonus_scale', type=float, default=0.01)
args = parser.parse_known_args()[0]

exp_name = 'go_prob_' + str(args.go_prob) + '_bonus_scale_' + str(args.bonus_scale)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

metadrive_macro_config = dict(
    exp_name=exp_name,
    env=dict(
        metadrive=dict(use_render=False, ),
        manager=dict(
            shared_memory=False,
            max_retry=50,
            retry_type='renew',
            context='spawn',
        ),
        n_evaluator_episode=5,
        stop_value=99999,
        # collector_env_num=20,
        collector_env_num=4,
        evaluator_env_num=1,
        wrapper=dict(),
    ),
    policy=dict(
        cuda=True,
        action_space='discrete',
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=5,
            action_space='discrete',
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
        ),
        collect=dict(
            # n_sample=300,
            n_sample=1000,
        ),
    ),
)

rnd_config = dict(
        type='rnd',
        intrinsic_reward_type='add',
        learning_rate=1e-3,
        batch_size=64,
        hidden_size_list=[64, 64, 128],
        update_per_collect=100,
        #obs_shape=[200, 200, 5],
        obs_shape=[5,200,200],
        obs_norm=True,
        obs_norm_clamp_min=-1,
        obs_norm_clamp_max=1,
        intrinsic_reward_weight=None,
        # means the relative weight of RND intrinsic_reward.
        # If intrinsic_reward_weight=None, we will automatically set it based on
        # the absolute value of the difference between max and min extrinsic reward in the sampled mini-batch
        # please refer to  estimate() method for details.
        intrinsic_reward_rescale=args.bonus_scale
        # means the rescale value of RND intrinsic_reward only used when intrinsic_reward_weight is None
    )

# rnd_config = dict(
#         # (str) Reward model register name, refer to registry ``REWARD_MODEL_REGISTRY``.
#         type='rnd',
#         # (str) The intrinsic reward type, including add, new, or assign.
#         intrinsic_reward_type='add',
#         # (float) The step size of gradient descent.
#         learning_rate=1e-3,
#         # (float) Batch size.
#         batch_size=64,
#         # (list(int)) Sequence of ``hidden_size`` of reward network.
#         # If obs.shape == 1,  use MLP layers.
#         # If obs.shape == 3,  use conv layer and final dense layer.
#         hidden_size_list=[64, 64, 128],
#         # (int) How many updates(iterations) to train after collector's one collection.
#         # Bigger "update_per_collect" means bigger off-policy.
#         # collect data -> update policy-> collect data -> ...
#         update_per_collect=100,
#         obs_shape=[200, 200, 5],
#         # (bool) Observation normalization: transform obs to mean 0, std 1.
#         obs_norm=True,
#         # (int) Min clip value for observation normalization.
#         obs_norm_clamp_min=-1,
#         # (int) Max clip value for observation normalization.
#         obs_norm_clamp_max=1,
#         # Means the relative weight of RND intrinsic_reward.
#         # (float) The weight of intrinsic reward
#         # r = intrinsic_reward_weight * r_i + r_e.
#         intrinsic_reward_weight=0.01,
#         # (bool) Whether to normlize extrinsic reward.
#         # Normalize the reward to [0, extrinsic_reward_norm_max].
#         extrinsic_reward_norm=False,
#         # (int) The upper bound of the reward normalization.
#         extrinsic_reward_norm_max=1,
#     )

main_config = EasyDict(metadrive_macro_config)
rnd_config = EasyDict(rnd_config)


def custom_env(env_cfg, wrapper_cfg=None):
    return CustomWrapper(gym.make("Macro-v1", config=env_cfg), wrapper_cfg, go_prob = args.go_prob)

def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(gym.make("Macro-v1", config=env_cfg), wrapper_cfg)


def main(cfg):
    cfg = compile_config(
        cfg, SyncSubprocessEnvManager, PPOPolicy, BaseLearner, SampleSerialCollector, InteractionSerialEvaluator
    )

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(custom_env, cfg.env.metadrive) for _ in range(collector_env_num)],
        cfg=cfg.env.manager,
    )
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )
    device = "cuda"
    tb_path = os.path.join('./rnd_log/{}/'.format(cfg.exp_name), str(args.seed))
    tb_logger = SummaryWriter(tb_path)
    rnd_policy = RndRewardModel(rnd_config, device, tb_logger=SummaryWriter(tb_path))

    policy = PPOPolicy(cfg.policy)
    state_dict = torch.load('/home/zck7060/DI-drive/baseline/metadrive_metaenv_ppo/ckpt/ckpt_best.pth.tar', map_location='cpu')
    policy._model.load_state_dict(state_dict['model'], strict=True)

    tb_path = os.path.join('./retrain_log/{}/'.format(cfg.exp_name), str(args.seed))
    tb_logger = SummaryWriter(tb_path)
    learner = BaseLearner(cfg.policy.learn.learner, rnd_policy, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    learner.call_hook('before_run')

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, 1)
            if stop:
                break
        # Sampling data from environments
        new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
        learner.train(new_data, collector.envstep)
    learner.call_hook('after_run')

    collector.close()
    evaluator.close()
    learner.close()


if __name__ == '__main__':
    main(main_config)
