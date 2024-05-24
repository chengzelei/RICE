import metadrive
import gym
import torch
import numpy as np
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import PPOPolicy, DQNPolicy
from ding.utils import get_world_size, get_rank
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from core.envs import DriveEnvWrapper, MetaDriveMacroEnv

from typing import Optional, Callable, Tuple
from ding.worker.collector.base_serial_evaluator import ISerialEvaluator, VectorEvalMonitor
from ding.torch_utils.data_helper import to_ndarray, to_tensor
from ding.utils.data import default_collate
from ding.torch_utils import to_device


metadrive_macro_config = dict(
    # exp_name='metadrive_metaenv_masknet',
    env=dict(
        metadrive=dict(use_render=False, ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=1,
        stop_value=99999,
        collector_env_num=4,
        evaluator_env_num=1,
        wrapper=dict(),
    ),
    policy=dict(
        cuda=True,
        #continuous=False,
        action_space='discrete',
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=2,
            #continuous=False,
            action_space='discrete',
            encoder_hidden_size_list=[128, 128, 64],
        ),
    ),
)

metadrive_macro_agent_config = dict(
    # exp_name='metadrive_metaenv_agent',
    env=dict(
        metadrive=dict(use_render=False, ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=1,
        stop_value=99999,
        collector_env_num=20,
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
            n_sample=300,
        ),
    ),
)

class Traj:
    def __init__(self):
        self.eps_len = 0
        self.act_seq = []
        self.reward_seq = []
        self.mask_probs = []

    
    def set(self, eps_len, act_seq, reward_seq, mask_probs):
        self.eps_len = eps_len
        self.act_seq = act_seq
        self.reward_seq = reward_seq
        self.mask_probs = mask_probs


main_config = EasyDict(metadrive_macro_config)
agent_config = EasyDict(metadrive_macro_agent_config)

def fid_play(obs, env, masknet, agent, seed):
    traj = Traj()
    reward_record = []
    importance_record = []
    action_record = []
    # env.seed(seed)
    # obs = env.reset()
    epr, step = 0, 0

    while True:
        obs_copy = obs.transpose((2, 0, 1))
        obs_dict = {i: obs_copy for i in range(1)}
        data = default_collate(list(obs_dict.values()))
        data = torch.tensor(list(obs_dict.values()), dtype=torch.float32)
        data = to_device(data, masknet._device)

        with torch.no_grad():
            masknet_output = masknet._model.forward(data, mode='compute_actor_critic')
            agent_output = agent._model.forward(data, mode='compute_actor_critic')
            agent_logit = agent_output['logit'][0]
            assert isinstance(agent_logit, torch.Tensor) or isinstance(agent_logit, list)
            if isinstance(agent_logit, torch.Tensor):
                agent_logit = [agent_logit]
            if 'action_mask' in agent_output:
                mask = agent_output['action_mask']
                if isinstance(mask, torch.Tensor):
                    mask = [mask]
                agent_logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(agent_logit, mask)]

            agent_action = [l.argmax(dim=-1) for l in agent_logit]
            if len(agent_action) == 1:
                agent_action = agent_action[0]
            

        mask_logit = masknet_output['logit']
        importance_score = torch.nn.functional.softmax(mask_logit[0])[0].cpu().numpy()

            #agent_obs = to_tensor(obs_dict, dtype=torch.float32)
            #agent_policy_output = agent.forward(agent_obs)
            #agent_actions = {i: a['action'] for i, a in agent_policy_output.items()}
            #agent_actions = to_ndarray(agent_actions)

        obs, rew, done, info = env.step(agent_action)
        epr += rew
        step += 1
            
        reward_record.append(epr)
        action_record.append(agent_action)
        importance_record.append(importance_score)

        if done or info["arrive_dest"]:
            traj.set(step, action_record, reward_record, importance_record)
            break

    return traj
    





def gen_one_traj(obs, env, seed):

    cfg = compile_config(
        main_config, SyncSubprocessEnvManager, PPOPolicy, BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, save_cfg=False
    )

    agent_cfg = compile_config(
        agent_config, SyncSubprocessEnvManager, PPOPolicy, BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, save_cfg=False
    )

    policy = PPOPolicy(cfg.policy)    

    import torch
    state_dict = torch.load('/home/zck7060/DI-drive/masknet_True_0.0001/ckpt/iteration_10000.pth.tar', map_location='cpu')
    #policy._model.load_state_dict(state_dict['model'], strict=True)
    policy.eval_mode.load_state_dict(state_dict)


    agent_policy = PPOPolicy(agent_cfg.policy)
    import torch
    state_dict = torch.load('/home/zck7060/DI-drive/baseline/metadrive_metaenv_ppo/ckpt/ckpt_best.pth.tar', map_location='cpu')
    #agent_policy._model.load_state_dict(state_dict['model'], strict=True)
    agent_policy.eval_mode.load_state_dict(state_dict)

    config = dict(
        use_render=False,
    )

    # env = gym.make("Macro-v1", config=config)
    traj = fid_play(obs, env, policy, agent_policy, seed)
    # env.close()

    return traj
