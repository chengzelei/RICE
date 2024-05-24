import gym
import copy
import numpy as np
from typing import Any, Dict, Optional
from easydict import EasyDict
from itertools import product
import torch

from core.data.benchmark import ALL_SUITES
from core.eval.carla_benchmark_evaluator import get_suites_list, read_pose_txt, get_benchmark_dir
from core.envs.base_drive_env import BaseDriveEnv
from ding.utils.default_helper import deep_merge_dicts
from ding.envs.env.base_env import BaseEnvTimestep
from ding.envs.common.env_element import EnvElementInfo
from ding.torch_utils.data_helper import to_ndarray, to_tensor

from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import PPOPolicy

class DriveEnvWrapper(gym.Wrapper):
    """
    Environment wrapper to make ``gym.Env`` align with DI-engine definitions, so as to use utilities in DI-engine.
    It changes ``step``, ``reset`` and ``info`` method of ``gym.Env``, while others are straightly delivered.
    :Arguments:
        - env (BaseDriveEnv): The environment to be wrapped.
        - cfg (Dict): Config dict.
    :Interfaces: reset, step, info, render, seed, close
    """

    config = dict()

    def __init__(self, env: BaseDriveEnv, cfg: Dict = None, **kwargs) -> None:
        if cfg is None:
            self._cfg = self.__class__.default_config()
        elif 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg
        self.env = env
        if not hasattr(self.env, 'reward_space'):
            self.reward_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, ))
        
        metadrive_macro_agent_config = dict(
            exp_name='metadrive_metaenv_ppo_agent',
            env=dict(
                metadrive=dict(use_render=False),
                manager=dict(
                    shared_memory=False,
                    max_retry=2,
                    context='spawn',
                ),
                n_evaluator_episode=1,
                stop_value=99999,
                collector_env_num=1,
                evaluator_env_num=1,
                wrapper=dict(),
            ),
            policy=dict(
                cuda=True,
                model=dict(
                    obs_shape=[5, 200, 200],
                    action_shape=5,
                    encoder_hidden_size_list=[128, 128, 64],
                ),
                learn=dict(
                    #epoch_per_collect=10,
                    batch_size=64,
                    learning_rate=1e-3,
                    update_per_collect=100,
                    hook=dict(
                        load_ckpt_before_run='',
                    ),
                ),
                collect=dict(
                    n_sample=1000,
                ),
                eval=dict(evaluator=dict(eval_freq=50, )),
                other=dict(
                    eps=dict(
                        type='exp',
                        start=0.95,
                        end=0.1,
                        decay=10000,
                    ),
                    replay_buffer=dict(
                        replay_buffer_size=10000,
                    ),
                ),
            ),
        )


        agent_config = EasyDict(metadrive_macro_agent_config)

        
        agent_cfg = compile_config(
            agent_config, SyncSubprocessEnvManager, PPOPolicy, BaseLearner, SampleSerialCollector, InteractionSerialEvaluator
        )
        self.agent_policy = PPOPolicy(agent_cfg.policy)
        state_dict = torch.load('/home/zck7060/DI-drive/baseline/metadrive_metaenv_ppo/ckpt/ckpt_best.pth.tar', map_location='cpu')
        self.agent_policy.eval_mode.load_state_dict(state_dict)
        self.agent = self.agent_policy.eval_mode

        self.obs = None
        self.lamb = 1e-4
        self.bonus_last = True
        self.accu_mask = 0


    def reset(self, *args, **kwargs) -> Any:
        """
        Wrapper of ``reset`` method in env. The observations are converted to ``np.ndarray`` and final reward
        are recorded.
        :Returns:
            Any: Observations from environment
        """
        obs = self.env.reset(*args, **kwargs)
        obs = to_ndarray(obs, dtype=np.float32)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            obs = obs.transpose((2, 0, 1))
        self.obs = obs
        # elif isinstance(obs, dict):
        #     if 'birdview' in obs:
        #         obs['birdview'] = obs['birdview'].transpose((2, 0, 1))
        #     if 'rgb' in obs:
        #         obs['rgb'] = obs['rgb'].transpose((2, 0, 1))
        self._final_eval_reward = 0.0
        self.accu_mask = 0
        return obs

    def step(self, action: Any = None) -> BaseEnvTimestep:
        """
        Wrapper of ``step`` method in env. This aims to convert the returns of ``gym.Env`` step method into
        that of ``ding.envs.BaseEnv``, from ``(obs, reward, done, info)`` tuple to a ``BaseEnvTimestep``
        namedtuple defined in DI-engine. It will also convert actions, observations and reward into
        ``np.ndarray``, and check legality if action contains control signal.
        :Arguments:
            - action (Any, optional): Actions sent to env. Defaults to None.
        :Returns:
            BaseEnvTimestep: DI-engine format of env step returns.
        """
        action = to_ndarray(action)
  
        agent_obs_dict = {i: self.obs for i in range(5)}
        agent_obs = to_tensor(agent_obs_dict, dtype=torch.float32)
        agent_policy_output = self.agent.forward(agent_obs)
        agent_actions = {i: a['action'] for i, a in agent_policy_output.items()}
        agent_actions = to_ndarray(agent_actions)

        real_actions = []
        for i in range(1):
            if action[i] == 1:
                real_actions.append(np.random.choice(5))
                mask = 1
            else:
                real_actions.append(agent_actions[i])
                mask = 0

        self.accu_mask += mask
        obs, rew, done, info = self.env.step(np.array(real_actions))
        self._final_eval_reward += rew
        obs = to_ndarray(obs, dtype=np.float32)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            obs = obs.transpose((2, 0, 1))
        self.obs = obs
        # elif isinstance(obs, dict):
        #     if 'birdview' in obs:
        #         obs['birdview'] = obs['birdview'].transpose((2, 0, 1))
        #     if 'rgb' in obs:
        #         obs['rgb'] = obs['rgb'].transpose((2, 0, 1))
        if not self.bonus_last:
            rew += self.lamb * mask
        rew = to_ndarray([rew], dtype=np.float32)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
            if self.bonus_last:
                rew += self.lamb * self.accu_mask
        return BaseEnvTimestep(obs, rew, done, info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self.env = gym.wrappers.Monitor(self.env, self._replay_path, video_callable=lambda episode_id: True, force=True)

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)

    def __repr__(self) -> str:
        return repr(self.env)

    def render(self):
        self.env.render()
