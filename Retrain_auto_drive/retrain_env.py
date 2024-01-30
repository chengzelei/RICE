from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import PPOPolicy
import gym
from core.envs.base_drive_env import BaseDriveEnv
from typing import Any, Dict, Optional
from ding.envs.env.base_env import BaseEnvTimestep
from easydict import EasyDict
import copy
import numpy as np
from gen_retrain_traj import gen_one_traj
from ding.torch_utils.data_helper import to_ndarray

class CustomWrapper(gym.Wrapper):
    """
    Environment wrapper to make ``gym.Env`` align with DI-engine definitions, so as to use utilities in DI-engine.
    It changes ``step``, ``reset`` and ``info`` method of ``gym.Env``, while others are straightly delivered.
    :Arguments:
        - env (BaseDriveEnv): The environment to be wrapped.
        - cfg (Dict): Config dict.
    :Interfaces: reset, step, info, render, seed, close
    """

    config = dict()

    def __init__(self, env: BaseDriveEnv, cfg: Dict = None, go_prob = 0.5, **kwargs) -> None:
        self.p = go_prob
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

        self.random_sampling = True
        self.obs = None
        self.seed = 0


    def reset(self, *args, **kwargs) -> Any:
        """
        Wrapper of ``reset`` method in env. The observations are converted to ``np.ndarray`` and final reward
        are recorded.
        :Returns:
            Any: Observations from environment
        """



        if np.random.rand() < self.p:
            obs = self.env.reset(force_seed=0)
            obs = to_ndarray(obs, dtype=np.float32)
            traj = gen_one_traj(obs, self.env,self.seed)
            if self.random_sampling:
                start_idx = np.random.choice(traj.eps_len)
                print("random sampling start idx: ", start_idx)
            else:
                start_idx = np.argmax(traj.mask_probs)

            # self.env.close()
            print("complete traj collect")
            obs = self.env.reset()
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

            if start_idx > 0:
                step = 0
                while step < start_idx:
                    action = int(traj.act_seq[step])
                    obs, rew, done, info = self.env.step(action)
                    self._final_eval_reward += rew
                    step += 1
                obs = to_ndarray(obs, dtype=np.float32)
                if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
                    obs = obs.transpose((2, 0, 1))
        
        else:
            print("environment: ", self.env)
            obs = self.env.reset()
            obs = to_ndarray(obs, dtype=np.float32)
            if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
                obs = obs.transpose((2, 0, 1))
            self._final_eval_reward = 0.0
        
        self.seed += 1
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

        obs, rew, done, info = self.env.step(action)
        self._final_eval_reward += rew
        obs = to_ndarray(obs, dtype=np.float32)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            obs = obs.transpose((2, 0, 1))
        # elif isinstance(obs, dict):
        #     if 'birdview' in obs:
        #         obs['birdview'] = obs['birdview'].transpose((2, 0, 1))
        #     if 'rgb' in obs:
        #         obs['rgb'] = obs['rgb'].transpose((2, 0, 1))
        rew = to_ndarray([rew], dtype=np.float32)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
            self.env.close()
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

