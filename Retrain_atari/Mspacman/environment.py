import gym
from gym import Wrapper
from stable_baselines3 import PPO
import numpy as np
import torch
import torch.nn as nn
import random
from stable_baselines3.common.env_util import make_atari_env

import os
from typing import Any, Callable, Dict, Optional, Type, Union

import gym

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


def unwrap_wrapper(env: gym.Env, wrapper_class: Type[gym.Wrapper]) -> Optional[gym.Wrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env: Type[gym.Env], wrapper_class: Type[gym.Wrapper]) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.

    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


def make_vec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    random_net, 
    bonus_scale, 
    inv_cov,
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
        Note: the wrapper specified by this parameter will be applied after the ``Monitor`` wrapper.
        if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
        See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs)
                
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            env = RetrainEnv(env, random_net, bonus_scale, inv_cov)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm1d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
           module.bias.data.zero_()

class MLP_Net(nn.Module):
    def __init__(self, input_dim, hiddens):

        super(MLP_Net, self).__init__()

        self.RND = nn.Sequential()

        for i in range(len(hiddens)):
            if i == 0:
                self.RND.add_module('mlp_%d' %i, nn.Linear(input_dim, hiddens[i]))
            else:
                self.RND.add_module('mlp_%d' %i, nn.Linear(hiddens[i-1], hiddens[i]))

            if i != len(hiddens) - 1:
                self.RND.add_module('relu_%d' %i, nn.ReLU())

        self.RND.apply(initialize_weights)

    
    def eval(self, obs, act):
        # Preprocess the observation
        obs = np.asarray(obs[0]).flatten()

        np_act = np.zeros((3,))
        np_act.fill(act)
        #np_act = np.asarray(act)

        obs_act = np.concatenate((obs, np_act))
        obs_act = torch.FloatTensor(obs_act)

        # Expand dim
        obs_act = torch.unsqueeze(obs_act, 0)
        ret = self.RND(obs_act)
        ret = torch.squeeze(ret, 0)
        return ret.detach().numpy()

class training_pool():

    def __init__(self, losing_games_file, winning_games_file, ratio):
        self.total_num = 5000
        self.ratio = ratio
        self.losing_games_idxs = self.extract_idxs(losing_games_file)
        self.winning_games_idxs = self.extract_idxs(winning_games_file)
        self.candidates = self.create_pool()
    
    def extract_idxs(self, filename):
        idxs = np.loadtxt(filename)
        return idxs
    
    def create_pool(self):
        losing_idxs_selected = np.random.choice(self.losing_games_idxs, int(self.total_num * self.ratio))
        winning_idxs_selected = np.random.choice(self.winning_games_idxs, int(self.total_num * (1-self.ratio)))
        pool = np.concatenate((losing_idxs_selected, winning_idxs_selected), axis=None)
        return pool

def compute_bonus(features, inv_cov):
    bonus = np.sqrt(np.sum(np.dot(np.dot(features.T, inv_cov), features)))
    return bonus

def go_inv_cov(env, random_net, feat_sz, lamb):
    losing_games_file = 'losing_game.out'
    winning_games_file = 'winning_game.out'
    train_pool = training_pool(losing_games_file, winning_games_file, 1.0)
    idxs_list = train_pool.candidates
    critical_steps_starts = np.loadtxt("critical_steps_starts.out")
    losing_idx = train_pool.losing_games_idxs

    cov, pair_num = np.zeros((feat_sz, feat_sz)), 0 

    # Compute inv_cov from the losing trajectories
    for i in range(len(losing_idx)):

        if i % 10 == 0:
            print("Episode %d" %(i))

        idx = int(losing_idx[i])
        action_sequence_path = "weak_retrain_data/act_seq_" + str(idx) + ".npy"
        recorded_actions = np.load(action_sequence_path, allow_pickle=True)
        env.seed(idx)
        obs, done = env.reset(), False
        count = 0

        while True and count < 1000:
            act = recorded_actions[count]
            if count >= critical_steps_starts[idx]:
                feature = random_net.eval(obs[0], act[0])
                cov += np.outer(feature, feature)
                pair_num += 1
            obs, reward, done, info = env.step(act)
            count += 1
            if "episode" in info[0].keys(): break

    print('Pair num is %d' % pair_num)
    cov /= pair_num 
    cov = lamb * np.identity(feat_sz) + cov

    inv_cov = np.linalg.inv(cov)
    np.savez('inv_cov.npz', inv_cov=inv_cov)

class RetrainEnv(Wrapper):

    def __init__(self, env, random_net, bonus_scale, inv_cov):

        Wrapper.__init__(self, env)
        self.env = env
        self.random_net = random_net
        self.bonus_scale = bonus_scale
        self.inv_cov = inv_cov
        self.counter = 0

        # 0: losing 1: wining 
        self.idx = 0
        losing_games_file = 'losing_game.out'
        winning_games_file = 'winning_game.out'
        self.train_pool = training_pool(losing_games_file, winning_games_file, 0.5)
        self.idxs_list = self.train_pool.candidates
        self.critical_steps_starts = np.loadtxt("critical_steps_starts.out")



    def step(self, action):
            
        # obtain needed information from the environment.
        obs, reward, done, info = self.env.step(action)
        bonus = 0
        if done:
            self.flag = True
        if self.idx == 0 and not self.flag:
            feature = self.random_net.eval(self.obs, action)
            bonus = compute_bonus(feature, self.inv_cov)
            #print('bonus is', bonus * self.bonus_scale)
            reward += self.bonus_scale * bonus
        self.obs = obs
        return obs, reward, done, info



    def reset(self):
        self.counter += 1
        if self.counter % 100 == 0:
            self.bonus_scale *= 0.9
            self.counter = 0
        self.flag = False
        i_episode = int(random.choice(self.idxs_list))
        
        if i_episode in self.train_pool.losing_games_idxs:
            self.idx = 0
        else:
            self.idx = 1
        action_sequence_path = "weak_retrain_data/act_seq_" + str(i_episode) + ".npy"
        recorded_actions = np.load(action_sequence_path, allow_pickle=True)

        self.env.seed(i_episode)
        obs = self.env.reset()

        if np.random.rand() > 0.5:
            count = 0
            while count < self.critical_steps_starts[i_episode]:
                act = recorded_actions[count]
                obs, reward, done, info = self.env.step(act[0])
                if done: obs = self.env.reset()
                count += 1
        
        self.obs = obs
        return obs




def make_retrain_env(
    env_id: Union[str, Callable[..., gym.Env]],
    random_net, 
    bonus_scale, 
    inv_cov,
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[Type[DummyVecEnv], Type[SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    return make_vec_env(
        env_id,
        random_net, 
        bonus_scale, 
        inv_cov,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=AtariWrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    )


if __name__ == '__main__':

    PATH = 'RND'
    env = make_atari_env("MsPacman-v0", n_envs=1)
    input_dim = 84 + 3
    random_net = MLP_Net(input_dim, [500, 500])
    # save random_net
    torch.save(random_net.state_dict(), PATH)
    go_inv_cov(env, random_net, 500, 0.01)