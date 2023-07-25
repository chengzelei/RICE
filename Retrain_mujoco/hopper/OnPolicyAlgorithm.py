import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces

from base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from utils import RunningMeanStd
from models import MuJoCoStateEncoder, MuJoCoInverseDynamicNet, RNDModel

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        bonus: Union[None, str] = None,
        bonus_scale: float = 1,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[spaces.Space, ...]] = None,
    ):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )
   
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.bonus = bonus
        self.bonus_scale = bonus_scale

        if self.bonus == 'e3b':
            # init e3b bonus related
            self.feature_extractor = MuJoCoStateEncoder(self.device).to(self.device)
            self.feature_extractor_optimizer = th.optim.Adam(
                self.feature_extractor.parameters(), 
                lr=1e-3)
            self.lamb = 1e-1
            self.feat_sz = 500
            self.inv_cov = th.mul(th.eye(self.feat_sz), 1.0/self.lamb)
            self.cov = th.mul(th.eye(self.feat_sz), self.lamb)
            self.rank1_update = False
            self.inverse_net = MuJoCoInverseDynamicNet(self.device).to(self.device)
            self.inverse_net_optimizer = th.optim.Adam(
                self.inverse_net.parameters(), 
                lr=1e-3)

        elif self.bonus == 'rnd':
            # init rnd bonus related
            obs_shape = self.observation_space.shape
            self.obs_rms = RunningMeanStd(shape = obs_shape)

            input_dim, output_dim = obs_shape[0], 64
            self.rnd_model = RNDModel(self.device, input_dim, output_dim).to(self.device)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  
        )
        self.policy = self.policy.to(self.device)


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:

        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        total_next_obs = []
        bonuses = []


        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if self.bonus == 'e3b':
                # add elliptical bonus
                for i in range(len(new_obs)):
                    feature = self.feature_extractor(new_obs[i]).squeeze().detach().cpu()

                    if self.rank1_update:
                        u = th.mv(self.inv_cov, feature)
                        bonus = th.dot(feature, u).numpy()
                        outer_product_buffer = th.matmul(u, u.T)
                        th.add(self.inv_cov, outer_product_buffer, alpha=-(1./(1. + bonus)), out=self.inv_cov) 
                    else:
                        self.cov += th.outer(feature, feature) 
                        self.inv_cov = th.inverse(self.cov)
                        u = th.mv(self.inv_cov, feature)
                        bonus = th.dot(feature, u).numpy()
                    bonuses.append(bonus)
                    rewards[i] += self.bonus_scale * bonus
            
            elif self.bonus == 'rnd':
                total_next_obs.append(new_obs)
                # add rnd bonus
                new_norm_obs =  ((new_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5, 5)
                int_rewards = self.rnd_model.compute_bonus(new_norm_obs)
                rnd_reward = (int_rewards - np.min(int_rewards)) / (np.max(int_rewards) - np.min(int_rewards) + 1e-11)
                bonuses.append(np.mean(int_rewards))
                rewards = rewards + self.bonus_scale * int_rewards

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)


            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        if self.bonus == 'rnd':
            # Normalize the observation
            total_next_obs = np.vstack(total_next_obs)
            self.obs_rms.update(total_next_obs)

        return True, bonuses

    def train(self) -> None:
        raise NotImplementedError

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training, bonuses = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.record("rollout/bonus", np.mean(bonuses))
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []