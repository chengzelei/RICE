from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch
from tianshou.policy import BasePolicy

class RNDPolicy(BasePolicy):
    def __init__(
        self,
        policy: BasePolicy,
        model,
        optim: torch.optim.Optimizer,
        reward_scale: float,
        forward_loss_weight: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.policy = policy
        self.model = model
        self.optim = optim
        self.reward_scale = reward_scale
        self.forward_loss_weight = forward_loss_weight

    def train(self, mode: bool = True):
        """Set the module in training mode."""
        self.policy.train(mode)
        self.training = mode
        self.model.train(mode)
        return self

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data by inner policy.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        return self.policy.forward(batch, state, **kwargs)

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        return self.policy.exploration_noise(act, batch)

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        if hasattr(self.policy, "set_eps"):
            self.policy.set_eps(eps)  # type: ignore
        else:
            raise NotImplementedError()

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Pre-process the data from the provided replay buffer.

        Used in :meth:`update`. Check out :ref:`process_fn` for more information.
        """
        mse_loss = self.model.forward(batch.obs_next)
        bonus = self.model.compute_bonus(batch.obs_next)
        bonus = (bonus - np.min(bonus)) / (np.max(bonus) - np.min(bonus) + 1e-11)
        batch.policy = Batch(orig_rew=batch.rew, bonus=bonus, mse_loss=mse_loss)
        batch.rew += to_numpy(bonus* self.reward_scale)
        return self.policy.process_fn(batch, buffer, indices)

    def post_process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> None:
        """Post-process the data from the provided replay buffer.

        Typical usage is to update the sampling weight in prioritized
        experience replay. Used in :meth:`update`.
        """
        self.policy.post_process_fn(batch, buffer, indices)
        batch.rew = batch.policy.orig_rew  # restore original reward

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        res = self.policy.learn(batch, **kwargs)
        self.optim.zero_grad()
        forward_loss = batch.policy.mse_loss.mean()
        bonus = np.mean(batch.policy.bonus)
        loss = self.forward_loss_weight * forward_loss
        loss.backward()
        self.optim.step()
        res.update(
            {
                "loss/rnd": loss.item(),
                "bonus": bonus
            }
        )
        return res