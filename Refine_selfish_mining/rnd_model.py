from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)

import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from tianshou.data.batch import Batch

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
                 Sequence[Dict[Any, Any]]]


class RNDModel(nn.Module):

    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        output_dim: Union[int, Sequence[int]] = 0,
        device = 'cpu',
    ) -> None:
        super().__init__()
        input_dim = int(np.prod(state_shape))
        self.device = device
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.target = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False


    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        obs = torch.from_numpy(obs).to(self.device)
        predict_feature = self.predictor(obs)
        target_feature = self.target(obs)
        forward_loss = F.mse_loss(predict_feature, target_feature.detach())
        return forward_loss

    def compute_bonus(self,next_obs):
        next_obs = torch.from_numpy(next_obs).to(self.device)
        predict_next_feature = self.predictor(next_obs)
        target_next_feature = self.target(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
        return intrinsic_reward.data.cpu().numpy()
