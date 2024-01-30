import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
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

class Net_ppo(nn.Module):
    def __init__(self):
        super(Net_ppo, self).__init__()
        self.input_length=1048576
        self.window_size=500
        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(4, 128, self.window_size, stride=self.window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, self.window_size, stride=self.window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(self.input_length/self.window_size))
        

        self.fc_1 = nn.Linear(128,128)
        # self.fc_2 = nn.Linear(128,action_shape)

        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax()
        self.output_dim = 128

    def forward(self, obs, state=None, info={}):
        # obs = obs.int() # data parallel needs this
        obs = obs.astype(int)
        obs = torch.from_numpy(obs).cuda() # without data parallel needs this

        x = self.embed(obs)
        # Channel first
        x = torch.transpose(x,-1,-2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.view(-1,128)
        x = self.fc_1(x)
        # x = self.fc_2(x)
        #x = self.sigmoid(x)

        return x, state

class Net_ppo_parallel(nn.Module):
    def __init__(self):
        super(Net_ppo_parallel, self).__init__()
        self.input_length=1048576
        self.window_size=500
        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(4, 128, self.window_size, stride=self.window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, self.window_size, stride=self.window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(self.input_length/self.window_size))
        

        self.fc_1 = nn.Linear(128,128)
        # self.fc_2 = nn.Linear(128,action_shape)

        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax()
        self.output_dim = 128

    def forward(self, obs, state=None, info={}):
        obs = obs.int() # data parallel needs this
        # obs = obs.astype(int)
        # obs = torch.from_numpy(obs).cuda() # without data parallel needs this

        x = self.embed(obs)
        # Channel first
        x = torch.transpose(x,-1,-2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.view(-1,128)
        x = self.fc_1(x)
        # x = self.fc_2(x)
        #x = self.sigmoid(x)

        return x, state


class RNDModel(nn.Module):
    def __init__(self, output_dim):
        super(RNDModel, self).__init__()
        self.input_length = 1048576
        self.window_size = 500

        self.pred_embed = nn.Embedding(257, 8, padding_idx=0)
        self.pred_conv_1 = nn.Conv1d(4, 64, self.window_size, stride=self.window_size, bias=True)
        self.pred_conv_2 = nn.Conv1d(4, 64, self.window_size, stride=self.window_size, bias=True)
        self.pred_pooling = nn.MaxPool1d(int(self.input_length/self.window_size))
        self.pred_fc_1 = nn.Linear(64,output_dim)


        self.target_embed = nn.Embedding(257, 8, padding_idx=0)
        self.target_conv_1 = nn.Conv1d(4, 64, self.window_size, stride=self.window_size, bias=True)
        self.target_conv_2 = nn.Conv1d(4, 64, self.window_size, stride=self.window_size, bias=True)
        self.target_pooling = nn.MaxPool1d(int(self.input_length/self.window_size))
        self.target_fc_1 = nn.Linear(64,output_dim)
        self.sigmoid = nn.Sigmoid()

        # Freeze the parameters of self.target_embed
        self.target_embed.weight.requires_grad = False

        # Freeze the parameters of self.target_conv_1
        self.target_conv_1.weight.requires_grad = False
        self.target_conv_1.bias.requires_grad = False

        # Freeze the parameters of self.target_conv_2
        self.target_conv_2.weight.requires_grad = False
        self.target_conv_2.bias.requires_grad = False

        # Freeze the parameters of self.target_pooling
        # There are no learnable parameters in MaxPool1d, so no action is needed here

        # Freeze the parameters of self.target_fc_1
        self.target_fc_1.weight.requires_grad = False
        self.target_fc_1.bias.requires_grad = False


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()


    def forward(self, obs):
        obs = obs.int()
        # obs = obs.astype(int)
        # obs = torch.from_numpy(obs).to(self.device)
        x = self.pred_embed(obs)
        # Channel first
        x = torch.transpose(x,-1,-2)

        cnn_value = self.pred_conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.pred_conv_2(x.narrow(-2, 4, 4)))

        x = cnn_value * gating_weight
        x = self.pred_pooling(x)

        x = x.view(-1,64)
        predict_feature = self.pred_fc_1(x)
        with torch.no_grad():
            x = self.target_embed(obs)
            # Channel first
            x = torch.transpose(x,-1,-2)

            cnn_value = self.target_conv_1(x.narrow(-2, 0, 4))
            gating_weight = self.sigmoid(self.target_conv_2(x.narrow(-2, 4, 4)))

            x = cnn_value * gating_weight
            x = self.target_pooling(x)

            x = x.view(-1,64)
            target_feature = self.target_fc_1(x)
        
        return predict_feature, target_feature

class DataParallelNet(nn.Module):
    """DataParallel wrapper for training agent with multi-GPU.

    This class does only the conversion of input data type, from numpy array to torch's
    Tensor. If the input is a nested dictionary, the user should create a similar class
    to do the same thing.

    :param nn.Module net: the network to be distributed in different GPUs.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = nn.DataParallel(net)

    def forward(self, obs: Union[np.ndarray, torch.Tensor], *args: Any,
                **kwargs: Any) -> Tuple[Any, Any]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        return self.net(obs=obs.cuda(), *args, **kwargs)
    
    # def compute_bonus(self, obs, *args, **kwargs):
    #     if not isinstance(obs, torch.Tensor):
    #         obs = torch.as_tensor(obs, dtype=torch.float32)
    #     return self.net.compute_bonus(obs=obs.cuda(), *args, **kwargs)
