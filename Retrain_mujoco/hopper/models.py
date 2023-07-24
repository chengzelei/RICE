import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

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

class MuJoCoStateEncoder(nn.Module):
    def __init__(self, device, input_dim=11, hiddens=[500,500]):

        super(MuJoCoStateEncoder, self).__init__()
        self.device = device
        self.state_encoder = nn.Sequential(nn.Linear(input_dim, 500),
                                         nn.ReLU(),
                                         nn.Linear(500, 500))

        self.state_encoder.apply(initialize_weights)

    def forward(self, obs):
        # # Preprocess the observation
        # obs = np.asarray(obs)
        # obs = torch.FloatTensor(obs).to(self.device)

        # # Expand dim
        # obs = torch.unsqueeze(obs, 0)
        # ret = self.state_encoder(obs)
        # ret = torch.squeeze(ret, 0)

        # return ret

        x = obs
        state_embedding = self.state_encoder(torch.Tensor(x).to(self.device))

        return state_embedding
        
        
    def eval(self, obs):
        # Preprocess the observation
        obs = np.asarray(obs)
        obs = torch.FloatTensor(obs).to(self.device)

        # Expand dim
        obs = torch.unsqueeze(obs, 0)
        ret = self.state_encoder(obs)
        ret = torch.squeeze(ret, 0)
        return ret.detach().numpy()


class MuJoCoInverseDynamicNet(nn.Module):
    def __init__(self, device, num_actions=3, input_dim=1000):

        super(MuJoCoInverseDynamicNet, self).__init__()
        self.device = device
        self.inverse_net = nn.Sequential(nn.Linear(input_dim, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, num_actions))
        self.inverse_net.apply(initialize_weights)

    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=1)
        action_logits = self.inverse_net(inputs).to(self.device)
        return action_logits
        
class RNDModel(nn.Module):
    def __init__(self, device, input_size, output_size):
        super(RNDModel, self).__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size

        self.predictor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

        self.target = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

        # Initialize weights    
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        # Set target parameters as untrainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature

    def compute_bonus(self, next_obs):
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        target_next_feature = self.target(next_obs)
        predict_next_feature = self.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()
