import numpy as np
import torch
import torch.nn as nn

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
    def __init__(self, input_dim=11, hiddens=[500,500]):

        super(MuJoCoStateEncoder, self).__init__()

        self.state_encoder = nn.Sequential(nn.Linear(input_dim, 500),
                                         nn.ReLU(),
                                         nn.Linear(500, 500))

        self.state_encoder.apply(initialize_weights)

    def forward(self, obs):
        # Preprocess the observation
        obs = np.asarray(obs)
        obs = torch.FloatTensor(obs)

        # Expand dim
        obs = torch.unsqueeze(obs, 0)
        ret = self.state_encoder(obs)
        ret = torch.squeeze(ret, 0)
        return ret
        
    def eval(self, obs):
        # Preprocess the observation
        obs = np.asarray(obs)
        obs = torch.FloatTensor(obs)

        # Expand dim
        obs = torch.unsqueeze(obs, 0)
        ret = self.state_encoder(obs)
        ret = torch.squeeze(ret, 0)
        return ret.detach().numpy()


class MuJoCoInverseDynamicNet(nn.Module):
    def __init__(self, num_actions=3, input_dim=500):

        super(MuJoCoInverseDynamicNet, self).__init__()

        self.inverse_net = nn.Sequential(nn.Linear(input_dim, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, num_actions))


        self.inverse_net.apply(initialize_weights)

    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.inverse_net(inputs)
        return action_logits.detach().numpy()
        

