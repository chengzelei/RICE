import torch
import torch.nn as nn

class RNDModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RNDModel, self).__init__()

        self.predict_model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.target_model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
    
    def foward(self, obs):
        predict_feature = self.predict_model(obs)
        target_feature = self.target_model(obs)
        return predict_feature, target_feature
    
    def compute_bonus(self, next_obs):
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        target_next_feature = self.target(next_obs)
        predict_next_feature = self.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
        return intrinsic_reward.data.cpu().numpy()