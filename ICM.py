import torch
import torch.nn as nn
import torch.nn.functional as F

class ICM_CNN(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim=256):
        super(ICM_CNN, self).__init__()
        self.action_dim = action_dim
        channels, _, _ = obs_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            feature_size = self.encoder(dummy_input).shape[1]
        self.encoder_fc = nn.Linear(feature_size, feature_dim)
        
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim) 
        )
        
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, state, next_state, action):
        state_feature_raw = self.encoder(state)
        state_feature = self.encoder_fc(state_feature_raw)
        
        next_state_feature_raw = self.encoder(next_state)
        next_state_feature = self.encoder_fc(next_state_feature_raw)

        action_onehot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        
        pred_next_state_feature = self.forward_model(torch.cat((state_feature, action_onehot), dim=1))
        
        pred_action_logits = self.inverse_model(torch.cat((state_feature, next_state_feature), dim=1))

        return pred_next_state_feature, next_state_feature, pred_action_logits