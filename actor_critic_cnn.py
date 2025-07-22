# actor_critic_cnn.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class ActorCriticCNN(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super(ActorCriticCNN, self).__init__()
        
        channels, height, width = obs_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            feature_dim = self.features(dummy_input).shape[1]

        self.lstm_hidden_dim = 128
        self.lstm = nn.LSTM(feature_dim, self.lstm_hidden_dim, batch_first=True)

        self.actor = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.hidden = None

    def forward(self, state, reset_hidden=False):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        x = self.features(state)
        
        if self.hidden is None or reset_hidden or self.hidden[0].shape[1] != batch_size:
            self.hidden = (torch.zeros(1, batch_size, self.lstm_hidden_dim).to(state.device),
                           torch.zeros(1, batch_size, self.lstm_hidden_dim).to(state.device))
        
        x = x.unsqueeze(1)
        
        x, self.hidden = self.lstm(x, self.hidden)
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        x = x.squeeze(1)
        
        value = self.critic(x)
        action_logits = self.actor(x)

        dist = Categorical(logits=action_logits)
        
        return dist, value