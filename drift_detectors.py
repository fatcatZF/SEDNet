import numpy as np

import torch 
import torch.nn as nn 



class MLPDriftDetector(nn.Module):
    def __init__(self, obs_dim, num_actions, discrete_action: bool=True,
                 hidden_dim: int = 256,
                 action_embed_dim: None|int = None):
        """
        args:
            obs_dim: observation dimensions of RL environment
            num_actions: the number of possible actions in discrete action space
                         or the number of action dimensions in continuous action space
            
        """
        super(MLPDriftDetector, self).__init__()
        if discrete_action:
            assert action_embed_dim is not None
            self.fc1 = nn.Linear(2*obs_dim+action_embed_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(2*obs_dim+num_actions, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.discrete_action = discrete_action
        self.num_actions = num_actions
        self.action_embed_layer = None 
        if discrete_action:
            self.action_embed_layer = nn.Embedding(num_actions, action_embed_dim)
        self.batch_norm_layer1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm_layer2 = nn.BatchNorm1d(hidden_dim)

    
    def forward(self, x):
        if self.discrete_action:
            action_dim = 1
        else:
            action_dim = self.num_actions
        action = x[:, -action_dim:]
        transition = x[:, :-action_dim]
        transition_dim = transition.size(-1)
        if self.discrete_action:
            action_embedded = self.action_embed_layer(action.long().squeeze(-1))
        else:
            action_embedded = action 
        ot = transition[:, :transition_dim//2]
        delta_ot = transition[:, transition_dim//2:]
        x = torch.cat([ot, delta_ot, action_embedded], dim=-1)

        x = torch.relu(self.batch_norm_layer1(self.fc1(x)))
        x = torch.relu(self.batch_norm_layer2(self.fc2(x)))
        return self.fc3(x)
        