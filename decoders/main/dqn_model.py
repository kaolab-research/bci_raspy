import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=12, fc2_units=12):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        #nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        #nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(fc2_units, action_size)
        #nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def save(self):
        """saves model weights"""
        torch.save(self.state_dict(), "model.pt")
    
    def load(self):
        """loads model weights"""
        self.load_state_dict(torch.load("model.pt"))
        
        