import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=16, fc2_units=16):
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

        self.lstm = nn.LSTM(input_size=state_size, hidden_size=16)
        self.lstm_hidden_state = torch.zeros((1, 1, 16)).to(device)
        self.lstm_cell_state = torch.zeros((1, 1, 16)).to(device)

        self.fc1 = nn.Linear(16, fc1_units)
        #nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        #nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(fc2_units, action_size)
        #nn.init.xavier_uniform_(self.fc3.weight)

    def init_lstm():
        '''Initializes LSTM state'''
        self.lstm_hidden_state = torch.zeros((1, 1, 16))
        self.lstm_cell_state = torch.zeros((1, 1, 16))


    def forward(self, state):
        """Build a network that maps state -> action values."""

        state = torch.transpose(state, 0, 1)  # (seq, batch, features)

        output, cell_state = self.lstm(state)

        x = F.relu(self.fc1(output[-1,:,:]))  # only take end of sequence
        x = F.relu(self.fc2(x))
        Q_s_a = self.fc3(x)
        return Q_s_a
    
    def save(self, episode_num=0):
        """saves model weights"""
        torch.save(self.state_dict(), "model_checkpoint_" + str(episode_num) +"_.pt")
    
    def load(self, fname):
        """loads model weights"""
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(fname))          # trained using noisy state
        else:
            self.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
        #self.load_state_dict(torch.load("dqn_pilot_trained.pt"))   # trained using true state
        print("Previous model loaded")
        
        
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.lstm = nn.LSTM(input_size=state_size, hidden_size=16)
        self.lstm_hidden_state = torch.zeros((1, 1, 16)).to(device)
        self.lstm_cell_state = torch.zeros((1, 1, 16)).to(device)

        self.bn1 = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(16, fc1_units)
        self.bn2 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn3 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        state = torch.transpose(state, 0, 1)  # (seq, batch, features)
        output, cell_state = self.lstm(state)
        state = output[-1,:,:]
        state = self.bn1(state)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(self.bn2(x)))
        return torch.tanh(self.fc3(self.bn3(x)))

    def save(self, episode_num=0):
        """saves model weights"""
        torch.save(self.state_dict(), "training/actor_" + str(episode_num) +"_.pt")

    def load(self):
        """loads model weights"""
        dir = os.getcwd()
        self.load_state_dict(torch.load(f"{dir}/main/lstm_actor.pt",map_location=torch.device('cpu')))          # trained using noisy state
        #self.load_state_dict(torch.load("dqn_pilot_trained.pt"))   # trained using true state
        print("Previous model loaded")


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.lstm = nn.LSTM(input_size=state_size, hidden_size=16)
        self.lstm_hidden_state = torch.zeros((1, 1, 16)).to(device)
        self.lstm_cell_state = torch.zeros((1, 1, 16)).to(device)

        self.bn1 = nn.BatchNorm1d(16)
        self.fcs1 = nn.Linear(16, fcs1_units)
        self.bn2 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = torch.transpose(state, 0, 1)  # (seq, batch, features)
        output, cell_state = self.lstm(state)
        state = output[-1,:,:]
        xs = F.relu(self.fcs1(self.bn1(state)))
        x = torch.cat((self.bn2(xs), action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save(self, episode_num=0):
        """saves model weights"""
        torch.save(self.state_dict(), "training/critic_" + str(episode_num) +"_.pt")

    def load(self):
        """loads model weights"""
        dir = os.getcwd()
        self.load_state_dict(torch.load(f"{dir}/main/lstm_critic.pt",map_location=torch.device('cpu')))          # trained using noisy state
        #self.load_state_dict(torch.load("dqn_pilot_trained.pt"))   # trained using true state
        print("Previous model loaded")

