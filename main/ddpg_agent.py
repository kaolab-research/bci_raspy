import numpy as np
import random
import copy
from copy import deepcopy
from collections import namedtuple, deque
from os import mkdir
import datetime
import matplotlib.pyplot as plt
import pdb
from main.dqn_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1000000000000000 # hack to avoid updatess
UPDATES_PER_STEP = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, seed=12345):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.t_step = 0

        self.m = 16  # number of past observations to include in state
        self.state = deque(maxlen=self.m)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed, fc1_units=256, fc2_units=128).to(device)
        self.actor_target = Actor(state_size, action_size, seed, fc1_units=256, fc2_units=128).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed, fcs1_units=256, fc2_units=128).to(device)
        self.critic_target = Critic(state_size, action_size, seed, fcs1_units=256, fc2_units=128).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Replay memory
        self.memory = ReplayBuffer(state_size, action_size, BUFFER_SIZE, BATCH_SIZE, seed)


    def load_weights(self,  fpath, model_num):
        '''loads weights from existing model'''
        self.actor_local.load_state_dict(torch.load(fpath + '/ddpg_actor_local_' + str(model_num), map_location=torch.device('cpu')))
        self.actor_target.load_state_dict(torch.load(fpath + '/ddpg_actor_target_' + str(model_num), map_location=torch.device('cpu')))
        self.critic_local.load_state_dict(torch.load(fpath + '/ddpg_critic_local_' + str(model_num), map_location=torch.device('cpu')))
        self.critic_target.load_state_dict(torch.load(fpath + '/ddpg_critic_target_' + str(model_num), map_location=torch.device('cpu')))

    def save_actor_critic(self, episode_num=0):
        '''saves the actor and critic models'''
        torch.save(self.actor_local.state_dict(), self.log_dir + 'ddpg_actor_local_' + str(episode_num))
        torch.save(self.actor_target.state_dict(), self.log_dir + 'ddpg_actor_target_' + str(episode_num))
        torch.save(self.critic_local.state_dict(), self.log_dir + 'ddpg_critic_local_' + str(episode_num))
        torch.save(self.critic_target.state_dict(), self.log_dir + 'ddpg_critic_target_' + str(episode_num))
    
    def save_training_run(self, scores, episode_num):
        '''
        plots the learning curve for this training run and logs it to file
        '''
        # generate learning curve
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), np.mean(scores, axis=-1))
        plt.ylabel('Score')
        plt.xlabel('Training Episode')
        plt.savefig(self.log_dir + 'learning_curve')

        # save the final model
        self.save_actor_critic(episode_num=episode_num)

        # save the scores array
        with open(self.log_dir + 'scores.npy', 'wb') as f:
            np.save(f, np.array(scores))
        f.close()

        plt.show()

    def step(self, state, action, reward, new_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # construct the state
        state = np.array(self.state)
        next_state = deepcopy(self.state)
        next_state.append(new_state)
        next_state = np.array(next_state)

        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.t_step %UPDATE_EVERY == 0:
            for _ in range(UPDATES_PER_STEP) :
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True, episode_num=0, eps=0.):
        """Returns actions for given state as per current policy."""

        self.state.append(state)
        state = np.array(self.state)
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()  
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            noise_process = np.exp(-episode_num/100.0) * np.random.randn(1, action.shape[1])
            action += noise_process
        return np.clip(action, -1, 1)

    def reset(self, correct=False):
        state = np.zeros((self.state_size))
        for _ in range(self.m):
            self.state.append(state)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_model_weights(self, episode_num=0, log_dir=""):
        """saves Q network weights"""
        self.actor_local.save(episode_num)
        self.critic_local.save(episode_num)
        self.memory.save_experiences(log_dir="ddpg_replay_buffer", episode_num=episode_num)  # saves replay buffer

    def load_model_weights(self, dummy):
        self.actor_local.load()
        self.actor_target.load()
        self.critic_local.load()
        self.critic_target.load()

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) 

        self.state_memory = torch.zeros((buffer_size, 16, state_size), dtype=torch.float32).to(device) #deque(maxlen=buffer_size) 
        self.action_memory = torch.zeros((buffer_size, action_size), dtype=torch.float32).to(device) #deque(maxlen=buffer_size)
        self.reward_memory = torch.zeros((buffer_size, 1)).to(device) #deque(maxlen=buffer_size)
        self.next_state_memory = torch.zeros((buffer_size, 16, state_size), dtype=torch.float32).to(device) #deque(maxlen=buffer_size)
        self.done_memory = torch.zeros((buffer_size, 1)).to(device) #deque(maxlen=buffer_size)

        self.buffer_size = buffer_size
        self.replay_ix = 0  # counts where in the replay buffer we are
        self.buffer_full = False

        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def load_replay_buffer(self):
        '''loads all saved experiences for Neural Fitted
        Q-iteration'''

        # load the saved data
        states = np.vstack([np.load("NFQ_data/states_140.npy"), np.load("NFQ_data/states_100.npy")])
        actions = np.vstack([np.load("NFQ_data/actions_140.npy"), np.load("NFQ_data/actions_100.npy")])
        rewards = np.vstack([np.load("NFQ_data/rewards_140.npy"), np.load("NFQ_data/rewards_100.npy")])
        next_states = np.vstack([np.load("NFQ_data/next_states_140.npy"), np.load("NFQ_data/next_states_100.npy")])
        dones = np.vstack([np.load("NFQ_data/dones_140.npy"), np.load("NFQ_data/dones_100.npy")])




        # set replay buffer equal to these
        self.state_memory = torch.from_numpy(states).to(device)
        self.action_memory = torch.from_numpy(actions).to(device)
        self.reward_memory = torch.from_numpy(rewards).to(device)
        self.next_state_memory = torch.from_numpy(next_states).to(device)
        self.done_memory = torch.from_numpy(dones).to(device)

        self.replay_ix = self.done_memory.shape[0]
        self.buffer_size = self.done_memory.shape[0]
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        #e = self.experience(state, action, reward, next_state, done)
        #self.memory.append(e)
        # new implementation below
        buffer_ix = self.replay_ix %self.buffer_size
        self.state_memory[buffer_ix] = torch.from_numpy(state)
        self.action_memory[buffer_ix] = torch.from_numpy(action)
        self.reward_memory[buffer_ix] = reward
        self.next_state_memory[buffer_ix] = torch.from_numpy(next_state)
        self.done_memory[buffer_ix] = done
        self.replay_ix += 1
        if self.replay_ix %self.buffer_size == 0 and not self.buffer_full:
            self.buffer_full = True
            print("replay buffer full")
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        #experiences = random.sample(self.memory, k=self.batch_size)
        # we need at least 4 future frames to construct the next_state
        batch_ixs = np.random.choice(self.__len__(), self.batch_size)
        #print("\n", batch_ixs, "\n")
        #print("batch indices:", batch_ixs)
        #pdb.set_trace()
        states = self.state_memory[batch_ixs]#torch.stack((states1, states2, states3, states4), dim=1)  # (batch_size, 4, pixels, pixels)
        next_states = self.next_state_memory[batch_ixs]#torch.stack((next_states_1, next_states_2, next_states_3, next_states_4), dim=1)  # (batch_size, 4, pixels, pixels)

        #states = torch.from_numpy(states).float().to(device)
        actions = self.action_memory[batch_ixs]
        #actions = torch.from_numpy(actions).long().to(device)
        rewards = self.reward_memory[batch_ixs]
        #rewards = torch.from_numpy(rewards).float().to(device)
        #next_states = self.next_state_memory[batch_ixs]
        #next_states = torch.from_numpy(next_states).float().to(device)
        dones = self.done_memory[batch_ixs]
        #dones = torch.from_numpy(dones).float().to(device)

        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        #actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        #rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        #next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        #dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        if self.buffer_full:
            return self.buffer_size
        return self.replay_ix
        
    def save_experiences(self, log_dir='', episode_num=0):
        # is the replay buffer full ? then save the entire buffer
        # if its not full just save to the index
        if self.buffer_full:
            ix = self.buffer_size 
        else:
            ix = self.replay_ix
        states = self.state_memory[:ix].cpu().detach().numpy()
        actions = self.action_memory[:ix].cpu().detach().numpy()
        rewards = self.reward_memory[:ix].cpu().detach().numpy()
        next_states = self.next_state_memory[:ix].cpu().detach().numpy()
        dones = self.done_memory[:ix].cpu().detach().numpy()

        with open(log_dir + "states_" + str(episode_num) + ".npy", "wb") as f:
            np.save(f, states)
        with open(log_dir + "actions_" + str(episode_num) + ".npy", "wb") as f:
            np.save(f, actions)
        with open(log_dir + "rewards_" + str(episode_num) + ".npy", "wb") as f:
            np.save(f, rewards)
        with open(log_dir + "next_states_" + str(episode_num) + ".npy", "wb") as f:
            np.save(f, next_states)
        with open(log_dir + "dones_" + str(episode_num) + ".npy", "wb") as f:
            np.save(f, dones)
