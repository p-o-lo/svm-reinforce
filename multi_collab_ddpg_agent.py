import numpy as np
import random
import copy
from collections import namedtuple, deque

from actor_critic_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 6e-2              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1        # Update every time step
NUM_UPDATE = 1          # Update once at each time step

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.t_step = 0

        # Actor Network 0 (w/ Target Network)
        self.actor_local0 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target0 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer0 = optim.Adam(self.actor_local0.parameters(), lr=LR_ACTOR)

        # Actor Network 1 (w/ Target Network)
        self.actor_local1 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target1 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer1 = optim.Adam(self.actor_local1.parameters(), lr=LR_ACTOR)

        # Critic Network 0 (w/ Target Network)
        self.critic_local0 = Critic(state_size * 2, action_size * 2, random_seed).to(device)
        self.critic_target0 = Critic(state_size * 2, action_size * 2, random_seed).to(device)
        self.critic_optimizer0 = optim.Adam(self.critic_local0.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Critic Network 1 (w/ Target Network)
        self.critic_local1 = Critic(state_size * 2, action_size * 2, random_seed).to(device)
        self.critic_target1 = Critic(state_size * 2, action_size * 2, random_seed).to(device)
        self.critic_optimizer1 = optim.Adam(self.critic_local1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise((2, action_size), random_seed)

        # Replay memory 0
        self.memory0 = ReplayBuffer(action_size, BUFFER_SIZE, int(BATCH_SIZE/2), random_seed)

        # Replay memory 1
        self.memory1 = ReplayBuffer(action_size, BUFFER_SIZE, int(BATCH_SIZE/2), random_seed)

    def reset(self):
        self.noise.reset()

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy with noise"""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local0.eval()
        self.actor_local1.eval()
        with torch.no_grad():
            action0 = self.actor_local0(states[0]).cpu().data.numpy()
            action1 = self.actor_local1(states[1]).cpu().data.numpy()
        self.actor_local0.train()
        self.actor_local1.train()
        if add_noise:
            actions = np.array((action0, action1)).reshape((2,2))
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory0.add(states[0], actions[0], rewards[0], next_states[0], dones[0])
        self.memory1.add(states[1], actions[1], rewards[1], next_states[1], dones[1])

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if (len(self.memory0) + len(self.memory1) > BATCH_SIZE) and self.t_step == 0:
            for _ in range(NUM_UPDATE):
                experiences0 = self.memory0.sample()
                experiences1 = self.memory1.sample()
                self.learn(experiences0, experiences1, GAMMA)

    def learn(self, experiences0, experiences1, gamma):
        """Update policy and value parameters using given batch of experience tuples.
         Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
         where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences0 (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) of the agent 0 tuples 
            experiences1 (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) of the agent 1 tuples
            gamma (float): discount factor
        """
        states0, actions0, rewards0, next_states0, dones0 = experiences0
        states1, actions1, rewards1, next_states1, dones1 = experiences1


        states = torch.cat((states0, states1), dim=1).float().to(device)
        actions = torch.cat((actions0, actions1), dim=1).float().to(device)
        next_states = torch.cat((next_states0, next_states1), dim=1).float().to(device)
    
        actions_next0 = self.actor_target0(states0)
        actions_next1 = self.actor_target1(states1)
        actions_next = torch.cat((actions_next0, actions_next1), dim=1).float().to(device)
        
        # ---------------------------- update critic 0 ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        Q_targets_next0 = self.critic_target0(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets0 = rewards0 + (gamma * Q_targets_next0 * (1 - dones0))
        # Compute critic loss
        Q_expected0 = self.critic_local0(states, actions)
        critic_loss0 = F.mse_loss(Q_expected0, Q_targets0)
        # Minimize the loss
        self.critic_optimizer0.zero_grad()
        critic_loss0.backward(retain_graph=True)
        self.critic_optimizer0.step()

        # ---------------------------- update critic 1---------------------------- #
        # Get predicted next-state actions and Q values from target models
        Q_targets_next1 = self.critic_target1(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets1 = rewards1 + (gamma * Q_targets_next1 * (1 - dones1))
        # Compute critic loss
        Q_expected1 = self.critic_local1(states, actions)
        critic_loss1 = F.mse_loss(Q_expected1, Q_targets1)
        # Minimize the loss
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward(retain_graph=True)
        self.critic_optimizer1.step()


        actions_pred0 = self.actor_local0(states0)
        actions_pred1 = self.actor_local1(states1)
        actions_pred = torch.cat((actions_pred0, actions_pred1), dim=1).float().to(device)
        
        # ---------------------------- update actor 0 ---------------------------- #
        # Compute actor loss
        actor_loss0 = -self.critic_local0(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer0.zero_grad()
        actor_loss0.backward(retain_graph=True)
        self.actor_optimizer0.step()

        # ---------------------------- update actor 1 ---------------------------- #
        # Compute actor loss
        actor_loss1 = -self.critic_local1(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer1.zero_grad()
        actor_loss1.backward(retain_graph=True)
        self.actor_optimizer1.step()


        # ----------------------- update target networks 0 ----------------------- #
        self.soft_update(self.critic_local0, self.critic_target0, TAU)
        self.soft_update(self.actor_local0, self.actor_target0, TAU)                     

        # ----------------------- update target networks 1 ----------------------- #
        self.soft_update(self.critic_local1, self.critic_target1, TAU)
        self.soft_update(self.actor_local1, self.actor_target1, TAU)                     

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

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()
       
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size) 
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
