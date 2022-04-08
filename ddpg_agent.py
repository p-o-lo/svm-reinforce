import numpy as np
import random
import copy
from collections import namedtuple, deque

from actor_critic_model_ddpg import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 1          # minibatch size
BOOTSTRAP_SIZE = 1      # for n-step bootstrap
GAMMA = 1.0             # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
UPDATE_EVERY = 1        # update every c-step
TRANSFER_EVERY = 1      # transfer params every
NUM_UPDATE = 1          # num update per step
ADD_NOISE_EVERY = 1     # add noise every step
WEIGHT_DECAY = 0        # L2 weight decay


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG_agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed,
                name='DDPG',
                batch_size=BATCH_SIZE,
                bootstrap_size=BOOTSTRAP_SIZE,
                gamma=GAMMA,
                tau=TAU,
                lr_critic=LR_ACTOR,
                lr_actor=LR_CRITIC,
                update_every=UPDATE_EVERY,
                transfer_every=TRANSFER_EVERY,
                num_update=NUM_UPDATE,
                add_noise_every=ADD_NOISE_EVERY,
                weight_decay=WEIGHT_DECAY):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            bootstrap_size: length of the bootstrap
            gamma: discount factor
            tau: for soft update of target parameters
            lr_critic: learning rate of the critics
            lr_actor: learning rate of the actors
            update_every: how often to update the networks
            transfer_every: after how many update do we transfer from the online network to the targeted fixed network
            num_update: how many times update the network parameters at each time step
            add_noise_every: how often to add noise to favor exploration
            weight_decay: decay of network parameters
        """
        self.name = name
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.bootstrap_size = bootstrap_size
        self.gamma = gamma
        self.tau = tau
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.update_every = update_every
        self.transfer_every = transfer_every
        self.num_update = num_update
        self.add_noise_every = add_noise_every

        # for bootstrap purpose
        self.rewards = deque(maxlen=bootstrap_size)
        self.states = deque(maxlen=bootstrap_size)
        self.actions = deque(maxlen=bootstrap_size)
        self.gammas = np.array([gamma ** i for i in range(bootstrap_size)])

        # Actor networks
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.actor_target = Actor(state_size, action_size, seed).to(device)

        # Critic networks
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)
        self.critic_target = Critic(state_size, action_size, seed).to(device)

        # Ensure that at the begining, both target and local are having the same parameters
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)

        # Noise
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, batch_size, seed)

        # Initialize time steps (for updating every UPDATE_EVERY steps)
        self.u_step = 0
        self.n_step = 0

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        self.n_step = (self.n_step + 1) % self.add_noise_every

        with torch.no_grad():
            self.actor_local.eval()
            state = torch.from_numpy(state).float().to(device)
            action = self.actor_local(state).cpu().data.numpy()
            # Add noise each add noise_every
            if add_noise and self.n_step == 0:
                action += self.noise.sample()
            self.actor_local.train()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # Save experience in replay memory
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        if len(self.rewards) == self.bootstrap_size:
            reward = np.sum(self.rewards * self.gammas)
            self.memory.add(self.states[0], self.actions[0], reward, next_state, done)

        # Learn every UPDATE_EVERY time steps from a memory of size BATCH_SIZE.
        # Transfer parameters every TRANSFER EVERY
        self.u_step = (self.u_step + 1) % self.update_every
        if len(self.memory) > self.batch_size and self.u_step == 0:
            t_step = 0
            for _ in range(self.num_update):
                self.learn()
                t_step = (t_step + 1) % self.transfer_every
                if t_step == 0:
                    self.soft_update(self.actor_local, self.actor_target, self.tau)
                    self.soft_update(self.critic_local, self.critic_target, self.tau)

    def learn(self):
        """Update value parameters using given batch of experience tuples.
        """
        # shuffle all memory to disrupt the internal correlation and learn from all of them
        states, actions, rewards, next_states, dones = self.memory.sample()

        # The critic should estimate the value of the states to be equal to rewards plus
        # the estimation of the next_states value according to the critic_target and actor_target
        with torch.no_grad():
            self.actor_target.eval()
            next_actions = self.actor_local(next_states)
            self.critic_target.eval()
            targeted_value = rewards + (self.gamma**self.bootstrap_size)*self.critic_target(next_states, next_actions)*(1 - dones)
        current_value = self.critic_local(states, actions)

        # calculate the loss
        self.critic_optim.zero_grad()
        loss = F.mse_loss(current_value, targeted_value)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actions_pred = self.actor_local(states)
        mean = self.critic_local(states, actions_pred).mean()
        (-mean).backward()
        self.actor_optim.step()

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
        # np.array([random.random() for i in range(len(x))])
        # np.random.standard_normal(self.size)
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
