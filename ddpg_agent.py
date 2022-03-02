import numpy as np
import random
import copy
import time
from collections import namedtuple, deque

from actor_critic_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
BOOTSTRAP_SIZE = 5      # for n-step bootstrap
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


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, memory,
                    BOOTSTRAP_SIZE=BOOTSTRAP_SIZE,
                    GAMMA=GAMMA,
                    TAU=TAU,
                    LR_ACTOR=LR_ACTOR,
                    LR_CRITIC=LR_CRITIC,
                    UPDATE_EVERY=UPDATE_EVERY,
                    TRANSFER_EVERY=TRANSFER_EVERY,
                    NUM_UPDATE=NUM_UPDATE,
                    ADD_NOISE_EVERY=ADD_NOISE_EVERY,
                    WEIGHT_DECAY=WEIGHT_DECAY):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            num_agents: number of running agents
            memory: instance of ReplayBuffer
            Actor: a class inheriting from torch.nn.Module that define the structure of the actor neural network
            Critic: a class inheriting from torch.nn.Module that define the structure of the critic neural network
            device: cpu or cuda:0 if available
            BOOTSTRAP_SIZE: length of the bootstrap
            GAMMA: discount factor
            TAU: for soft update of target parameters
            LR_CRITIC: learning rate of the critics
            LR_ACTOR: learning rate of the actors
            UPDATE_EVERY: how often to update the networks
            TRANSFER_EVERY: after how many update do we transfer from the online network to the targeted fixed network
            ADD_NOISE_EVERY: how often to add noise to favor exploration
        """
        # Actor networks
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.actor_target = Actor(state_size, action_size, seed).to(device)

        # Critic networks
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic_target = Critic(state_size, action_size, seed).to(device)

        # Ensure that at the begining, both target and local are having the same parameters
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)

        # Noise
        self.noise = None

        # Replay memory
        self.memory = memory
        # Initialize time steps (for updating every UPDATE_EVERY steps)
        self.u_step = 0
        self.n_step = 0

        self.BOOTSTRAP_SIZE = BOOTSTRAP_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LR_CRITIC = LR_CRITIC
        self.LR_ACTOR = LR_ACTOR
        self.UPDATE_EVERY = UPDATE_EVERY
        self.TRANSFER_EVERY = TRANSFER_EVERY
        self.NUM_UPDATE = NUM_UPDATE
        self.ADD_NOISE_EVERY = ADD_NOISE_EVERY

        # for bootstrap purpose
        self.rewards = deque(maxlen=BOOTSTRAP_SIZE)
        self.states = deque(maxlen=BOOTSTRAP_SIZE)
        self.actions = deque(maxlen=BOOTSTRAP_SIZE)
        self.gammas = np.array([GAMMA ** i for i in range(BOOTSTRAP_SIZE)])

    def reset(self):
        if self.noise:
            self.noise.reset()

    def set_noise(self, noise):
        self.noise = noise

    def act(self, state, add_noise=True):
        """Returns actions of each actor for given states.

        Params
        ======
            state (array_like): current states
        """
        ret = None

        self.n_step = (self.n_step + 1) % self.ADD_NOISE_EVERY

        with torch.no_grad():
            if add_noise and self.noise and self.n_step == 0:
                self.actor_local.eval()
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                to_add = self.noise.apply(self.actor_local, state)
                if ret is None:
                    ret = to_add
                else:
                    ret = np.concatenate((ret, to_add))
                self.actor_local.train()
            else:
                self.actor_local.eval()
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                to_add = self.actor_local(state).cpu().data.numpy()
                if ret is None:
                    ret = to_add
                else:
                    ret = np.concatenate((ret, to_add))
                self.actor_local.train()
        return ret

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory

        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        if len(self.rewards) == self.BOOTSTRAP_SIZE:
            reward = np.sum(self.rewards * self.gammas)
            self.memory.add(self.states[0], self.actions[0], reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.u_step = (self.u_step + 1) % self.UPDATE_EVERY
        if len(self.memory) > self.memory.batch_size and self.u_step == 0:
            t_step = 0
            for _ in range(self.UPDATE_LOOP):
                self.learn()
                t_step = (t_step + 1) % self.TRANSFER_EVERY
                if t_step == 0:
                    self.soft_update(self.actor_local, self.actor_target, self.TAU)
                    self.soft_update(self.critic_local, self.critic_target, self.TAU)

    def learn(self):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """
        # shuffle all memory to disrupt the internal correlation and learn from all of them
        states, actions, rewards, next_states, dones = self.memory.sample()

        # The critic should estimate the value of the states to be equal to rewards plus
        # the estimation of the next_states value according to the critic_target and actor_target
        with torch.no_grad():
            self.actor_target.eval()
            next_actions = self.actor_local(next_states)
            self.critic_target.eval()
            targeted_value = rewards + (self.GAMMA**self.BOOTSTRAP_SIZE)*self.critic_target(next_states, next_actions)*(1 - dones)
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

    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        θ_target = θ_target + τ*(θ_local - θ_target)
        θ_local = r + gamma * θ_local(s+1)
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        # this is transferring gradually the parameters of the online Q Network to the fixed one
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ActionNoise:
    """ Noise generator that disturb the output of a network """
    def __init__(self, size, device, seed, mu=0., theta=0.15, sigma=0.2):
        self.noise = OUNoise(size, device, seed, mu, theta, sigma)

    def reset(self):
        self.noise.reset()

    def apply(self, model, state):
        action = model(state).cpu().data.numpy()
        action += self.noise.sample()
        return np.clip(action, -1, 1)


class ParameterNoise:
    """ Noise generator that disturb the weight of a network and thus its output """
    def __init__(self, model, device, seed, mu=0., theta=0.15, sigma=0.2):
        size = sum([np.array(param.data.size()).prod() for param in model.parameters()])
        self.noise = OUNoise(size, device, seed)

    def reset(self):
        self.noise.reset()

    def apply(self, model, state):
        noise_sample = self.noise.sample()
        # apply a noise to the parameters only for exploration purpose
        self.apply_noise(model, noise_sample)
        ret = model(state).cpu().data.numpy()
        # restore the previous parameters otherwise the noise will disturb the acquired knowldege
        self.apply_noise(model, -noise_sample)
        return ret

    def apply_noise(self, model, noise_sample):
        start = torch.tensor(0)
        for param in model.parameters():
            size = np.array(param.data.size()).prod().item()
            noise = noise_sample[start:start + size].reshape(param.data.size()).float()
            param.data.add_(noise)
            start += size


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, device, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.device = device
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return torch.from_numpy(self.state).to(self.device).requires_grad_(False)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, device, buffer_size, batch_size, seed):
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
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        return self.to_tensor(experiences)

    def shuffle_all(self):
        temp = list(self.memory)
        random.shuffle(temp)
        batch_count = int(len(temp) / self.batch_size)
        for a in range(batch_count):
            yield self.to_tensor(temp[a:(a+1)*self.batch_size])

    def to_tensor(self, experiences):
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device).requires_grad_(False)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device).requires_grad_(False)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device).requires_grad_(False)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device).requires_grad_(False)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device).requires_grad_(False)
        return (states, actions, rewards, next_states, dones)
