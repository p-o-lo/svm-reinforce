import random
import numpy as np
import copy

from actor_critic_model_ppo import Actor, Critic

import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import MultivariateNormal

BUFFER_SIZE = int(1e5)  # replay buffer size
CLIP = 0.2              # clip for surrogate loss of ppo
LAMBDA = 0.95           # for generalized advantage estimation
GAMMA = 0.99            # discount factor
LR_ACTOR = 3e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
NUM_UPDATE = 10         # num update per step
ADD_NOISE_EVERY = 1     # add noise every step
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO_agent():

    def __init__(self, state_size, action_size, seed,
                name='PPO',
                clip=CLIP,
                lambda_gae=LAMBDA,
                gamma=GAMMA,
                lr_critic=LR_ACTOR,
                lr_actor=LR_CRITIC,
                num_update=NUM_UPDATE,
                add_noise_every=ADD_NOISE_EVERY,
                weight_decay=WEIGHT_DECAY):

        """Initialize an Agent object.

            Params
            ======
                env: the env to train on
                state_size (int): dimension of each state
                action_size (int): dimension of each action
                seed (int): random seed
                clip: clip for surrogate loss of ppo
                lambda_gae: lambda for GAE
                gamma: discount factor
                lr_critic: learning rate of the critics
                lr_actor: learning rate of the actors
                num_update: how many times update the network parameters at each time step
                add_noise_every: how often to add noise to favor exploration
                weight_decay: decay of network parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.seed = random.seed(seed)
        self.clip = clip
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.num_update = num_update
        self.add_noise_every = add_noise_every

        # Actor networks
        self.actor_local = Actor(self.state_size, self.action_size, seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic networks
        self.critic_local = Critic(self.state_size, seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Initialize time steps and k iteration
        self.k_step = 0
        self.t_step = 0

        # Initialize noise
        self.noise = OUNoise(action_size, seed)

    def compute_return_fut(self, all_rews):
        """Compute the Reward-To-Go of each timestep in a batch given the rewards.

        Params
            ======
                all_rews: the rewards in all trajectories, Shape: (# of traj , number of timesteps per trajectory)
            Return
            =====
                rews_t_future: the future returns of each traj, Shape: (# of traj )

        """
        rews_t_future = []

        for rew_traj in reversed(all_rews):
            discounted_rew = 0

            for rew in reversed(rew_traj):
                discounted_rew = rew + discounted_rew * self.gamma
                rews_t_future.insert(0, discounted_rew)

        # Convert the rewards to future into a tensor
        rews_t_future = torch.tensor(rews_t_future, dtype=torch.float).to(device)

        return rews_t_future

    def act(self, state):
        """Queries an action from the actor network, should be called from collect_trajs.
           It uses a multivariate gaussian with mean the output of the actor network.

            Params
            ======
                state: the observation at the current timestep
            Return
            ======
                action: the action to take, as a numpy array
                log_pol: the log of the policy of the selected action in the distribution
        """

        # Calculate the action from the actor network
        state = torch.from_numpy(state).float().to(device)
        mean = self.actor_local(state)

        # Build a MultivariateNormal dist for log policy
        cov_var = torch.full(size=(self.action_size,), fill_value=0.5)
        cov_mat = torch.diag(cov_var)
        dist = MultivariateNormal(mean, cov_mat)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        log_pol = dist.log_prob(action)

        return action.detach().numpy(), log_pol.detach()

    def value_func(self, trajs_state, trajs_acts):
        """Estimate the value of each state and the log pol
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

            Params
            ======
                batch_obs: the observations from the most recently collected batch as a tensor. Shape: (number of timesteps in batch, dimension of observation)
                batch_acts: the actions from the most recently collected batch as a tensor. Shape: (number of timesteps in batch, dimension of action)
            Return
            =====
                V: the predicted values of traj_states
                log_pols: the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic_local(trajs_state).squeeze()

        # Calculate the log policies of actions in trajs using most recent actor network.
        # This segment of code is similar to that in act()
        means = self.actor_local(trajs_state)

        # Build a MultivariateNormal dist for log policy
        cov_var = torch.full(size=(self.action_size,), fill_value=0.5)
        cov_mat = torch.diag(cov_var)
        dist = MultivariateNormal(means, cov_mat)
        log_pols = dist.log_prob(trajs_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_pols

    def advantage_gae(self, all_rews, trajs_states, trajs_acts):
        """ Estimate the advantage function with generalized advantage
            esitmation

            Params
            ======
                all_rews: the list of all rewards for all trajectories

            Return
            ======
                A: a list with values of the advantage functions
        """

        # Calculate the value function
        value_func, _ = self.value_func(trajs_states, trajs_acts)
        value_func = value_func.detach()

        # initilization advantage an index of value func
        A = []
        j = 0

        for rew_traj in reversed(all_rews):

            advantage = 0

            for rew in reversed(rew_traj):
                if j == 0:
                    delta = value_func[-1-j]
                else:
                    delta = self.gamma * value_func[len(value_func) - j] - value_func[-1-j]
                advantage = rew + delta + (self.gamma * self.lambda_gae) * advantage
                A.insert(0, advantage)
                j += 1

        # Convert adavantage to a tensor
        A = torch.tensor(A, dtype=torch.float).to(device)

        return A.detach()

    def step(self, trajs_states, trajs_acts, trajs_log_pol, all_rews, lens_trajs):

        # Increment t step as the length of the total trajs and the k iter
        self.t_step += np.sum(lens_trajs)
        self.k_step += 1

        # Calculate the advantage function at k-th step (without gae lambda = 1)
        rews_t_future = self.compute_return_fut(all_rews)
        # value_func, _ = self.value_func(trajs_states, trajs_acts)
        # advantage = rews_t_future - value_func.detach()

        # Calculate the advantage function at k-th step (with gae lambda != 1)
        advantage = self.advantage_gae(all_rews, trajs_states, trajs_acts)

        # Normalizing advantages isn't theoretically necessary, but in practice
        # it decreases the variance of our advantages and makes convergence much
        # more stable and faster.
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

        # The training loop of the network
        for _ in range(self.num_update):
            self.learn(trajs_states, trajs_acts, trajs_log_pol, rews_t_future, advantage)

    def learn(self, trajs_states, trajs_acts, trajs_log_pol, rew_t_future, advantage):
        """Train the actor and critic networks. Here is where the main PPO algorithm resides.

                Params
                ======
                    total_timesteps: the total number of timesteps to train for
                Return
                ======
                    None
        """

        # Calculate the value function and current log policy
        V, curr_log_pol = self.value_func(trajs_states, trajs_acts)

        # Calcualte the ratio between the current policy and the k_th
        ratio = torch.exp(curr_log_pol - trajs_log_pol)

        # Calculate the surrogate loss function
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage

        # Calculate the loss for both actor and critic NN
        actor_loss = (-torch.min(surr1, surr2)).mean()
        critic_loss = nn.MSELoss()(V, rew_t_future)

        # Calculate gradients and perform backpropagation for actor network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # Calculate gradients and perform backpropagation for critic network
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()


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
