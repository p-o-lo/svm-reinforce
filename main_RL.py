import numpy as np
import gym
import torch

from ddpg_agent import Agent


env = gym.make('svm_env:svmEnv-v0', file_sigmas="./svmCodeSVD/sigmas.dat")
# Instance of the ddpg agent
agent = Agent(1, 3, random_seed=2)


# Save all rewards, energies and princip dims in files during training
def save_all(agent, rewards, energies, princip_dims):
    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

    name_rewards = 'rewards_RL_0.out'
    file_rewards = open(name_rewards, 'w')
    np.savetxt(file_rewards, rewards, fmt='%f')
    file_rewards.close()

    name_energies = 'energies_RL_0.out'
    file_energies = open(name_energies, 'w')
    np.savetxt(file_energies, energies, fmt='%f')
    file_energies.close()

    name_dim = 'princip_dims_RL_0.out'
    file_dim = open(name_dim, 'w')
    np.savetxt(file_dim, princip_dims, fmt='%f')
    file_dim.close()


def run_ddpg(max_t_step=300, n_episodes=700):
    # Inizialization
    rewards = []
    energies = []
    princip_dims = []

    for i_episode in range(n_episodes):
        state = env.reset()
        agent.reset()
        rew_per_i_episode = []
        energies_per_i_episode = []
        princip_dim_per_i_episode = []

        # Training loop of each episode
        for t_step in range(max_t_step):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state

            # Save
            rew_per_i_episode.append(reward)
            energies_per_i_episode.append(state[0])
            princip_dim_per_i_episode.append(env.princp_dim)
            if done:
                break

        # Save data during training (to not lose the work done)
        rewards.append(rew_per_i_episode)
        energies.append(energies_per_i_episode)
        princip_dims.append(princip_dim_per_i_episode)

        save_all(agent, rewards, energies, princip_dims)

        print('Episode {} ... Score: {:.3f}'.format(i_episode, np.sum(rewards[i_episode])))

    return rewards, energies, princip_dims


all_rewards, all_energies, all_princip_dim = run_ddpg()
