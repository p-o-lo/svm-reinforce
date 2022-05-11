import numpy as np
import gym
import torch
import os
import pickle

from ddpg_agent import DDPG_agent

env = gym.make('svm_env:svmEnv-v2', n_pairs=3, n_basis=250, file_sigmas="./svmCodeSVD/sigmas6.dat")
obs_space = env.observation_space
state_size = env.observation_space.shape[-1]
act_space = env.action_space.shape
act_size = env.action_space.shape[0]*env.action_space.shape[-1]
state = env.reset()


# Save all rewards, energies and princip dims in files during episode training
def create_run_fold_and_info(agent, env):

    # Check if folder exist and creat it
    i = 0
    while os.path.exists(f'runs_optim_envs/run_{i}/'):
        i += 1
    name_dir = f'runs_optim_envs/run_{i}/'
    os.makedirs(name_dir)

    # Create info.p to store info in pickle file
    info = {'alg': agent.name, 'env': env.unwrapped.spec.id, 'basis_size': env.n_basis,
            'batch_size': agent.batch_size, 'bootstrap_size': agent.bootstrap_size,
            'gamma': agent.gamma, 'tau': agent.tau, 'lr_critic': agent.lr_critic,
            'lr_actor': agent.lr_actor, 'update_every': agent.update_every,
            'transfer_every': agent.transfer_every, 'num_update': agent.num_update,
            'add_noise_every': agent.add_noise_every}

    pickle.dump(info, open(name_dir+'info.p', 'wb'))
    return name_dir


def save_all(name_run_dir, i_ep, sigmas_i_ep, rew_i_ep, en_i_ep, pri_dim_i_ep,
        full_dim_i_ep, act_model_i_ep, cr_model_i_ep):

    pickle.dump(sigmas_i_ep, open(name_run_dir+f'sigmas_{i_ep}.p', 'wb'))
    pickle.dump(rew_i_ep, open(name_run_dir+f'rew_{i_ep}.p', 'wb'))
    pickle.dump(en_i_ep, open(name_run_dir+f'en_{i_ep}.p', 'wb'))
    pickle.dump(pri_dim_i_ep, open(name_run_dir+f'pri_dim_{i_ep}.p', 'wb'))
    pickle.dump(full_dim_i_ep, open(name_run_dir+f'full_dim_{i_ep}.p', 'wb'))
    pickle.dump(act_model_i_ep, open(name_run_dir+f'act_model_{i_ep}.p', 'wb'))
    pickle.dump(cr_model_i_ep, open(name_run_dir+f'cr_model_{i_ep}.p', 'wb'))


def rm_useless_file(actor_model_file, critic_model_file, file_sigmas):
    os.remove(actor_model_file)
    os.remove(critic_model_file)
    os.remove(file_sigmas)


agent = DDPG_agent(state_size, act_size, seed=0)
actor_model_file = 'checkpoint_actor6.pth'
critic_model_file = 'checkpoint_critic6.pth'


# Run ddpg algs
def run_ddpg(max_t_step=200, n_episodes=600):

    # Create h5 file and store info about alg and its hypereparams
    name_run_dir = create_run_fold_and_info(agent, env)

    for i_ep in range(n_episodes):
        state = env.reset()
        agent.reset()
        rew_i_ep = []
        en_i_ep = []
        pri_dim_i_ep = []
        full_dim_i_ep = []
        action_i_episode = []

        # Training loop of each episode
        for t_step in range(max_t_step):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action.reshape((env.n_basis, env.n_pairs)))
            agent.step(state, action, reward, next_state, done)
            state = next_state

            # Save rew, energies, princip dims, act and crit models
            action_i_episode.append(action.reshape((env.n_basis, env.n_pairs)))
            rew_i_ep.append(reward)
            en_i_ep.append(state[0])
            pri_dim_i_ep.append(env.princp_dim)
            full_dim_i_ep.append(env.full_dim)
            torch.save(agent.actor_local.state_dict(), actor_model_file)
            torch.save(agent.critic_local.state_dict(), critic_model_file)
            if done:
                break

        # Save data during training (to not lose the work done) and remove useless
        save_all(name_run_dir=name_run_dir, i_ep=int(i_ep), sigmas_i_ep=action_i_episode,
                rew_i_ep=rew_i_ep, en_i_ep=en_i_ep, pri_dim_i_ep=pri_dim_i_ep,
                full_dim_i_ep=full_dim_i_ep, act_model_i_ep=actor_model_file,
                cr_model_i_ep=critic_model_file)

        rm_useless_file(actor_model_file, critic_model_file, env.file_sigmas)

        print('Episode {} ... Score: {:.3f}'.format(i_ep, np.sum(rew_i_ep)))

    return name_run_dir


all_data = run_ddpg()
