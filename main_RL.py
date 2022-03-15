import numpy as np
import gym
import torch
import os
import h5py
from ddpg_agent import DDPG_agent

# Helper functions to save all the data of a run


def create_info_h5(agent, env):
    # Check if file exist and creat it
    i = 0
    while os.path.exists(f'run_{i}.hdf5'):
        i += 1
    dataFile = h5py.File(f'run_{i}.hdf5', 'a')

    # Create dataset to store info in hdf5 file
    info = {'alg': agent.name, 'env': env.unwrapped.spec.id}
    st = h5py.string_dtype(encoding='utf-8')
    dataFile.create_dataset('info', dtype=st)
    for k in info.keys():
        dataFile['info'].attrs[k] = info[k]

    # Create dataset to store hyperparams of the model in hdf5 file
    hyperparams = {'batch_size': agent.batch_size, 'bootstrap_size': agent.bootstrap_size
                   , 'gamma': agent.gamma, 'tau': agent.tau, 'lr_critic': agent.lr_critic
                   , 'lr_actor': agent.lr_actor, 'update_every': agent.update_every
                   , 'transfer_every': agent.transfer_every, 'num_update': agent.num_update 
                   , 'add_noise_every': agent.add_noise_every}
    dataFile.create_dataset('hyperparams', dtype='f')
    for k in hyperparams.keys():
        dataFile['hyperparams'].attrs[k] = hyperparams[k]

    # Create group for rewards, energies, princip dims, actor and critic model
    dataFile.create_group('sigmas')
    dataFile.create_group('rewards')
    dataFile.create_group('energies')
    dataFile.create_group('princip_dims')
    dataFile.create_group('actor_models')
    dataFile.create_group('critic_models')

    return dataFile


def save_all(dat_file, i_ep, sigmas_i_ep, rew_i_ep, en_i_ep, pri_dim_i_ep, act_model_i_ep, cr_model_i_ep):
    # Create datasets for rewards, energies, pri dim and store data in it
    dat_file['sigmas'].create_dataset(f'sigmas_ep_{i_ep}', dtype='f', data=sigmas_i_ep)
    dat_file['rewards'].create_dataset(f'rew_ep_{i_ep}', dtype='f', data=rew_i_ep)
    dat_file['energies'].create_dataset(f'en_ep_{i_ep}', dtype='f', data=en_i_ep)
    dat_file['princip_dims'].create_dataset(f'pri_dim_ep_{i_ep}', dtype='i', data=pri_dim_i_ep)

    # Store in actor models group the network params at each ep
    actor_model = torch.load(act_model_i_ep)
    dat_file['actor_models'].create_dataset(f'act_mod_{i_ep}', dtype='f')
    for k in actor_model.keys():
        dat_file['actor_models'][f'act_mod_{i_ep}'].attrs.create(name=k, data=actor_model[k].numpy())

    # Store in actor models group the network params at each ep
    critic_model = torch.load(cr_model_i_ep)
    dat_file['critic_models'].create_dataset(f'cri_mod_{i_ep}', dtype='f')
    for k in critic_model.keys():
        dat_file['critic_models'][f'cri_mod_{i_ep}'].attrs.create(name=k, data=critic_model[k].numpy())


def close_file(dat_file, actor_model_file, critic_model_file):
    dat_file.close()
    os.remove(actor_model_file)
    os.remove(critic_model_file)


# Env declaration
env = gym.make('svm_env:svmEnv-v1', file_sigmas="./svmCodeSVD/sigmas1.dat")
# Instance of the ddpg agent
agent = DDPG_agent(state_size=1, action_size=3, seed=2)


def run_ddpg(max_t_step=250, n_episodes=400):

    # Create h5 file and store info about alg and its hypereparams
    dat_file = create_info_h5(agent, env)

    for i_ep in range(n_episodes):
        state = env.reset()
        agent.reset()
        rew_i_ep = []
        en_i_ep = []
        pri_dim_i_ep = []

        # Training loop of each episode
        for t_step in range(max_t_step):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state

            # Save rew, energies, princip dims, act and crit models
            rew_i_ep.append(reward)
            en_i_ep.append(state[0])
            pri_dim_i_ep.append(env.princp_dim)
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            if done:
                break

        # Save data during training (to not lose the work done)
        save_all(dat_file=dat_file, i_ep=int(i_ep), sigmas_i_ep=env.actions_taken
                 , rew_i_ep=rew_i_ep, en_i_ep=en_i_ep, pri_dim_i_ep=pri_dim_i_ep
                 , act_model_i_ep='checkpoint_actor.pth', cr_model_i_ep='checkpoint_critic.pth')

        print('Episode {} ... Score: {:.3f}'.format(i_ep, np.sum(rew_i_ep)))

    close_file(dat_file, 'checkpoint_actor.pth', 'checkpoint_critic.pth')
    return dat_file
