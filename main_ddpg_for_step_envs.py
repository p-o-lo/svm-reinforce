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
    while os.path.exists(f'runs_step_envs/run_{i}.hdf5'):
        i += 1
    dataFile = h5py.File(f'runs_step_envs/run_{i}.hdf5', 'a')

    # Create dataset to store info in hdf5 file
    info = {'alg': agent.name, 'env': env.unwrapped.spec.id}
    st = h5py.string_dtype(encoding='utf-8')
    dataFile.create_dataset('info', dtype=st)
    for k in info.keys():
        dataFile['info'].attrs[k] = info[k]

    # Create dataset to store hyperparams of the model in hdf5 file
    hyperparams = {'batch_size': agent.batch_size, 'bootstrap_size': agent.bootstrap_size,
            'gamma': agent.gamma, 'tau': agent.tau, 'lr_critic': agent.lr_critic,
            'lr_actor': agent.lr_actor, 'update_every': agent.update_every,
            'transfer_every': agent.transfer_every, 'num_update': agent.num_update,
            'add_noise_every': agent.add_noise_every}
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

    # Close and return data file name
    dataFile_name = dataFile.filename
    dataFile.close()

    return dataFile_name


def save_all(dat_file_name, i_ep, sigmas_i_ep, rew_i_ep, en_i_ep, pri_dim_i_ep, act_model_i_ep, cr_model_i_ep):
    # Open data file
    dat_file = h5py.File(dat_file_name, 'a')

    # Create datasets for rewards, energies, pri dim and store data in it
    dat_file['sigmas'].create_dataset(f'sigmas_ep_{i_ep}', dtype='f', data=sigmas_i_ep)
    dat_file['rewards'].create_dataset(f'rew_ep_{i_ep}', dtype='f', data=rew_i_ep)
    dat_file['energies'].create_dataset(f'en_ep_{i_ep}', dtype='f', data=en_i_ep)
    dat_file['princip_dims'].create_dataset(f'pri_dim_ep_{i_ep}', dtype='i', data=pri_dim_i_ep)

    # Store in actor models group the network params at each ep
    actor_model = torch.load(act_model_i_ep)
    dat_file['actor_models'].create_dataset(f'act_mod_{i_ep}', dtype='f')
    for k in actor_model.keys():
        dat_file['actor_models'][f'act_mod_{i_ep}'].attrs.create(name=k, data=actor_model[k].cpu().data.numpy())

    # Store in actor models group the network params at each ep
    critic_model = torch.load(cr_model_i_ep)
    dat_file['critic_models'].create_dataset(f'cri_mod_{i_ep}', dtype='f')
    for k in critic_model.keys():
        dat_file['critic_models'][f'cri_mod_{i_ep}'].attrs.create(name=k, data=critic_model[k].cpu().data.numpy())

    # Close data file
    dat_file.close()


def rm_useless_files(actor_model_file, critic_model_file, file_sigmas):
    os.remove(actor_model_file)
    os.remove(critic_model_file)
    os.remove(file_sigmas)

# Env declaration and print its features


env = gym.make('svm_env:svmEnv-v0', file_sigmas="./svmCodeSVD/sigmas1.dat")

obs_space = env.observation_space

state_size = env.observation_space.shape[-1]

act_space = env.action_space

act_size = env.action_space.shape[-1]

state = env.reset()


# Instance of the ddpg agent
agent = DDPG_agent(state_size, act_size, seed=0)

actor_model_file = 'checkpoint_actor1.pth'
critic_model_file = 'checkpoint_critic1.pth'


def run_ddpg(max_t_step=300, n_episodes=600):

    # Create h5 file and store info about alg and its hypereparams
    dat_file_name = create_info_h5(agent, env)

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
            torch.save(agent.actor_local.state_dict(), actor_model_file)
            torch.save(agent.critic_local.state_dict(), critic_model_file)
            if done:
                break

        # Save data during training (to not lose the work done) and remove useless
        save_all(dat_file_name=dat_file_name, i_ep=int(i_ep), sigmas_i_ep=env.actions_taken,
                rew_i_ep=rew_i_ep, en_i_ep=en_i_ep, pri_dim_i_ep=pri_dim_i_ep,
                act_model_i_ep=actor_model_file, cr_model_i_ep=critic_model_file)

        rm_useless_files(actor_model_file, critic_model_file, env.file_sigmas)

        print('Episode {} ... Score: {:.3f}'.format(i_ep, np.sum(rew_i_ep)))

    return dat_file_name


# Run
all_data = run_ddpg()
