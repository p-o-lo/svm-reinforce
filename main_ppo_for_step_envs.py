import numpy as np
import gym
import torch
import os
import h5py

from ppo_agent import PPO_agent

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
    hyperparams = {'lambda_gae': agent.lambda_gae, 'gamma': agent.gamma,
            'clip': agent.clip, 'lr_critic': agent.lr_critic,
            'lr_actor': agent.lr_actor, 'num_update': agent.num_update,
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


def save_all(dat_file_name, i_ep, sigmas_i_ep, rew_i_ep, en_i_ep, pri_dim_i_ep
             , act_model_i_ep, cr_model_i_ep):
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


def rm_useless_file(actor_model_file, critic_model_file, file_sigmas):
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


# Instance of the ppo agent and actor critic model name
agent = PPO_agent(state_size, act_size, seed=0)
actor_model_file = 'checkpoint_actor1.pth'
critic_model_file = 'checkpoint_critic1.pth'


def run_ppo(num_iterations=200, num_trajs=10, length_traj=250):
    dat_file_name = create_info_h5(agent, env)
    for k in range(num_iterations):
        # Data for trajectories
        trajs_states = []
        trajs_acts = []
        all_rews = []
        trajs_pri_dim = []
        trajs_log_pol = []
        len_trajs = []

        # Run to collect trajs for a maximum of length_traj
        for i in range(num_trajs):
            # Episodic data. Keeps track of rewards per traj
            print(f'##### {i}th Traj #####')
            ep_rews = []
            state = env.reset()
            done = False

            for t_traj in range(length_traj):
                # Track observations in this batch
                trajs_states.append(state)

                # Calculate action and log policy and perform a step of th env
                action, log_policy = agent.act(state)
                state, reward, done, info = env.step(action)

                # Track recent reward, action, and action log policy
                trajs_acts.append(action)
                trajs_log_pol.append(log_policy)
                trajs_pri_dim.append(env.princp_dim)
                ep_rews.append(reward)

                if done:
                    break

            len_trajs.append(1 + t_traj)
            all_rews.append(ep_rews)

        # Reshape data as tensors
        trajs_states = torch.tensor(trajs_states, dtype=torch.float)
        trajs_acts = torch.tensor(trajs_acts, dtype=torch.float)
        trajs_log_pol = torch.tensor(trajs_log_pol, dtype=torch.float)

        # Run step for learning
        agent.step(trajs_states, trajs_acts, trajs_log_pol, all_rews, len_trajs)
        torch.save(agent.actor_local.state_dict(), actor_model_file)
        torch.save(agent.critic_local.state_dict(), critic_model_file)

        # Save data during training (to not lose the work done)
        save_all(dat_file_name=dat_file_name, i_ep=int(k),
                 sigmas_i_ep=trajs_acts.reshape((num_trajs, length_traj, env.n_pairs)),
                 rew_i_ep=all_rews,
                 en_i_ep=trajs_states.reshape((num_trajs, length_traj)),
                 pri_dim_i_ep=np.reshape(trajs_pri_dim, (num_trajs, length_traj)),
                 act_model_i_ep=actor_model_file,
                 cr_model_i_ep=critic_model_file)

        # Remove useless files
        rm_useless_file(actor_model_file, actor_model_file, env.file_sigmas)

        # Calculate metrics to print
        avg_iter_lens = np.mean(len_trajs)
        avg_iter_retur = np.mean([np.sum(ep_rews) for ep_rews in all_rews])

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{agent.k_step} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_iter_lens}", flush=True)
        print(f"Average Episodic Return: {avg_iter_retur}", flush=True)
        print(f"Timesteps So Far: {agent.t_step}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)
    return dat_file_name


all_data = run_ppo()
