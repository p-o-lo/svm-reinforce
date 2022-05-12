import numpy as np
import gym
import torch
import os
import pickle

from ppo_agent import PPO_agent

env = gym.make('svm_env:svmEnv-v2', n_pairs=3, n_basis=250, file_sigmas="./svmCodeSVD/sigmas.dat")
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
            'lambda_gae': agent.lambda_gae, 'gamma': agent.gamma,
            'clip': agent.clip, 'lr_critic': agent.lr_critic,
            'lr_actor': agent.lr_actor, 'num_update': agent.num_update,
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


agent = PPO_agent(state_size, act_size, seed=0)
actor_model_file = 'checkpoint_actor6.pth'
critic_model_file = 'checkpoint_critic6.pth'


# Run ppo algs
def run_ppo(num_episodes=300, num_trajs=10, length_traj=100):
    name_run_dir = create_run_fold_and_info(agent, env)
    for k in range(num_episodes):
        # Data for trajectories
        trajs_states = []
        trajs_acts = []
        all_rews = []
        trajs_pri_dim = []
        trajs_full_dim = []
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

                # Calculate action and log policy and perform a step of the env
                action, log_policy = agent.act(state)
                state, reward, done, info = env.step(action.reshape((env.n_basis, env.n_pairs)))

                # Track recent reward, action, and action log policy, pri dim, full dim
                ep_rews.append(reward)
                trajs_acts.append(action)
                trajs_log_pol.append(log_policy)
                trajs_pri_dim.append(env.princp_dim)
                trajs_full_dim.append(env.full_dim)

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

        # Save energies (states), sigmas (actions), rew, pri dim, full dim
        # actor, critic models
        save_all(name_run_dir=name_run_dir, i_ep=int(k),
                sigmas_i_ep=trajs_acts.reshape((num_trajs, 1 + t_traj, env.n_basis, env.n_pairs)),
                rew_i_ep=all_rews, en_i_ep=trajs_states.reshape((num_trajs, 1 + t_traj)),
                pri_dim_i_ep=np.reshape(trajs_pri_dim, (num_trajs, 1 + t_traj)),
                full_dim_i_ep=np.reshape(trajs_full_dim, (num_trajs, 1 + t_traj)),
                act_model_i_ep=actor_model_file, cr_model_i_ep=critic_model_file)

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
    return name_run_dir


name_dir_ppo = run_ppo()
