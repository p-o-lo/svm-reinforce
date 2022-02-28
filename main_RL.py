import numpy as np
import gym
import svm_env as svm
import torch

from ddpg_agent import Agent


env = gym.make('svm_env:svmEnv-v0', file_sigmas = "./svmCodeSVD/sigmas.dat")
# Instance of the ddpg agent
agent = Agent(1, 3, random_seed=2)

def save_all(agent, scores, last_energies, princip_dim):
    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

    name_scores = 'scores_RL.out'
    file_scores = open(name_scores,'w')
    np.savetxt(file_scores, scores, fmt="%f")
    file_scores.close()

    name_energies = 'energies_RL.out'
    file_energies = open(name_energies,'w')
    np.savetxt(file_energies, last_energies, fmt="%f")
    file_energies.close()

    name_dim = 'princip_dim_RL.out'
    file_dim = open(name_dim,'w')
    np.savetxt(file_dim, princip_dim, fmt="%f")
    file_dim.close()

def run_ddpg(max_t_step = 300, n_episodes=700):
    ##Inizialization
    scores = []
    last_energies = []
    princip_dim = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()                  
        score = 0.0

        ## Training loop of each episode
        for t_step in range(max_t_step):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)                   
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state  
            if done:
                break

        print('Episode {} ... Score: {:.3f}'.format(i_episode, score))
        
        scores.append(score)
        last_energies.append(state[0])
        princip_dim.append(env.princp_dim)

        ## Save data during training (to not lose the work done)
        save_all(agent, scores, last_energies, princip_dim)

    return scores, last_energies, princip_dim


scores, last_energies, princip_dim = run_ddpg()
