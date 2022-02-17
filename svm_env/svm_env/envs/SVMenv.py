import numpy as np
import math
import gym
from gym import spaces
import subprocess


class svmEnv(gym.Env):  # inherit from super class gym (OpenAI)
    
    def __init__(self, n_pairs=3, file_sigmas="./svmCodeSVD/sigmas.dat"):
        
        self.file_sigmas = file_sigmas
        self.sigmas = open(self.file_sigmas, 'w')
        self.sigmas.truncate(0)
        self.sigmas.close()
        
        self.n_pairs = n_pairs
        self.action_space = spaces.Box(low=-1., high=1., shape=(n_pairs,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(1,), dtype=np.float32)
       
        self.princp_dim = 0

        self.actions_taken = []
        self.energies = [0.0]
        self.agent_pos = np.array([0.0]).astype(np.float32)
        self.diff_dim = [0.0]
        
    def reset(self):
        print("*****CALL RESET******")
        self.sigmas = open(self.file_sigmas, 'w')
        self.sigmas.truncate(0)
        self.sigmas.close()
        
        self.actions_taken = []
        self.energies = [0.0]
        self.agent_pos = np.array([0.0]).astype(np.float32)
        self.diff_dim = [0.0]
        
        print("Action chosen at reset: ", self.agent_pos)
        print("Actions taken at reset: ", self.actions_taken)
        print("Energies got at reset: ", self.energies)
        
        return self.agent_pos 

    def step(self, action):
        action = action*55.0 + 55.0  
        print("****CALL STEP****")
        print("Action chosen at step: ", action)
       
        info = {}
        done = False    
        if (action[0] == 0.0 or action[1] == 0.0 or action[2] == 0.0) or (action[0] <= action[1] + action[2] and action[1] <= action[0] + action[2] and action[2] <= action[1] + action[0]):
            print('**** ILLEGAL ACTION ****')
            reward = -10.0
            self.agent_pos = np.array([self.energies[-1]]).astype(np.float32) 


        else:
            self.actions_taken.append(action)
            print('## Basis size (it should be the same of full dim) =  ', len(self.actions_taken))
            self.sigmas = open(self.file_sigmas, 'w')
            np.savetxt(self.sigmas, self.actions_taken, fmt="%f")
            self.sigmas.close()
        
            result = subprocess.check_output(['./svmCodeSVD/svmThree', './svmCodeSVD/remmy.input', self.file_sigmas]).splitlines()
            result = np.array(result, dtype=float)
            result_en = result[0]
            self.agent_pos = np.array([result_en]).astype(np.float32)
        
            princp_dim = int(result[1])
            self.princp_dim = princp_dim
            full_dim = int(result[2])
            diff = full_dim - princp_dim 

            # reward = -10.0
            # self.agent_pos = np.array([self.energies[-1]]).astype(np.float32)
            # print("This action IS REMOVED from actions taken and sigmas, the enery is NOT STORED!")
            # self.actions_taken.pop()
            # self.sigmas = open(self.file_sigmas, 'w')
            # np.savetxt(self.sigmas, self.actions_taken, fmt="%f")
            # self.sigmas.close()

            print("With this action the energy is: ", result_en)
            print("With this action the full dim is: ", full_dim, " and princip dim is: ", princp_dim)
        
            if (math.isnan(result_en) or result_en >= self.energies[-1] or result_en <= -0.151):
                print("The new action: ", action, " makes the energy greater than:", self.energies[-1], " the previous one: ", result_en >= self.energies[-1])
                print("The new action: ", action, " makes the energy less than: -0.151", result_en <= -0.151)
                print("The new action: ", action, " makes the energy nan: ", math.isnan(result_en))
            
                reward = -1.0

                if math.isnan(result_en):
                    reward = -10.0
                    self.agent_pos = np.array([self.energies[-1]]).astype(np.float32)
                    print("IS NAN --> Set reward: ", reward)
                    print("IS NAN --> Set agent pos to previoius energy value:", self.agent_pos)
                    print("This action IS REMOVED from actions taken and sigmas, the energy is NOT STORED!")
                    self.actions_taken.pop()
                    self.sigmas = open(self.file_sigmas, 'w')
                    np.savetxt(self.sigmas, self.actions_taken, fmt="%f")
                    self.sigmas.close()

                elif result_en >= self.energies[-1]:
                    reward = reward - 10.0*(result_en - self.energies[-1])
                    print("The energy is greater than previous energy --> Set reward: ", reward)
                    print("Store the energy got!")
                    self.energies.append(result_en)
            
                elif result_en < -0.151:
                    reward = reward + 10.0*(self.energies[-1] - result_en)
                    self.energies.append(result_en)
                    print("Is less than target energy --> Set reward: ", reward)
            
                if bool(abs(-0.1504259 - result_en) < 1e-05):
                    reward = princp_dim*1.5
                    done = True
                            
                return self.agent_pos, reward, done, info
            
            else: 
                print("The new action: ", action, " makes the energy greater than: ", self.energies[-1], " the previous one: ", result_en >= self.energies[-1])
                print("Store the energy got!")
                self.energies.append(result_en)
            
                reward = 1.0
                reward = princp_dim*(reward - 10.0*(result_en - self.energies[-2]))
                print("Reward is positive!", reward)

                print("Calculate the diff between dim: ")
                self.diff_dim.append(diff)
                diff2 = self.diff_dim[-1] - self.diff_dim[-2]
                print("Diff 2: ", diff2)
            
                if (diff2 > 0):
                    print("Add a small PENALTY on the rewards!!")
                    reward = -0.2*diff2*reward
                    print("Reward is slightly negative!", reward)
                if (diff2 < 0):
                    print("INCREASE the reward")
                    reward = -diff2*reward
                    print("Reward is positive increased!", reward)
            
                if bool(abs(-0.1504259 - self.energies[-1]) < 1e-05):
                    reward = princp_dim*1.5
                    done = True
            
            return self.agent_pos, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print("The last action pos (sigmas): ", self.agent_pos)
        print("The energies explored are:", self.energies)
        print("The actions taken (list of sigmas):", self.actions_taken)
        
    def close(self):
        pass
