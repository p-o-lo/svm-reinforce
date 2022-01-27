import numpy as np
import math
import gym
from gym import spaces
import subprocess

class svmEnv(gym.Env): # inherit from super class gym (OpenAI)
    
    def __init__(self, n_pairs=3, file_sigmas="./svmCodeSVD/sigmas.dat"):
        
        self.file_sigmas = file_sigmas
        self.sigmas = open(self.file_sigmas,'w')
        self.sigmas.truncate(0)
        self.sigmas.close()
        
        self.n_pairs = n_pairs
        self.action_space = spaces.Box(low=-1., high=1., shape=(n_pairs,)\
                                       ,dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(1,), dtype=np.float32)
        
        self.actions_taken = []
        self.energies = [0.0]
        self.agent_pos = np.array([0.0]).astype(np.float32)
        self.diff_dim = [0.0]
        
    def reset(self):
        print("*****CALL RESET******")
        self.sigmas = open(self.file_sigmas,'w')
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
        action = (action + 111.0/109.0)*109.0/2.0 
        print("****CALL STEP****")
        print("Action chosen at step: ", action)
        info = {}
        
        self.actions_taken.append(action)
        
        self.sigmas = open(self.file_sigmas, 'w')
        np.savetxt(self.sigmas, self.actions_taken, fmt="%f")
        self.sigmas.close()
        
        result = subprocess.check_output(['./svmCodeSVD/svmThree', './svmCodeSVD/remmy.input', self.file_sigmas]).splitlines()
        result = np.array(result,dtype=float)
        result_en = result[0]
        self.agent_pos = np.array([result_en]).astype(np.float32)
        
        princp_dim = int(result[1])
        full_dim = int(result[2])
        diff = full_dim - princp_dim 

        print("With this action the energy is: ", result_en)
        print("With this action the full dim is: ", full_dim, " and princip dim is: ", princp_dim)
        
        
        if (math.isnan(result_en) or result_en >= 0 or result_en >= self.energies[-1] or \
            result_en < -0.1026):
            print("The new action: ", action, " makes the energy positive: ", result_en >= 0 )
            print("The new action: ", action, " makes the energy greater than: ", self.energies[-1] \
                  , " the previous one: ", result_en >= self.energies[-1])
            print("The new action: ", action, " makes the energy less than: -0.1026", result_en < -0.1026)
            print("The new action: ", action, " makes the energy nan: ", math.isnan(result_en))
            
            print("This action IS REMOVED from actions taken and sigmas, the enery is NOT STORED!")
            self.actions_taken.pop()
            self.sigmas = open(self.file_sigmas,'w')
            np.savetxt(self.sigmas, self.actions_taken, fmt="%f")
            self.sigmas.close()
            
            reward = -1.0

            if math.isnan(result_en):
                reward = 100*reward
                self.agent_pos = np.array([self.energies[-1]]).astype(np.float32)
                print("IS NAN --> Set reward: ", reward)
                print("IS NAN --> Set agent pos to prev energy: ", self.agent_pos)
        
            elif result_en >= self.energies[-1]:
                reward = 1000*reward*(result_en - self.energies[-1])
                print("The energy is greater than previous energy --> Set reward: ", reward)
                print("This action IS REMOVED from actions taken and sigmas, the energy is NOT STORED!")        
            
            elif result_en < -0.1026:
                reward = 1000*reward*(self.energies[-1] - result_en)
                print("Is less than target energy --> Set reward: ", reward)
            
            done = False
            return self.agent_pos, reward, done, info
            
        else: 
            print("The new action: ", action, " makes the energy positive: ", result_en >= 0 )
            print("The new action: ", action, " makes the energy greater than: ", self.energies[-1] \
                  , " the previous one: ", result_en >= self.energies[-1])
            
            print("This action is NOT REMOVED from actions taken and sigmas, the energy is STORED!")
            
            print("Store the energy got!")
            self.energies.append(result_en)
            
            reward = 1.0
            print("Reward is positive!", reward)
            
            print("Calculate the diff between dim: ")
            self.diff_dim.append(diff)
            diff2 = self.diff_dim[-1] - self.diff_dim[-2]
            print("Diff 2: ", diff2)
            
            if (diff2 > 0):
                print("Add a PENALTY on the rewards!!")
                reward = -0.2*diff2*reward
                print("Reward is slightly negative!", reward)
            if (diff2 < 0):
                print("INCREASE the reward")
                reward = -diff2*reward
                print("Reward is positive increased!", reward)
            
            done = bool(abs(-0.1024803 - self.energies[-1]) < 1e-06)
            
            return self.agent_pos, reward, done, info
        
        
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print("The last action pos (sigmas): ", self.agent_pos)
        print("The energies explored are:", self.energies)
        print("The actions taken (list of sigmas):", self.actions_taken)
        
    def close(self):
        pass
