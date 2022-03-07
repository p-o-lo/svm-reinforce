import numpy as np
import math
import gym
from gym import spaces
import subprocess


class svmEnv1(gym.Env):  # inherit from super class gym (OpenAI)
    def __init__(self, n_pairs=3, file_sigmas='./svmCodeSVD/sigmas1.dat'):

        self.file_sigmas = file_sigmas
        self.sigmas = open(self.file_sigmas, 'w')
        self.sigmas.truncate(0)
        self.sigmas.close()

        self.n_pairs = n_pairs
        self.action_space = spaces.Box(low=-1., high=1., shape=(n_pairs,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(1,), dtype=np.float32)

        self.princp_dim = 0
        self.i_step = 0

        self.actions_taken = []
        self.energies = [0.0]
        self.agent_pos = np.array([0.0]).astype(np.float32)
        self.diff_dim = [0.0]

    def reset(self):
        print('#### CALL RESET ####')
        self.sigmas = open(self.file_sigmas, 'w')
        self.sigmas.truncate(0)
        self.sigmas.close()

        self.actions_taken = []
        self.energies = [0.0]
        self.agent_pos = np.array([0.0]).astype(np.float32)
        self.diff_dim = [0.0]
        self.i_step = 0
        print('Action chosen at reset: ', self.agent_pos)
        print('Actions taken at reset: ', self.actions_taken)
        print('Energies got at reset: ', self.energies)

        return self.agent_pos

    def step(self, action):
        action = action*55.0 + 55.0
        self.i_step = self.i_step + 1
        print('#### CALL STEP ####', self.i_step)
        print('Action chosen at step: ', action)

        info = {}
        done = bool(abs(-0.1504259 - self.energies[-1]) < 1e-05)

        if (action[0] == 0.0 or action[1] == 0.0 or action[2] == 0.0):
            reward = -10.0
            print('**** ILLEGAL ACTION **** --> Set reward:', reward)
            print('This action IS REMOVED from actions taken and sigmas, the energy is NOT STORED!')
            self.agent_pos = np.array([self.energies[-1]]).astype(np.float32)
            return self.agent_pos, reward, done, info

        else:
            self.actions_taken.append(action)
            print('Basis size (it should be the same of full dim) =  ', len(self.actions_taken))
            self.sigmas = open(self.file_sigmas, 'w')
            np.savetxt(self.sigmas, self.actions_taken, fmt="%f")
            self.sigmas.close()

            try:
                result = subprocess.check_output(['./svmCodeSVD/svmThree', './svmCodeSVD/run_svm.input', self.file_sigmas]).splitlines()
            except subprocess.CalledProcessError:
                reward = -10.0
                print('**** WITH THIS ACTION THE SVM CODE RISE A CORE DUMP ERROR **** --> Set reward: ', reward)
                print('This action IS REMOVED from actions taken and sigmas, the energy is NOT STORED!')
                self.agent_pos = np.array([self.energies[-1]]).astype(np.float32)
                self.actions_taken.pop()
                self.sigmas = open(self.file_sigmas, 'w')
                np.savetxt(self.sigmas, self.actions_taken, fmt='%f')
                self.sigmas.close()
                return self.agent_pos, reward, done, info

            else:
                result = np.array(result, dtype=float)
                result_en = result[0]
                self.agent_pos = np.array([result_en]).astype(np.float32)

                princp_dim = int(result[1])
                self.princp_dim = princp_dim
                full_dim = int(result[2])

                # reward = -10.0
                # self.agent_pos = np.array([self.energies[-1]]).astype(np.float32)
                # print("This action IS REMOVED from actions taken and sigmas, the enery is NOT STORED!")
                # self.actions_taken.pop()
                # self.sigmas = open(self.file_sigmas, 'w')
                # np.savetxt(self.sigmas, self.actions_taken, fmt="%f")
                # self.sigmas.close()

                print('With this action the energy is: ', result_en)
                print('With this action the full dim is: ', full_dim, ' and princip dim is: ', princp_dim)

                if math.isnan(result_en):
                    reward = -10.0
                    self.agent_pos = np.array([self.energies[-1]]).astype(np.float32)
                    print('**** THE ENERGY IS NAN **** --> Set reward: ', reward)
                    print('This action IS REMOVED from actions taken and sigmas, the energy is NOT STORED!')
                    self.actions_taken.pop()
                    self.sigmas = open(self.file_sigmas, 'w')
                    np.savetxt(self.sigmas, self.actions_taken, fmt="%f")
                    self.sigmas.close()

                else:
                    print('#### THE ACTION IS A GOOD ONE #### --> Store the energy got!')
                    self.energies.append(result_en)

                    reward = -80.0*abs(result_en + 0.1504259)/1.1504259 + 10.0
                    print('Reward is ', reward)

                return self.agent_pos, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print("The last action pos (sigmas): ", self.agent_pos)
        print("The energies explored are:", self.energies)
        print("The actions taken (list of sigmas):", self.actions_taken)

    def close(self):
        pass
