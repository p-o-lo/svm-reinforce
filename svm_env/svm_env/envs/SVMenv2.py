import numpy as np
import math
import gym
from gym import spaces
import subprocess


class svmEnv2(gym.Env):  # inherit from super class gym (OpenAI)

    def __init__(self, n_pairs=3, n_basis=200, file_sigmas="./svmCodeSVD/sigmas.dat"):

        self.file_sigmas = file_sigmas
        self.sigmas = open(self.file_sigmas, 'w')
        self.sigmas.truncate(0)
        self.sigmas.close()

        self.n_pairs = n_pairs
        self.n_basis = n_basis
        self.action_space = spaces.Box(low=-1., high=1., shape=(n_basis, n_pairs), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(1,), dtype=np.float32)
        self.agent_pos = np.array([0.0]).astype(np.float32)

        self.princp_dim = 0
        self.full_dim = 0

    def reset(self):
        print("*****CALL RESET******")
        self.sigmas = open(self.file_sigmas, 'w')
        self.sigmas.truncate(0)
        self.sigmas.close()

        self.agent_pos = np.array([0.0]).astype(np.float32)
        print("Action chosen at reset: ", self.agent_pos)
        return self.agent_pos

    def step(self, action):
        action = (action*109 + 111)/2.0
        print("****CALL STEP****")
        print("Action chosen at step: ", action)

        info = {}
        done = bool(abs(-0.1504259 - self.agent_pos[0]) < 1e-05)

        self.sigmas = open(self.file_sigmas, 'w')
        np.savetxt(self.sigmas, action, fmt="%f")
        self.sigmas.close()

        try:
            result = subprocess.check_output(['./svmCodeSVD/svmThree', './svmCodeSVD/run_svm.input', self.file_sigmas]).splitlines()
        except subprocess.CalledProcessError:
            reward = -10.0
            print('**** WITH THIS ACTION THE SVM CODE RISE A CORE DUMP ERROR **** --> Set reward: ', reward)
            print('**** THE AGENT STATE IS THE PREVIOUS ENERGY ****')
            done = bool(abs(-0.1504259 - self.agent_pos[0]) < 1e-05)
            return self.agent_pos, reward, done, info

        else:
            result = np.array(result, dtype=float)
            result_en = result[0]
            self.princp_dim = int(result[1])
            self.full_dim = int(result[2])

            print('With this action the energy is: ', result_en)
            print('With this action the full dim is: ', self.full_dim, ' and princip dim is: ', self.princp_dim)
            if math.isnan(result_en):
                reward = -10.0
                print('**** THE ENERGY IS NAN **** --> Set reward: ', reward)
                print('**** THE AGENT STATE IS THE PREVIOUS ENERGY ****')

            else:
                print('#### THE ACTION IS A GOOD ONE ####')
                self.agent_pos = np.array([result_en]).astype(np.float32)
                reward = -80.0*abs(result_en + 0.1504259)/1.1504259 + 10.0

                print('**** THE AGENT STATE IS THE ENERGY ****', result_en)
                print('Set reward : ', reward)

            return self.agent_pos, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print("The last action pos (sigmas): ", self.agent_pos)
        print("The energies explored are:", self.energies)
        print("The actions taken (list of sigmas):", self.actions_taken)

    def close(self):
        pass
