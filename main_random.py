import numpy as np
import gym
import svm_env as svm
import subprocess

env = gym.make("svm_env:svmEnv-v0", file_sigmas = "./svmCodeSVD/sigmasRandom")

state = env.reset()
scores = []
step = 0
score = 0.0

while True:
    print(".....STEP.....", step)
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    step = step + 1
    score += reward
    scores.append(score)
    state = next_state
    if done:
        break

name_scores = 'scores_random.out'
file_scores = open(name_scores,'w')
np.savetxt(file_scores, scores, fmt="%f")
file_scores.close()
