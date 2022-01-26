from gym.envs.registration import register

register(id='svmEnv-v0',
        entry_point='svm_env.envs:svmEnv',
)

