from gym.envs.registration import register

register(id='svmEnv-v0', entry_point='svm_env.envs:svmEnv0',)
register(id='svmEnv-v1', entry_point='svm_env.envs:svmEnv1',)
register(id='svmEnv-v2', entry_point='svm_env.envs:svmEnv2',)
register(id='svmEnv-v3', entry_point='svm_env.envs:svmEnv3',)
