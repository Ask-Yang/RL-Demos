import parl
import paddle
import numpy as np

class MujocoAgent(parl.Agent):
    def __init__(self, algorithm, act_dim, expl_noise=0.1):
        assert isinstance(act_dim, int)
        super(MujocoAgent, self).__init__(algorithm)

        self.algorithm = algorithm
        self.expl_noise = expl_noise
        self.alg.sync_target(decay=0)

    def sample(self):

    def predict(self):

    def learn(self):