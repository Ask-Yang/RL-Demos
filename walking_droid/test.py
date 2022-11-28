from learning.actor_critic_model import *
import gym
import torch
import time
env = gym.make("MountainCarContinuous-v0")

class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, 16)

model = ActorNet()
path =  "../model/test_"+str(time.time())
torch.save(model.state_dict(),path)
