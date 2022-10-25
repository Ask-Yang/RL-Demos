import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from actor_critic_model import ActorNet, CriticNet


criterion = nn.MSELoss


class DDPG:
    def __init__(self,
                 num_inputs,
                 num_actions,
                 args,
                 gamma=0.99,):
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }

        self.actor = ActorNet(self.num_inputs, self.num_actions, **net_cfg)
        self.actor_target =ActorNet(self.num_inputs, self.num_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = CriticNet(self.num_inputs, self.num_actions, **net_cfg)
        self.critic_target = CriticNet(self.num_inputs, self.num_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

    def predict(self, state):

    def learn(self, state, action, reward, next_state, terminal):

    def _critic_learn(self, state, action, reward, next_state, terminal):

    def _actor_learn(self, state):

    def sync_target(self, decay=None):

