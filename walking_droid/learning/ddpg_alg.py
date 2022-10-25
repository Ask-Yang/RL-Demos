import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from actor_critic_model import ActorNet, CriticNet
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from utils.util import *

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
        self.actor_target = ActorNet(self.num_inputs, self.num_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = CriticNet(self.num_inputs, self.num_actions, **net_cfg)
        self.critic_target = CriticNet(self.num_inputs, self.num_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=num_actions, theta=args.ou_theta,
                                                       mu=args.ou_mu, sigma=args.ou_sigma)

        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.epsilon = 1.0
        self.s_t = None
        self.a_t = None
        self.is_training = True

        if USE_CUDA: self.cude()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target(
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(
                to_tensor(next_state_batch, volatile=True)
            )
        )
        next_q_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values



