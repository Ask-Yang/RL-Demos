from torch import nn


class DDPGPolicy:
    def __init__(self, actor, actor_target, actor_optim,
                 critic_optim, critic, critic_target):
        self.actor = actor
        self.actor_target = actor_target
        self.actor_optim = actor_optim
        self.critic = critic
        self.critic_target = critic_target
        self.critic_optim = critic_optim

