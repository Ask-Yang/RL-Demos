import random

from actor_critic_model import *
from env.sim_env import *
from pybullet_ddpg import make_env, parse_args

import gym
import numpy as np
import pybullet_envs  # noqa
import torch


def robot_display():
    args = parse_args()
    args.env_id = "droid"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, "GUI")
                                      for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # model setup
    actor = ActorNet(envs).to(device)
    actor_target = ActorNet(envs).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = CriticNet(envs).to(device)
    critic_target = CriticNet(envs).to(device)
    critic_target.load_state_dict(critic.state_dict())

    # model load
    cp_path = "model_state_1669680763.6998255"
    checkpoint = torch.load("models/" + cp_path)
    actor.load_state_dict(checkpoint["actor"])
    actor_target.load_state_dict(checkpoint["actor_target"])
    critic.load_state_dict(checkpoint["critic"])
    critic_target.load_state_dict(checkpoint["critic_target"])
    actor.eval()
    actor_target.eval()
    critic.eval()
    critic_target.eval()

    envs.single_observation_space.dtype = np.float32

    # setup game and start
    obs = envs.reset()
    global_step = 0
    while global_step < args.total_timesteps:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.normal(0, actor.action_scale * args.exploration_noise)  # noise default 0.1
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        next_obs, rewards, dones, infos = envs.step(actions)

        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        obs = next_obs
        global_step += envs.num_envs


if __name__ == "__main__":
    robot_display()
