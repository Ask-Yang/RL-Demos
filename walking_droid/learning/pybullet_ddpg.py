import argparse
import random
import sys
from distutils.util import strtobool
import os

from actor_critic_model import *
from collector import *
from ddpg import *
from trainer import *
from env.sim_env import *

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MountainCarContinuous-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name, pybullet_mode):
    def thunk():
        # env = gym.make(env_id)
        env = wdSim(pybullet_mode)
        # env = gym.make("MountainCarContinuous-v0")
        env = gym.wrappers.RecordEpisodeStatistics(env)  # 递归调用所有wrapper的step
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}",
                                               step_trigger=lambda t: t % 100 == 0
                                                                      or t % 100 == 30
                                                                      or t % 100 == 60)
                # 并行运行的时候录制的video会少一些，因为wrapper的global_step是内有的，数量为1/num_envs*global_step，
                # 也就是并行的环境生成的video名字相同会覆盖之前生成的，但迭代的video仍覆盖全部global_step区间
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def get_time():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def mujoco_ddpg():
    args = parse_args()
    args.env_id = "droid"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{get_time()}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, "DIRECT")
                                      for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # model setup
    actor = ActorNet(envs).to(device)
    actor_target = ActorNet(envs).to(device)
    actor_target.load_state_dict(actor.state_dict())
    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)

    critic = CriticNet(envs).to(device)
    critic_target = CriticNet(envs).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_optim = optim.Adam(list(critic.parameters()), lr=args.learning_rate)
    # policy = DDPGPolicy(actor, actor_target, actor_optim,
    #                     critic, critic_target, critic_optim)

    envs.single_observation_space.dtype = np.float32
    # replay buffer setup
    replay_buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True
    )

    # setup game and start
    obs = envs.reset()
    learning_starts = args.total_timesteps // 40
    global_max_return = sys.float_info.min
    global_step = 0
    # model_save_num = 0
    model_save_path = "models/models_" + get_time()
    os.makedirs(model_save_path)
    while global_step < args.total_timesteps:
        # ALGO LOGIC: put action logic here
        if global_step < learning_starts:  # collect rollout, starts defalut 25000
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            # actions 对应的是 envs。单个action也是一个数组，表示机器人的各个运动
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)  # noise default 0.1
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        start_time = time.time()
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                if info['episode']['r'] > global_max_return:  # and global_step / args.total_timesteps * 10 > model_save_num:
                    # model_save_num = (model_save_num + 1) % 10  # 保存每10%step的model，与最优的model
                    global_max_return = info['episode']['r']
                    torch.save({
                        'actor': actor.state_dict(),
                        'actor_target': actor_target.state_dict(),
                        'critic': critic.state_dict(),
                        'critic_target': critic_target.state_dict(),
                        'actor_optim': actor_optim.state_dict(),
                        'critic_optim': critic_optim.state_dict()
                    }, model_save_path+"/checkpoint_"+get_time())
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d and "terminal_observation" in infos[idx]:
                real_next_obs[idx] = infos[idx]["terminal_observation"]  # idx是第几个env的next_obs，d=True，而且这个env的状态是terminal_observation
        for i in range(envs.num_envs):
            replay_buffer.add(obs[i], real_next_obs[i], actions[i], rewards[i], dones[i], [infos[i]])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > learning_starts:
            data = replay_buffer.sample(args.batch_size)  # batch_size最好要比learning_starts大，但比它小也不是不可以
            with torch.no_grad():
                next_state_actions = actor_target(data.next_observations)
                qf1_next_target = critic_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = critic(data.observations, data.actions).view(-1)
            critic_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -critic(data.observations, actor(data.observations)).mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        global_step += envs.num_envs

    envs.close()
    writer.close()


if __name__ == "__main__":
    mujoco_ddpg()