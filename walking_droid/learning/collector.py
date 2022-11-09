import numpy as np
import torch


class Collector:
    def __init__(self, envs, args, actor, writer, replay_buffer, device):
        self.envs = envs
        self.args = args
        self.actor = actor
        self.writer = writer
        self.replay_buffer = replay_buffer
        self.device = device

    def collect(self):
        # TRY NOT TO MODIFY: start the game
        obs = self.envs.reset()
        for global_step in range(self.args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.args.learning_starts:
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                with torch.no_grad():
                    actions = self.actor(torch.Tensor(obs).to(self.device))
                    actions += torch.normal(0,  self.actor .action_scale * self.args.exploration_noise)
                    actions = actions.cpu().numpy().clip(self.envs.single_action_space.low, self.envs.single_action_space.high)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = self.envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            self.replay_buffer.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
