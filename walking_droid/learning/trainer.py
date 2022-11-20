import torch
import torch.nn.functional as F
import time
import numpy as np

class Trainer:
    def __init__(self, policy, envs, device):
        self.policy = policy
        self.envs = envs
        self.device = device

    def run(self, rb, args, writer):
        # ALGO LOGIC: training.
        start_time = time.time()
        # TRY NOT TO MODIFY: start the game
        obs = self.envs.reset()
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                with torch.no_grad():
                    actions = self.policy.actor(torch.Tensor(obs).to(self.device))
                    actions += torch.normal(0, self.policy.actor.action_scale * args.exploration_noise)
                    actions = actions.cpu().numpy().clip(self.envs.single_action_space.low, self.envs.single_action_space.high)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = self.envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = self.policy.actor_target(data.next_observations)
                qf1_next_target = self.policy.critic_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    qf1_next_target).view(-1)

            qf1_a_values = self.policy.critic(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            self.policy.critic_optimizer.zero_grad()
            qf1_loss.backward()
            self.policy.critic_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -self.policy.critic(data.observations, self.policy.actor(data.observations)).mean()
                self.policy.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.policy.actor_optimizer.step()

                # update the target network
                for param, target_param in zip(self.policy.actor.parameters(), self.policy.actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(self.policy.critic.parameters(), self.policy.critic_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)