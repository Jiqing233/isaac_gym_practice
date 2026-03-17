import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super().__init__()

        hidden = 256

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # Log std (learnable)
        self.log_std = nn.Parameter(torch.zeros(act_dim) * -0.5)

    def forward(self, obs):
        raise NotImplementedError

    def act(self, obs):

        mean = self.actor(obs)
        std = torch.exp(self.log_std)

        dist = Normal(mean, std)

        action = dist.sample()
        action = torch.tanh(dist.sample())
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob

    def evaluate(self, obs, action):

        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        value = self.critic(obs).squeeze(-1)

        return log_prob, entropy, value

class RolloutBuffer:

    def __init__(self, num_steps, num_envs, obs_dim, act_dim, device):

        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, act_dim, device=device)
        self.logprobs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)

        self.step = 0

    def add(self, obs, action, logprob, reward, done, value):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step += 1

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):

        num_steps = self.rewards.size(0)
        self.advantages = torch.zeros_like(self.rewards)
        last_adv = 0

        for t in reversed(range(num_steps)):

            if t == num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]

            last_adv = delta + gamma * lam * (1 - self.dones[t]) * last_adv
            self.advantages[t] = last_adv

        self.returns = self.advantages + self.values

class PPO:
    def __init__(self,
                 model,
                 device="cuda",
                 lr=3e-4,
                 clip_coef=0.2,
                 vf_coef=0.5,
                 ent_coef=0.01,
                 epochs=4,
                 batch_size=16384):

        self.device = device

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.epochs = epochs
        self.batch_size = batch_size

    def update(self, buffer):

        num_steps, num_envs = buffer.rewards.shape
        total_samples = num_steps * num_envs

        obs = buffer.obs.reshape(-1, buffer.obs.size(-1))
        actions = buffer.actions.reshape(-1, buffer.actions.size(-1))
        old_logprobs = buffer.logprobs.reshape(-1)
        advantages = buffer.advantages.reshape(-1)
        returns = buffer.returns.reshape(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):

            indices = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, self.batch_size):

                end = start + self.batch_size
                mb_idx = indices[start:end]

                new_logprob, entropy, value = self.model.evaluate(
                    obs[mb_idx],
                    actions[mb_idx]
                )

                ratio = torch.exp(new_logprob - old_logprobs[mb_idx])

                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_coef,
                    1 + self.clip_coef
                ) * advantages[mb_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = ((returns[mb_idx] - value) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    - self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()