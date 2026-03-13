from envs.humanoid_env import HumanoidEnv
from ppo import ActorCritic, RolloutBuffer, PPO
import torch

device = "cuda"

env = HumanoidEnv(num_envs=1024, device=device)
print("Environment Observation:", env.obs_dim)

model = ActorCritic(env.obs_dim, env.act_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

num_steps = 256

buffer = RolloutBuffer(
    num_steps,
    env.num_envs,
    env.obs_dim,
    env.act_dim,
    device
)

ppo = PPO(
    obs_dim=env.obs_dim,
    act_dim=env.act_dim,
    device=device
)

obs = env.compute_observations()
print("OBS:", obs.shape)

for iteration in range(10000):

    buffer.step = 0

    for step in range(num_steps):

        with torch.no_grad():
            action, logprob = model.act(obs)
            value = model.critic(obs).squeeze(-1)

        next_obs, reward, done = env.step(action)

        buffer.add(obs, action, logprob, reward, done, value)

        obs = next_obs

    with torch.no_grad():
        last_value = model.critic(obs).squeeze(-1)

    buffer.compute_returns(last_value)

    ppo.update(buffer)

    print("Iteration:", iteration,
          "Reward:", buffer.rewards.mean().item())