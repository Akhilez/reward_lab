import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from torch.utils.data import DataLoader

from evo_devo_nano.interaction import BatchedInteractions
from evo_devo_nano.model.model import CraftaxModel


def evaluate(interaction, model, device):
    temperature = 100  # lower temperature means more greedy sampling
    model.eval()
    interaction.reset()
    done = False
    total_reward = 0
    while not done:
        obs = torch.tensor(interaction.obs, device=device)  # [64, 64, 3] (0-1)
        obs = obs.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 64, 64]

        with torch.no_grad():
            obs_embeddings = model.obs_encoder(obs, return_reconstruction=False)  # [1, 36, 64]
            obs_embeddings = model.time_encoder(obs_embeddings, time_steps=[interaction.time_step])  # [1, 37, 64]

            action_key = model.vocab["action_key"]  # int
            action_key = model.embeddings(torch.tensor(action_key).unsqueeze(0).to(device))  # [1, 64]
            action_key = action_key.view(1, 1, -1)

            input_sequence = torch.cat([obs_embeddings, action_key], dim=1)  # [1, 38, 64]

            output_sequence = model.transformer(input_sequence)  # [1, 38, 64]

            action_probs = model.classifier(output_sequence[:, -1, :])  # [1, n_classes]
            action_probs = action_probs[:, torch.where(model.vocab.actions_mask)[0]]  # [1, n_actions]
            action_probs = torch.softmax(action_probs / temperature, dim=-1)  # [1, n_actions]
            action_dist = Categorical(probs=action_probs)
            action = action_dist.sample()

        interaction.step(action)
        done = interaction.memory.dones[-1]
        total_reward += interaction.memory.rewards[-1]
        # print(f"Action: {action.item()}, Reward: {interaction.memory.rewards[-1]}")
    return total_reward


def compute_gae(next_value, rewards, masks, values, gamma, tau):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


@torch.inference_mode()
def collect_data(interactions, model, device, time_delay_steps):
    model.train()
    data = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "masks": [],
        "values": [],
        "log_probs": [],
        "advantages": [],
        "returns": [],
        "time_steps": [],
    }
    for _ in range(time_delay_steps):
        obs = torch.from_numpy(np.asarray(interactions.obs)).to(device)  # [n_envs, 64, 64, 3] (0-1)
        obs = obs.permute(0, 3, 1, 2)  # [n_envs, 3, 64, 64]

        obs_embeddings = model.obs_encoder(obs, return_reconstruction=False)  # [n_envs, 36, 64]
        obs_embeddings = model.time_encoder(obs_embeddings, time_steps=interactions.time_step)  # [n_envs, 37, 64]

        action_key = model.vocab["action_key"]  # int
        action_key = torch.tensor(action_key, device=device).unsqueeze(0)  # [1]
        action_key = model.embeddings(action_key)  # [1, 64]
        action_key = action_key.repeat(len(interactions), 1)  # [n_envs, 64]
        action_key = action_key.unsqueeze(1)  # [n_envs, 1, 64]

        value_key = model.vocab["value_key"]  # int
        value_key = torch.tensor(value_key, device=device).unsqueeze(0)  # [1]
        value_key = model.embeddings(value_key)  # [1, 64]
        value_key = value_key.repeat(len(interactions), 1)  # [n_envs, 64]
        value_key = value_key.unsqueeze(1)  # [n_envs, 1, 64]

        input_sequence = torch.cat([
            torch.cat([obs_embeddings, action_key], dim=1),  # [n_envs, 38, 64]
            torch.cat([obs_embeddings, value_key], dim=1),  # [n_envs, 38, 64]
        ], dim=0)  # [2*n_envs, 38, 64]

        output_sequence = model.transformer(input_sequence)  # [2*n_envs, 38, 64]

        action_probs = model.classifier(output_sequence[:len(obs), -1, :])  # [n_envs, n_classes]
        action_probs = action_probs[:, torch.where(model.vocab.actions_mask)[0]]  # [n_envs, n_actions]
        action_probs = torch.softmax(action_probs, dim=-1)  # [n_envs, n_actions]
        action_dist = Categorical(probs=action_probs)
        action = action_dist.sample()  # [n_envs]
        log_prob = action_dist.log_prob(action)  # [n_envs]

        value = model.critic(output_sequence[len(obs):, -1, :])  # [n_envs, 1]
        value = value.view(-1)  # [n_envs]

        data["log_probs"].append(log_prob.cpu())
        data["values"].append(value.cpu())
        data["obs"].append(obs.cpu())
        data["actions"].append(action.cpu())
        data["time_steps"].append(interactions.time_step)

        interactions.step(action)

        data["rewards"].append(torch.tensor([float(i.memory.rewards[-1]) for i in interactions.interactions]))
        data["dones"].append(torch.tensor([bool(i.memory.dones[-1]) for i in interactions.interactions]))
        data["masks"].append(1 - data["dones"][-1].to(torch.int32))

    obs = torch.from_numpy(np.asarray(interactions.obs)).to(device)  # [n_envs, 64, 64, 3] (0-1)
    obs = obs.permute(0, 3, 1, 2)  # [n_envs, 3, 64, 64]
    obs_embeddings = model.obs_encoder(obs, return_reconstruction=False)  # [n_envs, 36, 64]
    obs_embeddings = model.time_encoder(obs_embeddings, time_steps=interactions.time_step)  # [n_envs, 37, 64]

    value_key = model.vocab["value_key"]  # int
    value_key = torch.tensor(value_key, device=device).unsqueeze(0)  # [1]
    value_key = model.embeddings(value_key)  # [1, 64]
    value_key = value_key.repeat(len(interactions), 1)  # [n_envs, 64]
    value_key = value_key.unsqueeze(1)  # [n_envs, 1, 64]

    input_sequence = torch.cat([obs_embeddings, value_key], dim=1)  # [n_envs, 38, 64]

    output_sequence = model.transformer(input_sequence)  # [2*n_envs, 38, 64]

    value = model.critic(output_sequence[:, -1, :])  # [n_envs, 1]
    value = value.view(-1)  # [n_envs]

    returns = compute_gae(value.cpu(), data["rewards"], data["masks"], data["values"], gamma=0.99, tau=0.95)
    advantages = [r - v for r, v in zip(returns, data["values"])]

    data["returns"] = returns
    data["advantages"] = advantages

    # data is of shape (time_delay_steps, n_envs, ...). We want to flatten it to (time_delay_steps*n_envs, ...)
    # It's gonna be a list of dictionaries. Each dictionary is an item in the batch.
    data_flatten = []
    for td_i in range(time_delay_steps):
        for env_i in range(len(interactions)):
            data_flatten.append({
                "obs": data["obs"][td_i][env_i],
                "actions": data["actions"][td_i][env_i],
                "log_probs": data["log_probs"][td_i][env_i],
                "returns": data["returns"][td_i][env_i],
                "advantages": data["advantages"][td_i][env_i],
                "time_steps": data["time_steps"][td_i][env_i],
            })

    return data_flatten


def train_ppo(data, model, optimizer, device):
    loss_agg = 0
    model.train()
    data_loader = DataLoader(data, batch_size=32, shuffle=True)
    for batch in data_loader:
        obs = batch["obs"].to(device)  # [batch, 3, 64, 64]
        actions = batch["actions"].to(device)  # [batch]
        log_probs = batch["log_probs"].to(device)  # [batch]
        returns = batch["returns"].to(device)  # [batch]
        advantages = batch["advantages"].to(device)  # [batch]

        obs_embeddings = model.obs_encoder(obs, return_reconstruction=False)  # [batch, 36, 64]
        obs_embeddings = model.time_encoder(obs_embeddings, time_steps=batch["time_steps"])  # [batch, 37, 64]

        action_key = model.vocab["action_key"]  # int
        action_key = torch.tensor(action_key, device=device).unsqueeze(0)  # [1]
        action_key = model.embeddings(action_key)  # [1, 64]
        action_key = action_key.repeat(len(obs), 1)  # [batch, 64]
        action_key = action_key.unsqueeze(1)  # [batch, 1, 64]

        value_key = model.vocab["value_key"]  # int
        value_key = torch.tensor(value_key, device=device).unsqueeze(0)  # [1]
        value_key = model.embeddings(value_key)  # [1, 64]
        value_key = value_key.repeat(len(obs), 1)  # [batch, 64]
        value_key = value_key.unsqueeze(1)  # [batch, 1, 64]

        input_sequence = torch.cat([
            torch.cat([obs_embeddings, action_key], dim=1),  # [batch, 38, 64]
            torch.cat([obs_embeddings, value_key], dim=1),  # [batch, 38, 64]
        ], dim=0)  # [2*batch, 38, 64]

        output_sequence = model.transformer(input_sequence)  # [2*batch, 38, 64]

        action_probs = model.classifier(output_sequence[:len(obs), -1, :])  # [batch, n_classes]
        action_probs = action_probs[:, torch.where(model.vocab.actions_mask)[0]]  # [batch, n_actions]
        action_probs = torch.softmax(action_probs, dim=-1)  # [batch, n_actions]
        action_dist = Categorical(probs=action_probs)
        
        entropy = action_dist.entropy().mean()
        new_log_probs = action_dist.log_prob(actions)  # [batch]

        value = model.critic(output_sequence[len(obs):, -1, :])  # [batch, 1]
        value = value.view(-1)  # [batch]

        ratio = (new_log_probs - log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (returns - value).pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_agg += loss.item()
    loss_agg /= len(data_loader)
    return loss_agg


def ppo_craftax():
    device = "mps"
    n_envs = 2
    max_steps = 6000
    time_delay_steps = 5

    interactions = BatchedInteractions(batch_size=n_envs)
    model = CraftaxModel(h=interactions.h, w=interactions.w, n_actions=interactions.n_actions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # total_reward = evaluate(interactions[0], model, device)
    # print(f"Total reward: {total_reward}")

    step = 0
    while step < max_steps:
        data = collect_data(interactions, model, device, time_delay_steps)
        loss = train_ppo(data, model, optimizer, device)
        print(f"Step: {step}, Loss: {loss}")
        step += time_delay_steps

        if step//time_delay_steps % 100 == 0:
            total_reward = evaluate(interactions[0], model, device)
            print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    ppo_craftax()
