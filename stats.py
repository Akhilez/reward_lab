from typing import List
import numpy as np
import torch


class PGStats:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.rewards: List[List[float]] = [[] for _ in range(self.batch_size)]
        self.probs: List[List[torch.Tensor]] = [[] for _ in range(self.batch_size)]
        self.dones: List[bool] = [False for _ in range(self.batch_size)]
        self.end_steps = np.zeros(self.batch_size, dtype=int)

    def record(self, rewards, actions, p_pred, done_list):
        for env_index, (reward, action, probs, done) in enumerate(
            zip(rewards, actions, p_pred, done_list)
        ):
            prob = torch.tensor(0) if self.dones[env_index] else probs[action]

            self.rewards[env_index].append(reward)
            self.probs[env_index].append(prob)

            if done and not self.dones[env_index]:
                self.end_steps[env_index] = len(self.rewards[env_index])
                self.dones[env_index] = True

    def get_returns(self, gamma: float):
        returns = torch.tensor(np.zeros_like(self.rewards), dtype=torch.float32)
        for i in range(self.batch_size):
            rewards = self.rewards[i]
            end_step = self.end_steps[i]
            discount = 1.0
            return_ = 0.0
            for step in range(end_step - 1, -1, -1):
                return_ = rewards[step] + discount * return_
                discount *= gamma
                returns[i][step] = return_
        return returns

    def get_credits(self, gamma: float):
        credits = torch.tensor(np.zeros_like(self.rewards), dtype=torch.float32)
        batch_size, steps = credits.shape
        discounts = gamma ** torch.arange(steps)  # [1, 0.9, 0.8, ...]
        discounts = reversed(discounts)

        for i in range(batch_size):
            end_step = self.end_steps[i]
            credits[i, :end_step] = discounts[steps - end_step :]

        return credits

    def get_probs(self):
        probs = [torch.stack(prob) for prob in self.probs]
        probs = torch.stack(probs).squeeze()
        return probs

    def get_mean_rewards(self):
        rewards = []
        for i in range(self.batch_size):
            end_step = self.end_steps[i]
            rewards.append(np.mean(self.rewards[i][:end_step]))
        return float(np.mean(rewards))
