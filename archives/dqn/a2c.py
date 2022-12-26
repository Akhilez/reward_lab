"""
Advantage Actor Critic - Monte Carlo

- A dual head model
- Training happens at the end

What's the idea behind a2c?
- It is basically policy grad except
- instead of return, we have (R - V)
- And we don't have credits anymore

"""

from datetime import datetime
from typing import List
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from dqn.action_sampler import ProbabilityActionSampler
from libs.env_recorder import EnvRecorder
from libs.env_wrapper import BatchEnvWrapper
from libs.stats import PGStats
from settings import BASE_DIR
from torch.nn import functional as F


class A2CStats(PGStats):
    def __init__(self, batch_size: int):
        super().__init__(batch_size)

        self.qs: List[List[torch.Tensor]] = [[] for _ in range(self.batch_size)]

        self.cumulative_reward = 0
        self.cumulative_done = 0

    def record(self, rewards, actions, p_pred, qs, dones):
        super().record(rewards, actions, p_pred, dones)

        for env_index, (action, done, q) in enumerate(zip(actions, dones, qs)):
            q = torch.tensor(0) if self.dones[env_index] else q[action]
            self.qs[env_index].append(q)

            if done and not self.dones[env_index]:
                self.end_steps[env_index] = len(self.rewards[env_index])
                self.dones[env_index] = True

    def get_values(self):
        qs = [torch.stack(q) for q in self.qs]
        qs = torch.stack(qs).squeeze()
        return qs


class TrainA2CMonteCarlo:
    def __init__(
        self,
        env: BatchEnvWrapper,
        model: nn.Module,
        config: DictConfig,
        project_name: str,
        run_name: str,
    ):
        self.env = env
        self.model = model
        self.config = config

        self.init_wandb(project_name, run_name)

        # TODO: Episodic env recorder?
        self.env_recorder = EnvRecorder(
            config.env_record_freq, config.env_record_duration
        )
        self.sample_actions = ProbabilityActionSampler()

        self.stats = A2CStats(config.batch_size)

    def init_wandb(self, project_name, run_name):
        wandb.init(
            name=f"{run_name}_{str(datetime.now().timestamp())[5:10]}",
            project=project_name or "testing_dqn",
            config=dict(self.config),
            save_code=True,
            group=None,
            tags=None,  # List of string tags
            notes=None,  # longer description of run
            dir=BASE_DIR,
        )
        wandb.watch(self.model)

    def train(self):
        config = self.config
        env = self.env

        optim = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        for episode in range(config.episodes):
            step = 0
            env.reset()

            # Monte Carlo loop
            while not env.is_done("all"):
                log = DictConfig({"step": step})

                states = env.get_state_batch()
                p_pred, q_pred = self.model(states)
                p_pred = F.softmax(p_pred, 1)

                actions = self.sample_actions(
                    valid_actions=env.get_legal_actions(), probs=p_pred, noise=0.1
                )

                _, rewards, dones, _ = env.step(actions)

                self.stats.record(rewards, actions, p_pred, q_pred, dones)

                # ===== Logging =====

                mean_reward = float(np.mean(rewards))
                log.mean_reward = mean_reward

                self.stats.cumulative_done += mean_reward
                log.cumulative_reward = self.stats.cumulative_reward

                self.stats.cumulative_done += float(np.sum(dones))
                log.cumulative_done = self.stats.cumulative_done

                # TODO: Log policy histograms

                wandb.log(log)
                step += 1

            # ======= Learn =======

            returns = self.stats.get_returns(config.gamma_discount_returns)
            probs = self.stats.get_probs()
            values = self.stats.get_values()

            loss_p = -1 * probs * (returns - values)
            loss_q = F.mse_loss(values, returns, reduction="none")
            loss = loss_p + loss_q
            loss = torch.sum(loss)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # ======== Episodic logging ========

            log = DictConfig({"episode": episode})
            log.episodic_reward = self.stats.get_mean_rewards()

            wandb.log(log)
