"""
Vanilla Policy Gradients

- Wait for all envs to finish episode yo!

"""
from typing import Type
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from datetime import datetime
from torch.optim import Adam
from dqn.action_sampler import ProbabilityActionSampler
from libs.env_recorder import EnvRecorder
from libs.env_wrapper import EnvWrapper, DoneIgnoreBatchedEnvWrapper
from settings import BASE_DIR
from libs.stats import PGStats


def train_pg(
    env_class: Type[EnvWrapper],
    model: nn.Module,
    config: DictConfig,
    project_name=None,
    run_name=None,
):
    env = DoneIgnoreBatchedEnvWrapper(env_class, config.batch_size)
    optim = Adam(model.parameters(), lr=config.lr)
    wandb.init(
        name=f"{run_name}_{str(datetime.now().timestamp())[5:10]}",
        project=project_name or "testing_dqn",
        config=dict(config),
        save_code=True,
        group=None,
        tags=None,  # List of string tags
        notes=None,  # longer description of run
        dir=BASE_DIR,
        mode='disabled',
    )
    wandb.watch(model)
    # TODO: Episodic env recorder?
    env_recorder = EnvRecorder(config.env_record_freq, config.env_record_duration)
    sample_actions = ProbabilityActionSampler()

    cumulative_reward = 0
    cumulative_done = 0

    # ======= Start training ==========

    for episode in range(config.episodes):
        stats = PGStats(config.batch_size)  # Stores (reward, policy prob)
        step = 0
        env.reset()

        # Monte Carlo loop
        while not env.is_done("all"):
            log = DictConfig({"step": step})

            states = env.get_state_batch()
            p_pred = model(states)
            p_pred = F.softmax(p_pred, 1)

            legal_actions = env.get_legal_actions()
            for legal_actions_i in legal_actions:
                assert len(legal_actions_i) > 0
            actions = sample_actions(valid_actions=legal_actions, probs=p_pred, noise=0.1)

            _, rewards, done_list, _ = env.step(actions)

            stats.record(rewards, actions, p_pred, done_list)

            # ======== Step logging =========

            mean_reward = float(np.mean(rewards))
            log.mean_reward = mean_reward

            cumulative_done += mean_reward
            log.cumulative_reward = cumulative_reward

            cumulative_done += float(np.sum(done_list))
            log.cumulative_done = cumulative_done

            # TODO: Log policy histograms

            wandb.log(log)

            step += 1

        returns = stats.get_returns(config.gamma_discount_returns)
        credits = stats.get_credits(config.gamma_discount_credits)
        probs = stats.get_probs()

        loss = -1 * (probs * credits * returns)
        loss = torch.sum(loss)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # ======== Episodic logging ========

        log = DictConfig({"episode": episode})
        log.episodic_reward = stats.get_mean_rewards()

        wandb.log(log)
