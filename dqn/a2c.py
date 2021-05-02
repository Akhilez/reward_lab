"""
Advantage Actor Critic - Monte Carlo

- A dual head model
- Training happens at the end

"""

from datetime import datetime
from typing import Type
import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from dqn.action_sampler import ProbabilityActionSampler
from env_recorder import EnvRecorder
from envs.env_wrapper import EnvWrapper, DoneIgnoreBatchedEnvWrapper
from settings import BASE_DIR


def train_a2c_mc(
        env_class: Type[EnvWrapper],
        model: nn.Module,
        config: DictConfig,
        project_name=None,
        run_name=None,
):
    env = DoneIgnoreBatchedEnvWrapper(env_class, config.batch_size)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    wandb.init(
        name=f"{run_name}_{str(datetime.now().timestamp())[5:10]}",
        project=project_name or "testing_dqn",
        config=dict(config),
        save_code=True,
        group=None,
        tags=None,  # List of string tags
        notes=None,  # longer description of run
        dir=BASE_DIR,
    )
    wandb.watch(model)
    # TODO: Episodic env recorder?
    env_recorder = EnvRecorder(config.env_record_freq, config.env_record_duration)
    sample_actions = ProbabilityActionSampler()

    cumulative_reward = 0
    cumulative_done = 0

    # ======= Start training ==========

    for episode in range(config.episodes):
        stats = Stats(config.batch_size)  # Stores (reward, policy prob)
        step = 0
        env.reset()

        # Monte Carlo loop
        while not env.is_done("all"):
            log = DictConfig({"step": step})

            states = env.get_state_batch()
