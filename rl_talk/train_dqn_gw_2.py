import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam
from envs.train_gridworld import GridWorldEnvWrapper
import wandb
from datetime import datetime
from settings import BASE_DIR

wandb.init(
    name=f"gw_dqn_{str(datetime.now().timestamp())[5:10]}",
    project="rl_talk_gridworld",
    config={},
    save_code=True,
    group=None,
    tags=['dqn'],  # List of string tags
    notes=None,  # longer description of run
    dir=BASE_DIR,
)

env = GridWorldEnvWrapper()
model = nn.Sequential(
    nn.Linear(4 * 4 * 4, 100),
    nn.ReLU(),
    nn.Linear(100, 4)
)
optimizer = Adam(model.parameters())
cumulative_reward = 0

for episode in range(15000):
    state = env.reset()
    rewards = []
    while True:
        state = torch.FloatTensor([state.flatten()])
        qs = model(state)[0]

        if torch.rand(1) < 0.1:
            action = int(torch.randint(low=0, high=4, size=(1,)))
        else:
            action = int(qs.argmax())

        state, reward, is_done, info = env.step(action)

        with torch.no_grad():
            qs2 = model(torch.FloatTensor([state.flatten()]))[0]

        target = reward + 0.9 * qs2.amax()
        loss = (target - qs[action]) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log = DictConfig({"episode": episode})
        log.ep_loss = loss.item()

        cumulative_reward += reward
        log.cumulative_reward = cumulative_reward

        rewards.append(reward)
        if is_done:
            log.ep_mean_reward = float(np.mean(rewards))
            log.ep_length = len(rewards)

        wandb.log(log)

        if is_done:
            break
