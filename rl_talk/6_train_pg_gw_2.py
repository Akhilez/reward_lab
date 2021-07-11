from copy import deepcopy
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam
from envs.train_gridworld import GridWorldEnvWrapper
import wandb
from datetime import datetime
from settings import BASE_DIR
import numpy as np


def _format_video(video):
    video = np.array(video)
    video = np.moveaxis(video, -1, 1)
    return video


wandb.init(
    name=f"gw_pg_{str(datetime.now().timestamp())[5:10]}",
    project="rl_talk_gridworld",
    config={},
    save_code=True,
    group=None,
    tags=['policy_grad'],  # List of string tags
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

for episode in range(10000):
    state = env.reset()

    rewards = []
    probabilities = []
    video_buffer = []
    must_record = episode % 1000 == 0
    is_done = False

    while not is_done:
        state = torch.FloatTensor([state.flatten()])
        prob = model(state)
        prob = torch.softmax(prob, dim=1)
        action = int(torch.multinomial(prob[0], num_samples=1)[0])

        state, reward, is_done, info = env.step(action)

        rewards.append(reward)
        probabilities.append(prob[0][action])
        if must_record:
            video_buffer.append(deepcopy(env.render("rgb_array")))

    rewards = torch.FloatTensor(rewards)
    probabilities = torch.stack(probabilities)
    loss_terms = -1 * rewards * probabilities

    loss = torch.mean(loss_terms)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    log = DictConfig({"episode": episode})
    log.ep_loss = loss.item()
    log.ep_mean_reward = rewards.mean().item()
    log.ep_length = len(rewards)

    if must_record:
        log = dict(log)
        log[f"video_ep{episode}_reward{reward}"] = wandb.Video(_format_video(video_buffer), fps=4, format="gif")

    wandb.log(log)
