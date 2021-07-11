import torch
from torch import nn
from torch.optim import Adam
from envs.train_gridworld import GridWorldEnvWrapper

env = GridWorldEnvWrapper()
state = env.reset()

model = nn.Sequential(
    nn.Linear(4 * 4 * 4, 100),
    nn.ReLU(),
    nn.Linear(100, 4)
)
optimizer = Adam(model.parameters())

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

    if is_done:
        break
