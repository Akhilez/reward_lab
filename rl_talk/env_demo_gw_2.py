import torch
from torch import nn
from envs.train_gridworld import GridWorldEnvWrapper

env = GridWorldEnvWrapper()
state = env.reset()  # state shape: (4, 4, 4)

model = nn.Sequential(
    nn.Linear(4 * 4 * 4, 100),
    nn.ReLU(),
    nn.Linear(100, 4)
)

while True:
    # action = np.random.randint(4)

    state = torch.FloatTensor([state.flatten()])  # state is a tensor of shape (1, 64)
    action = model(state)  # action is a tensor of shape (1, 4)
    action = int(action[0].argmax())  # find the max action index

    state, reward, is_done, info = env.step(action)
    print(f'{action=}, {reward=}')
    env.render()

    if is_done:
        break


