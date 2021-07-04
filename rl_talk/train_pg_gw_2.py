import torch
from torch import nn
from torch.optim import Adam
from envs.train_gridworld import GridWorldEnvWrapper

env = GridWorldEnvWrapper()

model = nn.Sequential(
    nn.Linear(4 * 4 * 4, 100),
    nn.ReLU(),
    nn.Linear(100, 4)
)
optimizer = Adam(model.parameters())

for epoch in range(10):
    state = env.reset()  # state shape: (4, 4, 4)

    rewards = []
    probabilities = []

    while True:
        state = torch.FloatTensor([state.flatten()])
        prob = model(state)
        prob = torch.softmax(prob, dim=1)
        action = int(torch.multinomial(prob[0], num_samples=1)[0])

        state, reward, is_done, info = env.step(action)

        rewards.append(reward)
        probabilities.append(prob[0][action])

        print('.', end='')
        if is_done:
            break

    rewards = torch.tensor(rewards)
    probabilities = torch.stack(probabilities)
    loss_terms = -1 * rewards * probabilities

    # print(f'\n{rewards=}\n{probabilities=}\n{loss_terms=}')

    loss = torch.mean(loss_terms)
    print(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


