from envs.train_gridworld import GridWorldEnvWrapper
import numpy as np

env = GridWorldEnvWrapper()
env.reset()
env.render()

while True:
    action = np.random.randint(4)
    state2, reward, is_done, info = env.step(action)
    print(f'{action=}, {reward=}')
    env.render()

    if is_done:
        break

