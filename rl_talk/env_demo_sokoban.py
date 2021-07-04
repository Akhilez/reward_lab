from griddly import gd
from matplotlib import pyplot as plt
import gym


def render(env):
    state = env.render(mode='rgb_array')
    plt.imshow(state)
    plt.show()


env = gym.make(
    "GDY-Sokoban---2-v0",
    global_observer_type=gd.ObserverType.SPRITE_2D,
    player_observer_type=gd.ObserverType.SPRITE_2D,
    level=0,
)

env.reset()
render(env)

while True:
    action = env.action_space.sample()
    state, reward, is_done, info = env.step(action)
    print(f'{action=}, {reward=}')
    render(env)

    if is_done:
        break
