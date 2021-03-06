import gym
from griddly import gd
from omegaconf import DictConfig
from dqn.dqn_default import train_dqn
from libs.env_wrapper import GriddlyEnvWrapper, NumpyStateMixin, TimeOutLostMixin
from libs.models import GenericLinearModel
from settings import device


class SokobanV2L0EnvWrapper(TimeOutLostMixin, GriddlyEnvWrapper, NumpyStateMixin):
    max_steps = 500
    reward_range = (-10, 10)  # TODO: Fix this

    def __init__(self):
        super().__init__()
        self.env = gym.make(
            "GDY-Sokoban---2-v0",
            global_observer_type=gd.ObserverType.VECTOR,
            player_observer_type=gd.ObserverType.VECTOR,
            level=0,
        )


if __name__ == "__main__":

    hp = DictConfig({})

    hp.steps = 10000
    hp.batch_size = 1000
    hp.env_record_freq = 500
    hp.env_record_duration = 50
    hp.max_steps = 200
    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    model = GenericLinearModel(5 * 7 * 8, [10], 5, flatten=True).float().to(device)

    train_dqn(SokobanV2L0EnvWrapper, model, hp, name="SokobanV2L0")
