from omegaconf import DictConfig
from pettingzoo.classic import connect_four_v3
from dqn.dqn_double import train_dqn_double
from libs import decay_functions
from libs.env_wrapper import (
    PettingZooEnvWrapper,
    petting_zoo_random_player,
    NumpyStateMixin,
)
from libs.models import GenericLinearModel
from settings import device


class ConnectXEnvWrapper(PettingZooEnvWrapper, NumpyStateMixin):
    max_steps = 42
    reward_range = (-1, 1)

    def __init__(self):
        super(ConnectXEnvWrapper, self).__init__(
            env=connect_four_v3.env(),
            opponent_policy=petting_zoo_random_player,
        )


def train_dqn_connect4():

    hp = DictConfig({})

    hp.steps = 10_000
    hp.batch_size = 516
    # hp.max_steps = 40
    hp.lr = 1e-3
    hp.gamma_discount = 0.9

    # hp.epsilon_exploration = 0.1
    hp.epsilon_flatten_step = 1500
    hp.epsilon_start = 1
    hp.epsilon_end = 0.1
    hp.epsilon_decay_function = decay_functions.LINEAR
    hp.target_model_sync_freq = 50
    hp.replay_batch = 50
    hp.replay_size = 1000
    hp.delete_freq = 50 * (hp.batch_size + hp.replay_size)  # every 100 steps
    hp.delete_percentage = 0.2
    hp.env_record_freq = 100
    hp.env_record_duration = 50

    model = GenericLinearModel(2 * 6 * 7, [50], 7, flatten=True).float().to(device)

    train_dqn_double(ConnectXEnvWrapper, model, hp, run_name="Connect4")


if __name__ == "__main__":
    train_dqn_connect4()
