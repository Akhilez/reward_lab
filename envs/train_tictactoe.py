from omegaconf import DictConfig
from pettingzoo.classic import tictactoe_v3
from dqn.dqn_default import train_dqn
from libs.env_wrapper import (
    PettingZooEnvWrapper,
    NumpyStateMixin,
    petting_zoo_random_player,
)
from libs.models import GenericLinearModel
from settings import device


class TicTacToeEnvWrapper(PettingZooEnvWrapper, NumpyStateMixin):
    def __init__(self):
        super(TicTacToeEnvWrapper, self).__init__(
            env=tictactoe_v3.env(), opponent_policy=petting_zoo_random_player
        )


if __name__ == "__main__":

    hp = DictConfig({})

    hp.steps = 20
    hp.batch_size = 2
    hp.max_steps = 10
    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    model = GenericLinearModel(18, [10], 9, flatten=True).float().to(device)

    train_dqn(TicTacToeEnvWrapper, model, hp, name="TicTacToe")
