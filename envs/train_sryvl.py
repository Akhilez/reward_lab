from omegaconf import DictConfig
from sryvl.envs.sryvl_v0.sryvl import SrYvlLvl0Env
from dqn.pg import train_pg
from libs.env_wrapper import GymEnvWrapper, NumpyStateMixin
import numpy as np
from libs.models import GenericConvModel
from settings import device


class SrYvlLvl0EnvWrapper(GymEnvWrapper, NumpyStateMixin):
    max_steps = np.inf
    reward_range = (0, 1)

    def __init__(self, env=SrYvlLvl0Env()):
        super(SrYvlLvl0EnvWrapper, self).__init__(env)

    def get_legal_actions(self):
        legal_actions = self.env.legal_actions.nonzero()[0]
        return legal_actions


def train_sryvl_pg():
    hp = DictConfig({})

    hp.episodes = 2
    hp.batch_size = 2

    hp.lr = 1e-3

    hp.gamma_discount_credits = 0.9
    hp.gamma_discount_returns = 0.9

    hp.env_record_freq = 500
    hp.env_record_duration = 50

    model = (
        GenericConvModel(height=7, width=7, in_channels=6, channels=[50, 50], linear_units=[10], out_size=5)
        .float()
        .to(device)
    )

    train_pg(
        SrYvlLvl0EnvWrapper, model, hp, project_name="SrYvl", run_name="pg_test",
    )


if __name__ == "__main__":
    train_sryvl_pg()



