from unittest import TestCase, mock

from omegaconf import DictConfig

from envs.train_breakout import BreakoutEnvWrapper
from envs.train_connect_x import ConnectXEnvWrapper
from envs.train_frozen_lake import FrozenLakeEnvWrapper
from envs.train_gridworld import GridWorldEnvWrapper
from envs.train_mario import MarioEnvWrapper
from envs.train_sokoban import SokobanV2L0EnvWrapper
from libs.models import GenericLinearModel

env_cases = [
    {
        "env": GridWorldEnvWrapper,  # Custom made env
        "input": 4 * 4 * 4,
        "output": 4,
        "flatten": True,
    },
    {
        "env": ConnectXEnvWrapper,  # PettingZoo
        "input": 2 * 6 * 7,
        "output": 7,
        "flatten": True,
    },
    {
        "env": FrozenLakeEnvWrapper,  # OpenAI toy text
        "input": 16,
        "output": 4,
    },
    {
        "env": MarioEnvWrapper,
        "input": 3 * 42 * 42,
        "output": 12,
        "flatten": True,
    },
    {
        "env": SokobanV2L0EnvWrapper,  # Griddly
        "input": 5 * 7 * 8,
        "output": 5,
        "flatten": True,
    },
    {
        "env": BreakoutEnvWrapper,  # Atari
        "input": 42 * 42 * 3,
        "output": 4,
        "flatten": True,
    },
]


class TestRuns(TestCase):
    @mock.patch("dqn.dqn.wandb")
    def test_dqn_vanilla(self, *_):
        from dqn.dqn import train_dqn

        hp = DictConfig({})

        hp.steps = 2
        hp.batch_size = 2
        hp.env_record_freq = 0
        hp.env_record_duration = 0

        hp.max_steps = 50
        hp.grid_size = 4

        hp.lr = 1e-3
        hp.epsilon_exploration = 0.1
        hp.gamma_discount = 0.9

        for case in env_cases:
            print(case["env"].__name__)

            model = GenericLinearModel(
                in_size=case["input"],
                units=[10],
                out_size=case["output"],
                flatten=case.get("flatten", False),
            )

            train_dqn(case["env"], model, hp)

    @mock.patch("dqn.pg.wandb")
    def test_pg(self, *_):
        from dqn.pg import train_pg

        hp = DictConfig({})

        hp.episodes = 2
        hp.batch_size = 2

        hp.lr = 1e-3

        hp.gamma_discount_returns = 0.9
        hp.gamma_discount_credits = 0.9

        for case in env_cases:
            print(case["env"].__name__)

            model = GenericLinearModel(
                in_size=case["input"],
                units=[10],
                out_size=case["output"],
                flatten=case.get("flatten", False),
            )

            train_pg(case["env"], model, hp)
