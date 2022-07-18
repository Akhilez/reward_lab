from typing import Tuple, Any
import gym
import numpy as np
import torch
from omegaconf import DictConfig
from skimage.transform import resize
from dqn.dqn_default import train_dqn
from dqn.dqn_double import train_dqn_double
from libs import decay_functions
from libs.env_wrapper import (
    TensorStateMixin,
    GymEnvWrapper,
    reset_incrementer,
)
from libs.models import GenericLinearModel, GenericConvModel


class BreakoutEnvWrapper(GymEnvWrapper, TensorStateMixin):
    def __init__(self):
        super().__init__()
        self.env = gym.make("Breakout-v0")
        self.history_size = 3
        self.action_repeats = 2

    def step(self, action: int, **kwargs) -> Tuple[Any, Any, bool, dict]:
        for _ in range(self.action_repeats):
            frame, self.reward, self.done, self.info = self.env.step(action)
            self.state = prepare_multi_state(self.state, frame)
            if self.done:
                break
        return self.state, self.reward, self.done, self.info

    @reset_incrementer
    def reset(self):
        frame = self.env.reset()
        self.state = prepare_initial_state(frame, self.history_size)
        self.done = False
        return self.state

    def get_legal_actions(self):
        return list(range(4))


def prepare_initial_state(frame: np.ndarray, history_size: int):
    """
    state = ndarray of frame 1 full size. shape(3, 240, 256)
    """
    state = prepare_state(frame).repeat((history_size, 1, 1))
    return state


def prepare_state(frame: np.ndarray):
    return (
        torch.from_numpy(downscale_obs(frame, new_size=(42, 42), to_gray=True))
        .float()
        .unsqueeze(dim=0)
    )


def downscale_obs(obs, new_size=(42, 42), to_gray=True):
    resized = resize(obs, new_size, anti_aliasing=True)
    if to_gray:
        resized = resized.mean(axis=2)
    return resized


def prepare_multi_state(state: torch.Tensor, new_frame: np.ndarray):
    """
    state = tensor of 3 frames. Shape(3, 42, 42)
    new_frame = ndarray of 1 new frame full-size. shape(3, 240, 256)
    """
    new_frame = prepare_state(new_frame)
    state = torch.cat((state[1:].clone(), new_frame), dim=0)
    return state


def breakout_dqn():

    hp = DictConfig({})

    hp.steps = 2000
    hp.batch_size = 32
    hp.env_record_freq = 500
    hp.env_record_duration = 100
    hp.max_steps = 1000
    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    model = GenericLinearModel(42 * 42 * 3, [100, 100], 4, flatten=True)

    train_dqn(
        BreakoutEnvWrapper, model, hp, project_name="Breakout", run_name="vanilla_dqn"
    )


def breakout_double_dqn():
    hp = DictConfig({})

    hp.steps = 2000
    hp.batch_size = 50

    hp.replay_batch = 50
    hp.replay_size = 1000

    hp.delete_freq = 50 * (hp.batch_size + hp.replay_size)  # every 100 steps
    hp.delete_percentage = 0.2

    hp.env_record_freq = 100
    hp.env_record_duration = 50

    hp.lr = 1e-3
    hp.gamma_discount = 0.9

    # hp.epsilon_exploration = 0.1
    hp.epsilon_flatten_step = 1500
    hp.epsilon_start = 1
    hp.epsilon_end = 0.1
    hp.epsilon_decay_function = decay_functions.LINEAR

    hp.target_model_sync_freq = 50

    model = GenericConvModel(42, 42, 3, [50, 50, 50], [100], 4)

    train_dqn_double(
        BreakoutEnvWrapper, model, hp, project_name="Breakout", run_name="double_dqn"
    )


if __name__ == "__main__":
    breakout_double_dqn()
