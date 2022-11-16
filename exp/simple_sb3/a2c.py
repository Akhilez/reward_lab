import random
from typing import Callable, Union, Tuple, Dict, Optional, Any
import torch.backends.cudnn
import numpy as np
from gym import Env, make, Space, spaces
import torch
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
from torch.nn import Tanh
from torch.optim import Adam


def constant_fn(val: float) -> Callable[[float], float]:
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: constant value
    :return: Constant schedule function.
    """

    def func(_):
        return val

    return func


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_obs_shape(
    observation_space: Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, Box):
        return observation_space.shape
    elif isinstance(observation_space, Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Used by the ``FlattenExtractor`` to compute the input shape.

    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)


class DiagGaussianDistribution:
    def __init__(self, action_dim: int):
        self.distribution = None
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None


class CategoricalDistribution:
    def __init__(self, action_dim: int):
        self.distribution = None
        self.action_dim = action_dim


def make_proba_distribution(action_space: Space):
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """

    if isinstance(action_space, spaces.Box):
        return DiagGaussianDistribution(get_action_dim(action_space))
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(action_space.n)
    # elif isinstance(action_space, spaces.MultiDiscrete):
    #     return MultiCategoricalDistribution(action_space.nvec, **dist_kwargs)
    # elif isinstance(action_space, spaces.MultiBinary):
    #     return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_space: Space,
        action_space: Space,
        device: Union[torch.device, str],
        gamma: float = 0.99,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device

        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False

        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.generator_ready = False

        self.pos = 0
        self.full = False


class ActorCriticPolicy(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
    ):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = None
        self.normalize_images = True

        self.optimizer_class = Adam
        self.optimizer_kwargs = {}
        self.optimizer = None  # type: Optional[torch.optim.Optimizer]

        self.features_extractor_kwargs = {}

        self._squash_output = False

        self.net_arch = [dict(pi=[64, 64], vf=[64, 64])]
        self.activation_fn = Tanh
        self.ortho_init = True

        self.features_extractor = nn.Flatten()
        self.features_dim = get_flattened_obs_dim(observation_space)

        self.normalize_images = True
        self.log_std_init = 0.0

        # Action distrubution
        self.action_dist = make_proba_distribution(action_space)

        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )


class SimplePPO:
    def __init__(self, env: Env):
        self.device = torch.device('cpu')
        self.env = None
        self.learning_rate = 0.0003

        # TODO: Remove this later
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env

        self.n_steps = 2048
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.rollout_buffer = None

        self.batch_size = 64
        self.n_epochs = 10
        self.clip_range = 0.2
        self.clip_range_vf = None
        self.normalize_advantage = True
        self.target_kl = None

        self.lr_schedule = constant_fn(float(self.learning_rate))
        self.seed = None
        if self.seed is not None:
            set_random_seed(self.seed, using_cuda=self.device.type == torch.device("cuda").type)
            self.action_space.seed(self.seed)
            self.env.seed(self.seed)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
        )
        self.policy = ActorCriticPolicy(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
        )


_env = make("CartPole-v1")
_model = SimplePPO(_env)
_model.learn(total_timesteps=25000)
