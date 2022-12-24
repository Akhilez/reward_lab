import os
import random
from functools import partial
from itertools import zip_longest
from typing import (
    Callable,
    Union,
    Tuple,
    Dict,
    Optional,
    List,
    Type,
    NamedTuple,
    Generator,
)
import torch.backends.cudnn
import numpy as np
from gym import Env, make, Space, spaces
import torch
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import Tanh
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import glob
import tempfile
import datetime


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
        return {
            key: get_obs_shape(subspace)
            for (key, subspace) in observation_space.spaces.items()
        }

    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported"
        )


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

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = 0.0
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(
            torch.ones(self.action_dim) * log_std_init, requires_grad=True
        )
        return mean_actions, log_std

    def proba_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)

        """
        Continuous actions are usually considered to be independent,
        so we can sum components of the ``log_prob`` or the entropy.

        :param tensor: shape: (n_batch, n_actions) or (n_batch,)
        :return: shape: (n_batch,)
        """
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=1)
        else:
            log_prob = log_prob.sum()

        return log_prob

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def entropy(self) -> torch.Tensor:
        tensor = self.distribution.entropy()
        if len(tensor.shape) > 1:
            tensor = tensor.sum(dim=1)
        else:
            tensor = tensor.sum()
        return tensor


class CategoricalDistribution:
    def __init__(self, action_dim: int):
        self.distribution = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(
        self, action_logits: torch.Tensor
    ) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)


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


# TODO: Fix this later because logging is quite lengthy for some reason.
# def configure_logger(
#     verbose: int = 0,
#     tensorboard_log: Optional[str] = None,
#     tb_log_name: str = "",
#     reset_num_timesteps: bool = True,
# ):
#     """
#     Configure the logger's outputs.
#
#     :param verbose: Verbosity level: 0 for no output, 1 for the standard output to be part of the logger outputs
#     :param tensorboard_log: the log location for tensorboard (if None, no logging)
#     :param tb_log_name: tensorboard log
#     :param reset_num_timesteps:  Whether the ``num_timesteps`` attribute is reset or not.
#         It allows to continue a previous learning curve (``reset_num_timesteps=False``)
#         or start from t=0 (``reset_num_timesteps=True``, the default).
#     :return: The logger object
#     """
#     save_path, format_strings = None, ["stdout"]
#
#     if tensorboard_log is not None and SummaryWriter is None:
#         raise ImportError("Trying to log data to tensorboard but tensorboard is not installed.")
#
#     if tensorboard_log is not None and SummaryWriter is not None:
#         # ------------ utils.get_latest_run_id # -------------
#         def get_latest_run_id(log_path: str = "", log_name: str = "") -> int:
#             """
#             Returns the latest run number for the given log name and log path,
#             by finding the greatest number in the directories.
#
#             :param log_path: Path to the log folder containing several runs.
#             :param log_name: Name of the experiment. Each run is stored
#                 in a folder named ``log_name_1``, ``log_name_2``, ...
#             :return: latest run number
#             """
#             max_run_id = 0
#             for path in glob.glob(os.path.join(log_path, f"{glob.escape(log_name)}_[0-9]*")):
#                 file_name = path.split(os.sep)[-1]
#                 ext = file_name.split("_")[-1]
#                 if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
#                     max_run_id = int(ext)
#             return max_run_id
#
#         latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
#         if not reset_num_timesteps:
#             # Continue training in the same directory
#             latest_run_id -= 1
#         save_path = os.path.join(tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")
#         if verbose >= 1:
#             format_strings = ["stdout", "tensorboard"]
#         else:
#             format_strings = ["tensorboard"]
#     elif verbose == 0:
#         format_strings = [""]
#
#     # return configure(save_path, format_strings=format_strings)
#     folder = save_path
#
#     """
#         Configure the current logger.
#
#         :param folder: the save location
#             (if None, $SB3_LOGDIR, if still None, tempdir/SB3-[date & time])
#         :param format_strings: the output logging format
#             (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
#         :return: The logger object.
#         """
#     if folder is None:
#         folder = os.getenv("SB3_LOGDIR")
#     if folder is None:
#         folder = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S-%f"))
#     assert isinstance(folder, str)
#     os.makedirs(folder, exist_ok=True)
#
#     log_suffix = ""
#     if format_strings is None:
#         format_strings = os.getenv("SB3_LOG_FORMAT", "stdout,log,csv").split(",")
#
#     format_strings = list(filter(None, format_strings))
#     output_formats = [make_output_format(f, folder, log_suffix) for f in format_strings]
#
#     logger = Logger(folder=folder, output_formats=output_formats)
#     # Only print when some files will be saved
#     if len(format_strings) > 0 and format_strings != ["stdout"]:
#         logger.log(f"Logging to {folder}")
#     return logger


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_space: Space,
        action_space: Space,
        device: Union[torch.device, str],
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.observations, self.actions, self.rewards, self.advantages = (
            None,
            None,
            None,
            None,
        )
        self.returns, self.episode_starts, self.values, self.log_probs = (
            None,
            None,
            None,
            None,
        )
        self.generator_ready = False

        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros(
            (self.buffer_size,) + self.obs_shape, dtype=np.float32
        )
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

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # Same reshape, for actions
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(
        self, last_values: torch.Tensor, dones: np.ndarray
    ) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def get(self, batch_size: int) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            data = (
                self.observations[batch_inds],
                self.actions[batch_inds],
                self.values[batch_inds].flatten(),
                self.log_probs[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.returns[batch_inds].flatten(),
            )
            yield RolloutBufferSamples(
                *tuple(map(lambda x: torch.tensor(x).to(self.device), data))
            )

            start_idx += batch_size

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


class MlpExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        # device: Union[torch.device, str],
    ):
        super().__init__()
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = (
            []
        )  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = (
            []
        )  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                shared_net.append(
                    nn.Linear(last_layer_dim_shared, layer)
                )  # add linear of size layer
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer
            else:
                assert isinstance(
                    layer, dict
                ), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(
                        layer["pi"], list
                    ), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]
                if "vf" in layer:
                    assert isinstance(
                        layer["vf"], list
                    ), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for pi_layer_size, vf_layer_size in zip_longest(
            policy_only_layers, value_only_layers
        ):
            if pi_layer_size is not None:
                assert isinstance(
                    pi_layer_size, int
                ), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(
                    vf_layer_size, int
                ), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net)  # .to(device)
        self.policy_net = nn.Sequential(*policy_net)  # .to(device)
        self.value_net = nn.Sequential(*value_net)  # .to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.shared_net(features))


def is_image_space_channels_first(observation_space: spaces.Box) -> bool:
    """
    Check if an image observation space (see ``is_image_space``)
    is channels-first (CxHxW, True) or channels-last (HxWxC, False).

    Use a heuristic that channel dimension is the smallest of the three.
    If second dimension is smallest, raise an exception (no support).

    :param observation_space:
    :return: True if observation space is channels-first image, False if channels-last.
    """
    smallest_dimension = np.argmin(observation_space.shape).item()
    if smallest_dimension == 1:
        print(
            "Treating image space as channels-last, while second dimension was smallest of the three."
        )
    return smallest_dimension == 0


def is_image_space(
    observation_space: spaces.Space,
    check_channels: bool = False,
) -> bool:
    """
    Check if a observation space has the shape, limits and dtype
    of a valid image.
    The check is conservative, so that it returns False if there is a doubt.

    Valid images: RGB, RGBD, GrayScale with values in [0, 255]

    :param observation_space:
    :param check_channels: Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected.
    :return:
    """
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if observation_space.dtype != np.uint8:
            return False

        # Check the value range
        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]
        # RGB, RGBD, GrayScale
        return n_channels in [1, 3, 4]
    return False


def preprocess_obs(
    obs: torch.Tensor,
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.

    :param obs: Observation
    :param observation_space:
    :param normalize_images: Whether to normalize images or not
        (True by default)
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        if is_image_space(observation_space) and normalize_images:
            return obs.float() / 255.0
        return obs.float()

    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(obs.long(), num_classes=observation_space.n).float()

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return torch.cat(
            [
                F.one_hot(
                    obs_.long(), num_classes=int(observation_space.nvec[idx])
                ).float()
                for idx, obs_ in enumerate(torch.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()

    elif isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(
                _obs, observation_space[key], normalize_images=normalize_images
            )
        return preprocessed_obs

    else:
        raise NotImplementedError(
            f"Preprocessing not implemented for {observation_space}"
        )


# From stable baselines
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


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

        self.features_extractor_kwargs = {}

        self._squash_output = False

        self.net_arch = [dict(pi=[64, 64], vf=[64, 64])]
        self.activation_fn = Tanh
        self.ortho_init = True

        self.features_extractor = nn.Flatten()
        self.features_dim = get_flattened_obs_dim(observation_space)

        self.normalize_images = True
        self.log_std_init = 0.0

        # Action distribution
        self.action_dist = make_proba_distribution(action_space)

        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            # device=self.device,
        )
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution,)):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = Adam(self.parameters(), lr=lr_schedule(1))

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        return self.features_extractor(preprocessed_obs)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, MultiCategoricalDistribution):
        #     # Here mean_actions are the flattened logits
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, BernoulliDistribution):
        #     # Here mean_actions are the logits (before rounding to get the binary actions)
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)


class SimplePPO:
    def __init__(self, env: Env):
        self.device = torch.device("cpu")
        self.env = None
        self.learning_rate = 0.0003

        # TODO: Remove this later
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env
        self.n_envs = env.num_envs

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
            set_random_seed(
                self.seed, using_cuda=self.device.type == torch.device("cuda").type
            )
            self.action_space.seed(self.seed)
            self.env.seed(self.seed)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.n_envs,
        )
        self.policy = ActorCriticPolicy(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
        ).to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = constant_fn(self.clip_range)
        # if self.clip_range_vf is not None:
        #     if isinstance(self.clip_range_vf, (float, int)):
        #         assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
        #     self.clip_range_vf = constant_fn(self.clip_range_vf)

        # --------- BaseAlgorithm.__init__ --------------
        # self._vec_normalize_env = unwrap_vec_normalize(env)

        self.num_timesteps = 0
        self._episode_num = 0

        # Buffers for logging
        # self.ep_info_buffer = None  # type: Optional[deque]
        # self.ep_success_buffer = None  # type: Optional[deque]

        self.action_noise = None

        # Used for updating schedules
        self._total_timesteps = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0

        self._current_progress_remaining = 1

        self._last_obs = (
            None
        )  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]

        # self._logger = None

        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int

    def collect_rollouts(
        self,
        env,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.train(False)

        n_steps = 0
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                # obs_tensor = obs_as_tensor(self._last_obs, self.device)
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            n_steps += 1

            # for idx, info in enumerate(infos):
            #     maybe_ep_info = info.get("episode")
            #     maybe_is_success = info.get("is_success")
            #     if maybe_ep_info is not None:
            #         self.ep_info_buffer.extend([maybe_ep_info])
            #     if maybe_is_success is not None and dones[idx]:
            #         self.ep_success_buffer.append(maybe_is_success)

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            # for idx, done in enumerate(dones):
            #     # If reached terminal state and not abruptly stopped.
            #     if (
            #             done
            #             and infos[idx].get("terminal_observation") is not None
            #             and infos[idx].get("TimeLimit.truncated", False)
            #     ):
            #         terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
            #         with torch.no_grad():
            #             terminal_value = self.policy.predict_values(terminal_obs)[0]
            #         rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(
                # obs_as_tensor(new_obs, self.device)
                torch.as_tensor(self._last_obs).to(self.device)
            )

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=self._last_episode_starts
        )

        # return True

    def learn(
        self,
        total_timesteps: int,
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
    ):
        log_interval: int = 1
        eval_freq: int = -1
        n_eval_episodes: int = 5
        reset_num_timesteps: bool = True
        progress_bar: bool = False

        iteration = 0

        # ----------- BaseAlgorithm._setup_learn() ----------
        # if self.ep_info_buffer is None or reset_num_timesteps:
        #     # Initialize buffers if they don't exist, or reinitialize if resetting counters
        #     self.ep_info_buffer = deque(maxlen=100)
        #     self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        self.num_timesteps = 0
        self._episode_num = 0
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
        self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
        # Retrieve unnormalized observation for saving into the buffer
        # if self._vec_normalize_env is not None:
        #     self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # self._logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # ----------- END _setup_learn() ----------

        while self.num_timesteps < total_timesteps:
            # continue_training = \
            self.collect_rollouts(
                self.env,
                self.rollout_buffer,
                n_rollout_steps=self.n_steps,
            )

            # if continue_training is False:
            #     break

            iteration += 1

            # ------------- BaseAlgorithm._update_current_progress_remaining() -------------
            self._current_progress_remaining = 1.0 - float(self.num_timesteps) / float(
                total_timesteps
            )
            # ------------- END --------------

            # Display training infos
            # if log_interval is not None and iteration % log_interval == 0:
            #     time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
            #     fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
            #     self.logger.record("time/iterations", iteration, exclude="tensorboard")
            #     if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            #         self.logger.record("rollout/ep_rew_mean",
            #                            safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            #         self.logger.record("rollout/ep_len_mean",
            #                            safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            #     self.logger.record("time/fps", fps)
            #     self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
            #     self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            #     self.logger.dump(step=self.num_timesteps)

            self.train_ppo()

    def train_ppo(self):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.train(True)

        # Update optimizer learning rate
        # Log the current learning rate
        # self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))
        lr = self.lr_schedule(self._current_progress_remaining)
        for param_group in self.policy.optimizer.param_groups:
            param_group["lr"] = lr

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                rollout_data: RolloutBufferSamples = rollout_data
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                values_pred = values
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    print(
                        f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                    )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        # self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        # self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        # self.logger.record("train/value_loss", np.mean(value_losses))
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        # self.logger.record("train/loss", loss.item())
        # self.logger.record("train/explained_variance", explained_var)
        # if hasattr(self.policy, "log_std"):
        #     self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        #
        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/clip_range", clip_range)


_env = make("CartPole-v1")
_model = SimplePPO(_env)
_model.learn(total_timesteps=25000)
