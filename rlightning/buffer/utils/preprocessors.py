"""Preprocessors for transforming observations and actions."""

import abc
import functools
import operator
from abc import abstractmethod
from typing import Any, Dict, List, Sequence, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


class Preprocessor(abc.ABC):
    """Transform raw data into vectorized representations."""

    def __init__(self, space: gym.Space) -> None:
        """Initialize with the original space definition."""
        self.original_space = space

    @abstractmethod
    def transform(
        self, data: Union[np.ndarray, torch.Tensor, Dict]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform raw data as preprocessed"""

    @abstractmethod
    def batch_transform(
        self, batched_data: Union[np.ndarray, torch.Tensor, Tuple, Dict]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform a batched raw data as preprocessed"""

    @property
    def shape(self) -> Sequence[int]:
        """Return the data shape of preprocessed data."""

        raise NotImplementedError

    def __call__(
        self, data: Union[np.ndarray, torch.Tensor, Dict]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply the preprocessor to a single sample."""
        assert isinstance(data, (np.ndarray, torch.Tensor, Dict)), type(data)
        return self.transform(data)


class NonPreprocessor(Preprocessor):
    """No-op preprocessor that returns input unchanged."""

    def __init__(self, space: gym.Space) -> None:
        """Initialize the no-op preprocessor."""
        super().__init__(space)

    def transform(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        """Return the input unchanged."""
        return data

    def batch_transform(self, batched_data: Union[np.ndarray, torch.Tensor, Tuple, Dict]) -> Any:
        """Return the batched input unchanged."""
        return batched_data

    @property
    def shape(self) -> Sequence[int]:
        """Return the original space shape."""
        return self.original_space.shape


class BoxFlattenPreprocessor(Preprocessor):
    """Flatten Box observations into 1D vectors."""

    def __init__(self, space: gym.Space) -> None:
        """Initialize the box flattener."""
        assert isinstance(space, gym.spaces.Box), type(space)
        super().__init__(space)

        self.out_dim = functools.reduce(operator.mul, space.shape)

    @property
    def shape(self) -> Sequence[int]:
        """Return the flattened output shape."""
        return (self.out_dim,)

    def transform(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        """Flatten a single observation."""
        assert not isinstance(data, Dict)

        if isinstance(data, torch.Tensor):
            data = data.flatten()
        elif isinstance(data, np.ndarray):
            data = data.reshape(-1)

        return data

    def batch_transform(self, batched_data: Union[np.ndarray, torch.Tensor, Tuple, Dict]) -> Any:
        """Flatten a batch of observations."""
        assert not isinstance(batched_data, Dict)

        if isinstance(batched_data, torch.Tensor):
            batched_data = batched_data.flatten(1, -1)
        elif isinstance(batched_data, np.ndarray):
            batch_dim = batched_data.shape[0]
            batched_data = batched_data.reshape((batch_dim, -1))
        elif isinstance(batched_data, Tuple):
            batched_data = tuple(map(self.transform, batched_data))
        else:
            raise RuntimeError(f"unexpected type: {type(batched_data)}")

        return batched_data


class DiscretePreprocessor(Preprocessor):
    """One-hot encoder for discrete observations."""

    def __init__(self, space: gym.Space) -> None:
        """Initialize the discrete one-hot encoder."""
        assert isinstance(space, gym.spaces.Discrete), type(space)
        super().__init__(space)

        self.out_dim = space.n

    @property
    def shape(self) -> Sequence[int]:
        """Return the output shape."""
        return (self.out_dim,)

    def transform(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        """One-hot encode a single observation."""
        assert not isinstance(data, Dict)

        if isinstance(data, np.ndarray):
            tmp_data = torch.tensor(data)
        else:
            tmp_data = data

        ret = F.one_hot(tmp_data.long(), num_classes=self.original_space.n).float()

        if isinstance(data, np.ndarray):
            ret = ret.cpu().numpy()

        return ret

    def batch_transform(self, batched_data: Union[np.ndarray, torch.Tensor, Tuple, Dict]) -> Any:
        """One-hot encode a batch of observations."""
        if isinstance(batched_data, Tuple):
            return tuple(map(self.transform, batched_data))
        return self.transform(batched_data)


def default_obs_preprocessor(obs_seq: List[Any]) -> List[Any]:
    """Return observations unchanged."""
    return obs_seq


def default_reward_preprocessor(rew_seq: List[Any]) -> List[Any]:
    """Return rewards unchanged."""
    return rew_seq


def get_preprocessor_cls(space: gym.Space) -> Type[Preprocessor]:
    """Select an appropriate preprocessor class based on space type."""
    if isinstance(space, gym.spaces.Box):
        return BoxFlattenPreprocessor
    elif isinstance(space, gym.spaces.Discrete):
        return DiscretePreprocessor
    else:
        return NonPreprocessor
