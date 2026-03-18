"""Sampling strategies for selecting data from buffers."""

import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseSampler(ABC):
    """Abstract base class for buffer sampling strategies."""

    def __init__(self, replacement: bool) -> None:
        """Initialize the sampler.

        Args:
            replacement: Whether to sample with replacement.
        """
        self.replacement = replacement  # whether to sample with replacement

    @abstractmethod
    def sample(
        self, batch_size: int, data_size: int, shuffle: Optional[bool] = False
    ) -> np.ndarray:
        """Sample indices from a dataset."""
        raise NotImplementedError


class UniformSampler(BaseSampler):
    """Uniformly sample data from the buffer with replacement."""

    def __init__(self) -> None:
        """Initialize a uniform sampler with replacement."""
        super().__init__(replacement=True)

    def sample(
        self, batch_size: int, data_size: int, shuffle: Optional[bool] = False
    ) -> np.ndarray:
        """Sample indices uniformly."""
        indice = np.random.choice(data_size, batch_size, replace=self.replacement)
        return indice


class AllDataSampler(BaseSampler):
    """Sample all data from the buffer without replacement."""

    def __init__(self) -> None:
        """Initialize a sampler that returns all indices."""
        super().__init__(replacement=False)

    def sample(
        self, batch_size: int, data_size: int, shuffle: Optional[bool] = False
    ) -> np.ndarray:
        """Return all indices, optionally shuffled."""
        if batch_size is not None and batch_size != data_size:
            warnings.warn(
                "AllDataSampler is designed to sample all data from the buffer, "
                "but provided batch_size is not equal to data_size. ",
                UserWarning,
            )

        indice = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indice)
        return indice


class BatchSampler(BaseSampler):
    """Sample a batch of data from the buffer without replacement."""

    def __init__(self) -> None:
        """Initialize a batch sampler without replacement."""
        super().__init__(replacement=False)

    def sample(
        self, batch_size: int, data_size: int, shuffle: Optional[bool] = False
    ) -> np.ndarray:
        """Sample a batch of indices without replacement."""
        if batch_size > data_size:
            raise ValueError("batch_size must be less than or equal to data_size for BatchSampler")
        indice = np.random.choice(data_size, batch_size, replace=self.replacement)
        return indice
