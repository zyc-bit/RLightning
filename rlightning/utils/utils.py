"""General utility functions and classes for RLightning.

This module provides utility functions for device handling, type conversions,
and internal flags for controlling runtime behavior.
"""

import os
from typing import Any, Dict, Union

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils._pytree import tree_map


def torch_dtype_from_precision(precision: Union[int, str, None]) -> Union[torch.dtype, None]:
    """Convert precision specification to torch dtype.

    Args:
        precision: Precision specification (int, str, or None).

    Returns:
        Corresponding torch dtype, or None if precision is None.

    Raises:
        ValueError: If precision cannot be parsed to a valid dtype.
    """
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    elif precision in [16, "16", "fp16", "16-mixed"]:
        return torch.float16
    elif precision in [32, "32", "32-true"]:
        return torch.float32
    elif precision in [None]:
        return None
    else:
        raise ValueError(f"Could not parse the precision of `{precision}` to a valid torch.dtype")


class _InternalFlagMeta(type):
    """Metaclass for InternalFlag providing dynamic property access."""

    @property
    def DEBUG(cls) -> bool:
        """Check if debug mode is enabled."""
        return os.getenv("RLIGHTNING_DEBUG", "0") == "1"

    @property
    def VERBOSE(cls) -> bool:
        """Check if verbose mode is enabled."""
        return os.getenv("RLIGHTNING_VERBOSE", "0") == "1"

    @property
    def REMOTE_TRAIN(cls) -> bool:
        """Check if remote training is enabled."""
        return os.getenv("RLIGHTNING_REMOTE_TRAIN", "0") == "1"

    @property
    def REMOTE_EVAL(cls) -> bool:
        """Check if remote evaluation is enabled."""
        return os.getenv("RLIGHTNING_REMOTE_EVAL", "0") == "1"

    @property
    def REMOTE_STORAGE(cls) -> bool:
        """Check if remote storage is enabled."""
        return os.getenv("RLIGHTNING_REMOTE_STORAGE", "0") == "1"

    @property
    def REMOTE_ENV(cls) -> bool:
        """Check if remote environment is enabled."""
        return os.getenv("RLIGHTNING_REMOTE_ENV", "0") == "1"


class InternalFlag(metaclass=_InternalFlagMeta):
    """Dynamic access to internal runtime flags.

    Provides class-level properties to check various runtime modes
    controlled by environment variables.

    Example:
        >>> if InternalFlag.DEBUG:
        ...     # do something related with profiling and debugging
        ...     pass
    """

    @classmethod
    def get_env_vars(cls) -> Dict[str, str]:
        """Get all internal flags as environment variable dictionary.

        Returns:
            Dictionary mapping environment variable names to values.
        """
        return {
            "RLIGHTNING_DEBUG": "1" if cls.DEBUG else "0",
            "RLIGHTNING_VERBOSE": "1" if cls.VERBOSE else "0",
            "RLIGHTNING_REMOTE_TRAIN": "1" if cls.REMOTE_TRAIN else "0",
            "RLIGHTNING_REMOTE_EVAL": "1" if cls.REMOTE_EVAL else "0",
            "RLIGHTNING_REMOTE_STORAGE": "1" if cls.REMOTE_STORAGE else "0",
            "RLIGHTNING_REMOTE_ENV": "1" if cls.REMOTE_ENV else "0",
        }


def to_device(data: Any, device: Union[str, torch.device]) -> Any:
    """Move tensors in data structure to specified device.

    Recursively traverses the data structure and moves any Tensor
    or TensorDict objects to the specified device.

    Args:
        data: Any data structure that may contain Tensor or TensorDict.
        device: Target device (e.g., 'cpu', 'cuda:0').

    Returns:
        Data structure with tensors moved to the specified device.
    """

    def _tensor_to_device(x: Any) -> Any:
        if isinstance(x, (torch.Tensor, TensorDict)):
            return x.to(device)
        elif isinstance(x, np.ndarray):
            if x.dtype == np.uint16:
                return torch.from_numpy(x).view(torch.bfloat16).to(device)
            return torch.from_numpy(x).to(device)
        return x

    return tree_map(_tensor_to_device, data)


def to_numpy(data: Any) -> Any:
    """Convert tensors in data structure to NumPy arrays.

    Recursively traverses the data structure and converts any Tensor
    objects to NumPy arrays.

    Args:
        data: Any data structure that may contain Tensor objects.

    Returns:
        Data structure with tensors converted to NumPy arrays.
    """

    def _tensor_to_numpy(x):
        if isinstance(x, torch.Tensor):
            if x.dtype == torch.bfloat16:
                return x.cpu().view(torch.uint16).numpy()
            elif x.dtype == torch.uint16:
                raise ValueError(
                    "Sorry. We haven't support converting uint16 tensor to numpy. Since we treat uint16 as bfloat16."
                )
            return x.cpu().numpy()
        return x

    return tree_map(_tensor_to_numpy, data)
