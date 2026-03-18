"""Distributed environment initialization utilities.

This module provides functions for initializing PyTorch distributed
training environments using the CommContext singleton.
"""

from .comm_context import CommContext


def initialize_distributed_env(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    dist_url: str = "env://",
    timeout: int = 1800,
) -> None:
    """Initialize the PyTorch distributed environment.

    This function wraps CommContext.init_distributed_env to provide a
    convenient interface for initializing distributed training.

    Args:
        rank: Global rank of this process.
        world_size: Total number of processes in the distributed group.
        backend: Communication backend ('nccl' or 'gloo'). Defaults to 'nccl'.
        dist_url: URL for distributed initialization. Defaults to 'env://'.
        timeout: Timeout in seconds for initialization. Defaults to 1800.
    """
    CommContext().init_distributed_env(
        world_size=world_size,
        rank=rank,
        backend=backend,
        dist_url=dist_url,
        timeout=timeout,
    )
