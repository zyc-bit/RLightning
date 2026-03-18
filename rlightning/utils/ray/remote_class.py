"""Ray actor mixin classes for distributed execution.

This module provides mixin classes for Ray actors, including utilities
for distributed environment initialization and communication group setup.
"""

import os
import socket
from typing import Any, Dict, List, Optional, Tuple, Type

import ray
import torch

from rlightning.utils.distributed import initialize_distributed_env
from rlightning.utils.distributed.comm_context import CommContext
from rlightning.utils.distributed.group_initializer import ParallelMode
from rlightning.utils.logger import get_logger

logger = get_logger(__name__)


class RayActorMixin:
    """Mixin class providing Ray actor functionality.

    This class provides common methods for Ray actors, including GPU
    management, node identification, and network address retrieval.
    Should be used as a base class for any class that needs to be a Ray actor.
    """

    def init(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the remote class.

        Must be implemented in derived classes.

        Raises:
            NotImplementedError: Always, as this must be overridden.
        """
        raise NotImplementedError("The 'init' method must be implemented in the derived class.")

    def _get_gpu_ids(self) -> List[int]:
        """Get the GPU IDs assigned to this actor.

        Returns:
            List of GPU IDs available to this actor.

        Raises:
            RuntimeError: If no GPU is found or multiple GPUs are assigned.
        """
        gpu_id = ray.get_gpu_ids()
        if len(gpu_id) == 0:
            raise RuntimeError("No GPU found on the current node.")
        elif len(gpu_id) > 1:
            raise RuntimeError("Multiple GPUs found on the current node, only support single GPU per node currently.")

        return gpu_id

    @classmethod
    def as_remote(
        cls,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        memory: Optional[int] = None,
        object_store_memory: Optional[int] = None,
        resources: Optional[Dict[str, float]] = None,
    ) -> Type:
        """Create a remote class for Ray Actor initialization.

        Args:
            num_cpus: Number of CPUs required.
            num_gpus: Number of GPUs required.
            memory: Memory required in bytes.
            object_store_memory: Object store memory required.
            resources: Custom resource requirements.

        Returns:
            Ray remote class.
        """

        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

    def _get_node_id(self) -> str:
        """Get the node ID where this actor is running.

        Returns:
            Node ID string.
        """
        return ray.get_runtime_context().get_node_id()

    def _get_addr_and_port(self) -> Tuple[str, int]:
        """Get the IP address and a free port for this node.

        Returns:
            Tuple of (master_addr, master_port).
        """

        def _get_free_port() -> int:
            with socket.socket() as sock:
                sock.bind(("", 0))
                return sock.getsockname()[1]

        master_addr = ray.util.get_node_ip_address()
        master_port = _get_free_port()
        return master_addr, master_port


class DistributedMixin:
    """Mixin class for distributed training functionality.

    Provides methods for initializing distributed environments and
    communication groups across Ray actors.
    """

    def init_distributed_env(
        self,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        master_addr: str,
        master_port: int,
        backend: str = "nccl",
        dist_url: str = "env://",
        timeout: int = 1800,
    ) -> None:
        """Initialize the distributed environment for this actor.

        Sets up environment variables and initializes PyTorch distributed.

        Args:
            rank: Global rank of this process.
            world_size: Total number of processes.
            local_rank: Local rank within the node.
            local_world_size: Number of processes on this node.
            master_addr: Address of the master node.
            master_port: Port of the master node.
            backend: Communication backend ('nccl' or 'gloo').
            dist_url: URL for distributed initialization.
            timeout: Timeout in seconds for initialization.
        """
        # setup distribute os environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)

        initialize_distributed_env(rank, world_size, backend, dist_url, timeout)
        logger.debug("Distributed environment initialized.")

    def init_single_comm_group(
        self, ranks: List[int], mode: ParallelMode, backend: str = "nccl", use_cpu: bool = False
    ) -> None:
        """Initialize a single communication group for the given ranks.

        Args:
            ranks: List of global ranks to include in the group.
            mode: The parallel mode for this communication group.
            backend: Communication backend ('nccl' or 'gloo').
            use_cpu: If True, use CPU tensors for communication.

        Raises:
            AssertionError: If distributed environment is not initialized.
        """
        assert torch.distributed.is_initialized(), "Distributed environment is not initialized."
        CommContext().init_group(ranks, mode, backend, use_cpu)
        logger.debug(f"Communication Group {mode} initialized.")

    def get_rank(self) -> int:
        """Get the global rank of this process.

        Returns:
            Global rank.

        Raises:
            AssertionError: If distributed environment is not initialized.
        """
        assert torch.distributed.is_initialized(), "Distributed environment is not initialized."
        return CommContext().get_global_rank()

    def dist_barrier(self, mode: ParallelMode = None, device_ids: List[int] = None):
        """Barrier for distributed synchronization.

        By default, this barriers on the *global* process group.

        When `mode` is provided (e.g., ParallelMode.TRAIN_DATA_PARALLEL), it will
        barrier on that specific communication group. This is important when only
        a subset of ranks participate in the group; using the global barrier would
        deadlock.
        """

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return

        group = None
        if mode is not None:
            group = CommContext().get_group(mode)
            # If this rank is not part of the group, there's nothing to sync on.
            if group is None:
                return

        # For NCCL, passing device_ids avoids warnings/hangs when the device mapping
        # is not inferred (common for barrier-only calls).
        if device_ids is None and torch.cuda.is_available():
            try:
                device_ids = [torch.cuda.current_device()]
            except Exception:
                device_ids = None

        if device_ids is not None:
            torch.distributed.barrier(group=group, device_ids=device_ids)
        else:
            torch.distributed.barrier(group=group)
