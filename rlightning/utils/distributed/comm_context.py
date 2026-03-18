"""Communication context for distributed training.

This module provides the CommContext singleton class for managing
PyTorch distributed communication groups and process ranks.
"""

import datetime
import gc
import os
from typing import Dict, List, Optional

import torch
from torch import distributed as dist
from torch.distributed import ProcessGroup

from .group_initializer import (
    CommMode,
    IntraNodeGroupInitializer,
    ParallelMode,
    TrainDPGroupInitializer,
    WeightTransferGroupInitializer,
)
from .utils import SingletonMeta


class CommContext(metaclass=SingletonMeta):
    """Singleton communication context for PyTorch distributed.

    Manages process groups, ranks, and world sizes for various
    communication modes in distributed training.

    Attributes:
        _local_ranks: Mapping from CommMode to local rank.
        _global_ranks: Mapping from CommMode to global rank.
        _world_sizes: Mapping from CommMode to world size.
        _ranks_in_group: Mapping from CommMode to list of ranks.
        _groups: Mapping from CommMode to ProcessGroup.
    """

    def __init__(self) -> None:
        """Initialize the communication context."""
        self._local_ranks: Dict[CommMode, int] = {}
        self._global_ranks: Dict[CommMode, int] = {}
        self._world_sizes: Dict[CommMode, int] = {}
        self._ranks_in_group: Dict[CommMode, List[int]] = {}
        self._node_cuda_num: int = torch.cuda.device_count()
        # Build communication groups
        self._groups: Dict[CommMode, "ProcessGroup"] = {}

        self._tensor_parallel_size: int = 1
        self._train_data_parallel_size: int = 1
        self._data_parallel_size: int = 1
        self._sequence_parallel_size: int = 1

    def is_initialized(self) -> bool:
        """Check if the global communication group is initialized.

        Returns:
            True if initialized, False otherwise.
        """
        return self._groups.get(CommMode.GLOBAL) is not None

    def init_distributed_env(
        self,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        backend: str = "nccl",
        dist_url: str = "env://",
        timeout: int = 1800,
    ) -> None:
        """Initialize the PyTorch distributed process group.

        Args:
            world_size: Global world size.
            rank: Global rank of this process.
            backend: Communication backend ('nccl' or 'gloo').
            dist_url: Initialization method URL.
            timeout: Timeout in seconds for initialization.
        """
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(0, timeout),
        )
        torch.distributed.barrier()

        logic_gpu_id = int(os.environ["LOCAL_RANK"]) % torch.cuda.device_count()
        torch.cuda.set_device(f"cuda:{logic_gpu_id}")
        ranks = list(range(world_size))
        # Register global group
        self._register_group(rank, world_size, dist.GroupMember.WORLD, ranks, CommMode.GLOBAL)
        self._global_ranks[CommMode.GLOBAL] = rank

    def _register_group(
        self,
        local_rank: Optional[int],
        world_size: int,
        process_group: "ProcessGroup",
        ranks_in_group: List[int],
        mode: CommMode,
    ) -> None:
        """Register a communication group.

        Args:
            local_rank: Local rank within the group.
            world_size: Size of the group.
            process_group: PyTorch ProcessGroup instance.
            ranks_in_group: List of global ranks in this group.
            mode: Communication mode identifier.
        """
        if local_rank is None:
            return
        self._local_ranks[mode] = local_rank
        self._world_sizes[mode] = world_size
        self._groups[mode] = process_group
        self._ranks_in_group[mode] = ranks_in_group

    def get_group(self, comm_mode: CommMode) -> "ProcessGroup":
        """Get the process group for a communication mode.

        Args:
            comm_mode: The communication mode.

        Returns:
            The corresponding ProcessGroup.
        """
        return self._groups[comm_mode]

    def get_world_size(self, comm_mode: CommMode) -> int:
        """Get the world size for a communication mode.

        Args:
            comm_mode: The communication mode.

        Returns:
            World size for the mode.
        """
        return self._world_sizes[comm_mode]

    def get_global_rank(self) -> int:
        """Get the global rank of this process.

        Returns:
            Global rank.
        """
        return self._global_ranks[CommMode.GLOBAL]

    def get_local_rank(self, comm_mode: CommMode) -> int:
        """Get the local rank for a communication mode.

        Args:
            comm_mode: The communication mode.

        Returns:
            Local rank within the group.
        """
        return self._local_ranks[comm_mode]

    def get_in_node_rank(self) -> int:
        """Get the local rank within the current node.

        Returns:
            Local rank from LOCAL_RANK environment variable.
        """
        return int(os.environ.get("LOCAL_RANK"))

    def get_ranks_in_group(self, comm_mode: CommMode) -> List[int]:
        """Get all ranks in a communication group.

        Args:
            comm_mode: The communication mode.

        Returns:
            List of global ranks in the group.
        """
        return self._ranks_in_group[comm_mode]

    def get_local_world_size(self) -> int:
        """Get the world size within the current node.

        Returns:
            Local world size from LOCAL_WORLD_SIZE environment variable.
        """
        return int(os.environ.get("LOCAL_WORLD_SIZE"))

    def get_intra_node_process_group(self) -> "ProcessGroup":
        """Get the intra-node process group.

        Returns:
            Intra-node ProcessGroup.
        """
        return self.get_group(CommMode.INTRA_NODE)

    def get_inter_node_process_group(self) -> "ProcessGroup":
        """Get the inter-node process group.

        Returns:
            Inter-node ProcessGroup.
        """
        return self.get_group(CommMode.INTER_NODE)

    def is_main_rank(self) -> bool:
        """Check if this is the main (rank 0) process.

        Returns:
            True if global rank is 0.
        """
        return self.get_global_rank() == 0

    def init_group(
        self,
        ranks: List[int],
        mode: Optional[ParallelMode] = None,
        backend: str = "nccl",
        use_cpu: bool = False,
    ) -> None:
        """Initialize a communication group for the given mode.

        Args:
            ranks: List of global ranks to include.
            mode: The parallel mode for this group.
            backend: Communication backend.
            use_cpu: If True, use CPU tensors.

        Raises:
            NotImplementedError: If the parallel mode is not supported.
        """
        rank = self.get_global_rank()
        world_size = self.get_world_size(CommMode.GLOBAL)

        if mode == ParallelMode.TRAIN_DATA_PARALLEL:
            initializer = TrainDPGroupInitializer(rank, world_size, ranks)
        elif mode == ParallelMode.INTRA_NODE:
            initializer = IntraNodeGroupInitializer(rank, world_size, ranks)
        elif mode == ParallelMode.WEIGHT_TRANSFER:
            initializer = WeightTransferGroupInitializer(rank, world_size, ranks)
        else:
            raise NotImplementedError(
                f"Parallel mode {mode} is not supported for group initialization currently."
            )

        group_info_to_register = initializer.init_dist_group(backend, use_cpu)
        self._register_group(*group_info_to_register)

    def destroy(self) -> None:
        """Destroy the distributed environment and clear all groups."""
        if not self.is_initialized():
            return

        dist.barrier()
        dist.destroy_process_group()
        self._local_ranks.clear()
        self._global_ranks.clear()
        self._world_sizes.clear()
        self._ranks_in_group.clear()
        self._groups.clear()

        self._tensor_parallel_size = 1
        self._data_parallel_size = 1
        self._sequence_parallel_size = 1
        gc.collect()
