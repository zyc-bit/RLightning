"""Process group initializers for distributed training.

This module provides classes for initializing PyTorch distributed process
groups with various parallel modes including data parallel, tensor parallel,
sequence parallel, and weight transfer groups.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from torch import distributed as dist
from torch.distributed import ProcessGroup


class CommMode:
    """Communication mode identifiers for process groups.

    Attributes:
        GLOBAL: Global communication group spanning all processes.
        INTRA_NODE: Communication group within a single node.
        INTER_NODE: Communication group across nodes (one process per node).
    """

    GLOBAL: str = "GLOBAL"
    INTRA_NODE: str = "INTRA_NODE"
    INTER_NODE: str = "INTER_NODE"


class ParallelMode(CommMode):
    """Parallel mode identifiers extending CommMode.

    Attributes:
        TRAIN_DATA_PARALLEL: Data parallel group for training.
        DATA_PARALLEL: General data parallel group.
        TENSOR_PARALLEL: Tensor parallel group for model parallelism.
        SEQUENCE_PARALLEL: Sequence parallel group for long sequences.
        WEIGHT_TRANSFER: Group for transferring model weights.
    """

    TRAIN_DATA_PARALLEL: str = "TRAIN_DATA_PARALLEL"
    DATA_PARALLEL: str = "DATA_PARALLEL"
    TENSOR_PARALLEL: str = "TENSOR_PARALLEL"
    SEQUENCE_PARALLEL: str = "SEQUENCE_PARALLEL"
    WEIGHT_TRANSFER: str = "WEIGHT_TRANSFER"


class ProcessGroupInitializer(ABC):
    """Abstract base class for process group initialization.

    This class provides the foundation for creating PyTorch distributed
    process groups with various configurations.

    Attributes:
        rank: Global rank of this process.
        world_size: Total number of processes.
        local_rank: Local rank within the initialized group.
        ranks_in_group: List of global ranks in the group.
        process_group: The initialized ProcessGroup instance.
        group_world_size: Size of the initialized group.
        mode: Communication mode identifier for this group.
    """

    def __init__(self, rank: int, world_size: int) -> None:
        """Initialize the process group initializer.

        Args:
            rank: Global rank of this process.
            world_size: Total number of processes.
        """
        self.rank: int = rank
        self.world_size: int = world_size

        self.local_rank: Optional[int] = None
        self.ranks_in_group: Optional[List[int]] = None
        self.process_group: Optional["ProcessGroup"] = None
        self.group_world_size: Optional[int] = None
        self.mode: Optional[str] = None

    @abstractmethod
    def init_dist_group(
        self, backend: str = "nccl", use_cpu: bool = False
    ) -> Tuple[
        Optional[int], Optional[int], Optional["ProcessGroup"], Optional[List[int]], Optional[str]
    ]:
        """Initialize the distributed group.

        Args:
            backend: Communication backend ('nccl' or 'gloo').
            use_cpu: If True, use CPU tensors (forces 'gloo' backend).

        Returns:
            Tuple of (local_rank, group_world_size, process_group, ranks_in_group, mode).
        """
        pass

    def _new_and_update_group_info(
        self, ranks: List[int], use_cpu: bool = False, backend: str = "nccl"
    ) -> None:
        """Create a new process group and update internal state.

        Args:
            ranks: List of global ranks to include in the group.
            use_cpu: If True, use 'gloo' backend for CPU tensors.
            backend: Communication backend to use.
        """
        backend = "gloo" if use_cpu else backend
        group = dist.new_group(ranks, backend=backend)

        if self.rank in ranks:
            self.local_rank = ranks.index(self.rank)
            self.group_world_size = len(ranks)
            self.process_group = group
            self.ranks_in_group = ranks


class TrainDPGroupInitializer(ProcessGroupInitializer):
    """Data parallel group initializer for training.

    Creates a process group containing specified ranks for data parallel
    training synchronization.

    Attributes:
        ranks: List of global ranks to include in this group.
    """

    def __init__(self, rank: int, world_size: int, ranks: List[int]) -> None:
        """Initialize the training data parallel group initializer.

        Args:
            rank: Global rank of this process.
            world_size: Total number of processes.
            ranks: List of global ranks to include in the group.
        """
        super().__init__(rank, world_size)
        self.ranks: List[int] = ranks
        self.mode = ParallelMode.TRAIN_DATA_PARALLEL

    def init_dist_group(
        self, backend: str = "nccl", use_cpu: bool = False
    ) -> Tuple[Optional[int], Optional[int], Optional["ProcessGroup"], Optional[List[int]], str]:
        """Initialize the training data parallel group.

        Args:
            backend: Communication backend ('nccl' or 'gloo').
            use_cpu: If True, use CPU tensors.

        Returns:
            Tuple of (local_rank, group_world_size, process_group, ranks_in_group, mode).
        """
        ranks = self.ranks
        self._new_and_update_group_info(ranks, use_cpu, backend)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


class IntraNodeGroupInitializer(ProcessGroupInitializer):
    """Intra-node process group initializer.

    Creates a process group for communication within a single node.

    Attributes:
        ranks: List of global ranks within the node.
    """

    def __init__(self, rank: int, world_size: int, ranks: List[int]) -> None:
        """Initialize the intra-node group initializer.

        Args:
            rank: Global rank of this process.
            world_size: Total number of processes.
            ranks: List of global ranks within the same node.
        """
        super().__init__(rank, world_size)
        self.ranks: List[int] = ranks
        self.mode = CommMode.INTRA_NODE

    def init_dist_group(
        self, backend: str = "nccl", use_cpu: bool = False
    ) -> Tuple[Optional[int], Optional[int], Optional["ProcessGroup"], Optional[List[int]], str]:
        """Initialize the intra-node communication group.

        Args:
            backend: Communication backend ('nccl' or 'gloo').
            use_cpu: If True, use CPU tensors.

        Returns:
            Tuple of (local_rank, group_world_size, process_group, ranks_in_group, mode).
        """
        self._new_and_update_group_info(self.ranks, use_cpu, backend)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


class WeightTransferGroupInitializer(ProcessGroupInitializer):
    """Weight transfer process group initializer.

    Creates a process group for transferring model weights between processes.

    Attributes:
        ranks: List of global ranks participating in weight transfer.
    """

    def __init__(self, rank: int, world_size: int, ranks: List[int]) -> None:
        """Initialize the weight transfer group initializer.

        Args:
            rank: Global rank of this process.
            world_size: Total number of processes.
            ranks: List of global ranks for weight transfer.
        """
        super().__init__(rank, world_size)
        self.ranks: List[int] = ranks
        self.mode = ParallelMode.WEIGHT_TRANSFER

    def init_dist_group(
        self, backend: str = "nccl", use_cpu: bool = False
    ) -> Tuple[Optional[int], Optional[int], Optional["ProcessGroup"], Optional[List[int]], str]:
        """Initialize the weight transfer communication group.

        Args:
            backend: Communication backend ('nccl' or 'gloo').
            use_cpu: If True, use CPU tensors.

        Returns:
            Tuple of (local_rank, group_world_size, process_group, ranks_in_group, mode).
        """
        self._new_and_update_group_info(self.ranks, use_cpu, backend)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


class DPGroupInitializer(ProcessGroupInitializer):
    """Data parallel group initializer.

    Creates data parallel groups based on the specified parallelism size.
    Groups are formed by selecting processes at regular intervals.

    Attributes:
        data_parallel_size: Number of processes in each data parallel group.
        process_num_between_dp_rank: Stride between data parallel ranks.
    """

    def __init__(self, rank: int, world_size: int, data_parallel_size: int) -> None:
        """Initialize the data parallel group initializer.

        Args:
            rank: Global rank of this process.
            world_size: Total number of processes.
            data_parallel_size: Number of processes per data parallel group.
        """
        super().__init__(rank, world_size)
        self.data_parallel_size: int = data_parallel_size
        self.process_num_between_dp_rank: int = world_size // data_parallel_size
        self.mode = ParallelMode.DATA_PARALLEL

    def init_dist_group(
        self, backend: str = "nccl", use_cpu: bool = False
    ) -> Tuple[Optional[int], Optional[int], Optional["ProcessGroup"], Optional[List[int]], str]:
        """Initialize data parallel groups.

        Creates multiple groups where each group contains processes
        separated by process_num_between_dp_rank stride.

        Args:
            backend: Communication backend ('nccl' or 'gloo').
            use_cpu: If True, use CPU tensors.

        Returns:
            Tuple of (local_rank, group_world_size, process_group, ranks_in_group, mode).
        """
        for j in range(self.process_num_between_dp_rank):
            ranks = [
                i * self.process_num_between_dp_rank + j for i in range(self.data_parallel_size)
            ]
            self._new_and_update_group_info(ranks, use_cpu, backend)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


class TPGroupInitializer(ProcessGroupInitializer):
    """Tensor parallel group initializer.

    Creates tensor parallel groups for model parallelism where
    model parameters are sharded across processes.

    Attributes:
        tensor_parallel_size: Number of processes in each tensor parallel group.
        tensor_parallel_group_num: Total number of tensor parallel groups.
    """

    def __init__(self, rank: int, world_size: int, tensor_parallel_size: int) -> None:
        """Initialize the tensor parallel group initializer.

        Args:
            rank: Global rank of this process.
            world_size: Total number of processes.
            tensor_parallel_size: Number of processes per tensor parallel group.
        """
        super().__init__(rank, world_size)
        self.tensor_parallel_size: int = tensor_parallel_size
        self.tensor_parallel_group_num: int = world_size // tensor_parallel_size
        self.mode = ParallelMode.TENSOR_PARALLEL

    def init_dist_group(
        self, backend: str = "nccl", use_cpu: bool = False
    ) -> Tuple[Optional[int], Optional[int], Optional["ProcessGroup"], Optional[List[int]], str]:
        """Initialize tensor parallel groups.

        Creates groups of consecutive processes for tensor parallelism.

        Args:
            backend: Communication backend ('nccl' or 'gloo').
            use_cpu: If True, use CPU tensors.

        Returns:
            Tuple of (local_rank, group_world_size, process_group, ranks_in_group, mode).
        """
        for i in range(self.tensor_parallel_group_num):
            ranks = [i * self.tensor_parallel_size + j for j in range(self.tensor_parallel_size)]
            self._new_and_update_group_info(ranks, use_cpu, backend)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


class SPGroupInitializer(ProcessGroupInitializer):
    """Sequence parallel group initializer.

    Creates sequence parallel groups for parallelizing long sequence
    processing across multiple processes.

    Attributes:
        sequence_parallel_size: Number of processes in each sequence parallel group.
        sequence_parallel_group_num: Total number of sequence parallel groups.
    """

    def __init__(self, rank: int, world_size: int, sequence_parallel_size: int) -> None:
        """Initialize the sequence parallel group initializer.

        Args:
            rank: Global rank of this process.
            world_size: Total number of processes.
            sequence_parallel_size: Number of processes per sequence parallel group.
        """
        super().__init__(rank, world_size)
        self.sequence_parallel_size: int = sequence_parallel_size
        self.sequence_parallel_group_num: int = world_size // sequence_parallel_size
        self.mode = ParallelMode.SEQUENCE_PARALLEL

    def init_dist_group(
        self, backend: str = "nccl", use_cpu: bool = False
    ) -> Tuple[Optional[int], Optional[int], Optional["ProcessGroup"], Optional[List[int]], str]:
        """Initialize sequence parallel groups.

        Creates groups of consecutive processes for sequence parallelism.

        Args:
            backend: Communication backend ('nccl' or 'gloo').
            use_cpu: If True, use CPU tensors.

        Returns:
            Tuple of (local_rank, group_world_size, process_group, ranks_in_group, mode).
        """
        for i in range(self.sequence_parallel_group_num):
            ranks = [
                (i * self.sequence_parallel_size + j) for j in range(self.sequence_parallel_size)
            ]
            self._new_and_update_group_info(ranks, use_cpu, backend)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


# class IntraNodeGroupInitializer(ProcessGroupInitializer):
#     """intra node group initializer"""

#     def __init__(self, rank, world_size, local_world_size) -> None:
#         super().__init__(rank, world_size)
#         self.local_world_size = local_world_size
#         self.node_count = world_size // local_world_size
#         self.mode = CommMode.INTRA_NODE

#     def init_dist_group(self):
#         for i in range(self.node_count):
#             ranks = list(range(i * self.local_world_size, (i + 1) * self.local_world_size))
#             self._new_and_update_group_info(ranks)

#         return (
#             self.local_rank,
#             self.group_world_size,
#             self.process_group,
#             self.ranks_in_group,
#             self.mode,
#         )


class InterNodeGroupInitializer(ProcessGroupInitializer):
    """Inter-node process group initializer.

    Creates process groups for communication across nodes, with one
    representative process per node in each group.

    Attributes:
        local_world_size: Number of processes per node.
        node_count: Total number of nodes in the cluster.
    """

    def __init__(self, rank: int, world_size: int, local_world_size: int) -> None:
        """Initialize the inter-node group initializer.

        Args:
            rank: Global rank of this process.
            world_size: Total number of processes.
            local_world_size: Number of processes per node.
        """
        super().__init__(rank, world_size)
        self.local_world_size: int = local_world_size
        self.node_count: int = world_size // local_world_size
        self.mode = CommMode.INTER_NODE

    def init_dist_group(
        self, backend: str = "nccl", use_cpu: bool = False
    ) -> Tuple[Optional[int], Optional[int], Optional["ProcessGroup"], Optional[List[int]], str]:
        """Initialize inter-node communication groups.

        Creates groups where each group contains processes with the same
        local rank across different nodes.

        Args:
            backend: Communication backend ('nccl' or 'gloo').
            use_cpu: If True, use CPU tensors.

        Returns:
            Tuple of (local_rank, group_world_size, process_group, ranks_in_group, mode).
        """
        for i in range(self.local_world_size):
            ranks = [i + j * self.local_world_size for j in range(self.node_count)]
            self._new_and_update_group_info(ranks, use_cpu, backend)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )
