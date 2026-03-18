"""Collective communication operations for distributed training.

This module provides wrappers around PyTorch distributed collective
operations that work with the CommContext singleton for group management.
Includes scatter, broadcast, gather, all_reduce, all_gather, and
sequence parallel communication utilities.
"""

# pylint: disable=W0613
from typing import Any, List, Optional

import torch
from torch import distributed as dist
from torch.distributed import ReduceOp

from .comm_context import CommContext
from .group_initializer import CommMode, ParallelMode


def scatter(
    tensor: torch.Tensor,
    comm_mode: CommMode,
    scatter_list: Optional[List[torch.Tensor]] = None,
    src: int = 0,
    async_op: bool = False,
) -> Optional[Any]:
    """
    custom scatter operation.

    Args:
        tensor(Tensor): Output tensor.
        comm_mode (CommMode): Communication mode registered in CommContext.
        scatter_list(list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank).
        src(int): Src rank.
        async_op(bool): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.scatter(
        tensor=tensor,
        scatter_list=scatter_list,
        src=src,
        group=group,
        async_op=async_op,
    )


def broadcast_object_list(object_list: List[Any], comm_mode: CommMode, src: int = 0) -> None:
    """
    Broadcasts python objects based on torch.distributed.broadcast_object_list

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
        comm_mode (CommMode): Communication mode registered in CommContext.

    Returns:
        ``None``
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.broadcast_object_list(object_list, src=src, group=group)


def all_gather(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    comm_mode: CommMode,
    async_op: bool = False,
) -> Optional[Any]:
    """
    Single tensor all gather. Gathers a single tensor from all ranks, and puts them in a single output tensor.

    Args:
        output_tensor (Tensor): Output tensor. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor (Tensor): Tensor to be broadcast from current process.
        comm_mode (CommMode): Communication mode registered in CommContext.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.all_gather_into_tensor(
        output_tensor=output_tensor, input_tensor=input_tensor, group=group, async_op=async_op
    )


def all_reduce(
    tensor: torch.Tensor,
    comm_mode: CommMode,
    op: ReduceOp = ReduceOp.SUM,
    async_op: bool = False,
) -> Optional[Any]:
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Complex tensors are supported.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        comm_mode (CommMode): Communication mode registered in CommContext.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)


def all_reduce_dict(dictionary: dict, comm_mode: CommMode, op=ReduceOp.SUM, dtype=torch.float32):
    """
    Reduces the dictionary data across all machines in such a way that all get
    the final result.
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    keys = sorted(dictionary)
    tensor = torch.as_tensor(
        [dictionary[k] for k in keys], dtype=dtype, device=torch.cuda.current_device()
    )
    dist.all_reduce(tensor, op=op, group=group)
    return dict(zip(keys, tensor.tolist()))


def broadcast(
    tensor: torch.Tensor, comm_mode: CommMode, src: int = 0, async_op: bool = False
) -> Optional[Any]:
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Args:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        comm_mode (CommMode): Communication mode registered in CommContext.
        src (int): Source rank.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)


def gather(
    tensor: torch.Tensor,
    comm_mode: CommMode,
    gather_list: Optional[List[torch.Tensor]] = None,
    dst: int = 0,
    async_op: bool = False,
) -> Optional[Any]:
    """
    Gathers a list of tensors in a single process.

    Args:
        tensor (Tensor): Input tensor.
        comm_mode (CommMode): Communication mode registered in CommContext.
        gather_list (list[Tensor], optional): List of appropriately-sized
            tensors to use for gathered data (default is None, must be specified
            on the destination rank)
        dst (int, optional): Destination rank (default is 0)
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.gather(tensor, gather_list=gather_list, dst=dst, group=group, async_op=async_op)


class _AllToAllFunction(torch.autograd.Function):
    """Autograd function for all-to-all communication.

    Implements forward and backward passes for all-to-all collective
    operation with proper gradient handling.
    """

    @staticmethod
    def forward(
        ctx: Any, input_: torch.Tensor, gather_dim: int, scatter_dim: int, group: Any
    ) -> torch.Tensor:
        """Forward pass for all-to-all communication.

        Args:
            ctx: Autograd context for saving tensors.
            input_: Input tensor to redistribute.
            gather_dim: Dimension to gather along.
            scatter_dim: Dimension to scatter along.
            group: Process group for communication.

        Returns:
            Redistributed tensor.
        """
        assert gather_dim != scatter_dim
        assert 0 <= gather_dim < input_.ndim
        assert 0 <= scatter_dim < input_.ndim
        world_size = dist.get_world_size(group)
        assert input_.size(scatter_dim) % world_size == 0

        ctx.gather_dim = gather_dim
        ctx.scatter_dim = scatter_dim
        ctx.group = group

        if world_size == 1:
            return input_

        inputs = [x.contiguous() for x in input_.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(x) for x in inputs]
        dist.all_to_all(outputs, inputs, group=group)

        return torch.cat(outputs, dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Backward pass for all-to-all communication.

        Args:
            ctx: Autograd context with saved tensors.
            grad_output: Gradient from downstream.

        Returns:
            Tuple of gradients (grad_input, None, None, None).
        """
        group = ctx.group
        world_size = dist.get_world_size(group)
        gather_dim = ctx.gather_dim
        scatter_dim = ctx.scatter_dim

        if world_size == 1:
            return grad_output, None, None, None

        grad_outputs = [x.contiguous() for x in grad_output.chunk(world_size, dim=gather_dim)]
        grad_inputs = [torch.empty_like(x) for x in grad_outputs]

        dist.all_to_all(grad_inputs, grad_outputs, group=group)

        return torch.cat(grad_inputs, dim=scatter_dim), None, None, None


def all_to_all(
    input_: torch.Tensor, gather_dim: int, scatter_dim: int, group: Any
) -> torch.Tensor:
    """Perform all-to-all communication with autograd support.

    Redistributes tensor data across processes by scattering along one
    dimension and gathering along another.

    Args:
        input_: Input tensor to redistribute.
        gather_dim: Dimension to gather along.
        scatter_dim: Dimension to scatter along.
        group: Process group for communication.

    Returns:
        Redistributed tensor.
    """
    return _AllToAllFunction.apply(input_, gather_dim, scatter_dim, group)


def _sp_split(input_: torch.Tensor) -> torch.Tensor:
    """Split tensor for sequence parallel by selecting local chunk.

    Args:
        input_: Input tensor to split.

    Returns:
        Local chunk of the input tensor.
    """
    sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)
    sp_rank = CommContext().get_local_rank(ParallelMode.SEQUENCE_PARALLEL)
    if sp_size == 1:
        return input_
    assert input_.size(1) % sp_size == 0
    return input_.chunk(sp_size, dim=1)[sp_rank].contiguous()


def _sp_scatter(input_: torch.Tensor) -> torch.Tensor:
    """Scatter tensor from rank 0 to all sequence parallel ranks.

    Args:
        input_: Input tensor (only valid on rank 0).

    Returns:
        Scattered tensor chunk for this rank.
    """
    sp_group = CommContext().get_group(ParallelMode.SEQUENCE_PARALLEL)
    sp_src = CommContext().get_ranks_in_group(ParallelMode.SEQUENCE_PARALLEL)[0]
    sp_rank = CommContext().get_local_rank(ParallelMode.SEQUENCE_PARALLEL)
    sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)

    if sp_size == 1:
        return input_
    assert input_.size(1) % sp_size == 0
    output = torch.empty(
        [x if i != 1 else x // sp_size for i, x in enumerate(input_.size())],
        dtype=input_.dtype,
        device=input_.device,
    )
    dist.scatter(
        output,
        [x.contiguous() for x in input_.chunk(sp_size, dim=1)] if sp_rank == 0 else None,
        src=sp_src,
        group=sp_group,
    )
    return output


def _sp_gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensor chunks from all sequence parallel ranks.

    Args:
        input_: Local tensor chunk.

    Returns:
        Concatenated tensor from all ranks.
    """
    sp_group = CommContext().get_group(ParallelMode.SEQUENCE_PARALLEL)
    sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)

    if sp_size == 1:
        return input_
    output = [torch.empty_like(input_) for _ in range(sp_size)]
    dist.all_gather(output, input_, group=sp_group)
    return torch.cat(output, dim=1)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Autograd function for scattering to sequence parallel region.

    Scatters input tensor along sequence dimension to distributed ranks
    with proper gradient handling.
    """

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, rank0_only: bool = True) -> torch.Tensor:
        """Forward pass: scatter tensor to sequence parallel ranks.

        Args:
            ctx: Autograd context.
            input_: Input tensor to scatter.
            rank0_only: If True, scatter from rank 0; otherwise, split locally.

        Returns:
            Scattered tensor chunk for this rank.
        """
        if rank0_only:
            return _sp_scatter(input_)
        else:
            return _sp_split(input_)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Backward pass: gather gradients from all ranks.

        Args:
            ctx: Autograd context.
            grad_output: Gradient from downstream.

        Returns:
            Tuple of (gathered gradient, None).
        """
        sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)
        return _sp_gather(grad_output / sp_size), None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Autograd function for gathering from sequence parallel region.

    Gathers tensor chunks from all sequence parallel ranks with proper
    gradient handling.
    """

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, rank0_only: bool = True) -> torch.Tensor:
        """Forward pass: gather tensors from all sequence parallel ranks.

        Args:
            ctx: Autograd context.
            input_: Local tensor chunk.
            rank0_only: If True, only rank 0 had the full tensor originally.

        Returns:
            Concatenated tensor from all ranks.
        """
        ctx.rank0_only = rank0_only
        return _sp_gather(input_)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Backward pass: scatter gradients to all ranks.

        Args:
            ctx: Autograd context.
            grad_output: Gradient from downstream.

        Returns:
            Tuple of (scattered gradient, None).
        """
        sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)
        if ctx.rank0_only:
            return _sp_scatter(grad_output) * sp_size, None
        else:
            return _sp_split(grad_output) * sp_size, None


def scatter_to_sequence_parallel_region(
    input_: torch.Tensor, rank0_only: bool = True
) -> torch.Tensor:
    """Scatter tensor to sequence parallel region.

    Args:
        input_: Input tensor to scatter.
        rank0_only: If True, scatter from rank 0; otherwise, split locally.

    Returns:
        Scattered tensor chunk for this rank.
    """
    return _ScatterToSequenceParallelRegion.apply(input_, rank0_only)


def gather_from_sequence_parallel_region(
    input_: torch.Tensor, rank0_only: bool = True
) -> torch.Tensor:
    """Gather tensor from sequence parallel region.

    Args:
        input_: Local tensor chunk.
        rank0_only: If True, indicates the original tensor was only on rank 0.

    Returns:
        Concatenated tensor from all ranks.
    """
    return _GatherFromSequenceParallelRegion.apply(input_, rank0_only)
