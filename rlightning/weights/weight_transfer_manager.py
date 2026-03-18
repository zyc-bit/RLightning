import gc
from collections import defaultdict
from typing import Dict, List, Sequence, Union

import ray
import torch
from ray.actor import ActorHandle

from rlightning.utils.distributed.comm_context import CommContext
from rlightning.utils.distributed.group_initializer import ParallelMode
from rlightning.utils.logger import get_logger
from rlightning.utils.utils import InternalFlag
from rlightning.weights.utils import tensor_state_dict_to_numpy
from rlightning.weights.weight_buffer import CPUWeightBuffer

from .weight_buffer import WeightBuffer

logger = get_logger(__name__)


class WeightTransferManager:
    def __init__(self, buffer_strategy: str):
        self.buffer_strategy = buffer_strategy

    def clear_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

    def record_gpu_memory(self, record_name: str):
        allocated_memory = torch.cuda.memory_allocated()
        max_reserved_memory = torch.cuda.max_memory_reserved()
        logger.debug(f"Allocated Memory: {allocated_memory/1024/1024/1024} GB {record_name}")
        logger.debug(f"Max Reserved Memory: {max_reserved_memory/1024/1024/1024} GB {record_name}")

    def _group_params_by_dtype(self, state_dict: Dict[str, Dict]):
        params_by_dtype = defaultdict(list)
        metadata_by_dtype = defaultdict(list)
        for module_name, module_state_dict in state_dict.items():
            for name, param in module_state_dict.items():
                dtype = param.dtype
                params_by_dtype[dtype].append(param)
                metadata_by_dtype[dtype].append(
                    {
                        "module_name": module_name,
                        "name": name,
                        "shape": param.shape,
                        "numel": param.numel(),
                    }
                )
        return params_by_dtype, metadata_by_dtype

    def _flat_tensor_to_state_dict(
        self,
        params_by_dtype: Dict[torch.dtype, torch.Tensor],
        metadata_by_dtype: Dict[torch.dtype, Sequence],
        to_numpy: bool = False,
    ):
        state_dict = {}
        for dtype, metadata_list in metadata_by_dtype.items():
            flat_tensor = params_by_dtype[dtype]
            offset = 0
            for meta in metadata_list:
                if meta["module_name"] not in state_dict:
                    state_dict[meta["module_name"]] = {}
                n_param = meta["numel"]
                param_flat = flat_tensor.narrow(0, offset, n_param).view(meta["shape"])
                if to_numpy:
                    if dtype == torch.bfloat16:
                        state_dict[meta["module_name"]][meta["name"]] = param_flat.to("cpu").view(torch.uint16).numpy()
                    else:
                        state_dict[meta["module_name"]][meta["name"]] = param_flat.to("cpu").numpy()
                else:
                    state_dict[meta["module_name"]][meta["name"]] = param_flat
                offset += n_param
        return state_dict

    def send_weights_cpu(self, state_dict: Dict[str, Dict[str, torch.Tensor]], cpu_weight_buffer: ActorHandle):
        """
        Sends weights to the CPU weight buffer actor.
        """
        cpu_state_dict = tensor_state_dict_to_numpy(state_dict)
        if isinstance(cpu_weight_buffer, ActorHandle):
            cpu_state_dict = ray.put(cpu_state_dict)
            ray.get(cpu_weight_buffer.add_state_dict.remote(cpu_state_dict))
        else:
            cpu_weight_buffer.add_state_dict(cpu_state_dict)

    def send_weights(self, state_dict: Dict[str, Dict[str, torch.Tensor]], eval_policies: List[ActorHandle]):
        """Broadcasts the state dict to the eval policies.

        This function is called on the TrainPolicy under sync mode. It dispatches to the correct sending strategy based on the eval policies' configuration.

        Args:
            state_dict (Dict[str, Dict[str, torch.Tensor]]): A dict of module state dicts.
            eval_policies (List[ActorHandle]): A list of Actors which related to policies for evaluation

        Raises:
            ValueError: Invalid buffer strategy
        """

        if self.buffer_strategy == "None":
            # Broadcast weights layer by layer
            context_group = CommContext().get_group(ParallelMode.WEIGHT_TRANSFER)
            # The sender is always ranks_in_group[0] (train policy's global rank).
            src_rank = CommContext().get_ranks_in_group(ParallelMode.WEIGHT_TRANSFER)[0]
            num_layers = sum(len(module_state_dict) for module_state_dict in state_dict.values())
            cur_layer = 0
            for module_name, module_state_dict in state_dict.items():
                for name, param in module_state_dict.items():
                    cur_layer += 1
                    shape, dtype = param.shape, param.dtype
                    refs = [
                        eval_policy.sync_weights_layer_by_layers.remote(
                            module_name, name, dtype, shape, is_last=(cur_layer == num_layers)
                        )
                        for eval_policy in eval_policies
                    ]
                    torch.distributed.broadcast(param.data, src=src_rank, group=context_group)
            ray.get(refs)
        else:
            # 1. Group parameters by dtype
            params_by_dtype, metadata_by_dtype = self._group_params_by_dtype(state_dict)

            # 2. Asynchronously call recv_weights on all eval policies to get them ready
            if not InternalFlag.REMOTE_EVAL:
                # single process mode
                for eval_policy in eval_policies:
                    eval_policy.recv_weights_local(state_dict)

                return

            refs = [eval_policy.recv_weights.remote(metadata_by_dtype) for eval_policy in eval_policies]

            # 3. Send weights from trainer. The corresponding receive logic is on the policy.
            if self.buffer_strategy in ["Double", "Shared"]:
                self._send_weights_full(params_by_dtype)
            elif self.buffer_strategy == "Sharded":
                self._send_weights_sharded(params_by_dtype)
            else:
                raise ValueError(f"Invalid buffer strategy: {self.buffer_strategy}")

            ray.get(refs)

    def recv_weights(self, weight_buffer, metadata_by_dtype):
        """
        Receive weights from the train policy, grouped by dtype.
        Delegates the buffering and application logic to the configured WeightBuffer.
        This method is called by policies that do NOT use the shared buffer actor.
        """
        if self.buffer_strategy in ["Double", "Shared"]:
            self._recv_weights_full(weight_buffer, metadata_by_dtype)
        elif self.buffer_strategy == "Sharded":
            self._recv_weights_sharded(weight_buffer, metadata_by_dtype)
        else:
            raise ValueError(f"Invalid buffer strategy: {self.buffer_strategy}")

    def recv_weights_local(self, weight_buffer, state_dict):
        """
        Receive weights directly for single process mode (non-remote).
        """
        if isinstance(weight_buffer, ray.actor.ActorHandle):
            weight_buffer.clear.remote()
        else:
            weight_buffer.clear()

        if isinstance(weight_buffer, ray.actor.ActorHandle):
            assert self.buffer_strategy == "Shared", " Only Shared buffer strategy support ray actor"
            ray.get(weight_buffer.add_state_dict.remote(state_dict))
        else:
            if isinstance(weight_buffer, CPUWeightBuffer):
                state_dict = tensor_state_dict_to_numpy(state_dict)
            weight_buffer.add_state_dict(state_dict)

    def _send_weights_full(self, params_by_dtype):
        """
        Sends full weight tensors for the double-buffer strategy.
        """
        context_group = CommContext().get_group(ParallelMode.WEIGHT_TRANSFER)
        # Use the global rank of the sender (train, always at position 0 in the group).
        src_rank = CommContext().get_ranks_in_group(ParallelMode.WEIGHT_TRANSFER)[0]
        sorted_dtypes = sorted(list(params_by_dtype.keys()), key=lambda x: str(x))
        work_handles = []
        for dtype in sorted_dtypes:
            params = params_by_dtype[dtype]
            flat_params = torch.nn.utils.parameters_to_vector(params).contiguous()

            # Use asynchronous broadcast so that trainer and all evals can pipeline the transfers
            work = torch.distributed.broadcast(flat_params, src=src_rank, group=context_group, async_op=True)
            work_handles.append(work)

        # Synchronize all outstanding broadcasts to guarantee completion before we exit.
        for work in work_handles:
            work.wait()

    def _recv_weights_full(
        self,
        weight_buffer: Union[WeightBuffer, ActorHandle],
        metadata_by_dtype: Dict[torch.Tensor, Sequence],
    ):
        """Implementation for the double buffer strategy.

        Receives full weights and applies them directly.

        Args:
            weight_buffer (Union[WeightBuffer, ActorHandle]): Weight buffer or an actor of that.
            metadata_by_dtype (Dict[torch.Tensor, Sequence]): Metadata grouped by dtype for
                reconstructing flattened tensors back to per-module state dicts.
        """
        context_group = CommContext().get_group(ParallelMode.WEIGHT_TRANSFER)
        # The sender is always ranks_in_group[0] (train policy's global rank).
        src_rank = CommContext().get_ranks_in_group(ParallelMode.WEIGHT_TRANSFER)[0]
        # prevent peak memory usage three times memory of weights
        if isinstance(weight_buffer, ActorHandle):
            weight_buffer.clear.remote()
        else:
            weight_buffer.clear()

        sorted_dtypes = sorted(list(metadata_by_dtype.keys()), key=lambda x: str(x))

        received_flat_tensors = {}
        work_handles = {}
        for dtype in sorted_dtypes:
            metadata_list = metadata_by_dtype[dtype]

            numel_total = sum(m["numel"] for m in metadata_list)
            flat_weights = torch.empty(numel_total, dtype=dtype, device="cuda")

            # Asynchronous receive side of broadcast
            work = torch.distributed.broadcast(flat_weights, src=src_rank, group=context_group, async_op=True)
            received_flat_tensors[dtype] = flat_weights
            work_handles[dtype] = work

        # Wait for all broadcasts to finish
        for work in work_handles.values():
            work.wait()

        # Add state dict to weight buffer
        if isinstance(weight_buffer, ActorHandle):
            assert self.buffer_strategy == "Shared", " Only Shared buffer strategy support ray actor"
            state_dict = self._flat_tensor_to_state_dict(received_flat_tensors, metadata_by_dtype, to_numpy=True)
            state_dict = ray.put(state_dict)
            ray.get(weight_buffer.add_state_dict.remote(state_dict))
        else:
            to_numpy = isinstance(weight_buffer, CPUWeightBuffer)
            state_dict = self._flat_tensor_to_state_dict(received_flat_tensors, metadata_by_dtype, to_numpy=to_numpy)
            weight_buffer.add_state_dict(state_dict)

    def _send_weights_sharded(self, params_by_dtype):
        """
        Sends sharded weights for the sharded-buffer strategy.
        """
        context_group = CommContext().get_group(ParallelMode.WEIGHT_TRANSFER)
        # The sender (train) is always at position 0 in the WEIGHT_TRANSFER ranks list.
        src_rank = CommContext().get_ranks_in_group(ParallelMode.WEIGHT_TRANSFER)[0]
        total_world_size = CommContext().get_world_size(ParallelMode.WEIGHT_TRANSFER)
        eval_world_size = total_world_size - 1

        sorted_dtypes = sorted(list(params_by_dtype.keys()), key=lambda x: str(x))

        for dtype in sorted_dtypes:
            params = params_by_dtype[dtype]
            flat_params = torch.nn.utils.parameters_to_vector(params)

            # Pad tensor to be divisible by eval_world_size（只考虑eval policies）
            numel = flat_params.numel()
            padded_numel = numel
            if padded_numel % eval_world_size != 0:
                padding_size = eval_world_size - (padded_numel % eval_world_size)
                padding = torch.zeros(padding_size, dtype=flat_params.dtype, device=flat_params.device)
                flat_params = torch.cat([flat_params, padding])

            # 为eval policies创建分片
            eval_param_shards = list(flat_params.chunk(eval_world_size, dim=0))

            # 为源rank 0（train policy）创建一个dummy分片 这个分片不会被使用，但scatter操作需要它
            dummy_shard = torch.empty_like(eval_param_shards[0])

            # 构建完整的scatter_list：[train_shard, eval_shard1, eval_shard2, ...]
            param_shards = [dummy_shard] + eval_param_shards
            output_tensor = torch.empty_like(dummy_shard)

            # 执行scatter操作
            torch.distributed.scatter(
                tensor=output_tensor,
                scatter_list=param_shards,
                src=src_rank,
                group=context_group,
                async_op=False,
            )

    def _recv_weights_sharded(self, weight_buffer, metadata_by_dtype):
        """
        Implementation for the sharded buffer strategy.
        Receives only a slice of the weights from the trainer.
        """
        context_group = CommContext().get_group(ParallelMode.WEIGHT_TRANSFER)
        # prevent peak memory usage three times memory of weights
        weight_buffer.clear()

        # The trainer is always at position 0 in the WEIGHT_TRANSFER ranks list.
        trainer_rank = CommContext().get_ranks_in_group(ParallelMode.WEIGHT_TRANSFER)[0]
        total_world_size = CommContext().get_world_size(ParallelMode.WEIGHT_TRANSFER)
        eval_world_size = total_world_size - 1  # 只有eval policies参与权重分片

        sorted_dtypes = sorted(list(metadata_by_dtype.keys()), key=lambda x: str(x))

        for dtype in sorted_dtypes:
            is_last = dtype == sorted_dtypes[-1]
            metadata_list = metadata_by_dtype[dtype]

            numel_total = sum(m["numel"] for m in metadata_list)

            # Pad to be divisible by eval_world_size（只考虑eval policies）
            padded_numel_total = numel_total
            if padded_numel_total % eval_world_size != 0:
                padded_numel_total += eval_world_size - (padded_numel_total % eval_world_size)

            numel_shard = padded_numel_total // eval_world_size

            shard_tensor = torch.empty(numel_shard, dtype=dtype, device="cuda")

            # 接收端不需要提供scatter_list
            torch.distributed.scatter(
                tensor=shard_tensor,
                scatter_list=None,
                src=trainer_rank,
                group=context_group,
                async_op=False,
            )

            weight_buffer.add_shard(dtype, shard_tensor, metadata_list, numel_total, is_last)
