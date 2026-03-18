import gc
import threading
from copy import deepcopy
from typing import Dict, List, Type

import numpy as np
import torch

from rlightning.utils.config import WeightBufferConfig
from rlightning.utils.distributed.comm_context import CommContext
from rlightning.utils.distributed.group_initializer import ParallelMode
from rlightning.utils.registry import WEIGHTS


@WEIGHTS.register("WeightBuffer")
class WeightBuffer:
    """policy weight buffer, for update weights from train policy to eval policy"""

    VALUE_TYPE: Type = torch.Tensor

    def __init__(self, config: WeightBufferConfig):
        self.config = deepcopy(config)

        self.buffer = {}
        self._is_ready = False
        self._lock = threading.Lock()

    def preprocess(self, *args, **kwargs):
        pass

    def is_ready(self):
        return self._is_ready

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]  # 不序列化 lock
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()  # 反序列化时新建 lock

    def add(self, module_name: str, name: str, weight: torch.Tensor, is_last: bool = False):
        if module_name not in self.buffer:
            self.buffer[module_name] = {}
        self.buffer[module_name][name] = weight

    def _validate_state_dict(self, state_dict):
        """
        Ensures that every value in the nested state_dict matches VALUE_TYPE.
        """
        for module_name, module_state_dict in state_dict.items():
            if not isinstance(module_state_dict, dict):
                raise TypeError(
                    f"State dict for module '{module_name}' must be a dict, "
                    f"got {type(module_state_dict)}"
                )
            for name, param in module_state_dict.items():
                if not isinstance(param, self.VALUE_TYPE):
                    raise TypeError(
                        f"Parameter '{module_name}.{name}' must be of type "
                        f"{self.VALUE_TYPE}, got {type(param)}"
                    )
                return

    def add_state_dict(self, state_dict):
        """
        Add a complete state dictionary to the buffer.
        """
        self._validate_state_dict(state_dict)
        with self._lock:
            self.buffer = state_dict
        self._is_ready = True

    def get_state_dict(self):
        """
        Retrieves the state dictionary from the buffer.
        """
        assert self._is_ready, "Weights buffer is not ready"
        with self._lock:
            return self.buffer

    def sample(self):
        raise NotImplementedError

    def clear(self):
        self.buffer = {}
        self._is_ready = False
        torch.cuda.empty_cache()
        gc.collect()


@WEIGHTS.register("CPUWeightBuffer")
class CPUWeightBuffer(WeightBuffer):
    """
    Implements a CPU buffer strategy. Weights are transferred to and stored in
    CPU memory on the evaluator nodes and moved to the GPU just-in-time for
    model updates.
    """

    VALUE_TYPE: Type = np.ndarray


@WEIGHTS.register("ShardedWeightBuffer")
class ShardedWeightBuffer(WeightBuffer):
    """
    Implements a sharded buffer strategy for memory efficiency.
    Each GPU holds only a fraction of the total weights. Weights are gathered
    across GPUs on the same node just-in-time for model updates.
    """

    def __init__(self, config: WeightBufferConfig):
        super().__init__(config)
        self.sharded_buffers = {}  # Internal buffer for shards, mapping dtype -> tensor

    def add_shard(
        self,
        dtype: torch.dtype,
        flat_tensor_shard: torch.Tensor,
        metadata_list: List[Dict],
        numel_total: int,
        is_last: bool,
    ):
        """
        Adds a weight shard for a specific dtype to the buffer.
        """
        self.sharded_buffers[dtype] = (flat_tensor_shard, metadata_list, numel_total)
        if is_last:
            self._is_ready = True

    def apply_to_model(self, policy):
        """
        Called by ShardedWeightBuffer. This reconstructs the full weights from shards
        and applies them to the model.
        """
        with self._lock:
            node_size = CommContext().get_world_size(ParallelMode.EVAL_PARALLEL)

            for dtype in self.sharded_buffers:
                shard, metadata_list, unpadded_numel = self.sharded_buffers[dtype]

                shard_contiguous = shard.contiguous()
                full_tensor_size = shard_contiguous.numel() * node_size
                full_flat_tensor = torch.empty(full_tensor_size, dtype=dtype, device="cuda")

                torch.distributed.all_gather_into_tensor(
                    full_flat_tensor, shard_contiguous, group=self.context_group_sharded
                )

                # Unpad the full tensor before applying weights
                full_flat_tensor = full_flat_tensor.narrow(0, 0, unpadded_numel)

                self.add_flat_tensor(full_flat_tensor, metadata_list, is_last=False)

            policy.load_state_dict(self.buffer)
            self.clear()

    def clear(self):
        self.sharded_buffers = {}
        super().clear()
