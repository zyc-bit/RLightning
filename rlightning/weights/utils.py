from typing import Optional

import numpy as np
import ray
import torch
from ray.actor import ActorHandle

from rlightning.utils.config import WeightBufferConfig
from rlightning.utils.registry import WEIGHTS
from rlightning.utils.utils import InternalFlag

from .weight_buffer import WeightBuffer


def build_weight_buffer(
    weight_buffer_cls: str,
    weight_buffer_cfg: Optional[WeightBufferConfig] = None,
    node_id: Optional[str] = None,
) -> WeightBuffer | ActorHandle:
    """Build an instance of WeightBuffer.

    If the configuration for building indicates the buffer strategy is `shared`, then the returned instance would be an instance of `ray.actor.ActorHandle`.

    Args:
        weight_buffer_cls (str): Registered name of a weight buffer class
        weight_buffer_cfg (Optional[WeightBufferConfig], optional): Configuration for building the weight buffer. Defaults to None.
        node_id (Optional[str], optional): Node id. Defaults to None.

    Returns:
        WeightBuffer | ActorHandle: An instance of weightbuffer or an Actor.
    """

    assert weight_buffer_cfg is not None, "Weight buffer config must be provided"
    weight_buffer_cls = WEIGHTS.get(weight_buffer_cls)

    if weight_buffer_cfg.buffer_strategy == "Shared" and InternalFlag.REMOTE_EVAL:
        assert node_id is not None, "Node id must be provided"
        weight_buffer_actor = (
            ray.remote(weight_buffer_cls)
            .options(
                num_cpus=1,
                name=f"{weight_buffer_cfg.type}_{node_id}",
                namespace="weight_buffer",
                runtime_env={"env_vars": InternalFlag.get_env_vars()},
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
            )
            .remote(weight_buffer_cfg)
        )
    else:
        weight_buffer_actor = weight_buffer_cls(weight_buffer_cfg)

    return weight_buffer_actor


def is_numpy_state_dict(state_dict):
    for module_state in state_dict.values():
        for param in module_state.values():
            return isinstance(param, np.ndarray)
    return False


def numpy_state_dict_to_tensor(state_dict):
    tensor_state_dict = {}
    for module_name, module_state in state_dict.items():
        tensor_state_dict[module_name] = {}
        for name, param in module_state.items():
            if isinstance(param, np.ndarray):
                param = param.copy()
                if param.dtype == np.uint16:
                    tensor_state_dict[module_name][name] = torch.from_numpy(param).view(
                        torch.bfloat16
                    )
                else:
                    tensor_state_dict[module_name][name] = torch.from_numpy(param)
            else:
                tensor_state_dict[module_name][name] = param
    return tensor_state_dict


def tensor_state_dict_to_numpy(state_dict):
    numpy_state_dict = {}
    for module_name, module_state_dict in state_dict.items():
        numpy_state_dict[module_name] = {}
        for name, param in module_state_dict.items():
            if param.dtype == torch.bfloat16:
                numpy_state_dict[module_name][name] = param.to("cpu").view(torch.uint16).numpy()
            else:
                numpy_state_dict[module_name][name] = param.to("cpu").numpy()
    return numpy_state_dict
