"""Builder utilities for constructing RL components.

This module provides factory functions for building the main components
of a reinforcement learning system: data buffers, environment groups,
policy groups, and training engines.
"""

import copy
import uuid
from typing import Callable, List, Optional, Type, Union

from rlightning.buffer import DataBuffer
from rlightning.buffer.utils.preprocessors import Preprocessor
from rlightning.buffer.utils.utils import (
    default_env_ret_preprocess_fn,
    default_obs_preprocessor,
    default_policy_resp_preprocess_fn,
    default_postprocess_fn,
    default_preprocess_fn,
    default_reward_preprocessor,
)
from rlightning.engine import BaseEngine
from rlightning.env.env_group import EnvGroup
from rlightning.env.utils.utils import default_env_preprocess_fn
from rlightning.policy.base_policy import PolicyRole
from rlightning.policy.policy_group import PolicyGroup
from rlightning.utils.config import (
    BufferConfig,
    ClusterConfig,
    EnvConfig,
    MainConfig,
    PolicyConfig,
)
from rlightning.utils.logger import get_logger
from rlightning.utils.ray.launcher import launch_ray_actor
from rlightning.utils.ray.remote_class import DistributedMixin
from rlightning.utils.registry import BUFFERS, ENGINE, POLICIES
from rlightning.utils.utils import InternalFlag

logger = get_logger(__name__)


def build_data_buffer(
    buffer_cls: str,
    buffer_cfg: BufferConfig,
    obs_preprocessor: Optional[Preprocessor] = default_obs_preprocessor,
    reward_preprocessor: Optional[Preprocessor] = default_reward_preprocessor,
    env_ret_preprocess_fn: Optional[Callable] = default_env_ret_preprocess_fn,
    policy_resp_preprocess_fn: Optional[Callable] = default_policy_resp_preprocess_fn,
    preprocess_fn: Optional[Callable] = default_preprocess_fn,
    postprocess_fn: Optional[Callable] = default_postprocess_fn,
) -> DataBuffer:
    """Build a data buffer instance.

    Args:
        buffer_cls: Name of the buffer class to instantiate.
        buffer_cfg: Buffer configuration object.
        obs_preprocessor: Observation preprocessor function.
        reward_preprocessor: Reward preprocessor function.
        env_ret_preprocess_fn: Preprocessing function for environment returns.
        policy_resp_preprocess_fn: Preprocessing function for policy responses.
        preprocess_fn: General preprocessing function for each timestep.
        postprocess_fn: Post-processing function for completed episodes.

    Returns:
        Configured DataBuffer instance.
    """
    buffer_cls: Type[DataBuffer] = BUFFERS.get(buffer_cls)
    return buffer_cls(
        config=buffer_cfg,
        obs_preprocessor=obs_preprocessor,
        reward_preprocessor=reward_preprocessor,
        env_ret_preprocess_fn=env_ret_preprocess_fn,
        policy_resp_preprocess_fn=policy_resp_preprocess_fn,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
    )


def define_env_instance_cfgs(
    env_cfg: EnvConfig,
    num_workers: int = 1,
) -> List[EnvConfig]:
    """Factory function to define duplicated env instance configs.

    The number of replicas is determined by `num_workers`.

    Args:
        env_cfg: Base environment configuration.
        num_workers: Number of environment instances to create.

    Returns:
        List of environment configurations, one per worker.
    """
    env_instance_cfg = copy.deepcopy(env_cfg)
    return [env_instance_cfg for _ in range(num_workers)]


def build_env_group(
    env_cfgs: Union[List[EnvConfig], EnvConfig],
    preprocess_fn: Optional[Union[Callable, List[Callable]]] = default_env_preprocess_fn,
) -> EnvGroup:
    """Build an environment group from configuration.

    Args:
        env_cfgs (Union[List[EnvConfig], EnvConfig]): Env config(s) to build from.
        preprocess_fn (Optional[Union[Callable, List[Callable]]]): Preprocess function(s) for
            observations, either a single callable or one per env config.

    Returns:
        EnvGroup instance managing the configured environments.
    """

    env_instance_cfgs: List[EnvConfig] = []
    preprocess_fn_list: List[Callable] = []

    if isinstance(env_cfgs, EnvConfig):
        env_cfgs = [env_cfgs]
    if not isinstance(preprocess_fn, list):
        preprocess_fn = [preprocess_fn] * len(env_cfgs)

    for env_cfg, _pre_fn in zip(env_cfgs, preprocess_fn):
        preprocess_fn_list.extend([_pre_fn] * env_cfg.num_workers)
        env_instance_cfgs.extend(
            define_env_instance_cfgs(
                env_cfg=env_cfg,
                num_workers=env_cfg.num_workers,
            )
        )

    env_group = EnvGroup(
        env_cfg_list=env_instance_cfgs,
        preprocess_fn=preprocess_fn_list,
    )

    return env_group


def create_distributed_policy_class(policy_cls: type, role_type: PolicyRole) -> type:
    """
    Create a distributed policy class by mixing policy_cls with DistributedMixin.
    """
    return type(f"{policy_cls.__name__}_{role_type}", (policy_cls, DistributedMixin), {})


def build_policy_group(
    policy_cls: str,
    policy_cfg: PolicyConfig,
    cluster_cfg: ClusterConfig,
    backend: str = "nccl",
    is_colocated: bool = False,
) -> PolicyGroup:
    """Build a policy group with train and eval workers.

    Creates and configures policy workers for training and evaluation,
    supporting distributed execution with Ray.

    Args:
        policy_cls: Name of the policy class to instantiate.
        policy_cfg: Policy configuration object.
        cluster_cfg: Cluster configuration object.
        backend: Communication backend for distributed training.
        is_colocated: If True, only initialize training policies for comm groups.

    Returns:
        PolicyGroup containing configured train and eval workers.
    """
    policy_cls = POLICIES.get(policy_cls)

    # Create distributed policy for train and eval
    DistributedTrainPolicyCls = create_distributed_policy_class(policy_cls, role_type=PolicyRole.TRAIN)
    DistributedEvalPolicyCls = create_distributed_policy_class(policy_cls, role_type=PolicyRole.EVAL)

    # initialize train policy
    policy_train_workers = []
    for i in range(cluster_cfg.train_worker_num):
        if InternalFlag.REMOTE_TRAIN:
            policy_id = str(uuid.uuid4())[:8]
            policy_name = f"policy-train-{policy_cfg.type}-{policy_id}"

            options = {
                "num_gpus": cluster_cfg.train_each_gpu_num,
                "num_cpus": 1,
                "name": policy_name,
                "namespace": "policy",
                "runtime_env": {"env_vars": InternalFlag.get_env_vars()},
            }
            policy_actor = launch_ray_actor(
                DistributedTrainPolicyCls,
                policy_cfg,
                role_type=PolicyRole.TRAIN,
                worker_index=i,
                options=options,
            )
        else:
            policy_actor = policy_cls(policy_cfg, role_type=PolicyRole.TRAIN)

        policy_train_workers.append(policy_actor)

    policy_eval_workers = []
    for i in range(cluster_cfg.eval_worker_num):
        if InternalFlag.REMOTE_EVAL:
            policy_id = str(uuid.uuid4())[:8]
            policy_name = f"policy-eval-{policy_cfg.type}-{policy_id}"
            options = {
                "num_gpus": cluster_cfg.eval_each_gpu_num,
                "num_cpus": 1,
                "name": policy_name,
                "namespace": "policy",
                "runtime_env": {"env_vars": InternalFlag.get_env_vars()},
            }
            if getattr(policy_cfg, "rollout_mode", None) == "async":
                options["max_concurrency"] = getattr(policy_cfg, "max_concurrency", 16)

            policy_actor = launch_ray_actor(
                DistributedEvalPolicyCls,
                policy_cfg,
                role_type=PolicyRole.EVAL,
                worker_index=i,
                options=options,
            )
        else:
            policy_actor = policy_cls(policy_cfg, role_type=PolicyRole.EVAL)

        policy_eval_workers.append(policy_actor)

    policy_group = PolicyGroup(
        train_policy_list=policy_train_workers,
        eval_policy_list=policy_eval_workers,
        config=policy_cfg,
    )

    is_colocated = cluster_cfg.is_colocated
    policy_group.init_placement_info(is_colocated=is_colocated)

    # weight_buffer initialization must be after placement info initialization
    if len(policy_train_workers) > 0:
        policy_group.init_weight_buffer()

    if InternalFlag.REMOTE_TRAIN or InternalFlag.REMOTE_EVAL:
        policy_group.init_comm_group(backend=backend, is_colocated=is_colocated)

    return policy_group


def build_engine(
    config: MainConfig,
    env_group: EnvGroup,
    policy_group: PolicyGroup,
    buffer: DataBuffer,
) -> BaseEngine:
    """
    Build engine instance

    Args:
        config: main config
        env_group: env group
        policy_group: policy group
        buffer: data buffer

    Returns:
        BaseEngine: engine instance
    """
    engine_cls = ENGINE.get(config.engine)

    return engine_cls(
        config=config,
        env_group=env_group,
        policy_group=policy_group,
        buffer=buffer,
    )
