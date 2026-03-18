"""Base data buffer module.

This module implements the DataBuffer class for storing and sampling
experience data, with support for distributed storage, sharding, and
various preprocessing pipelines.
"""

import math
import random
import uuid
from abc import ABC
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import ray
from tensordict import TensorDict

from rlightning.buffer.sampler import (
    AllDataSampler,
    BaseSampler,
    BatchSampler,
    UniformSampler,
)
from rlightning.types import BatchedData, EnvRet, PolicyResponse
from rlightning.utils.config import BufferConfig
from rlightning.utils.logger import get_logger
from rlightning.utils.placement import get_global_resource_manager
from rlightning.utils.ray import TaskSubmitter
from rlightning.utils.utils import InternalFlag

from .utils import EpisodeTable, Storage
from .utils.preprocessors import (
    Preprocessor,
    default_obs_preprocessor,
    default_reward_preprocessor,
)
from .utils.utils import (
    default_env_ret_preprocess_fn,
    default_policy_resp_preprocess_fn,
    default_postprocess_fn,
    default_preprocess_fn,
)

logger = get_logger(__name__)


class DataBuffer(ABC):
    """Data buffer for storing and sampling reinforcement learning experience.

    This class provides a unified interface for episode and transition storage,
    supporting both unified and sharded storage backends, various sampling
    strategies, and flexible preprocessing pipelines.

    """

    def __init__(
        self,
        config: BufferConfig,
        obs_preprocessor: Optional[Preprocessor] = default_obs_preprocessor,
        reward_preprocessor: Optional[Preprocessor] = default_reward_preprocessor,
        env_ret_preprocess_fn: Optional[Callable] = default_env_ret_preprocess_fn,
        policy_resp_preprocess_fn: Optional[Callable] = default_policy_resp_preprocess_fn,
        preprocess_fn: Optional[Callable] = default_preprocess_fn,
        postprocess_fn: Optional[Callable] = default_postprocess_fn,
    ) -> None:
        """Instantiate the data buffer.

        Args:
            config: Buffer configuration object.
            obs_preprocessor: Observation preprocessor function.
            reward_preprocessor: Reward preprocessor function.
            env_ret_preprocess_fn: Environment return preprocessing function.
            policy_resp_preprocess_fn: Policy response preprocessing function.
            preprocess_fn: General preprocessing function.
            postprocess_fn: Post-processing function for completed episodes.
        """
        self.config = deepcopy(config)

        self.timing_raw: Dict[str, Dict[str, Any]] = {}

        # init storage
        self.table: EpisodeTable = None
        self.storages: Dict[int, Union[Storage, ray.actor.ActorHandle]] = {}
        self.sampler: BaseSampler = None

        # process functions
        self._obs_preprocessor = obs_preprocessor
        self._reward_preprocessor = reward_preprocessor
        self._env_ret_preprocess_fn = env_ret_preprocess_fn
        self._policy_resp_preprocess_fn = policy_resp_preprocess_fn
        self._preprocess_fn = preprocess_fn

        self._postprocess_fn = postprocess_fn

        self.task_submitter = TaskSubmitter(max_pending_tasks_each_worker=8)
        self._global_resource_manager = get_global_resource_manager()

        self._sanity_check()

    def _sanity_check(self) -> None:
        """Validate buffer configuration and preprocess functions.

        Raises:
            ValueError: If conflicting preprocess functions are provided.
        """
        # Validates user-provided preprocess functions to prevent invalid configurations
        # where a custom outer preprocess_fn overrides custom inner preprocess_fns.
        if (
            self._obs_preprocessor is not default_obs_preprocessor
            or self._reward_preprocessor is not default_reward_preprocessor
        ):
            if self._env_ret_preprocess_fn is not default_env_ret_preprocess_fn:
                raise ValueError(
                    "Custom env_ret_preprocess_fn is set while the obs_preprocessor / "
                    "reward_preprocessor is also provided. Please ensure that observation and "
                    "reward preprocessing are handled within your custom env_ret_preprocess_fn, "
                    "as the default env_ret_preprocess_fn implementation which is expected to call"
                    " the provided obs/reward preprocessor will be overridden and not called."
                )
            elif self._preprocess_fn is not default_preprocess_fn:
                raise ValueError(
                    "Custom preprocess_fn is set while the obs_preprocessor / reward_preprocessor "
                    "is also provided. Please ensure that observation and reward preprocessing are"
                    " handled within your custom preprocess_fn, as the default preprocess_fn "
                    "implementation which is expected to call the provided obs/reward preprocessor"
                    " will be overridden and not called."
                )
        if self._env_ret_preprocess_fn is not default_env_ret_preprocess_fn:
            if self._preprocess_fn is not default_preprocess_fn:
                raise ValueError(
                    "Custom preprocess_fn is set while the env_ret_preprocess_fn is also provided."
                    " Please ensure that env_ret preprocessing are handled within your custom "
                    "preprocess_fn, as the default preprocess_fn implementation which is expected "
                    "to call the provided env_ret_preprocess_fn will be overridden and not called."
                )
        if self._policy_resp_preprocess_fn is not default_policy_resp_preprocess_fn:
            if self._preprocess_fn is not default_preprocess_fn:
                raise ValueError(
                    "Custom preprocess_fn is set while the policy_resp_preprocess_fn is also "
                    "provided. Please ensure that policy_resp preprocessing are handled within "
                    " your custom preprocess_fn, as the default preprocess_fn implementation which"
                    " is expected to call the provided policy_resp_preprocess_fn will be "
                    "overridden and not called."
                )

    def init(
        self,
        env_meta_list: Optional[List[Dict]] = None,
        env_ids: Optional[List[str]] = None,
    ) -> None:
        """Initialize the buffer with environment metadata.

        Args:
            env_meta_list: List of environment metadata dictionaries.
            env_ids: List of environment identifiers.
            train_worker_num: Number of training workers for data distribution.
        """
        self._init_sampler()
        self._init_storage(env_meta_list)
        self.init_storage_table(env_ids)

    def _init_sampler(self) -> None:
        """Initialize the sampler based on configuration."""
        if self.config.sampler.type == "uniform":
            self.sampler = UniformSampler()
        elif self.config.sampler.type == "all":
            self.sampler = AllDataSampler()
        elif self.config.sampler.type == "batch":
            self.sampler = BatchSampler()
        else:
            raise NotImplementedError(f"Sampler {self.config.sampler.type} not implemented")

    def _init_storage(self, env_meta_list: Optional[List[Dict]] = None) -> None:
        """Initialize storage backends based on configuration.

        Args:
            env_meta_list: List of environment metadata dictionaries.
            env_ids: List of environment identifiers.

        Raises:
            ValueError: If unknown storage type is specified.
        """
        if self.config.storage.type == "unified":
            if InternalFlag.REMOTE_STORAGE:
                device_str = str(self.config.storage.device)
                gpu_requirement = 0 if device_str == "cpu" else 1
                scheduling_kwargs = {}
                if self._global_resource_manager is not None:
                    strategy = self._global_resource_manager.get_scheduling_strategy("buffer", 0)
                    if strategy != "DEFAULT":
                        scheduling_kwargs["scheduling_strategy"] = strategy

                self.storages[0] = self._init_storage_actor(env_meta_list, gpu_requirement, scheduling_kwargs)
            else:
                self.storages[0] = Storage(
                    capacity=self.config.capacity,
                    mode=self.config.storage.mode,
                    unit=self.config.storage.unit,
                    env_meta_list=env_meta_list,
                    device=self.config.storage.device,
                    obs_preprocessor=self._obs_preprocessor,
                    reward_preprocessor=self._reward_preprocessor,
                    env_ret_preprocess_fn=self._env_ret_preprocess_fn,
                    policy_resp_preprocess_fn=self._policy_resp_preprocess_fn,
                    preprocess_fn=self._preprocess_fn,
                    postprocess_fn=self._postprocess_fn,
                    auto_truncate_episode=self.config.auto_truncate_episode,
                )

        elif self.config.storage.type == "sharded":
            assert InternalFlag.REMOTE_STORAGE, "Sharded storage is only supported with remote storage"
            buffer_worker_num = self._global_resource_manager.get_scheduling().buffer_worker.worker_num
            for i in range(buffer_worker_num):
                device_str = str(self.config.storage.device)
                gpu_requirement = 0 if device_str == "cpu" else 1
                scheduling_kwargs = {}
                if self._global_resource_manager is not None:
                    strategy = self._global_resource_manager.get_scheduling_strategy("buffer", i)
                    if strategy != "DEFAULT":
                        scheduling_kwargs["scheduling_strategy"] = strategy
                self.storages[i] = self._init_storage_actor(env_meta_list, gpu_requirement, scheduling_kwargs)
        else:
            raise ValueError(f"Unknown storage type: {self.config.storage.type}. " "Please use 'unified' or 'sharded'.")

    def _init_storage_actor(
        self,
        env_meta_list: Optional[List[Dict]],
        gpu_requirement: int,
        scheduling_kwargs: Dict[str, Any],
    ) -> ray.actor.ActorHandle:
        """Initialize a remote storage actor.

        Args:
            env_meta_list: List of environment metadata.
            gpu_requirement: Number of GPUs required (0 or 1).
            scheduling_kwargs: Ray scheduling options.

        Returns:
            Remote storage actor handle.
        """
        storage_id = str(uuid.uuid4())[:8]
        storage_name = f"storage-{self.config.storage.type}-{storage_id}"
        return (
            ray.remote(Storage)
            .options(
                num_gpus=gpu_requirement,
                num_cpus=1,
                name=storage_name,
                namespace=f"storage-{self.config.storage.type}",
                runtime_env={"env_vars": InternalFlag.get_env_vars()},
                **scheduling_kwargs,
            )
            .remote(
                capacity=self.config.capacity,
                mode=self.config.storage.mode,
                unit=self.config.storage.unit,
                env_meta_list=env_meta_list,
                device=self.config.storage.device,
                obs_preprocessor=self._obs_preprocessor,
                reward_preprocessor=self._reward_preprocessor,
                env_ret_preprocess_fn=self._env_ret_preprocess_fn,
                policy_resp_preprocess_fn=self._policy_resp_preprocess_fn,
                preprocess_fn=self._preprocess_fn,
                postprocess_fn=self._postprocess_fn,
                auto_truncate_episode=self.config.auto_truncate_episode,
            )
        )

    def __len__(self) -> int:
        """
        Get the number of transitions stored in the buffer.

        Returns:
            int: The number of transitions in the buffer.
        """
        return np.sum(
            [self.task_submitter.submit(storage.get_size, _block=True) for storage in self.storages.values()]
        ).item()

    def size(self) -> int:
        """
        Get the number of transitions stored in the buffer.

        Returns:
            int: The number of transitions in the buffer.
        """
        return len(self)

    def add_episode(self, episode: Union[Dict, TensorDict, ray.ObjectRef], num_envs: int = 1) -> None:
        """
        Adds a complete episode to the replay buffer.

        Args:
            episode (Union[Dict, TensorDict, ray.ObjectRef]): The episode data to be added to the
                buffer.
            num_envs (int): Number of environments represented in the episode.

        """
        storage_idx = random.choice(list(self.storages.keys()))
        storage = self.storages[storage_idx]
        self.task_submitter.submit(storage.add_episode, episode, num_envs)

    def add_transition(
        self,
        env_id: str,
        env_ret: Union[EnvRet, ray.ObjectRef],
        policy_resp: Union[PolicyResponse, ray.ObjectRef],
        truncated: Optional[bool] = False,
    ) -> None:
        """
        Add transition with a pair of env_ret and policy_resp to the buffer. This method will
        automatically handle the preprocess of given env_ret and policy_resp, add and organize
        them in an internal episode_buffer, and finally store the episode_buffer to the storage.
        It is useful when training with in distributed.

        Args:
            env_id (str): The env id.
            env_ret (Union[EnvRet, ray.ObjectRef]): An environment return.
            policy_resp (Union[PolicyResponse, ray.ObjectRef]): A policy response.
            truncated (Optional[bool]): Whether the episode is truncated after this transition.
                Defaults to False.
            is_eval (Optional[bool]): Whether the transition is from evaluation. Defaults to False.
        """
        storage_idx = self._get_storage_idx(env_id)
        storage = self.storages[storage_idx]
        self.task_submitter.submit(storage.add_transition, env_ret, policy_resp)
        if truncated:
            self.task_submitter.submit(storage.truncate_one_episode, env_ret)

    def add_data_async(
        self,
        env_id: str,
        data: Union[EnvRet, PolicyResponse, ray.ObjectRef],
        truncated: Optional[bool] = False,
    ) -> None:
        """
        Add data which is either env ret or policy resp to the buffer. This method is useful
        when async rollout between env and eval policy.

        Args:
            env_id (str): The env id.
            data (Union[EnvRet, PolicyResponse, ray.ObjectRef]): An environment return or a policy
                response.
            truncated (Optional[bool]): Whether the episode is truncated after this data. Default
                to False.
        """
        storage_idx = self._get_storage_idx(env_id)
        storage = self.storages[storage_idx]
        self.task_submitter.submit(storage.add_data_async, data)
        if truncated:
            self.task_submitter.submit(storage.truncate_one_episode, data)

    def add_batched_transition(
        self,
        batched_env_ret: BatchedData,
        batched_policy_resp: BatchedData,
        truncations: Optional[List[bool]] = None,
    ) -> "DataBuffer":
        """
        Add transition with pairs of env_ret and policy_resp to the buffer. This method will
        automatically handle the preprocess of given env_ret and policy_resp, add and organize
        them in an internal episode_buffer, and finally store the episode_buffer to the storage.
        It is useful when training with in distributed.

        Args:
            batched_env_ret (BatchedData): A batch of environment returns.
            batched_policy_resp (BatchedData): A batch of policy responses.
            truncations (Optional[List[bool]]): A list indicating whether each episode that
                corresponds to the given env_ret and policy_resp pair should be truncated.
                Defaults to False.
            is_eval (Optional[bool]): Whether the transitions are from evaluation. Defaults to False.
        """
        if len(batched_env_ret) != len(batched_policy_resp):
            raise ValueError("Length of batched_env_ret must match length of batched_policy_resp")
        if batched_env_ret.ids() != batched_policy_resp.ids():
            raise ValueError(
                "Mismatched batched_env_ret/batched_policy_resp env IDs, got "
                f"{batched_env_ret.ids()}, {batched_policy_resp.ids()}"
            )

        if truncations is None:
            truncations = [False] * len(batched_env_ret)
        else:
            if not isinstance(truncations, list):
                raise TypeError(f"truncations must be a list, but got {type(truncations)}")
            else:
                if len(truncations) != len(batched_env_ret):
                    raise ValueError(
                        "Length of truncations must match length of batched_env_ret and " "batched_policy_resp"
                    )
        env_ids = batched_env_ret.ids()
        env_ret_list = batched_env_ret.values()
        policy_resp_list = batched_policy_resp.values()
        for env_id, env_ret, policy_resp, truncated in zip(env_ids, env_ret_list, policy_resp_list, truncations):
            self.add_transition(env_id, env_ret, policy_resp, truncated)

        return self

    def add_batched_data_async(self, batched_data: BatchedData, truncations: Optional[List[bool]] = None) -> None:
        """
        Add batched data which is either batched env ret or batched policy resp to the buffer.
        This method is useful when async rollout between env and eval policy.

        Args:
            batched_data (BatchedData): A batch of environment returns or policy responses.
            truncations (Optional[List[bool]]): A list indicating whether each episode that
                corresponds to the given data should be truncated. Defaults to None.
        """
        if truncations is None:
            truncations = [False] * len(batched_data)
        else:
            if isinstance(truncations, bool):
                truncations = [truncations] * len(batched_data)
            elif isinstance(truncations, list):
                if len(truncations) != len(batched_data):
                    raise ValueError("Length of truncations must match length of batched_data")
            else:
                raise TypeError("truncations must be a list or a bool")

        for env_id, data, truncated in zip(batched_data.ids(), batched_data.values(), truncations):
            self.add_data_async(env_id, data, truncated)

    def truncate_one_episode(self, item: Union[str, Any]) -> None:
        """
        Manually truncate a single episode, finalize its collection, and trigger postprocessing
        and saving to the replay buffer.

        Args:
            item (Union[str, Any]): env_id or object with env_id attribute
        """
        env_id = item if isinstance(item, str) else getattr(item, "env_id", None)
        if env_id is None:
            raise TypeError(f"Expected env_id str or object with env_id attribute, got {type(item)}")
        storage_idx = self._get_storage_idx(env_id)
        storage = self.storages[storage_idx]
        self.task_submitter.submit(storage.truncate_one_episode, item)

    def truncate_episodes(self, items: List[Union[str, Any]]) -> None:
        """
        Manually truncate multiple episodes, finalize their collection, and trigger postprocessing
        and saving to the replay buffer.

        Args:
            items (List[Union[str, Any]]): List of env_ids or objects with env_id attribute
        """
        env_ids = []
        for item in items:
            env_id = item if isinstance(item, str) else getattr(item, "env_id", None)
            if env_id is None:
                raise TypeError(f"Expected env_id str or object with env_id attribute, got {type(item)}")
            env_ids.append(env_id)

        shard_buckets: Dict[int, List[str]] = {}
        for env_id in env_ids:
            storage_idx = self._get_storage_idx(env_id)
            shard_buckets.setdefault(storage_idx, []).append(env_id)

        for storage_idx, shard_env_ids in shard_buckets.items():
            storage = self.storages[storage_idx]
            self.task_submitter.submit(storage.truncate_episodes, shard_env_ids)

    def __getitem__(self, item: Union[int, Sequence[int], Tuple[int, Sequence[int]]]) -> TensorDict:
        """
        Get data from the buffer by index.
        """
        if isinstance(item, tuple):
            storage_idx, idx = item
        else:
            storage_idx, idx = 0, item

        storage = self.storages[storage_idx]
        return self.task_submitter.submit(storage.__getitem__, idx)

    def get_all(self) -> Union[TensorDict, Dict[int, TensorDict]]:
        """Get all data in the storage.

        Warning: This is a debugging function and should be used with caution
        as it may return large amounts of data.

        Returns:
            All stored data, or dict mapping storage index to data if sharded.
        """
        if len(self.storages) == 1:
            storage = next(iter(self.storages.values()))
            return self.task_submitter.submit(storage.get_data)

        data_per_shard = {}
        for idx, storage in self.storages.items():
            data_per_shard[idx] = self.task_submitter.submit(storage.get_data)
        return data_per_shard

    def get(self, item: Union[int, Sequence[int], Dict[str, Any]]) -> TensorDict:
        """Get data from the buffer by sample info.

        Args:
            item: Index, sequence of indices, or dict with storage_idx and indices.

        Returns:
            TensorDict containing the requested data.

        Raises:
            TypeError: If item is not a valid type.
        """
        if isinstance(item, dict):
            storage_idx = item["storage_idx"]
            indices = item["indices"]
            return self[storage_idx, indices]
        elif isinstance(item, (int, Sequence)):
            return self[item]
        else:
            raise TypeError(f"Expected dict or (int, Sequence[int]), got {type(item)}")

    def sample(
        self, batch_size: Optional[int] = None, shuffle: Optional[bool] = True, drop_last: Optional[bool] = True
    ) -> List[Dict]:
        """
        Sample a batch of data from the buffer.

        Sharded storage support:
        - Each storage samples indices based on its own size.
        - DistributedSampler then splits those indices to the workers bound to that storage.

        Returns:
            worker_sample_info (Dict[int, Dict[str, np.ndarray]]): A dictionary mapping worker rank
                to a dictionary containing the storage index and indices for the sampled data.
        """
        num_storages = len(self.storages)
        batch_size_per_storage = batch_size // num_storages if batch_size else None

        # Gather shard sizes
        shard_sizes: Dict[int, int] = {
            idx: self.task_submitter.submit(storage.get_size, _block=True) for idx, storage in self.storages.items()
        }

        # Check if all shard sizes are equal
        shard_size_values = list(shard_sizes.values())
        if len(set(shard_size_values)) > 1:
            raise RuntimeError("All shared storage sizes must be equal")

        storage_to_workers = self.table.get_storage_to_train_workers()
        num_workers = sum([len(workers) for workers in storage_to_workers.values()])
        sample_data = [None] * num_workers
        for storage_idx, data_size in shard_sizes.items():
            indices = self.sampler.sample(batch_size_per_storage, data_size, shuffle=shuffle)
            n = len(indices)

            workers = storage_to_workers[storage_idx]
            world_size = len(workers)

            # Evenly split data across workers(drop_last=True)
            if drop_last:
                per_worker = n // world_size
                indices = indices[: per_worker * world_size]
            else:
                per_worker = math.ceil(n / world_size)
                pad = per_worker * world_size - n
                if pad > 0:
                    indices = list(indices) + list(indices[:pad])
            # Ensure indices remain array-like for downstream slicing and storage access.
            indices = np.asarray(indices)
            for local_rank, worker_id in enumerate(workers):
                local_indices = indices[local_rank::world_size]
                local_data = self[storage_idx, local_indices]
                sample_data[worker_id] = local_data

        return sample_data

    def print_timing_summary(self, reset: bool = False) -> None:
        """Print timing summary for profiling.

        Args:
            reset: If True, reset timing statistics after printing.
        """
        for idx, storage in self.storages.items():
            logger.debug(f"[Storage {idx}] timing summary")
            self.task_submitter.submit(storage.print_timing_summary, reset, _block=True)

    def clear(self) -> None:
        """Clear all data from the buffer."""
        size_before = len(self)
        logger.debug(f"buffer size before clear: {size_before}")
        for storage in self.storages.values():
            self.task_submitter.submit(storage.clear)
        after_size = len(self)
        logger.debug(f"buffer size after clear: {after_size}")

    def _get_storage_idx(self, env_id: str) -> int:
        """Return the storage index for the given environment ID."""
        return self.table.get_storage_idx_for_env(env_id)

    def init_storage_table(self, env_ids: Optional[List[str]] = None, train_worker_num=1) -> None:
        """
        Initialize storage table for env -> storage and storage -> train workers mapping.
        """
        if not self.storages:
            return
        num_storages = len(self.storages)

        component_distribution = {}
        if self._global_resource_manager is not None:
            component_distribution = self._global_resource_manager.get_component_distribution()
            scheduling = self._global_resource_manager.get_scheduling()
            assert (
                scheduling.train_worker is not None
            ), "To init storage_to_train_workers, train worker must be configured"
            train_worker_num = scheduling.train_worker.worker_num

        self.table = EpisodeTable(
            num_storages=num_storages,
            env_ids=env_ids,
            num_train_workers=train_worker_num,
            component_distribution=component_distribution,
            node_affinity_env=self.config.node_affinity_env,
            node_affinity_train=self.config.node_affinity_train,
        )

        logger.info(f"[DataBuffer] storage_to_train_workers: {self.table.get_storage_to_train_workers()}")
        logger.info(f"[DataBuffer] env_to_storage: {self.table.get_env_to_storage()}")
