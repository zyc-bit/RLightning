"""Environment group module for managing multiple environments.

This module provides the EnvGroup class for coordinating multiple environments
in distributed RL training, with support for synchronous and asynchronous
stepping, auto-reset, and progress tracking.
"""

import time
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import gymnasium as gym
import ray

from rlightning.types import BatchedData, EnvMeta
from rlightning.utils.config import EnvConfig
from rlightning.utils.logger import get_logger
from rlightning.utils.placement import get_global_resource_manager
from rlightning.utils.progress import get_progress
from rlightning.utils.ray import TaskSubmitter, resolve_object
from rlightning.utils.ray.launcher import launch_ray_actor
from rlightning.utils.registry import ENVS
from rlightning.utils.utils import InternalFlag

from .base_env import BaseEnv
from .utils.utils import default_env_preprocess_fn

logger = get_logger(__name__)


class EnvStepCounter:
    """
    Tracks the number of steps taken in each environment.
    """

    def __init__(self, env_ids: List[str], max_steps_list: List[int]) -> None:
        """
        Initialize the environment step counter. Setup the max_steps for each environment.

        Args:
            env_ids: List of environment IDs to track.
            max_steps_list: Maximum number of steps for each environment.
        """
        self._env_ids = env_ids

        if isinstance(max_steps_list, List):
            if len(max_steps_list) != len(env_ids):
                raise ValueError("Length of max_steps_list must match length of env_ids.")
        else:
            raise ValueError("max_steps_list must be a List of integers.")
        self._max_steps = dict(zip(env_ids, max_steps_list))

        self._id_to_step = {env_id: -1 for env_id in self._env_ids}

        if InternalFlag.VERBOSE:
            # setup progress bar
            self._env_id_to_task_id: dict = None  # mapping from env_id to task_id in progress bar
            self._progress = get_progress()
            if self._env_id_to_task_id is None:
                self._env_id_to_task_id = {
                    env_id: self._progress.add_task(
                        f"[cyan]Env {self._env_ids.index(env_id)}",
                        total=self.get_max_steps(env_id),
                    )
                    for env_id in self._env_ids
                }

    def __del__(self) -> None:
        """
        Destructor to clean up progress bar tasks when the counter is no longer in use.
        """
        if InternalFlag.VERBOSE and self._progress is not None:
            if self._env_id_to_task_id:
                for task_id in self._env_id_to_task_id.values():
                    self._progress.remove_task(task_id)

    def reset(self, env_id: str) -> None:
        """
        Reset the step count for the specified environment or all environments.

        Args:
            env_id (int, optional): The ID of the environment to reset. If None, all environments
                                    are reset.
        Returns:
            None
        """
        self._id_to_step[env_id] = -1

        if InternalFlag.VERBOSE:
            self._progress.reset(self._env_id_to_task_id[env_id])

    def reset_all(self) -> None:
        """
        Reset the step count for all environments.
        """
        for env_id in self._env_ids:
            self.reset(env_id)

    def step(self, env_id: str) -> None:
        """
        Increment the step count for the provided environment.

        Args:
            env_id (str): The ID of the environment to step.
        """
        step = self._id_to_step[env_id] + 1
        if step > self.get_max_steps(env_id):
            raise ValueError(
                f"environment {env_id} reached its maximum steps: "
                f"{self.get_max_steps(env_id)}, cannot step further."
            )
        self._id_to_step[env_id] = step
        if InternalFlag.VERBOSE:
            self._progress.update(
                self._env_id_to_task_id[env_id],
                completed=self.get_steps(env_id) + 1,
            )

    def step_all(self) -> None:
        """
        Increment the step count for all environments.
        """
        for env_id in self._env_ids:
            self.step(env_id)

    def get_steps(self, env_id: str) -> int:
        """
        Get the number of steps taken in the environment.

        Args:
            env_id (str): The ID of the environment.
        Returns:
            int: The number of steps taken in the environment.
        """
        return self._id_to_step[env_id]

    def __getitem__(self, env_id: str) -> int:
        """Return the current step count for an environment."""
        return self._id_to_step[env_id]

    def get_max_steps(self, env_id: str) -> int:
        """
        Get the maximum number of steps for the environment.

        Args:
            env_id (str): The ID of the environment.
        Returns:
            int: The maximum number of steps for the environment.
        """

        return self._max_steps[env_id]

    def get_truncations(self, env_ids: List[str]) -> List[bool]:
        """
        Get whether the environment has reached its maximum steps.

        Args:
            env_ids (List[str]): List of environment IDs to check.
        Returns:
            List[bool]: List of booleans indicating whether each environment has reached its
                        maximum steps.
        """
        truncations = [self.is_reached_max_steps(env_id) for env_id in env_ids]
        return truncations

    def is_reached_max_steps(self, env_id: str) -> bool:
        """
        Check if the environment has reached its maximum steps. This method is used

        Args:
            env_id (str): The ID of the environment.
        Returns:
            bool: True if the environment has reached its maximum steps, False otherwise.
        """
        return self.get_steps(env_id) == self.get_max_steps(env_id)


class ThroughputTracker:
    """
    Track and log throughput metrics over a rolling time window. Default time window is 1 second.
    """

    __slots__ = ("_log_interval_s", "_state", "_executor", "_throughput")

    def __init__(self, log_interval_s: float = 5.0):
        self._log_interval_s = log_interval_s
        self._state: Dict[str, List[float]] = {}
        self._throughput: Dict[str, float] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="throughput")

    def record(self, name: str, count: int, step: Optional[int] = None) -> None:
        if count <= 0:
            return
        now = time.perf_counter()
        state = self._state.get(name)
        if state is None:
            self._state[name] = [float(count), now]
            return
        state[0] += float(count)
        elapsed = now - state[1]
        if elapsed < self._log_interval_s:
            return
        throughput = state[0] / elapsed if elapsed > 0.0 else 0.0
        self._throughput[name] = throughput
        state[0] = 0.0
        state[1] = now

    def get_stats(self):
        return self._throughput.copy()

    def record_async(self, name: str, count: int, step: Optional[int] = None) -> None:
        if count <= 0:
            return
        self._executor.submit(self.record, name, count, step)


class EnvGroup:
    """Group manager for multiple reinforcement learning environments.

    This class coordinates multiple environments for distributed RL training,
    supporting both synchronous and asynchronous stepping, auto-reset with
    step counting, and progress tracking.

    Attributes:
        env_list: List of local environment instances.
        env_servers: List of remote environment server instances.
        env_ids: List of unique environment identifiers.
        id_to_env: Mapping from env_id to environment instance.
        env_to_id: Mapping from environment instance to env_id.
        step_counter: Counter for tracking steps in auto-reset context.
    """

    _STEP_ASYNC_WARNED: bool = False
    throughput = ThroughputTracker()

    def __init__(
        self,
        env_cfg_list: List[EnvConfig],
        preprocess_fn: Optional[Union[Callable, List[Callable]]] = default_env_preprocess_fn,
    ) -> None:
        """
        Initialize the environment group.

        Args:
            env_cfg_list (List[EnvConfig]): List of each environment configurations.
            preprocess_fn (Optional[Union[Callable, List[Callable]]]): Function(s) to
                preprocess observations before calling step() function of the environment. Each
                environment can have its own preprocess function if a list is provided, otherwise shares the same function.
        """
        self._global_resource_manager = get_global_resource_manager()
        self._task_submitter = TaskSubmitter()

        if len(env_cfg_list) <= 0:
            raise ValueError("Environment config list cannot be empty.")

        # build envs (without actually initialization)
        if isinstance(preprocess_fn, Callable):
            preprocess_fns = [preprocess_fn] * len(env_cfg_list)
        elif isinstance(preprocess_fn, List):
            if len(env_cfg_list) != len(preprocess_fn):
                raise ValueError("Length of preprocess_fn list must match length of env_cfg_list.")
            preprocess_fns = preprocess_fn
        else:
            raise ValueError(
                f"preprocess_fn must be a Callable or a List of Callables, but got " f"{type(preprocess_fn)}"
            )

        self.env_list = []
        self.env_servers = []  # for dealing with remote env servers

        env_nums = []
        for idx, (env_cfg, preprocess_fn) in enumerate(zip(env_cfg_list, preprocess_fns)):
            env_instance, is_server = self.build_env(env_cfg, preprocess_fn, worker_index=idx)
            env_nums.append(env_cfg.num_envs)
            if not is_server:
                self.env_list.append(env_instance)
            else:
                self.env_servers.append(env_instance)

        self.env_ids = [
            self._task_submitter.submit(env.get_env_id, _block=True) for env in (self.env_list + self.env_servers)
        ]
        self.id_to_env = dict(zip(self.env_ids, self.env_list + self.env_servers))
        self.env_to_id = dict(zip(self.env_list + self.env_servers, self.env_ids))
        self.id_to_num_envs = dict(zip(self.env_ids, env_nums))

        self.observation_spaces: List[gym.Spaces] = None
        self.action_spaces: List[gym.Spaces] = None

        # for stepping envs async
        self._env_ret_future = []
        self._env_ret_future_to_env_id = {}

        # context related
        self.step_counter: EnvStepCounter = None
        self._in_auto_reset_context: bool = False

    def init(self) -> List[EnvMeta]:
        """
        Actually init for all environments. It returns the metadata of all environments.

        Returns:
            List[EnvMeta]: List of metadata for each environment.
        """

        env_metas = [self._task_submitter.submit(env.init, _block=True) for env in self.env_list]

        # check env meta consistency
        if len(env_metas) > 1:
            _env_metas: List[EnvMeta] = resolve_object(env_metas)
            _shape_check_keys = ["action_space"]

            env_0_start_shape = 1 if _env_metas[0].num_envs > 1 else 0
            for meta_info in _env_metas[1:]:
                env_i_start_shape = 1 if meta_info.num_envs > 1 else 0

                for key in _shape_check_keys:
                    if not hasattr(getattr(meta_info, key), "shape"):
                        continue  # no shape, skip check
                    if (
                        getattr(meta_info, key).shape[env_i_start_shape:]
                        != getattr(_env_metas[0], key).shape[env_0_start_shape:]
                    ):
                        raise ValueError(
                            f"All environments must have the same {key} shape, but got "
                            f"{getattr(meta_info, key).shape} and "
                            f"{getattr(_env_metas[0], key).shape}"
                        )
        # init env servers
        for server in self.env_servers:
            self._task_submitter.submit(server.init, _block=True)

        return env_metas

    @classmethod
    def build_env(
        cls,
        env_cfg: EnvConfig,
        preprocess_fn: Optional[Callable] = default_env_preprocess_fn,
        worker_index: Optional[int] = None,
    ) -> Tuple[Union[ray.actor.ActorHandle, BaseEnv], bool]:
        """Create an environment instance with given config and preprocess function.

        The returns are the environment instance and a boolean indicating whether the environment is a remote environment server.

        Args:
            cls: Environment group class.
            env_cfg (EnvConfig): Environment configuration.
            preprocess_fn (Optional[Callable]): Preprocess function for observations.
            worker_index (Optional[int]): Env worker index for placement grouping.

        Returns:
            Union[ray.actor.ActorHandle, BaseEnv]: Environment instance.
            bool: Whether the environment is a remote environment server.
        """

        env_name = f"env-{env_cfg.name}-{str(uuid.uuid4())[:8]}"
        env_cls = ENVS.get(env_cfg.backend)

        is_server = True if env_cfg.backend in ["env_server"] else False
        if InternalFlag.REMOTE_ENV:
            options = {
                "num_gpus": env_cfg.num_gpus,
                "num_cpus": env_cfg.num_cpus,
                "name": env_name,
                "namespace": f"env-{env_cfg.backend}",
                "runtime_env": {"env_vars": InternalFlag.get_env_vars()},
            }
            env_actor = launch_ray_actor(
                env_cls,
                env_cfg,
                role_type="env",
                worker_index=worker_index,
                options=options,
                preprocess_fn=preprocess_fn,
            )
            return (env_actor, is_server)
        else:
            return env_cls(env_cfg, worker_index=worker_index, preprocess_fn=preprocess_fn), is_server

    def __len__(self) -> int:
        """
        Get the number of environments in the group.

        Returns:
            int: The number of environments.
        """
        return len(self.env_list) + len(self.env_servers)

    def __getitem__(self, env_id: str) -> Union[ray.actor.ActorHandle, BaseEnv]:
        """Get environment by ID.

        Args:
            env_id: Environment identifier.

        Returns:
            The environment instance (local or remote).
        """
        return self.id_to_env[env_id]

    def get_observation_spaces(self) -> List[gym.Space]:
        """
        Retrieve a list of environment observation spaces

        Returns:
            List of observation spaces for each environment.
        """

        if self.observation_spaces is None:
            ref_list = []
            for env in self.env_list:
                observation_space = self._task_submitter.submit(env.get_observation_space)
                ref_list.append(observation_space)
            self.observation_spaces = ref_list

        return self.observation_spaces

    def get_action_spaces(self) -> List[gym.Space]:
        """
        Retrieve a list of action spaces

        Returns:
            List of action spaces for each environment.
        """

        if self.action_spaces is None:
            ref_list = []
            for env in self.env_list:
                action_space = self._task_submitter.submit(env.get_action_space)
                ref_list.append(action_space)
            self.action_spaces = ref_list

        return self.action_spaces

    def close(self) -> None:
        """
        Close all environments.
        """
        for env in self.env_list:
            self._task_submitter.submit(env.close, _block=True)
        for env_server in self.env_servers:
            self._task_submitter.submit(env_server.close, _block=True)

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[BatchedData, List[bool]]:
        """
        Reset all environments.

        Args:
            *args: Arguments to pass to the reset function of each environment.
            **kwargs: Keyword arguments to pass to the reset function of each environment.
        Returns:
            BatchedData: The batched env_ret from the environments.
            List[bool]: List of truncation flags indicating whether each environment was truncated.

        """
        env_ret_list = [self._task_submitter.submit(env._reset, *args, **kwargs) for env in self.env_list]
        env_ids = [self.env_to_id[env] for env in self.env_list]

        for env_server in self.env_servers:
            # _block = True to get env ret list
            env_server_rets = self._task_submitter.submit(env_server._reset, *args, **kwargs, _block=True)

            env_ret_list.extend(env_server_rets)
            env_ids.extend([self.env_to_id[env_server]] * len(env_server_rets))

        truncations = [False] * len(env_ids)  # default

        if self._in_auto_reset_context:
            # before state
            for env_id in env_ids:
                self.step_counter.reset(env_id)

            # after state
            for env_id in env_ids:
                self.step_counter.step(env_id)

            truncations = self.step_counter.get_truncations(env_ids)

        batched_env_ret = BatchedData(env_ids, env_ret_list)

        return batched_env_ret, truncations

    def print_timing_summary(self, reset: bool = False) -> None:
        """
        Print the timing summary of the environment group.
        """
        for env in self.env_list:
            self._task_submitter.submit(env.print_timing_summary, reset, _block=True)

    def get_stats(self) -> Dict[str, float]:
        """Retrieve throughput statistics"""

        return EnvGroup.throughput.get_stats()

    def apply_evaluate_cfg(self) -> None:
        """Apply evaluation-time config overrides for all environments."""
        resolve_object(
            [self._task_submitter.submit(env.apply_evaluate_cfg) for env in self.env_list + self.env_servers]
        )

    def restore_evaluate_cfg(self) -> None:
        """Restore environment members changed by apply_evaluate_cfg."""
        resolve_object(
            [self._task_submitter.submit(env.restore_evaluate_cfg) for env in self.env_list + self.env_servers]
        )

    def get_env_stats(self, reset: bool = False) -> Dict[str, float]:
        """Retrieve aggregated episode-level environment statistics.

        Collects per-env stats from all environments and computes the mean
        for each metric key.

        Args:
            reset: If True, clear recorded info in each env after retrieval.

        Returns:
            Dict mapping metric name to its mean value.
        """
        total = defaultdict(lambda: [0.0, 0])

        for env in self.env_list + self.env_servers:
            stats = self._task_submitter.submit(env.get_env_stats, reset, _block=True)
            for k, (s, c) in stats.items():
                total[k][0] += s
                total[k][1] += c
        env_stats = {k: s / c for k, (s, c) in total.items() if c > 0}

        resolve_object([self._task_submitter.submit(env.finish_rollout) for env in self.env_list + self.env_servers])
        return env_stats

    def step(self, batched_policy_resp: BatchedData) -> Tuple[BatchedData, List[bool]]:
        """
        Call step function for environments provided in batched_policy_resp synchronously. If this
        calling is wrappered by max_rollout_context context manager, it will also handle auto reset
        and progress updates.

        Args:
            batched_policy_resp (BatchedData): The batched policy_resp from the policies.

        Returns:
            BatchedData: The batched env_ret from the environments.
            List[bool]: List of truncation flags indicating whether each environment was truncated.
        """
        truncations = None  # default
        if self._in_auto_reset_context:
            env_ret_list = []
            for env_id, policy_resp in batched_policy_resp.items():
                if self.step_counter.is_reached_max_steps(env_id):
                    self.step_counter.reset(env_id)
                    env_ret_list.append(self._task_submitter.submit(self[env_id]._reset))
                else:
                    env_ret_list.append(self._task_submitter.submit(self[env_id]._step, policy_resp))

            # after state
            for env_id in self.env_ids:
                self.step_counter.step(env_id)

            truncations = self.step_counter.get_truncations(batched_policy_resp.ids())

        else:
            env_ret_list = [
                (self._task_submitter.submit(self[env_id.split("/")[0]]._step, policy_resp))
                for env_id, policy_resp in batched_policy_resp.items()
            ]

        batched_env_ret = BatchedData(batched_policy_resp.ids(), env_ret_list)

        EnvGroup.throughput.record_async("env_throughput_sync", sum(self.id_to_num_envs.values()))

        return batched_env_ret, truncations

    def step_async(self, batched_policy_resp: BatchedData) -> None:
        """
        Asynchronously submits step operations for environments specified in batched_policy_resp.
        It is non-blocking and doesn't return the results immediately. The results are stored as
        futures and can be retrieved later using collect_async.

        Args:
            batched_policy_resp (BatchedData): Batched policy responses for the environments.

        Examples:
            >>> env_group.step_async(batched_policy_resp)
            >>> # do other work
            >>> batched_env_ret, truncations = env_group.collect_async()
        """
        if not EnvGroup._STEP_ASYNC_WARNED and not InternalFlag.REMOTE_ENV:
            warnings.warn(
                "step_async method is mainly designed for remote environments. Please use "
                "synchronous step method instead.",
                stacklevel=2,
            )
            EnvGroup._STEP_ASYNC_WARNED = True

        envs_to_step = []
        envs_to_reset = []

        new_env_ret_list = []
        new_env_ids = []
        if self._in_auto_reset_context:
            for env_id, policy_resp in batched_policy_resp.items():
                if self.step_counter.is_reached_max_steps(env_id):
                    self.step_counter.reset(env_id)
                    envs_to_reset.append((env_id, policy_resp))
                else:
                    envs_to_step.append((env_id, policy_resp))

        else:
            envs_to_step = [(env_id, policy_resp) for env_id, policy_resp in batched_policy_resp.items()]

        if len(envs_to_reset) > 0:
            new_env_ret_list.extend([self._task_submitter.submit(self[env_id]._reset) for env_id, _ in envs_to_reset])
            new_env_ids.extend([env_id for env_id, _ in envs_to_reset])

        env_servers_id_and_policy_resp = {self.env_to_id[env]: [] for env in self.env_servers}
        for env_id, policy_resp in envs_to_step:
            env = self[env_id]
            if env in self.env_servers:
                env_servers_id_and_policy_resp[env_id].append(policy_resp)
            else:
                # self._task_submitter.submit(env._step_async, policy_resp)
                # new_env_ret_list.append(self._task_submitter.submit(env.collect_async))
                new_env_ret_list.append(self._task_submitter.submit(env._step, policy_resp))
                new_env_ids.append(env_id)

        # not add env server step async into env_ret_future and env_ret_future_to_env_id since we will call env server's collect_async in env_group's collect_async
        for env_id, policy_resp_list in env_servers_id_and_policy_resp.items():
            if len(policy_resp_list) == 0:
                continue
            env_server = self[env_id]
            self._task_submitter.submit(env_server._step_async, policy_resp_list)
            new_env_ret_list.append(self._task_submitter.submit(env_server._collect_async))
            new_env_ids.append(env_id)

        self._env_ret_future.extend(new_env_ret_list)
        self._env_ret_future_to_env_id.update(dict(zip(new_env_ret_list, new_env_ids)))

    def collect_async(self, timeout: Optional[float] = None, wait_all: bool = False) -> Tuple[BatchedData, List[bool]]:
        """
        Collects the results of previously submitted asynchronous step operations.

        This method blocks until at least one environment's computation is complete and its result
        (env_ret) is available. If timeout is assigned, it will wait and try to get more until the
        specified timeout is reached.
        Returns the batched results and truncation flags for the environments that have completed
        their steps.

        Args:
            timeout (Optional[int]): Maximum time to wait for more results in seconds.
            wait_all (bool): If True, waits for all submitted operations to complete before
                             returning. Default is False.

        Returns:
            BatchedData: Batched env_ret from the environments that have completed their step
            operations.
            List[bool]: List of truncation flags indicating whether each environment was truncated.

        Example:
            >>> env_group.step_async(batched_policy_resp)
            >>> # do other work
            >>> batched_env_ret, truncations = env_group.collect_async()
        """
        if len(self._env_ret_future) == 0:
            logger.warning("No pending environment step operations to collect.")

        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be a positive number. Got {timeout}.")

        if InternalFlag.REMOTE_ENV is False:
            if not EnvGroup._STEP_ASYNC_WARNED:
                warnings.warn(
                    "collect_async method is mainly supported for remote environments. Please use "
                    "synchronous step method instead.",
                    stacklevel=2,
                )
                EnvGroup._STEP_ASYNC_WARNED = True
            ready_list = self._env_ret_future
            self._env_ret_future = []

        elif wait_all:
            if timeout is not None:
                warnings.warn("timeout is ignored when wait_all is True.", stacklevel=2)
            ready_list, _ = ray.wait(self._env_ret_future, num_returns=len(self._env_ret_future))
            self._env_ret_future = []

        else:
            # Ensure at least has one ready return, and try to get more returns
            start = time.monotonic()
            ready_list, unready_list = ray.wait(self._env_ret_future, num_returns=1)
            elapsed = time.monotonic() - start

            if timeout is not None:
                remaining = max(0.0, timeout - elapsed)
                if unready_list and remaining > 0:
                    more_ready_list, unready_list = ray.wait(
                        unready_list, num_returns=len(unready_list), timeout=remaining
                    )
                    ready_list.extend(more_ready_list)
            else:
                if unready_list:
                    more_ready_list, unready_list = ray.wait(unready_list, num_returns=len(unready_list), timeout=0)
                    ready_list.extend(more_ready_list)

            self._env_ret_future = unready_list

        env_ids = [self._env_ret_future_to_env_id.pop(future) for future in ready_list]

        # flatten env server's returns
        new_env_ids = []
        new_ready_list = []
        n_collected_step = 0
        for env_id, env_ret in zip(env_ids, ready_list):
            if self[env_id] in self.env_servers:
                env_ret_list = resolve_object(env_ret)
                new_env_ids.extend([env_id] * len(env_ret_list))
                new_ready_list.extend(env_ret_list)
                n_collected_step += self.id_to_num_envs[env_id] * len(env_ret_list)
            else:
                new_env_ids.append(env_id)
                new_ready_list.append(env_ret)
                n_collected_step += self.id_to_num_envs[env_id]

        batched_env_ret = BatchedData(new_env_ids, new_ready_list)

        EnvGroup.throughput.record_async("env_throughput_async", n_collected_step)

        truncations = None
        if self._in_auto_reset_context:
            # after state
            for env_id in env_ids:
                self.step_counter.step(env_id)
            truncations = self.step_counter.get_truncations(env_ids)
        else:
            truncations = [False] * len(batched_env_ret)

        return batched_env_ret, truncations

    def size(self) -> int:
        """
        Get the number of environments in the group.

        Returns:
            int: The number of environments.
        """
        return len(self.env_list)

    def offload(self):
        """
        Offload the environment group to free GPU memory.
        """
        resolve_object([self._task_submitter.submit(env.offload) for env in self.env_list])

    def reload(self):
        """
        Reload the environment group to load GPU memory.
        """
        resolve_object([self._task_submitter.submit(env.reload) for env in self.env_list])

    @contextmanager
    def auto_reset(self, max_episode_steps: Optional[int] = None) -> Generator[None, None, None]:
        """
        Context manager for enforcing a maximum number of steps per environment.
        When used, environments that reach max episode steps during step or step_async will be
        automatically reset. Progress tracking and step counting are handled internally.

        Args:
            max_episode_steps (Optional[int]): Maximum number of steps per environment episode.
                                               If None, uses each environment's configured
                                               max_episode_steps.

        Examples:
            >>> with env_group.auto_reset(max_episode_steps=100):
            >>>     for _ in range(1000):
            >>>         batched_policy_resp = policy_group.rollout_batch(batched_env_ret)
            >>>         batched_env_ret = env_group.step(batched_policy_resp)
            >>>         # do other work
        """
        self._in_auto_reset_context = True

        if max_episode_steps is None:
            max_episode_steps = [
                self._task_submitter.submit(env.get_max_episode_steps, _block=True) for env in self.env_list
            ]
            # check all envs have max_episode_steps set
            for steps in max_episode_steps:
                if steps is None:
                    raise ValueError(
                        "All environments must have max_episode_steps set, or max_steps must be" "provided."
                    )
        else:
            if not isinstance(max_episode_steps, int) or max_episode_steps <= 0:
                raise ValueError("max_steps must be a positive integer")
            # raise warning if any env has max_episode_steps set
            for env in self.env_list:
                env_max_episode_steps = self._task_submitter.submit(env.get_max_episode_steps, _block=True)
                if env_max_episode_steps is not None:
                    warnings.warn(
                        "max_steps is provided, but some environments have max_episode_steps set. "
                        "The provided max_steps will override the environment's max_episode_steps."
                    )

            max_episode_steps = [max_episode_steps] * len(self.env_list)

        self.step_counter = EnvStepCounter(self.env_ids, max_episode_steps)

        try:
            yield
        except Exception:  # pylint: disable=W0706
            raise
        finally:
            self._in_auto_reset_context = False
            self.step_counter = None
