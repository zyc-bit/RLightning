"""Storage backend for buffer data and episode handling."""

import time
from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from intervaltree import IntervalTree
from tensordict import TensorDict

from rlightning.env.base_env import EnvMeta
from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.logger import get_logger
from rlightning.utils.profiler import profiler
from rlightning.utils.utils import InternalFlag, to_device, to_numpy

logger = get_logger(__name__)


class ActiveEpisodeBuffer:
    """A helper class to manage active episode buffers for multiple environments."""

    def __init__(
        self,
        env_meta_list: List[EnvMeta],
        auto_truncate_episode: bool,
        postprocess_fn: Callable,
        device: str | torch.device,
    ) -> None:
        """Initialize the active episode buffer."""
        self.env_meta_list = env_meta_list
        if self.env_meta_list is not None:
            self.env_meta_dict = {env_meta.env_id: env_meta for env_meta in env_meta_list}

        self.auto_truncate_episode = auto_truncate_episode
        self._postprocess_fn = postprocess_fn
        self._device = device

        self._buffer: Dict[str, Dict[str, List[Any]]] = defaultdict(list)
        """ env_id -> episode that composed of list of transitions """
        self._episodes_done_flag: Dict[str, bool] = defaultdict(bool)
        """ env_id -> episode that is done or not """

    def _get_num_envs(self, env_id: str) -> int:
        """Resolve num_envs of the env with env_id."""
        if self.env_meta_list is None:
            return 1

        if env_id not in self.env_meta_dict:
            # possibly a sub-env id
            env_id = "/".join(env_id.split("/")[:-1])
            if env_id not in self.env_meta_dict:
                # check is a sub-env id and its parent env_id in env_meta_dict
                raise KeyError(f"Env ID {env_id} not found in env_meta_dict.")

            return 1  # num_envs is 1 for sub-env

        return self.env_meta_dict[env_id].num_envs

    @staticmethod
    def _judge_done(transition: Dict[str, Any]) -> bool:
        """Judge whether the episode is done based on a transition dict."""

        last_terminated = transition.get("last_terminated", False)
        last_truncated = transition.get("last_truncated", False)

        return bool(last_terminated or last_truncated)

    def add_transition(self, env_id: str, transition: Dict[str, Any]) -> None:
        """Add a transition to the episode buffer for a given environment ID."""
        num_envs = self._get_num_envs(env_id)

        if self.auto_truncate_episode:
            if num_envs > 1:
                # split sub env transitions
                transition_td = TensorDict(transition, batch_size=[num_envs])
                for i in range(num_envs):
                    full_env_id = f"{env_id}/{i}"
                    sub_trans = transition_td[i].to_dict()
                    done = self._judge_done(sub_trans)
                    if done:
                        self._episodes_done_flag[full_env_id] = True
                    self._buffer[full_env_id].append(sub_trans)
            else:
                done = self._judge_done(transition)
                if done:
                    self._episodes_done_flag[env_id] = True
                self._buffer[env_id].append(transition)
        else:
            self._buffer[env_id].append(transition)

    def pop(self, env_id: str) -> List[TensorDict]:
        """Pop and return a list of episodes correlated to an env tagged with given environment ID.

        Args:
            env_id (str): Environment id.

        Returns:
            List[TensorDict]: A list of episode
        """

        num_envs = self._get_num_envs(env_id)

        def merge_transitions_to_episode(transition_list: List[Dict]) -> Dict:
            """Merge a list of transition dicts into an episode dict."""
            keys = transition_list[0].keys()
            return {k: [d[k] for d in transition_list] for k in keys}

        if num_envs > 1 and self.auto_truncate_episode:
            episode_list = [self._buffer.pop(f"{env_id}/{i}") for i in range(num_envs)]
            episode_list = [merge_transitions_to_episode(episode) for episode in episode_list]
        else:
            try:
                episode_list = [self._buffer.pop(env_id)]
            except Exception as e:
                print(f"Buffer keys: {list(self._buffer.keys())}")
                raise e

            episode_list = [merge_transitions_to_episode(episode) for episode in episode_list]

        # postprocess episodes
        episode_list = to_device(episode_list, self._device)
        episode_list = [self._postprocess_fn(episode) for episode in episode_list]
        episode_length_list = [len(next(iter(episode.values()))) for episode in episode_list]

        if num_envs > 1 and not self.auto_truncate_episode:
            # when num_envs > 1 and auto_truncate_episode is False, episode contains data from
            # all sub-envs. Here split the episode into sub-env episodes
            combined_episodes = []
            for episode, length in zip(episode_list, episode_length_list):
                episode_td = TensorDict(episode, batch_size=[length, num_envs], device=self._device)

                flat_td = episode_td.permute(1, 0).reshape(num_envs * length)

                flat_td._metadata = {
                    "_num_episodes": num_envs,
                    "_episode_length": length,
                }
                combined_episodes.append(flat_td)

            episode_list = combined_episodes

        return episode_list

    def pop_done_episode(self, env_id: Optional[str] = None) -> Dict[str, List[Any]] | List[Dict[str, List[Any]]]:
        """Pop and return the done episode buffer for a given environment ID."""

        done_env_ids = []

        if env_id is None:
            for eid in list(self._buffer.keys()):
                if self._episodes_done_flag[eid]:
                    done_env_ids.append(eid)
                    self._episodes_done_flag.pop(eid)
        else:
            if self._episodes_done_flag.get(env_id, False):
                done_env_ids.append(env_id)
                self._episodes_done_flag.pop(env_id)
            else:
                for eid in self._buffer.keys():  # eid is full env_id
                    if eid.startswith(f"{env_id}/") and self._episodes_done_flag[eid]:
                        done_env_ids.append(eid)
                        self._episodes_done_flag.pop(eid)

        return list(chain.from_iterable(self.pop(eid) for eid in done_env_ids))


class BufferView:
    """A read-only, buffer-like view of a TensorDict for dataset creation."""

    def __init__(self, data: TensorDict) -> None:
        """Initialize the view with a TensorDict."""
        self._data = data
        self.capacity = len(data)

    def __len__(self) -> int:
        """Return the number of entries in the view."""
        return self.capacity

    def __getitem__(self, idx: int) -> TensorDict:
        """Return a single item by index."""
        return self._data[idx]


class DataContainer:
    def __init__(
        self,
        capacity: int,
        mode: Literal["circular", "fixed"],
        unit: Literal["transition", "episode"],
        device: Union[str, torch.device],
    ) -> None:
        """Initialize the backing TensorDict container."""
        self.capacity = capacity
        self.device = device
        self.mode = mode
        self.unit = unit

        self.data = TensorDict(batch_size=[capacity], device=self.device)
        self._schema_initialized = False

        self.size = 0  # number of transitions
        self.pointer = 0  # pointer to the next write position

        # for episode unit
        self.episode_registry = IntervalTree()

        # usage counter for each entry
        self.data_use_counter = np.ones(capacity, dtype=np.int32) * -1

    def _check_range(self, idx: Union[int, slice, np.ndarray, torch.Tensor]) -> None:
        """Check if the index is within the valid range of the storage size."""

        if self.unit == "transition":
            # For transition unit, check against self.size (number of transitions)
            if isinstance(idx, int):
                actual_idx = idx if idx >= 0 else self.capacity + idx
                if actual_idx < 0 or actual_idx >= self.size:
                    raise IndexError(f"Transition index {idx} out of range for storage of size {self.size}")
            elif isinstance(idx, slice):
                start, stop, _ = idx.indices(self.capacity)
                if start < 0 or stop > self.size:
                    raise IndexError(f"Transition slice {idx} out of range for storage of size {self.size}")

            elif isinstance(idx, np.ndarray):
                idx_copy = np.where(idx < 0, self.capacity + idx, idx)
                if np.any(idx_copy < 0) or np.any(idx_copy >= self.size):
                    raise IndexError(
                        f"Transition index array contains out of range indices for storage of size {self.size}"
                    )
            elif isinstance(idx, torch.Tensor):
                # Convert to numpy for easier checking
                idx_np = idx.cpu().numpy()
                idx_copy = np.where(idx_np < 0, self.capacity + idx_np, idx_np)
                if np.any(idx_copy < 0) or np.any(idx_copy >= self.size):
                    raise IndexError(
                        f"Transition index tensor contains out of range indices for storage of size {self.size}"
                    )

        elif self.unit == "episode":
            # For episode unit, check against number of episodes
            num_episodes = len(self.episode_registry)

            if isinstance(idx, int):
                # Support negative indexing
                actual_idx = idx if idx >= 0 else num_episodes + idx
                if actual_idx < 0 or actual_idx >= num_episodes:
                    raise IndexError(f"Episode index {idx} out of range for {num_episodes} episodes")
            elif isinstance(idx, slice):
                # slice.indices handles negative indices and bounds checking
                start, stop, _ = idx.indices(num_episodes)
                # No explicit check needed as slice.indices clamps to valid range
                pass

            elif isinstance(idx, (torch.Tensor, np.ndarray)):
                # Convert to numpy for easier checking
                if isinstance(idx, torch.Tensor):
                    idx = idx.cpu().numpy()
                # Handle negative indices
                idx_copy = np.where(idx < 0, num_episodes + idx, idx)
                if np.any(idx_copy < 0) or np.any(idx_copy >= num_episodes):
                    raise IndexError(f"Episode index array contains out of range indices for {num_episodes} episodes")
        else:
            raise ValueError(f"Unknown unit type: {self.unit}")

    def __len__(self) -> int:
        """Return the number of stored items based on the unit type."""
        if self.unit == "episode":
            return len(self.episode_registry)
        else:
            return self.size

    def __getitem__(self, idx: Union[str, int, slice, np.ndarray, torch.Tensor]) -> TensorDict | List[TensorDict]:
        """Get item by index, slice or key."""
        self._check_range(idx)

        if self.unit == "episode":
            intervals = list(self.episode_registry)
            num_episodes = len(intervals)

            if isinstance(idx, int):
                interval = intervals[idx]
                start_idx = interval.begin
                end_idx = interval.end
                self.data_use_counter[start_idx:end_idx] += 1
                ret = self.data[start_idx:end_idx].clone()
            elif isinstance(idx, (slice, np.ndarray, torch.Tensor)):
                if isinstance(idx, slice):
                    current_indices = list(range(*idx.indices(num_episodes)))
                else:
                    if isinstance(idx, torch.Tensor):
                        current_indices = idx.cpu().numpy()
                    else:
                        current_indices = idx
                    # handle negative indices
                    current_indices = np.where(current_indices < 0, num_episodes + current_indices, current_indices)

                selected_intervals = [intervals[i] for i in current_indices]
                starts = [interval.begin for interval in selected_intervals]
                ends = [interval.end for interval in selected_intervals]
                lengths = [end - start for start, end in zip(starts, ends)]

                flat_offsets = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])

                np.add.at(self.data_use_counter, flat_offsets, 1)

                all_data_flattened = self.data[flat_offsets]

                if len(set(lengths)) == 1:
                    ret = all_data_flattened.view(len(lengths), lengths[0])
                else:
                    cum_lengths = np.cumsum([0] + lengths)
                    ret = [all_data_flattened[cum_lengths[i] : cum_lengths[i + 1]] for i in range(len(lengths))]
                    # ret = list(all_data_flattened.split(lengths))
            else:
                raise TypeError(f"Unsupported index type for episode unit: {type(idx)}")

        elif self.unit == "transition":
            self.data_use_counter[idx] += 1
            ret = self.data[idx]

        logger.debug(
            "Data use counter stats: \n"
            f"      min={np.min(self.data_use_counter[self.data_use_counter>-1])}, \n"
            f"      max={np.max(self.data_use_counter)}, \n"
            f"      mean={np.mean(self.data_use_counter[self.data_use_counter>-1])} \n"
        )
        return ret

    def _ensure_schema(self, items: TensorDict) -> None:
        """Lazily initialize the storage schema from the first incoming batch."""
        if self._schema_initialized:
            return

        for key in items.keys(include_nested=True):
            value = items.get(key)
            if isinstance(value, TensorDict):
                continue

            storage_value = torch.empty(
                (self.capacity, *value.shape[1:]),
                dtype=value.dtype,
                device=self.device,
            )
            self.data.set(key, storage_value)

        self._schema_initialized = True

    def push(self, items: List | TensorDict | Dict[str, Any]) -> None:
        """Push items into the storage."""
        if isinstance(items, list):
            for item in items:
                self.push(item)
            return
        elif isinstance(items, TensorDict):
            data_size = items.batch_size[0]
            metadata = getattr(items, "_metadata", {})
            num_sub_episodes = metadata.get("_num_episodes", 1)
            episode_length = metadata.get("_episode_length", None)
        elif isinstance(items, dict):
            first_value = next(iter(items.values()))
            data_size = first_value.shape[0]
            num_sub_episodes = 1
            batch_size = [data_size]
            items = TensorDict(items, batch_size=batch_size, device=self.device).view(data_size)
        else:
            raise TypeError(f"Unsupported episode type: {type(items)}")

        self._ensure_schema(items)

        if self.pointer + data_size <= self.capacity:
            start_idx = self.pointer
            end_idx = self.pointer + data_size

            if self.unit == "episode":
                episode_overlaps = self.episode_registry.overlap(start_idx, end_idx)
                for interval in episode_overlaps:
                    self.data_use_counter[interval.begin : interval.end] = -1
                    self.episode_registry.remove(interval)

                if num_sub_episodes > 1:
                    for i in range(num_sub_episodes):
                        self.episode_registry.addi(start_idx + i * episode_length, start_idx + (i + 1) * episode_length)
                else:
                    self.episode_registry.addi(start_idx, end_idx)

            self.data_use_counter[start_idx:end_idx] = 0
        else:
            if self.mode == "circular":
                # Overwrite the oldest data in a circular manner
                if data_size > self.capacity:
                    logger.warning(
                        f"Episode size ({data_size}) exceeds storage capacity ({self.capacity}). "
                        f"Only the last {self.capacity} transitions will be stored."
                    )
                    items = items[-self.capacity :]

                    if self.unit == "episode":
                        self.episode_registry.clear()
                        if num_sub_episodes > 1:
                            curr_end = self.capacity
                            while curr_end >= episode_length:
                                self.episode_registry.addi(curr_end - episode_length, curr_end)
                                curr_end -= episode_length
                        else:
                            self.episode_registry.addi(0, self.capacity)

                    self.data_use_counter[:] = 0
                    data_size = self.capacity
                    self.pointer = 0

                elif data_size + self.pointer > self.capacity:
                    if self.unit == "episode":
                        # early wrap to the beginning of storage
                        start_idx = 0
                        end_idx = data_size

                        # although we are overwriting from the beginning, we still clear the old
                        # episodes from current pointer to end to make sure the overwrite sequence
                        # is continuous
                        overlaps = self.episode_registry.overlap(self.pointer, self.capacity)
                        overlaps |= self.episode_registry.overlap(start_idx, end_idx)
                        for interval in overlaps:
                            self.data_use_counter[interval.begin : interval.end] = -1
                            self.episode_registry.remove(interval)

                        if num_sub_episodes > 1:
                            for i in range(num_sub_episodes):
                                self.episode_registry.addi(
                                    start_idx + i * episode_length,
                                    start_idx + (i + 1) * episode_length,
                                )
                        else:
                            self.episode_registry.addi(start_idx, end_idx)

                        self.data_use_counter[start_idx:end_idx] = 0
                    else:
                        # write the first half to the end of the storage
                        self._write_data(
                            self.pointer,
                            self.capacity - self.pointer,
                            items[: self.capacity - self.pointer],
                        )
                        self.data_use_counter[self.pointer :] = 0
                        # set the second half
                        data_size = data_size - (self.capacity - self.pointer)
                        items = items[self.capacity - self.pointer :]
                        self.data_use_counter[:data_size] = 0
                    self.pointer = 0
                else:
                    raise NotImplementedError("Circular mode not implemented for this case. It may be a bug.")
            elif self.mode == "fixed":
                raise RuntimeError(
                    f"Cannot push {data_size} items to storage of capacity {self.capacity}. "
                    f"Current size is {self.size}."
                )

        self._write_data(self.pointer, data_size, items)
        self.pointer = (self.pointer + data_size) % self.capacity
        self.size = min(self.size + data_size, self.capacity)

    def _write_data(self, pointer: int, size: int, items: TensorDict) -> None:
        """Write a TensorDict batch into storage using ``update_at_``."""
        self.data.update_at_(items, idx=slice(pointer, pointer + size))

    def clear(self) -> None:
        """Clear the storage."""
        self.size = 0
        self.pointer = 0
        self._schema_initialized = False

        self.episode_registry.clear()

        self.data.clear()
        self.data_use_counter = np.ones(self.capacity, dtype=np.int32) * -1


class Storage:
    """Actual storage for data buffer to support shard or unshard storage"""

    def __init__(
        self,
        capacity: int,
        mode: Literal["circular", "standard"],
        unit: Literal["transition", "episode"],
        env_meta_list: List[EnvMeta],
        device: Union[str, torch.device],
        obs_preprocessor: Callable,
        reward_preprocessor: Callable,
        env_ret_preprocess_fn: Callable,
        policy_resp_preprocess_fn: Callable,
        preprocess_fn: Callable,
        postprocess_fn: Callable,
        auto_truncate_episode: bool,
    ) -> None:
        """Initialize the storage backend."""
        self.capacity = capacity
        self.mode = mode
        self.unit = unit
        self.device = torch.device(device) if isinstance(device, str) else device
        self.auto_truncate_episode = auto_truncate_episode

        self.transitions_buffer = defaultdict(defaultdict)
        self.episodes_buffer = ActiveEpisodeBuffer(
            env_meta_list, self.auto_truncate_episode, postprocess_fn, self.device
        )

        self._data = DataContainer(self.capacity, self.mode, self.unit, self.device)

        self._obs_preprocessor = obs_preprocessor
        self._reward_preprocessor = reward_preprocessor
        self._env_ret_preprocess_fn = env_ret_preprocess_fn
        self._policy_resp_preprocess_fn = policy_resp_preprocess_fn
        self._preprocess_fn = preprocess_fn

        self.timing_raw = {}

    def __len__(self) -> int:
        """Get the number of valid entries in the storage"""
        return self._data.__len__()

    @property
    def size(self) -> int:
        """Get the size of the storage"""
        return self.__len__()

    def get_size(self) -> int:
        """Get the size of the storage"""
        return self.__len__()

    def __getitem__(self, idx: Union[str, int, slice, np.ndarray, torch.Tensor]) -> Dict:
        """Return stored data by index."""
        ret = self._data[idx]
        if isinstance(ret, TensorDict):
            if InternalFlag.REMOTE_STORAGE:
                return to_numpy(ret.to_dict())
            return ret.to_dict()
        return ret

    def get_data(self) -> TensorDict:
        """Return all stored data as a TensorDict."""
        return self._data[: self.size]

    def clear(self) -> None:
        """Clear all stored data."""
        self._data.clear()

    def add_transition(self, env_ret: EnvRet, policy_resp: PolicyResponse) -> None:
        """
        Add a transition with an env_ret and policy_resp pair to the storage.

        Args:
            env_ret (EnvRet): The environment return.
            policy_resp (PolicyResponse): The policy response.
        """
        if InternalFlag.DEBUG:
            t_policy_to_buffer_s = policy_resp.compute_sent_latency()
            t_env_to_buffer_s = env_ret.compute_sent_latency()
            t_pair_to_buffer_s = min(t_policy_to_buffer_s, t_env_to_buffer_s)
            profiler.record_timing("transition_pair_to_buffer", t_pair_to_buffer_s, self.timing_raw, level="debug")

        if env_ret.env_id != policy_resp.env_id:
            raise ValueError(f"Mismatched env IDs: {env_ret.env_id}, {policy_resp.env_id}")

        env_id = env_ret.env_id
        transition_buffer = defaultdict()
        self.transitions_buffer[env_id] = self._preprocess_fn(
            transition_buffer,
            env_ret,
            policy_resp,
            self._obs_preprocessor,
            self._reward_preprocessor,
            self._env_ret_preprocess_fn,
            self._policy_resp_preprocess_fn,
        )

        self.episodes_buffer.add_transition(env_id, self.transitions_buffer[env_id])

        if self.auto_truncate_episode:
            done_episodes = self.episodes_buffer.pop_done_episode(env_id)
            if len(done_episodes):
                self._data.push(done_episodes)

        # (tmp) this will reset the ts_env_sent_ns to current time
        env_ret.ts_env_sent_ns = time.time_ns()

    def add_data_async(self, item: Union[EnvRet, PolicyResponse]) -> None:
        """
        Add a single EnvRet or PolicyResponse to the storage. It is usually used in async rollout
        that cannot collect env_ret and policy_resp pair at the same time.

        Args:
            item (Union[EnvRet, PolicyResponse]): The EnvRet or PolicyResponse to add
        """
        if not isinstance(item, (EnvRet, PolicyResponse)):
            raise TypeError(f"Expected EnvRet or PolicyResponse, got {type(item)}")

        env_id = item.env_id

        if isinstance(item, EnvRet):
            if InternalFlag.DEBUG:
                # env_ret -> buffer transfer time and record in timing_raw
                t_env_to_buffer_s = item.compute_sent_latency()
                profiler.record_timing("transition_env_to_buffer", t_env_to_buffer_s, self.timing_raw, level="debug")

            transition_buffer = defaultdict()
            self.transitions_buffer[env_id] = self._preprocess_fn(
                transition_buffer,
                item,
                None,
                self._obs_preprocessor,
                self._reward_preprocessor,
                self._env_ret_preprocess_fn,
                self._policy_resp_preprocess_fn,
            )
        elif isinstance(item, PolicyResponse):
            if InternalFlag.DEBUG:
                # policy_resp -> buffer transfer time and record in timing_raw
                t_policy_to_buffer_s = item.compute_sent_latency()
                profiler.record_timing(
                    "transition_policy_to_buffer",
                    t_policy_to_buffer_s,
                    self.timing_raw,
                    level="debug",
                )

            self.transitions_buffer[env_id] = self._preprocess_fn(
                self.transitions_buffer[env_id],
                None,
                item,
                self._obs_preprocessor,
                self._reward_preprocessor,
                self._env_ret_preprocess_fn,
                self._policy_resp_preprocess_fn,
            )
            self.episodes_buffer.add_transition(env_id, self.transitions_buffer[env_id])

        if self.auto_truncate_episode:
            done_episodes = self.episodes_buffer.pop_done_episode(env_id)
            if len(done_episodes):
                self._data.push(done_episodes)

    def add_episode(self, episode: Dict | TensorDict, num_envs: int = 1) -> None:
        """
        Add a complete episode to the storage.

        Args:
            episode (Dict | TensorDict): The episode to add.
            num_envs (int): Number of parallel environments represented in the episode.
        """
        if num_envs > 1:
            episode_length = len(episode[list(episode.keys())[0]])  # first key's length
            episode_td = TensorDict(episode, batch_size=[episode_length, num_envs])
            sub_episode_td_list = episode_td.unbind(1)
            episode = [sub_episode_td.to_dict() for sub_episode_td in sub_episode_td_list]

        self._data.push(episode)

    def truncate_episodes(self, env_ids: List[str]) -> None:
        """
        manually finish current episode to actually save into replay buffer's storage.
        Args:
            env_ids (List[str]): list of env ids to truncate
        """
        for env_id in env_ids:
            self.truncate_one_episode(env_id)

    @profiler.timer_wrap(level="debug")
    def truncate_one_episode(self, item: Union[str, Any]) -> None:
        """
        manually finish current episode to actually save into replay buffer's storage.
        Args:
            item (Union[str, Any]): env_id or object with env_id attribute
        """
        if isinstance(item, str):
            env_id = item
        else:
            if not hasattr(item, "env_id"):
                raise TypeError(f"Expected env_id str or object with env_id attribute, got {type(item)}")
            env_id = item.env_id

        episode_list = self.episodes_buffer.pop(env_id)

        self._data.push(episode_list)

    def print_timing_summary(self, reset: bool = False) -> None:
        """
        Print the timing summary of the storage.
        """
        logger.debug("Buffer storage timing:")
        # iterate over a snapshot to avoid concurrent modification during iteration
        for name, stats in dict(self.timing_raw).items():
            logger.debug(f"{name:28} count={stats['count']:<3} total={stats['total']:.6f}s avg={stats['avg']:.6f}s")
        if reset:
            self.timing_raw = {}
