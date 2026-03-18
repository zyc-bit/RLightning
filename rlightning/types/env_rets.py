"""Environment return data structures.

This module provides dataclasses for representing environment step/reset
returns, supporting both single-agent and multi-agent scenarios.
"""

import time
from dataclasses import _MISSING_TYPE, dataclass, field, fields, replace
from typing import Any, Dict, Optional, Tuple, Union

from rlightning.utils.utils import to_device, to_numpy


@dataclass
class EnvRet:
    """Environment return data structure for single-agent interactions.

    Represents the return value from environment step() or reset() calls,
    containing observation, reward, termination status, and additional info.

    Attributes:
        env_id: Unique identifier for the environment instance.
        observation: Observation after environment step/reset.
        last_reward: Reward received after the last step.
        last_terminated: Whether the episode terminated after last step.
        last_truncated: Whether the episode was truncated after last step.
        last_info: Additional info dictionary from the environment.
        _extra: Extra fields for extensibility.
        ts_env_sent_ns: Timestamp (ns) when this EnvRet was produced.
    """

    env_id: str
    """ Environment unique identifier """
    observation: Any
    """ Observation after environment step/reset """
    last_reward: float = 0.0
    """ Reward received after the last step """
    last_terminated: bool = False
    """ Whether the episode has terminated after the last step """
    last_truncated: bool = False
    """ Whether the episode has been truncated after the last step """
    info: Dict[str, Any] = field(default_factory=dict)
    """ Additional info from the environment """
    _extra: Dict[str, Any] = field(default_factory=dict)
    """ Extra fields for extensibility """

    ts_env_sent_ns: int = field(default_factory=time.time_ns)
    """ Timestamp (ns) when this EnvRet is produced and sent by the env actor """

    @classmethod
    def fields(cls) -> Tuple:
        """Get field names excluding internal fields.

        Returns:
            Tuple of field names for serialization.
        """
        field_names = [f.name for f in fields(cls)]
        field_names.remove("env_id")
        field_names.remove("ts_env_sent_ns")
        field_names.remove("_extra")
        return tuple(field_names)

    def mark_env_sent(self) -> "EnvRet":
        """Record the timestamp when the env return is sent.

        Returns:
            Self for method chaining.
        """
        self.ts_env_sent_ns = time.time_ns()
        return self

    def compute_sent_latency(self, now_ns: Optional[int] = None) -> float:
        """Compute latency in seconds from env-sent timestamp to now.

        Use for env->policy or env->buffer transfer time (e.g. in policy _rollout_hook or in buffer add_transition).

        Args:
            now_ns: Current time in nanoseconds. If None, uses time.time_ns().

        Returns:
            Latency in seconds, or 0.0 if ts_env_sent_ns is missing or invalid.
        """
        try:
            ts = int(self.ts_env_sent_ns)
            now_ns = time.time_ns() if now_ns is None else now_ns
        except Exception:
            return 0.0
        return max(0.0, (now_ns - ts) / 1e9)

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Get default values for all serializable fields.

        Returns:
            Dictionary mapping field names to default values.
        """
        defaults = {}
        for f in fields(cls):
            if f.name in cls.fields():
                if not isinstance(f.default, _MISSING_TYPE):
                    defaults[f.name] = f.default

        return defaults

    def to_dict(self) -> Dict[str, Any]:
        """Convert EnvRet to dictionary.

        Excludes 'env_id' and 'ts_env_sent_ns'. Includes _extra fields
        if present.

        Returns:
            Dictionary representation of the environment return.

        Raises:
            KeyError: If _extra contains keys conflicting with existing fields.
        """
        results = {}
        for key in self.fields():
            results[key] = getattr(self, key)

        if self._extra:
            for key, value in self._extra.items():
                if key in results:
                    raise KeyError(f"Key {key} in _extra conflicts with existing fields.")

                results[key] = value

        return results

    def cpu(self) -> "EnvRet":
        """Move all tensor attributes to CPU.

        Returns:
            Self with tensors moved to CPU.
        """
        changes = {}

        for key in self.fields():
            value = getattr(self, key)
            new_value = to_device(value, "cpu")
            if new_value is not value:
                changes[key] = new_value

        if self._extra:
            new_extra = to_device(self._extra, "cpu")
            if new_extra is not self._extra:
                changes["_extra"] = new_extra

        return replace(self, **changes)

    def cuda(self, device: Optional[Union[int, str]] = None) -> "EnvRet":
        """Move all tensor attributes to CUDA.

        Args:
            device: CUDA device index or string. Defaults to 'cuda'.

        Returns:
            Self with tensors moved to CUDA.
        """

        device = "cuda" if device is None else device

        changes = {}
        for key in self.fields():
            value = getattr(self, key)
            new_value = to_device(value, device)
            if new_value is not value:
                changes[key] = new_value

        if self._extra:
            new_extra = to_device(self._extra, device)
            if new_extra is not self._extra:
                changes["_extra"] = new_extra

        return replace(self, **changes)

    def numpy(self) -> "EnvRet":
        """Convert all tensor attributes to numpy arrays.

        Returns:
            Self with tensors converted to numpy arrays.
        """

        changes = {}
        for key in self.fields():
            value = getattr(self, key)
            new_value = to_numpy(value)
            if new_value is not value:
                changes[key] = new_value
        if self._extra:
            new_extra = to_numpy(self._extra)
            if new_extra is not self._extra:
                changes["_extra"] = new_extra

        return replace(self, **changes)

    def __hash__(self) -> int:
        """Hash EnvRet by its environment identifier."""
        return hash(self.env_id)


@dataclass
class MultiAgentEnvRet(EnvRet):
    """Environment return for multi-agent interactions.

    Extends EnvRet with dictionary-based rewards and termination flags
    for multiple agents.

    Attributes:
        last_reward: Dictionary mapping agent IDs to rewards.
        last_terminated: Dictionary mapping agent IDs to termination flags.
        last_truncated: Dictionary mapping agent IDs to truncation flags.
    """

    last_reward: Dict[str, float]
    last_terminated: Dict[str, bool]
    last_truncated: Dict[str, bool]


Processed_EnvRet_fields = ("next_observation", "reward", "terminated", "truncated", "info")
