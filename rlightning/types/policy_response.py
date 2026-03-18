"""Policy response data structures.

This module provides classes for representing policy action responses,
supporting both single-agent and multi-agent scenarios.
"""

import time
from types import SimpleNamespace
from typing import Any, Dict, KeysView, Optional, Union

import gymnasium as gym

from rlightning.utils.utils import to_device, to_numpy


class PolicyResponse(SimpleNamespace):
    """Policy response container for single-agent interactions.

    Holds the action and any additional data produced by a policy
    for a single environment step.

    Attributes:
        env_id: Environment identifier this response is for.
        Additional attributes are set dynamically via **data.
    """

    def __init__(self, env_id: str, **data: Any) -> None:
        """Initialize policy response.

        Args:
            env_id: Environment identifier.
            **data: Additional response data (action, log_prob, etc.).
        """
        super().__init__(**data)
        self.env_id = env_id

    @property
    def _fields(self) -> KeysView[str]:
        """Get field names excluding internal fields.

        Returns:
            View of field names for serialization.
        """
        return self.__dict__.keys() - {"env_id", "ts_policy_sent_ns"}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary excluding env_id.

        Returns:
            Dictionary of response fields.
        """
        return dict((key, getattr(self, key)) for key in self._fields)

    def cpu(self) -> "PolicyResponse":
        """Move all tensor attributes to CPU.

        Returns:
            Self with tensors moved to CPU.
        """
        for key in self._fields:
            value = getattr(self, key)
            setattr(self, key, to_device(value, "cpu"))
        return self

    def cuda(self, device: Optional[Union[int, str]] = None) -> "PolicyResponse":
        """Move all tensor attributes to CUDA.

        Args:
            device: CUDA device index or string. Defaults to 'cuda'.

        Returns:
            Self with tensors moved to CUDA.
        """
        device = "cuda" if device is None else device

        for key in self._fields:
            value = getattr(self, key)
            setattr(self, key, to_device(value, device))
        return self

    def numpy(self) -> "PolicyResponse":
        """Convert all tensor attributes to NumPy arrays.

        Returns:
            Self with tensors converted to NumPy arrays.
        """
        for key in self._fields:
            value = getattr(self, key)
            setattr(self, key, to_numpy(value))
        return self

    def mark_policy_sent(self) -> "PolicyResponse":
        """Record the timestamp when the policy response is sent.

        Returns:
            Self for method chaining.
        """
        self.ts_policy_sent_ns = time.time_ns()
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

    def __getstate__(self) -> Dict[str, Any]:
        """Get state for pickling."""
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set state from pickling."""
        self.__dict__.update(state)

    def __reduce__(self) -> tuple:
        """Support for pickle serialization."""
        return (self.__class__.__new__, (self.__class__,), self.__dict__)

    @staticmethod
    def make_example(action_space: gym.Space, env_id: Optional[str] = None, **data: Any) -> "PolicyResponse":
        """Create an example PolicyResponse.

        Args:
            action_space: Gymnasium action space for sampling.
            env_id: Optional environment identifier.
            **data: Additional response data.

        Returns:
            PolicyResponse with sampled action.
        """
        return PolicyResponse(env_id=env_id, action=action_space.sample(), **data)


class MultiAgentPolicyResponse(PolicyResponse):
    """Policy response for multi-agent interactions.

    Extends PolicyResponse with support for multiple agents,
    where actions are stored in a dictionary keyed by agent ID.
    """

    @staticmethod
    def make_example(
        action_spaces: Dict[str, gym.Space], env_id: Optional[str] = None, **data: Any
    ) -> "MultiAgentPolicyResponse":
        """Create an example MultiAgentPolicyResponse.

        Args:
            action_spaces: Dictionary mapping agent IDs to action spaces.
            env_id: Optional environment identifier.
            **data: Additional response data.

        Returns:
            MultiAgentPolicyResponse with sampled actions for all agents.
        """
        return MultiAgentPolicyResponse(env_id=env_id, action={k: v.sample() for k, v in action_spaces.items()}, **data)
