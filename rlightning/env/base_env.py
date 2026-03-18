"""Base environment module for reinforcement learning.

This module defines the abstract base class for all RL environments,
providing the common interface for environment interactions.
"""

import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import gymnasium as gym
import ray
import torch

from rlightning.types import EnvMeta, EnvRet, PolicyResponse
from rlightning.utils.config import EnvConfig
from rlightning.utils.logger import get_logger
from rlightning.utils.profiler import profiler
from rlightning.utils.utils import InternalFlag, to_device

logger = get_logger(__name__)


class BaseEnv(ABC):
    """Abstract base class for reinforcement learning environments.

    This class defines the common interface for all RL environments,
    including methods for reset, step, and environment metadata.

    Attributes:
        config: Environment configuration object.
        env_id: Unique identifier for this environment instance.
        env: The underlying gymnasium environment.
        num_envs: Number of parallel environments (1 by default).
        max_episode_steps: Maximum steps per episode.
        timing_raw: Dictionary for tracking timing statistics.
    """

    def __init__(
        self,
        config: EnvConfig,
        worker_index: Optional[int] = 0,
        preprocess_fn: Optional[Callable] = None,
    ) -> None:
        """Initialize the base environment.

        Args:
            config: Environment configuration object.
            worker_index: Index of the env worker for placement grouping.
            preprocess_fn: Optional function to preprocess observations.
        """
        self.config = deepcopy(config)
        self.env_id = str(uuid.uuid4()) + "-" + str(worker_index)

        self.env = None

        self.num_envs = self.config.num_envs
        self.max_episode_steps = self.config.max_episode_steps

        self._preprocess_fn: Callable = preprocess_fn

        # episode env info tracking
        self._episode_env_infos: Dict[str, list] = defaultdict(list)

        # timing
        self.timing_raw: Dict[str, Dict[str, Any]] = {}

    def get_env_id(self) -> str:
        """Get the unique environment identifier.

        Returns:
            The unique identifier string for this environment.
        """
        return self.env_id

    def get_observation_space(self) -> gym.Space:
        """
        Retrieve observation space.
        User can override this method as needed.
        """
        return self.env.observation_space

    def get_action_space(self) -> gym.Space:
        """
        Retrieve action space.
        User can override this method as needed.
        """
        return self.env.action_space

    def get_max_episode_steps(self) -> Optional[int]:
        """Retrieve max episode steps"""
        return self.max_episode_steps

    def get_metadata(self) -> EnvMeta:
        """
        Get the environment meta information.

        Returns:
            EnvMeta: the environment meta information
        """
        return EnvMeta(
            env_id=self.env_id,
            action_space=self.get_action_space(),
            observation_space=self.get_observation_space(),
            num_envs=self.num_envs,
        )

    @abstractmethod
    def reset(self, *args, **kwargs) -> EnvRet | List[EnvRet] | List[ray.ObjectRef]:
        """Reset the environment to initial state.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            Environment return containing observation and info.
        """
        pass

    def _reset(self, *args, **kwargs) -> EnvRet:
        """Internal reset method with post-step hook.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            Environment return after applying post-step hook.
        """
        env_ret = self.reset(*args, **kwargs)
        env_ret = self._post_step_hook(env_ret)
        return env_ret

    @abstractmethod
    def step(self, policy_resp: PolicyResponse) -> EnvRet:
        """Step the environment with the given action.

        Args:
            policy_resp: Policy response containing the action.

        Returns:
            Environment return containing observation, reward, done, and info.
        """
        pass

    def step_async(self, policy_resp_list: List[PolicyResponse]) -> None:
        """Asynchronous step interface (only for RemoteEnvServer).

        This interface also reserves for future integration with env that natively support asynchronous steps.

        Args:
            policy_resp_list: List of Policy response

        Raises:
            NotImplementedError: Always, as this is not native supported and only for RemoteEnvServer from now.
        """
        raise NotImplementedError(
            "The step_async interface in the BaseEnv class is overwritten, with its scope "
            "restricted to the RemoteEnvServer class only."
        )

    def collect_async(self) -> List[EnvRet]:
        """Asynchronous collect interface (only for RemoteEnvServer).

        This interface also reserves for future integration with env that natively support asynchronous steps.

        Raises:
            NotImplementedError: Always, as this is not native supported and only for RemoteEnvServer from now.
        """
        raise NotImplementedError(
            "The collect_async interface in the BaseEnv class is overwritten, with its scope "
            "restricted to the RemoteEnvServer class only."
        )

    def _pre_step_hook(self, policy_resp: PolicyResponse) -> PolicyResponse:
        """Pre-step hook for timing and debugging.

        Computes policy -> env transfer time and records in timing_raw.

        Args:
            policy_resp: Policy response being processed.
        """
        policy_resp = policy_resp.cpu()
        if InternalFlag.DEBUG:
            # compute policy -> env transfer time and record in timing_raw
            t_policy_to_env_s = policy_resp.compute_sent_latency()
            profiler.record_timing("transition_policy_to_env", t_policy_to_env_s, self.timing_raw, level="debug")

        return policy_resp

    def _post_step_hook(self, env_ret: EnvRet) -> EnvRet:
        """Post-step hook for data transfer optimization.

        Handles CPU transfer and Ray object storage based on internal flags.

        Args:
            env_ret: Environment return to process.

        Returns:
            Processed environment return.
        """
        self._record_episode_info(env_ret)

        if InternalFlag.REMOTE_STORAGE:
            env_ret = env_ret.numpy()

        env_ret.mark_env_sent()
        return env_ret

    def _record_episode_info(self, env_ret: EnvRet) -> None:
        """Record episode-level info from the environment return.

        Args:
            env_ret: Environment return possibly containing episode_info.
        """
        env_info = env_ret.info
        if isinstance(env_info, dict) and "episode_info" in env_info:
            for k, v in env_info["episode_info"].items():
                self._episode_env_infos[k].append(v)

    def get_env_stats(self, reset: bool = False) -> Dict[str, list]:
        """Get episode-level environment statistics.

        Computes sum and count for each recorded metric key so that the
        caller (e.g. EnvGroup) can aggregate across multiple environments.

        Args:
            reset: If True, clear recorded info after computing stats.

        Returns:
            Dict mapping metric name to [sum, count].
        """
        stats: Dict[str, list] = defaultdict(lambda: [0.0, 0])

        for k, v in self._episode_env_infos.items():
            if isinstance(v, list) and len(v) > 0:
                v = to_device(v, "cpu")
                if isinstance(v[0], torch.Tensor):
                    t = torch.cat([x.flatten() for x in v]).float()
                else:
                    t = torch.tensor(v).float()
            else:
                t = torch.empty(0)
            stats[k][0] += t.sum().item()
            stats[k][1] += t.numel()

        if reset:
            self._episode_env_infos.clear()

        return stats

    def _step(self, policy_resp: PolicyResponse) -> EnvRet:
        """Internal step method with pre and post hooks.

        Args:
            policy_resp: Policy response containing the action.

        Returns:
            Environment return after applying hooks.
        """
        policy_resp = self._pre_step_hook(policy_resp)
        env_ret = self.step(policy_resp)
        env_ret = self._post_step_hook(env_ret)
        return env_ret

    def _step_async(self, policy_resp_list: List[PolicyResponse]) -> None:
        """Internal async step method for remote env server.

        Args:
            policy_resp_list: List of policy responses containing actions.
        """
        # not calling pre_step_hook since it's done on remote env side on client
        self.step_async(policy_resp_list)

    def _collect_async(self) -> List[EnvRet]:
        """Internal async collect method for remote env server.

        Returns:
            List[EnvRet]: list of EnvRet from clients
        """
        # not calling post_step_hook since it's done on remote env side on client
        return self.collect_async()

    def init(self) -> EnvMeta:
        """Initialize the environment and return metadata.

        Returns:
            EnvMeta containing environment properties.
        """

        return self.get_metadata()

    def is_finish(self) -> bool:
        """Check if the environment should finish running.

        Override this method in RemoteEnvClient subclasses to determine
        when to stop the environment loop.

        Returns:
            True if the environment should stop, False otherwise.
        """
        return False

    def close(self) -> None:
        """Close the environment.

        Override this method if special cleanup is needed.
        Default implementation does nothing.
        """
        pass

    def offload(self):
        """
        Offload the environment to free GPU memory.
        """
        pass
        # raise NotImplementedError("Offload is not implemented for this environment.")

    def reload(self):
        """
        Reload the environment to load GPU memory.
        """
        pass
        # raise NotImplementedError("Reload is not implemented for this environment.")

    def apply_evaluate_cfg(self) -> None:
        """Apply evaluation-time config overrides for this environment.

        Default implementation is a no-op. Specific environments can override
        this to support temporary evaluate-only behaviors.
        """
        return

    def restore_evaluate_cfg(self) -> None:
        """Restore environment members changed by apply_evaluate_cfg.

        Default implementation is a no-op.
        """
        return

    def finish_rollout(self) -> None:
        """Finish the rollout.

        Override this method in subclasses to implement custom rollout finishing behavior.
        """
        return

    def print_timing_summary(self, reset: bool = False) -> None:
        """Print timing summary for profiling.

        Args:
            reset: If True, reset timing statistics after printing.
        """
        logger.debug(f"Env {self.env_id} timing:")
        # iterate over a snapshot to avoid concurrent modification during iteration
        for name, stats in dict(self.timing_raw).items():
            logger.debug(f"{name:28} count={stats['count']:<3} total={stats['total']:.6f}s avg={stats['avg']:.6f}s")
        if reset:
            self.timing_raw = {}
