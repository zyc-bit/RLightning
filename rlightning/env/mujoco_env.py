"""MuJoCo environment wrapper."""

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.profiler import profiler
from rlightning.utils.registry import ENVS

from .base_env import BaseEnv
from .utils.utils import default_env_preprocess_fn

REGISTERED_ENV = {"Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5"}


@ENVS.register("mujoco")
class MujocoEnv(BaseEnv):
    """Gymnasium MuJoCo environment wrapper."""

    def __init__(
        self,
        config: Any,
        worker_index: Optional[int] = 0,
        preprocess_fn: Optional[Callable] = default_env_preprocess_fn,
        **kwargs: Any,
    ) -> None:
        """Initialize the MuJoCo environment."""
        super().__init__(config, worker_index, preprocess_fn)

        self.env = gym.make(self.config.task, max_episode_steps=self.config.max_episode_steps)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    @profiler.timer_wrap(level="debug")
    def reset(self, *args: Any, **kwargs: Any) -> EnvRet:
        """Reset the environment and return an EnvRet."""
        observation, info = self.env.reset(*args, **kwargs)
        return EnvRet(
            env_id=self.env_id,
            observation=observation,
            info=info,
        )

    @profiler.timer_wrap(level="debug")
    def step(self, policy_resp: PolicyResponse) -> EnvRet:
        """Step the environment with a policy response."""
        action = self._preprocess_fn(policy_resp)
        action = np.asarray(action)
        observation, reward, terminated, truncated, info = self.env.step(action)

        return EnvRet(
            env_id=self.env_id,
            observation=observation,
            last_reward=reward,
            last_terminated=terminated,
            last_truncated=truncated,
            info=info,
        )
