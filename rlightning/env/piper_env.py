"""Piper environment wrapper."""

from typing import Any, Callable, Optional

from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.profiler import profiler
from rlightning.utils.registry import ENVS

from .base_env import BaseEnv
from .utils.utils import default_env_preprocess_fn


@ENVS.register("piper")
class PiperEnv(BaseEnv):
    """Piper Environment."""

    def __init__(
        self,
        config: Any,
        worker_index: Optional[int] = 0,
        preprocess_fn: Optional[Callable] = default_env_preprocess_fn,
        **kwargs: Any,
    ) -> None:
        """Initialize the Piper environment."""
        super().__init__(config, worker_index, preprocess_fn)

        from third_party.rw_rl.src.envs.last_frame_wrapper import LastFrameReturn
        from third_party.rw_rl.src.envs.piper_env import PiperEnv as Piper

        env = Piper(
            task_name=config.task,
            enable_piper=config.get("enable_piper", False),
            can_port=config.get("can_port", "can0"),
            camera_mapping=config.get("intel_viewer_mapping", None),
            move_speed=config.get("move_speed", 10),
            block=config.get("block", False),
            visdom_config=config.get("visdom_config", None),
            enable_joint_validation=config.get("enable_joint_validation", True),
            max_episode_steps=config.get("max_episode_steps", None),
            control_hz=config.get("control_hz", -1),
            env_reset_joint=config.get("env_reset_joint", None),
        )
        self.env = LastFrameReturn(env)

    @profiler.timer_wrap(level="debug")
    def reset(self, *args: Any, **kwargs: Any) -> EnvRet:
        """Reset the environment and return an EnvRet."""
        state, _ = self.env.reset(*args, **kwargs)

        return EnvRet(env_id=self.env_id, **state)

    @profiler.timer_wrap(level="debug")
    def step(self, policy_resp: PolicyResponse, *args: Any, **kwargs: Any) -> EnvRet:
        """Step the environment with a policy response."""
        action = self._preprocess_fn(policy_resp, *args, **kwargs)
        _state, _reward, _done, _truncated, info = self.env.step(action)
        return EnvRet(
            env_id=self.env_id,
            **info,
        )
