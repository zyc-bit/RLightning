"""IsaacLab environment wrappers."""

import asyncio
import copy
import logging
import os
from collections import namedtuple
from numbers import Number
from typing import Any, Callable, Optional

# Disable all logging below WARNING level
os.environ["ISAACSIM_LOG_LEVEL"] = "ERROR"  # or "FATAL", "OFF"
os.environ["OMNI_LOG_LEVEL"] = "ERROR"
os.environ["CARB_LOG_LEVEL"] = "ERROR"

# Set Python logging to only show errors
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# Disable matplotlib debug logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("AutoNode").setLevel(logging.WARNING)
logging.getLogger("h5py._conv").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("trimesh").setLevel(logging.WARNING)

import argparse
import asyncio
import copy
import importlib
import math

import gymnasium as gym
import torch

from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.config import EnvConfig
from rlightning.utils.logger import get_logger
from rlightning.utils.profiler import profiler
from rlightning.utils.registry import ENVS
from rlightning.utils.utils import to_device

from .base_env import BaseEnv, EnvMeta
from .utils.utils import default_env_preprocess_fn

RSLEnvMeta = namedtuple("RSLEnvMeta", EnvMeta._fields + ("get_observations", "num_actions"))

logger = get_logger(__name__)


@ENVS.register("isaac_manager_based")
class IsaacManagerBasedRLEnv(BaseEnv):
    """IsaacLab manager-based RL environment wrapper."""

    def __init__(
        self,
        config: EnvConfig,
        worker_index: Optional[int] = 0,
        preprocess_fn: Optional[Callable] = default_env_preprocess_fn,
        **kwargs: Any,
    ) -> None:
        """Initialize the IsaacLab manager-based environment."""
        super().__init__(config, worker_index, preprocess_fn)

        from isaaclab.app import AppLauncher

        # use standard asyncio for Isaac Sim to avoid api incompatibility caused by uvloop
        try:
            current_policy = asyncio.get_event_loop_policy()
            if "uvloop" in str(type(current_policy)).lower():
                logger.warning(
                    "Detected uvloop in Ray Actor. Reverting to standard asyncio for Isaac Sim compatibility."
                )
                asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        except Exception as e:
            logger.warning(f"Failed to check/switch event loop policy: {e}")

        app_launcher = None
        simulation_app = None

        isaac_env_kwargs = config.env_kwargs

        launcher_cfg = isaac_env_kwargs.launcher
        opts = argparse.Namespace(
            kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error",
            headless=launcher_cfg.headless,
            device=f"cuda:{torch.cuda.current_device()}",
            num_envs=config.num_envs,
        )
        app_launcher = AppLauncher(opts)
        simulation_app = app_launcher.app

        # import customized environment spec
        importlib.import_module(isaac_env_kwargs.env_spec)

        from isaaclab.envs import ManagerBasedRLEnvCfg

        # use import
        module_path, module_name = isaac_env_kwargs.env_cfg.module.split("::")

        task_cfg_cls = getattr(importlib.import_module(module_path), module_name)

        task_cfg: ManagerBasedRLEnvCfg = task_cfg_cls()

        # override num_envs here?
        assert hasattr(config, "num_envs")
        if hasattr(config, "num_envs"):
            task_cfg.scene.num_envs = config.num_envs

        task_cfg.episode_length_s = math.ceil(config.max_episode_steps * task_cfg.decimation / task_cfg.sim.dt)
        task_cfg.from_dict(isaac_env_kwargs.env_cfg.override.to_dict())

        self.env: gym.Env = gym.make(config.task, cfg=task_cfg, render_mode=None)

        assert self.num_envs == self.env.unwrapped.num_envs
        if isinstance(self.env.unwrapped.observation_space, gym.spaces.Dict):
            self._observation_space = self.env.unwrapped.observation_space["policy"]
        else:
            self._observation_space = self.env.unwrapped.observation_space

        self._action_space = self.env.unwrapped.action_space
        self.simulation_app = simulation_app

    def init(self) -> RSLEnvMeta:
        """
        init the environment and return the environment meta information
        """

        obs_dict, info = self.env.reset()
        info["observations"] = obs_dict

        obs_dict_cpu = obs_dict["policy"].cpu()
        info_cpu = to_device(info, "cpu")
        return RSLEnvMeta(
            env_id=self.env_id,
            action_space=self.get_action_space(),
            observation_space=self.get_observation_space(),
            num_envs=self.num_envs,
            get_observations=lambda: (
                obs_dict_cpu,
                info_cpu,
            ),
            num_actions=self._action_space.shape[-1],
        )

    @property
    def unwrapped(self) -> gym.Env:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    def get_observation_space(self) -> gym.Space:
        """Retrieve vectorized observation space

        Returns:
            Dict[str, gym.Space]: Observation space
        """

        return self._observation_space

    def get_action_space(self) -> gym.Space:
        """Retrive vectorized action space

        Returns:
            Dict[str, gym.Space]: Action space
        """

        return self._action_space

    @profiler.timer_wrap(level="debug")
    def reset(self, *args, **kwargs) -> EnvRet:
        """Reset the environment and return the initial EnvRet."""
        obs_dict, info = self.env.reset()
        info["observations"] = obs_dict

        log = info.pop("log", {})
        episode_info = {}
        for k, v in log.items():
            if isinstance(v, Number):
                episode_info[k] = v
            elif isinstance(v, torch.Tensor):
                episode_info[k] = v.mean().item()
        info["episode_info"] = episode_info
        self.last_obs = obs_dict["policy"]
        self.last_info = info

        return EnvRet(
            env_id=self.env_id,
            observation=obs_dict["policy"],
            last_reward=torch.zeros(
                self.unwrapped.num_envs, dtype=torch.float32, device=self.unwrapped.device
            ),  # type: ignore
            last_terminated=torch.zeros(
                self.unwrapped.num_envs, dtype=torch.long, device=self.unwrapped.device
            ).bool(),  # type: ignore
            last_truncated=torch.zeros(
                self.unwrapped.num_envs, dtype=torch.long, device=self.unwrapped.device
            ).bool(),  # type: ignore
            info=info,
            _extra={
                "last_observation": obs_dict["policy"],
                "last_info": info,
            },
        )

    @profiler.timer_wrap(level="debug")
    def step(self, policy_resp: PolicyResponse) -> EnvRet:
        """Step the environment with the given policy response.

        The returned EnvRet has items as a series of dict (mapping from agent ids to entries)

        Args:
            policy_resp (PolicyResponse): The response from the policy, containing the action to take

        Returns:
            EnvRet: The return from the environment after taking the action
        """

        actions = self._preprocess_fn(policy_resp)

        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        actions = actions.to(self.unwrapped.device)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        dones = (terminated | truncated).to(dtype=torch.long)

        extras["observations"] = obs_dict

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        log = extras.pop("log", {})
        episode_info = {}
        for k, v in log.items():
            if isinstance(v, Number):
                episode_info[k] = v
            elif isinstance(v, torch.Tensor):
                episode_info[k] = v.mean().item()
        extras["episode_info"] = episode_info

        last_obs = copy.deepcopy(self.last_obs)
        last_info = copy.deepcopy(self.last_info)

        # update last obs and info
        self.last_obs = obs_dict["policy"]
        self.last_info = extras

        return EnvRet(
            env_id=self.env_id,
            observation=obs_dict["policy"],
            last_reward=rew,
            last_terminated=dones,
            last_truncated=truncated,
            info=extras,
            _extra={
                "last_observation": last_obs,
                "last_info": last_info,
            },
        )

    def close(self) -> None:
        """Close the environment and its simulation app."""
        self.env.close()
        self.simulation_app.close()
