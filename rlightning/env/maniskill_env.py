# Derived from code copied from RLinf/RLinf (Apache-2.0):
# https://github.com/RLinf/RLinf
# Original path: rlightning/env/maniskill_env.py
# Modified in this repository.
# See THIRD_PARTY_NOTICES.md for details.

"""ManiSkill environment wrapper."""

import gc
import os
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

try:
    from mani_skill.envs.sapien_env import BaseEnv as Base
    from mani_skill.utils import common, gym_utils
    from mani_skill.utils.common import torch_clone_dict
    from mani_skill.utils.visualization.misc import (
        images_to_video,
        put_info_on_image,
        tile_images,
    )
except ImportError as e:
    print(f"ManiSkill is not installed: {e}")

from rlightning.env.base_env import BaseEnv
from rlightning.env.utils.utils import default_env_preprocess_fn
from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.profiler import profiler
from rlightning.utils.registry import ENVS


@ENVS.register("maniskill")
class ManiskillEnv(BaseEnv):
    """ManiSkill environment wrapper for vision-based tasks."""

    def __init__(
        self,
        config: Any,
        worker_index: Optional[int] = None,
        preprocess_fn: Optional[Callable] = default_env_preprocess_fn,
        **kwargs: Any,
    ) -> None:
        """Initialize the ManiSkill environment."""
        super().__init__(config, worker_index, preprocess_fn)
        env_seed = config.seed
        self.seed = env_seed + worker_index
        self.auto_reset = config.auto_reset
        self.use_rel_reward = config.use_rel_reward
        self.ignore_terminations = config.ignore_terminations
        self.use_fixed_reset_state_ids = config.use_fixed_reset_state_ids

        self.video_cfg = config.video_cfg.to_dict()
        self.video_cnt = 0
        self.render_images = []

        self.env_config = config.init_params.to_dict()

        self.env: Base = gym.make(**self.env_config)
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(self.device)  # [B, ]

        self.record_metrics = self.env_config.get("record_metrics", True)
        self.num_action_chunks = self.env_config.get("num_action_chunks", 1)
        self.action_dim = self.env_config.get("action_dim", 7)

        # todo: now only support one group of environments
        self.num_group = self.num_envs
        self.group_size = 1
        self._init_reset_state_ids()

        if self.record_metrics:
            self._init_metrics()

        # Track offload state
        self._is_offloaded = False
        self._evaluate_cfg_backup_stack: list[Dict[str, Any]] = []

    def _merge_override(self, current_value: Any, override_value: Any) -> Any:
        """Merge override value into current config-like value."""
        if isinstance(override_value, dict):
            assert isinstance(current_value, dict), f"current_value must be a dict, but got {type(current_value)}"
            merged_value = deepcopy(current_value)
            for key, value in override_value.items():
                if key in merged_value and isinstance(merged_value[key], dict) and isinstance(value, dict):
                    merged_value[key] = self._merge_override(merged_value[key], value)
                else:
                    merged_value[key] = deepcopy(value)
            return merged_value

        return deepcopy(override_value)

    def apply_evaluate_cfg(self) -> None:
        """Apply evaluate_cfg overrides to environment member variables."""
        evaluate_cfg = self.config.get("evaluate_cfg", None)
        if evaluate_cfg is None:
            return

        evaluate_cfg_dict = evaluate_cfg.to_dict()
        member_backup: Dict[str, Any] = {}

        for key, value in evaluate_cfg_dict.items():
            if not hasattr(self, key):
                continue
            current_value = getattr(self, key)
            member_backup[key] = deepcopy(current_value)
            setattr(self, key, self._merge_override(current_value, value))

        self._evaluate_cfg_backup_stack.append(member_backup)

    def restore_evaluate_cfg(self) -> None:
        """Restore previously overridden member variables from evaluate_cfg."""
        if not self._evaluate_cfg_backup_stack:
            return

        member_backup = self._evaluate_cfg_backup_stack.pop()
        for key, value in member_backup.items():
            setattr(self, key, value)

    @property
    def total_num_group_envs(self) -> int:
        """Return the total number of grouped environments."""
        if hasattr(self.env.unwrapped, "total_num_trials"):
            return self.env.unwrapped.total_num_trials
        assert hasattr(self.env, "xyz_configs") and hasattr(self.env, "quat_configs")
        return len(self.env.xyz_configs) * len(self.env.quat_configs)

    # @property
    # def num_envs(self):
    #     return self.env.unwrapped.num_envs

    @property
    def device(self) -> torch.device:
        """Return the device used by the underlying environment."""
        return self.env.unwrapped.device

    @property
    def elapsed_steps(self) -> torch.Tensor:
        """Return the elapsed steps tensor from the environment."""
        return self.env.unwrapped.elapsed_steps

    @property
    def instruction(self) -> Any:
        """Return the current language instruction."""
        return self.env.unwrapped.get_language_instruction()

    def get_action_space(self) -> gym.Space:
        """Return the environment action space."""
        return self.env.action_space

    def get_observation_space(self) -> gym.Space:
        """Return the environment observation space."""
        return self.env.observation_space

    def _init_reset_state_ids(self) -> None:
        """Initialize reset state IDs and the RNG generator."""
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self) -> None:
        """Update reset state IDs for fixed-reset episodes."""
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(repeats=self.group_size).to(self.device)

    def _extract_obs_image(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract image observations and task descriptions."""
        obs_image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        obs_image = obs_image.permute(0, 3, 1, 2)  # [B, C, H, W]
        extracted_obs = {"images": obs_image, "task_descriptions": self.instruction}
        return extracted_obs

    def _calc_step_reward(self, info: Dict[str, Any]) -> torch.Tensor:
        """Compute step reward and optionally return relative reward."""
        reward = torch.zeros(self.num_envs, dtype=torch.float32).to(self.device)  # [B, ]
        reward += info["is_src_obj_grasped"] * 0.1
        reward += info["consecutive_grasp"] * 0.1
        reward += (info["success"] & info["is_src_obj_grasped"]) * 1.0
        # diff
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _init_metrics(self) -> None:
        """Initialize per-episode metrics buffers."""
        self.success_once = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.fail_once = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _reset_metrics(self, env_idx: Optional[torch.Tensor] = None) -> None:
        """Reset metrics for specific environments or all."""
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(self, step_reward: torch.Tensor, infos: Dict[str, Any]) -> Dict[str, Any]:
        """Record per-step metrics into the infos dict."""
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
            episode_info["fail_once"] = self.fail_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def _process_action(
        self,
        raw_chunk_actions: np.ndarray,
        action_scale: float = 1.0,
        policy: str = "widowx_bridge",
    ) -> torch.Tensor:
        """Convert raw chunk actions into environment-ready actions."""
        reshaped_actions = raw_chunk_actions.reshape(-1, self.action_dim)
        batch_size = reshaped_actions.shape[0]
        raw_actions = {
            "world_vector": np.array(reshaped_actions[:, :3]),
            "rotation_delta": np.array(reshaped_actions[:, 3:6]),
            "open_gripper": np.array(reshaped_actions[:, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        actions = {}
        actions["world_vector"] = raw_actions["world_vector"] * action_scale  # [B, 3]
        actions["rot_axangle"] = raw_actions["rotation_delta"] * action_scale  # [B, 3]

        if policy == "google_robot":
            raise NotImplementedError
        elif policy == "widowx_bridge":
            actions["gripper"] = 2.0 * (raw_actions["open_gripper"] > 0.5) - 1.0  # [B, 1]

        actions["terminate_episode"] = np.array([0.0] * batch_size).reshape(-1, 1)  # [B, 1]

        actions = {k: torch.tensor(v, dtype=torch.float32) for k, v in actions.items()}
        actions = torch.cat([actions["world_vector"], actions["rot_axangle"], actions["gripper"]], dim=1).cuda()

        if self.num_action_chunks == 1:
            return actions
        else:
            chunk_actions = actions.reshape(-1, self.num_action_chunks, self.action_dim)
            return chunk_actions

    @profiler.timer_wrap(level="debug")
    def reset(self, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> EnvRet:
        """Reset the environment and return an EnvRet."""
        if self._is_offloaded:
            raise RuntimeError("Environment is offloaded. Call reload() first.")
        if self.use_fixed_reset_state_ids and "episode_id" not in options:
            options.update(episode_id=self.reset_state_ids)

        raw_obs, info = self.env.reset(seed=self.seed, options=options)
        extracted_obs = self._extract_obs_image(raw_obs)

        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(self.device)  # [B, ]

        return EnvRet(
            env_id=self.env_id,
            observation=extracted_obs,
            last_terminated=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            last_truncated=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )

    @profiler.timer_wrap(level="debug")
    def step(self, policy_resp: PolicyResponse, auto_reset: bool = True) -> EnvRet:
        """Step the environment with a policy response."""
        if self._is_offloaded:
            raise RuntimeError("Environment is offloaded. Call reload() first.")
        raw_actions = self._preprocess_fn(policy_resp)
        action = self._process_action(raw_actions)

        raw_obs, _reward, terminations, truncations, infos = self.env.step(action)
        extracted_obs = self._extract_obs_image(raw_obs)
        step_reward = self._calc_step_reward(infos)

        if self.video_cfg.get("save_video", False):
            self.add_new_frames(infos=infos, rewards=step_reward)

        infos = self._record_metrics(step_reward, infos)
        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    infos["episode"]["fail_at_end"] = infos["fail"].clone()

        dones = torch.logical_or(terminations, truncations)

        last_infos = {}
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)
            episode_info = {}
            if "final_info" in infos:
                final_info = infos["final_info"]

                for key in final_info["episode"]:
                    episode_info[key] = final_info["episode"][key][dones]

                last_infos = {
                    "final_observation": infos["final_observation"],
                    "episode_info": episode_info,
                }
        return EnvRet(
            env_id=self.env_id,
            observation=extracted_obs,
            last_reward=step_reward,
            last_terminated=terminations,
            last_truncated=truncations,
            info=last_infos,
        )

    def _handle_auto_reset(
        self, dones: torch.Tensor, extracted_obs: Dict[str, Any], infos: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset completed environments and return updated observations and infos."""
        final_obs = torch_clone_dict(extracted_obs)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = torch_clone_dict(infos)
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        # reset the environment
        raw_obs, infos = self.env.reset(options=options)
        extracted_obs = self._extract_obs_image(raw_obs)
        self._reset_metrics(env_idx)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    # render utils
    def capture_image(self, infos: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Render the environment and optionally overlay info."""
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]

        if infos is not None:
            for i in range(len(img)):
                info_item = {k: v if np.size(v) == 1 else v[i] for k, v in infos.items()}
                img[i] = put_info_on_image(img[i], info_item)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=int(np.sqrt(self.num_envs)))
        return img

    def render(self, info: Any, rew: Optional[Any] = None) -> np.ndarray:
        """Render a frame with optional info and rewards."""
        if self.video_cfg.get("info_on_video", False):
            scalar_info = gym_utils.extract_scalars_from_info(common.to_numpy(info), batch_size=self.num_envs)
            if rew is not None:
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [float(rew) for rew in scalar_info["reward"]]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
            image = self.capture_image(scalar_info)
        else:
            image = self.capture_image()
        return image

    def sample_action_space(self) -> Any:
        """Sample a random action from the action space."""
        return self.env.action_space.sample()

    def add_new_frames(self, infos: Dict[str, Any], rewards: Optional[torch.Tensor] = None) -> None:
        """Append a rendered frame to the video buffer."""
        image = self.render(infos, rewards)
        self.render_images.append(image)

    def add_new_frames_from_obs(self, raw_obs: Dict[str, Any]) -> None:
        """For debugging render"""
        raw_imgs = common.to_numpy(raw_obs["images"].permute(0, 2, 3, 1))
        raw_full_img = tile_images(raw_imgs, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(raw_full_img)

    def flush_video(self, video_sub_dir: Optional[str] = None) -> None:
        """Write buffered frames to disk and reset the buffer."""
        if self.video_cfg.get("save_video", False):
            output_dir = os.path.join(self.video_cfg.get("video_base_dir", "video"), f"seed_{self.seed}")
            if video_sub_dir is not None:
                output_dir = os.path.join(output_dir, f"{video_sub_dir}")
            images_to_video(
                self.render_images,
                output_dir=output_dir,
                video_name=f"{self.video_cnt}",
                fps=self.env_config.get("sim_config", {}).get("control_freq", 5),
                verbose=False,
            )
            self.video_cnt += 1
            self.render_images = []

    def offload(self):
        """
        Offload the environment to free GPU memory.
        Only deletes the environment object, other variables are preserved.
        """
        if self._is_offloaded:
            return

        # Delete environment to free GPU memory
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass

        del self.env
        self.env = None

        self.clear_memory()
        self._is_offloaded = True

    def reload(self):
        """
        Reload the environment.
        Recreates the environment object only, other variables are preserved.
        """
        if not self._is_offloaded:
            return

        # Recreate environment
        self.env: Base = gym.make(**self.env_config)

        self._is_offloaded = False

    @property
    def is_offloaded(self) -> bool:
        """Check if the environment is currently offloaded."""
        return self._is_offloaded

    def clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
