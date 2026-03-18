from typing import Sequence

import os
import math
import torch
import numpy as np

from isaaclab.utils.math import (
    quat_error_magnitude,
    sample_uniform,
)

from rlightning.humanoid import formatter
from rlightning.humanoid.utils.kinematics_model import torch_utils
from rlightning.utils.logger import get_logger

logger = get_logger(__name__)


class Motion:

    def __init__(
        self,
        env,
        motion_file: str,
        body_indexes: Sequence[int],
        motion_anchor_body_index: Sequence[int],
        adaptive_lambda: float,
        adaptive_kernel_size: int,
        adaptive_uniform_ratio: float,
        device: str = "cpu",
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        motion_data = np.load(motion_file)

        self.fps = motion_data["fps"]

        self._joint_pos = torch.tensor(
            motion_data["joint_pos"], dtype=torch.float32, device=device
        )
        logger.debug(f"joint_pos loaded as shape={self._joint_pos.shape}")

        self._joint_vel = torch.tensor(
            motion_data["joint_vel"], dtype=torch.float32, device=device
        )
        logger.debug(f"joint_vel loaded as shape={self._joint_vel.shape}")

        self._body_pos_w = torch.tensor(
            motion_data["body_pos_w"], dtype=torch.float32, device=device
        )
        logger.debug(f"body_pos_w loaded as shape={self._body_pos_w.shape}")

        self._body_quat_w = torch.tensor(
            motion_data["body_quat_w"], dtype=torch.float32, device=device
        )
        logger.debug(f"body_quat_w loaded as shape={self._body_quat_w.shape}")

        self._body_lin_vel_w = torch.tensor(
            motion_data["body_lin_vel_w"], dtype=torch.float32, device=device
        )
        logger.debug(f"body_lin_vel_w loaded as shape={self._body_lin_vel_w.shape}")

        self._body_ang_vel_w = torch.tensor(
            motion_data["body_ang_vel_w"], dtype=torch.float32, device=device
        )
        logger.debug(f"body_ang_vel_w loaded as shape={self._body_ang_vel_w.shape}")

        self._body_indexes = body_indexes
        self._motion_anchor_body_index = motion_anchor_body_index

        self._env = env

        self.time_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        self.time_step_total = self._joint_pos.shape[0]
        self.num_envs = env.num_envs

        runtime_fps = 1 / (env.cfg.decimation * env.cfg.sim.dt)
        self.bin_count = int(self.time_step_total // runtime_fps)
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=device)
        self.kernel = torch.tensor(
            [adaptive_lambda**i for i in range(adaptive_kernel_size)], device=device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.device = device
        self.adaptive_uniform_ratio = adaptive_uniform_ratio
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_kernel_size = adaptive_kernel_size
        self.metrics = dict(
            error_anchor_pos=torch.zeros(self.num_envs),
            error_anchor_rot=torch.zeros(self.num_envs),
            error_anchor_lin_vel=torch.zeros(self.num_envs),
            error_anchor_ang_vel=torch.zeros(self.num_envs),
            error_body_pos=torch.zeros(self.num_envs),
            error_body_rot=torch.zeros(self.num_envs),
            error_joint_pos=torch.zeros(self.num_envs),
            error_joint_vel=torch.zeros(self.num_envs),
            sampling_entropy=torch.zeros(self.num_envs),
            sampling_top1_prob=torch.zeros(self.num_envs),
            sampling_top1_bin=torch.zeros(self.num_envs),
        )

    def step(self):
        """Step time_steps with 1"""

        self.time_steps += 1

    def update_metrics(self, command):
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - command.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, command.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - command.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - command.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            command.body_pos_relative_w - command.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            command.body_quat_relative_w, command.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - command.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - command.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - command.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - command.robot_joint_vel, dim=-1
        )

    def update_failed_bins(self):
        """Update failed bins for sampling strategy update."""

        failed_env_ids = self._env.termination_manager.terminated
        if not torch.any(failed_env_ids):
            return
        extend_time_steps = self.time_steps * self.bin_count
        current_bin_index = torch.clamp(
            extend_time_steps // max(self.time_step_total, 1), 0, self.bin_count - 1
        )
        fail_bins = current_bin_index[failed_env_ids]
        self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    def reset_timesteps(self, env_ids: Sequence[int]):
        """Set timesteps with given started bins.

        Args:
            env_ids (Sequence[int]): A list of environment ids.
        """

        started_bins = self.sample_bins(len(env_ids))

        self.time_steps[env_ids] = (
            (started_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.time_step_total - 1)
        ).long()

    def sample_bins(self, num_samples: int) -> torch.Tensor:
        """Sample a batch of bins.

        Args:
            num_samples (int): A tensor of bin indexes

        Returns:
            torch.Tensor: A tensor of bin indexes.
        """

        sampling_probabilities: torch.Tensor = (
            self.bin_failed_count + self.adaptive_uniform_ratio / float(self.bin_count)
        )
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities, self.kernel.view(1, 1, -1)
        ).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        # metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

        return torch.multinomial(sampling_probabilities, num_samples, replacement=True)

    @property
    def joint_pos(self) -> torch.Tensor:
        """Joint pos at current timesteps"""

        return self._joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """Joint velocity at current timesteps"""

        return self._joint_vel[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self._motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self._motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self._motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self._motion_anchor_body_index]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Body pos under world frame at current timesteps"""

        tmp = self._body_pos_w[self.time_steps]
        return tmp[:, self._body_indexes] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Body quatation under world frame at current timesteps"""

        tmp = self._body_quat_w[self.time_steps]
        return tmp[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Body linear velocity under world frame at current timesteps"""

        tmp = self._body_lin_vel_w[self.time_steps]
        return tmp[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Body angular velocity under world frame at current timesteps"""

        tmp = self._body_ang_vel_w[self.time_steps]
        return tmp[:, self._body_indexes]
