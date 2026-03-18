"""Async RSL-RL engine implementation.

Provides an async engine specialization that wires environment initialization,
policy setup, buffer initialization, and initial weight synchronization for
training.
"""

import logging
from typing import Dict

from torch.utils._pytree import tree_map

from rlightning.buffer import DataBuffer
from rlightning.env import EnvGroup
from rlightning.policy import PolicyGroup
from rlightning.utils.config import MainConfig
from rlightning.utils.logger import get_logger, log_metric
from rlightning.utils.registry import ENGINE

from . import AsyncRLEngine

logger = get_logger(__name__)


@ENGINE.register("async_rsl")
class AsyncRSLRLEngine(AsyncRLEngine):
    """Async RSL-RL engine."""

    def __init__(
        self,
        config: MainConfig,
        env_group: EnvGroup,
        policy_group: PolicyGroup,
        buffer: DataBuffer,
    ) -> None:
        super().__init__(
            config,
            env_group,
            policy_group,
            buffer,
        )

        env_meta_list = self.env_meta_list
        self.config.train.batch_size = env_meta_list[0].num_envs * len(self.policy_group.train_list)

    def warm_up(self):
        self.sync_weights()
        logger.info("Warm up done, ready to run.")

    def rollout(self, *args, **kwargs):
        batched_policy_resp = None
        with self.env_group.auto_reset():
            while self.coordinator.is_running():
                if batched_policy_resp is None:
                    batched_env_ret, truncations = self.env_group.reset(seed=0)
                else:
                    self.env_group.step_async(batched_policy_resp)
                    batched_env_ret, truncations = self.env_group.collect_async()
                batched_policy_resp = self.policy_group.rollout_batch(batched_env_ret)
                self.buffer.add_batched_data_async(batched_env_ret)
                processed_batched_policy_resp = self.policy_group.postprocess(batched_env_ret, batched_policy_resp)
                self.buffer.add_batched_data_async(processed_batched_policy_resp, truncations)

    def update_dataset(self, *args, **kwargs):
        super().update_dataset(*args, **kwargs)

    def train(self) -> None:
        training_info = self.policy_group.train()

        if isinstance(training_info, list) and len(training_info):
            assert all(isinstance(x, dict) for x in training_info)

            def mean_leaf_fn(*vals):
                vals = list(vals)
                return sum(vals) / len(vals) if vals else 0.0

            training_info: Dict[str, float] = tree_map(mean_leaf_fn, *training_info)

        log_metric(training_info, level=logging.INFO, step=self.epoch, prefix="Train")

        # and log rollout metrics here
        rollout_metrics = self.env_group.get_env_stats()
        if len(rollout_metrics):
            log_metric(
                rollout_metrics,
                level=logging.INFO,
                step=self.epoch,
                prefix="Rollout",
            )

        # log performance here
        performance_metrics = self.env_group.get_stats()
        if len(performance_metrics):
            log_metric(performance_metrics, level=logging.INFO, step=self.epoch, prefix="Performance")

    def sync_weights(self) -> None:
        super().sync_weights()
