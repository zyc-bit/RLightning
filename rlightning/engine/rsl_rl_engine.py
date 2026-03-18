"""
RSL-RL engine module for training with RSL-RL policies.

This module implements the RSLRLEngine which inherits from SyncRLEngine
and provides compatibility with RSL-RL (Robot Learning) style policies.

"""

import logging
from collections import defaultdict
from typing import Dict

from torch.utils._pytree import tree_map

from rlightning.types import BatchedData
from rlightning.utils.logger import get_logger, log_metric
from rlightning.utils.profiler import profiler
from rlightning.utils.registry import ENGINE
from rlightning.utils.utils import InternalFlag

from .sync_rl_engine import SyncRLEngine

logger = get_logger(__name__)


@ENGINE.register("rsl")
class RSLRLEngine(SyncRLEngine):
    """RSL-RL training engine.

    This engine extends SyncRLEngine to support RSL-RL style policies,
    with modified rollout and training loops for on-policy learning.

    """

    def __init__(
        self,
        config,
        env_group=None,
        policy_group=None,
        buffer=None,
    ):
        """Initialize the RSL-RL engine."""
        super().__init__(
            config=config,
            env_group=env_group,
            policy_group=policy_group,
            buffer=buffer,
        )
        env_meta_list = self.env_meta_list
        self.num_envs = env_meta_list[0].num_envs

        # flags for flagging env reset
        self._is_env_reset = False
        self.last_batched_env_ret = None

    def warm_up(self):
        """Initialize runtime state without running training."""
        self.sync_weights()
        if InternalFlag.DEBUG:
            self.print_timing_summary(reset=True)
        logger.info("Warm up done, ready to run.")

    @profiler.timer_wrap("async_rollout", level="info", log_to_metric=True)
    def rollout(self, obj_set: str, prefix: str = "", is_eval: bool = False) -> None:
        """Perform rollout to collect experience from environments.

        Collects experience by stepping through environments, applying
        policy post-processing before storing transitions in the buffer.

        Args:
            obj_set: Object set identifier for environment reset options.
            prefix: Prefix for logging metrics.
            is_eval: If True, skips buffer storage (evaluation only).
        """

        if not self._is_env_reset:
            batched_env_ret, _ = self.env_group.reset(options={"obj_set": obj_set})
            self._is_env_reset = True
        else:
            batched_env_ret = self.last_batched_env_ret

        step_counter = defaultdict(int)

        last_env_rets_dict = {}

        while len(last_env_rets_dict.keys()) < len(self.env_group.env_ids):
            if len(batched_env_ret) > 0:
                self.buffer.add_batched_data_async(batched_env_ret)
                batched_policy_resp = self.policy_group.rollout_batch(batched_env_ret)
                self.env_group.step_async(batched_policy_resp)
                processed_batched_policy_resp = self.policy_group.postprocess(batched_env_ret, batched_policy_resp)
                self.buffer.add_batched_data_async(processed_batched_policy_resp)

            batched_env_ret, _ = self.env_group.collect_async()

            # step count
            inactive_env_id = []
            for env_id in batched_env_ret.ids():
                step_counter[env_id] += 1
                if step_counter[env_id] == self.config.train.max_rollout_steps:
                    last_env_rets_dict[env_id] = batched_env_ret[env_id]
                    inactive_env_id.append(env_id)

            if inactive_env_id:
                # process last rollout
                batched_env_ret = BatchedData.from_dict(
                    {
                        env_id: batched_env_ret[env_id]
                        for env_id in batched_env_ret.ids()
                        if env_id not in inactive_env_id
                    }
                )

        assert set(last_env_rets_dict.keys()) == set(self.env_group.env_ids)
        batched_env_ret = BatchedData.from_dict(last_env_rets_dict)
        self.buffer.add_batched_data_async(batched_env_ret)

        batched_policy_resp = self.policy_group.rollout_batch(batched_env_ret)
        processed_batched_policy_resp = self.policy_group.postprocess(batched_env_ret, batched_policy_resp)
        self.buffer.add_batched_data_async(processed_batched_policy_resp)
        self.buffer.truncate_episodes(batched_env_ret.ids())

        env_stats = self.env_group.get_env_stats(reset=True)
        log_metric(env_stats, level=logging.INFO, step=self.epoch, prefix="Rollout")

        performance_metrics = self.env_group.get_stats()
        if len(performance_metrics):
            log_metric(
                performance_metrics,
                level=logging.INFO,
                step=self.epoch,
                prefix="Performance",
            )

        self.last_batched_env_ret = batched_env_ret

    def update_dataset(self) -> None:
        """Update the dataset in the policy group from the buffer."""
        # prepare dataset
        if len(self.buffer) < 1:
            raise ValueError(
                "Not enough data in buffer to sample a batch of size "
                f"{self.config.train.batch_size}. Current buffer size: {len(self.buffer)}."
                "Please increase the max rollout steps, or decrease the batch size."
            )

        with profiler.timer("update_dataset", self.timing_raw, level="info", enable=InternalFlag.DEBUG):
            # if batch size is used to as -1, which means on-policy training
            if self.config.buffer.sampler.type == "all":
                batch_size = len(self.buffer)
            else:
                #   if on-policy, use batch_size as the num of environments
                batch_size = self.num_envs * len(self.policy_group.train_list)
            data = self.buffer.sample(batch_size=batch_size)
            self.policy_group.update_dataset(data)

    @profiler.timer_wrap("training", level=logging.INFO, log_to_metric=True)
    def train(self) -> None:
        """Perform training on collected experience.

        Samples data from the buffer, updates the dataset, and trains
        the policy. Designed for on-policy training with batch_size=-1.

        Raises:
            ValueError: If buffer is empty when training.
        """

        with profiler.timer("policy_train", self.timing_raw, level="info", enable=InternalFlag.DEBUG):
            training_info = self.policy_group.train()

        # Ensure the returned training_info is a dict. If it's a list of dicts, aggregate by mean.
        if isinstance(training_info, list) and len(training_info):
            assert all(isinstance(x, dict) for x in training_info)

            def mean_leaf_fn(*vals):
                vals = list(vals)
                return sum(vals) / len(vals) if vals else 0.0

            training_info: Dict[str, float] = tree_map(mean_leaf_fn, *training_info)

        log_metric(training_info, level=logging.INFO, step=self.epoch, prefix="Train")
