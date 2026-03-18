"""Evaluation engine module for policy evaluation.

This module implements the EvaluationEngine for running policy evaluation
without training, useful for testing trained policies.
"""

from typing import Any

import torch

from rlightning.types import PolicyResponse
from rlightning.utils.logger import get_logger
from rlightning.utils.progress import get_progress
from rlightning.utils.registry import ENGINE
from rlightning.utils.utils import InternalFlag

from .base_engine import BaseEngine

logger = get_logger(__name__)


@ENGINE.register("eval")
class EvaluationEngine(BaseEngine):
    """Evaluation engine for policy testing.

    This engine runs policy evaluation without training, useful for
    testing trained policies on various environments.

    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the evaluation engine."""
        super().__init__(*args, **kwargs)
        env_meta = self.env_group.init()
        self._env_meta = env_meta
        # init eval policy
        self.policy_group.init_eval(env_meta=env_meta)

        if InternalFlag.DEBUG:
            self.warm_up()

    def train(self) -> None:
        """No-op training method for evaluation engine."""
        pass

    def sync_weights(self) -> None:
        """No-op weight update method for evaluation engine."""
        pass

    def warm_up(self) -> None:
        """Initialize environments and evaluation policy."""
        logger.info("Evaluation engine warmed up, ready to run.")

    @torch.inference_mode()
    def run(self) -> None:
        """Run evaluation for a configured number of episodes."""

        logger.info(f"Running evaluation for {self.config.train.max_epochs} episodes.")

        for _ in range(5):
            self.rollout()

    def rollout(self) -> None:
        """Rollout one complete episode and collect metrics."""
        batched_env_ret, _ = self.env_group.reset()
        if InternalFlag.VERBOSE:
            progress = get_progress()
            task = progress.add_task("Evaluation", total=self.config.train.max_epochs)

        for _ in range(self.config.train.max_rollout_steps):
            batched_policy_resp = self.policy_group.rollout_batch(batched_env_ret)
            for policy_resp in batched_policy_resp.values():
                self.get_metrics(policy_resp)

            batched_env_ret, _ = self.env_group.step(batched_policy_resp)
            if InternalFlag.VERBOSE:
                progress.update(task, advance=1)

    def get_metrics(self, policy_resp: PolicyResponse) -> None:
        """Extract and process metrics from policy response.

        Args:
            policy_resp: Policy response containing evaluation info.
        """
        pass
