"""Base engine module for reinforcement learning training loops.

This module defines the abstract base class for all RL engines,
providing the common interface for training, rollout, and weight updates.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from rlightning.buffer import DataBuffer
from rlightning.env.env_group import EnvGroup
from rlightning.policy.policy_group import PolicyGroup
from rlightning.utils.config import MainConfig
from rlightning.utils.logger import get_logger
from rlightning.utils.profiler import profiler
from rlightning.utils.progress import get_progress
from rlightning.utils.utils import InternalFlag

logger = get_logger(__name__)


class BaseEngine(ABC):
    """Abstract base class for reinforcement learning engines.

    This class defines the common interface for all RL engines, including
    methods for warm-up, training, rollout, weight updates, and timing.

    """

    def __init__(
        self,
        config: MainConfig,
        env_group: Optional[EnvGroup] = None,
        policy_group: Optional[PolicyGroup] = None,
        buffer: Optional[DataBuffer] = None,
    ) -> None:
        """Initialize the base engine.

        Args:
            config: Main configuration object.
            env_group: Environment group for managing multiple environments.
            policy_group: Policy group for managing train and eval policies.
            buffer: Data buffer for storing and sampling experience.
        """
        super().__init__()

        self.config = config
        self.env_group = env_group
        self.policy_group = policy_group
        self.buffer: DataBuffer = buffer

        self.epoch = 0  # current epoch
        self.timing_raw: dict[str, dict[str, Any]] = {}

    @abstractmethod
    def warm_up(self) -> None:
        """Initialize and perform dummy run for constructing RL dataflow.

        This method should initialize all components (environment, policy,
        buffer) and perform a dummy training iteration to ensure the
        dataflow is properly constructed.
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """Run the RL training flow.

        This method should execute the complete RL training flow,
        coordinating rollout, training, and weight updates.
        """
        pass

    # ================== Rollout Functions ===================
    @abstractmethod
    def rollout(self, *args: Any, **kwargs: Any) -> None:
        """User defined rollout to collect experience.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.
        """
        pass

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> None:
        """User defined evaluation to collect experience.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.
        """
        pass

    def _pre_rollout_hook(self) -> None:
        """Hook function called before each rollout iteration."""
        if self.config.cluster.enable_offload:
            self.policy_group.reload_eval_model()
            self.env_group.reload()

    def _post_rollout_hook(self) -> None:
        """Hook function called after each rollout iteration."""
        if self.config.cluster.enable_offload:
            self.policy_group.offload_eval_model()
            self.env_group.offload()

    @torch.inference_mode()
    @profiler.timer_wrap(level="info")
    def _rollout(self, *args, **kwargs) -> None:
        """Internal rollout function called in each rollout iteration."""
        self._pre_rollout_hook()
        with profiler.timer("rollout", self.timing_raw, level="info", enable=InternalFlag.DEBUG):
            self.rollout(*args, **kwargs)
        self._post_rollout_hook()

    def _pre_evaluate_hook(self) -> None:
        """Hook function called before each evaluate iteration."""
        if self.config.cluster.enable_offload:
            self.policy_group.reload_eval_model()
            self.env_group.reload()

    def _post_evaluate_hook(self) -> None:
        """Hook function called after each evaluate iteration."""
        if self.config.cluster.enable_offload:
            self.policy_group.offload_eval_model()
            self.env_group.offload()

    @torch.inference_mode()
    @profiler.timer_wrap(level="info")
    def _evaluate(self, *args, **kwargs) -> None:
        """Internal evaluate function called in each evaluate iteration."""
        self._pre_evaluate_hook()
        with profiler.timer("evaluate", self.timing_raw, level="info", enable=InternalFlag.DEBUG):
            self.evaluate(*args, **kwargs)
        self._post_evaluate_hook()

    # ================== Dataset Update Functions ===================
    @abstractmethod
    def update_dataset(self) -> None:
        """User defined dataset update from buffer to train policy."""
        pass

    def _pre_update_dataset_hook(self) -> None:
        """A hook function called before each dataset update iteration."""
        pass

    def _post_update_dataset_hook(self) -> None:
        """A hook function called after each dataset update iteration."""
        pass

    def _update_dataset(self) -> None:
        """Internal dataset update function called in each dataset update iteration."""
        self._pre_update_dataset_hook()
        with profiler.timer("update_dataset", self.timing_raw, level="info", enable=InternalFlag.DEBUG):
            self.update_dataset()
        self._post_update_dataset_hook()

    # ================== Training Functions ===================
    @abstractmethod
    def train(self) -> None:
        """User defined training on collected experience.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.
        """
        pass

    def _pre_train_hook(self) -> None:
        """A hook function called before each training iteration."""
        if self.config.cluster.enable_offload:
            self.policy_group.reload_model_param_and_grad(load_grad=True, load_optimizer=True)

    def _post_train_hook(self) -> None:
        """A hook function called after each training iteration."""
        pass

    @profiler.timer_wrap(level="info")
    def _train(self) -> None:
        """Internal training function called in each training iteration."""
        self._pre_train_hook()
        with profiler.timer("train", self.timing_raw, level="info", enable=InternalFlag.DEBUG):
            self.train()
        self._post_train_hook()

    # ================== Weight Update Functions ===================
    def sync_weights(self) -> None:
        """sync weights from train policy to eval policy.

        Args:
            policy_group: Policy group containing train and eval policies.
        """
        self.policy_group.sync_weights()

    def _pre_sync_weights_hook(self) -> None:
        """A hook function called before each weights sync iteration."""
        if self.config.cluster.enable_offload:
            self.policy_group.offload_model_optimizer()

    def _post_sync_weights_hook(self) -> None:
        """A hook function called after each weights sync iteration."""
        if self.config.cluster.enable_offload:
            self.policy_group.offload_model_param_and_grad(offload_grad=True, offload_optimizer=False)

    @profiler.timer_wrap(level="info")
    def _sync_weights(self) -> None:
        """Internal weights sync function called in each weights sync iteration."""
        self._pre_sync_weights_hook()
        with profiler.timer("sync_weights", self.timing_raw, level="info", enable=InternalFlag.DEBUG):
            self.sync_weights()
        self._post_sync_weights_hook()

    def print_timing_summary(self, reset: bool = False) -> None:
        """Print timing summary for profiling.

        Args:
            reset: If True, reset timing statistics after printing.
        """
        if self.timing_raw:
            logger.debug("Timing summary:")
            logger.debug(f"{self.__class__.__name__}:")
            # iterate over a snapshot to avoid concurrent modification during iteration
            for name, stats in dict(self.timing_raw).items():
                logger.debug(
                    f"\t{name:15} count={stats['count']:<3} total={stats['total']:.6f}s avg={stats['avg']:.6f}s"
                )
        if reset:
            self.timing_raw = {}
        self.env_group.print_timing_summary(reset)
        self.policy_group.print_timing_summary(reset)
        self.buffer.print_timing_summary(reset)

    def iter_epochs(self, num_epochs: int) -> iter:
        """
        An iterator that yields epoch numbers up to num_epochs.

        Args:
            num_epochs (int): The number of epochs to iterate through.

        Returns:
            iter: An iterator over epoch numbers.
        """
        iterator = range(num_epochs)

        if InternalFlag.VERBOSE:
            progress = get_progress()
            task = progress.add_task("[red]Training", total=self.config.train.max_epochs)

        try:
            for epoch in iterator:
                yield epoch

                if InternalFlag.VERBOSE:
                    progress.update(task, advance=1)

                if InternalFlag.DEBUG:
                    self.print_timing_summary()

        finally:
            if InternalFlag.VERBOSE:
                progress.update(task, description="[bold green]Training completed")
