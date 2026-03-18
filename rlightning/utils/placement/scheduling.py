"""
Component Scheduling Configuration.

This module defines scheduling requirements for each component type and provides
validation logic to ensure configurations are consistent and feasible.
"""

import dataclasses
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

from rlightning.utils.config.config import MainConfig
from rlightning.utils.logger import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class Scheduling:
    """Scheduling configuration for a single worker type.

    Attributes:
        worker_num: Number of workers of this type.
        num_cpus: Number of CPUs per worker.
        num_gpus: Number of GPUs per worker.
        node_list: Optional comma-separated list of node IDs for placement.
    """

    worker_num: int
    num_cpus: int
    num_gpus: float
    # Optional: specify exact nodes for this worker type
    node_list: Optional[str] = None

    def total_gpus(self) -> float:
        """Total GPU requirements for all workers of this type."""
        if self.worker_num == "auto":
            return 0.0
        return self.worker_num * self.num_gpus

    def total_cpus(self) -> int:
        """Total CPU requirements for all workers of this type."""
        if self.worker_num == "auto":
            return 0
        return self.worker_num * self.num_cpus

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "worker_num": self.worker_num,
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "node_list": self.node_list,
        }


@dataclasses.dataclass
class ComponentScheduling:
    """
    Scheduling configuration for all components.

    This class holds the resource requirements for each component type:
    - env_worker: Environment workers (can have multiple groups)
    - train_worker: Training policy workers
    - eval_worker: Evaluation policy workers (rollout)
    - buffer_worker: Data buffer storage workers
    """

    env_worker: Optional[List[Scheduling]] = None
    train_worker: Optional[Scheduling] = None
    eval_worker: Optional[Scheduling] = None
    buffer_worker: Optional[Scheduling] = None

    def infer_auto_buffer_worker_num(self, cluster_info: Dict[str, Any]) -> None:
        """Infer the auto buffer worker number based on the train worker number."""
        if self.buffer_worker is not None and self.buffer_worker.worker_num == "auto":
            assert self.train_worker is not None, "Train worker must be configured when buffer worker is auto"
            num_train_gpus = self.train_worker.total_gpus()
            node_info = cluster_info["node_id_to_resources"]
            num_gpus_per_node = next(iter(node_info.values()))["GPU"]
            self.buffer_worker.worker_num = math.ceil(num_train_gpus / num_gpus_per_node)

    def adjust_buffer_worker_num(self, train_node_count: int) -> None:
        """Adjust the buffer worker number to match the train worker number."""
        assert self.buffer_worker is not None, "Buffer worker must be configured"
        if self.buffer_worker.worker_num != "auto" and self.buffer_worker.worker_num != train_node_count:
            logger.warning(f"Buffer worker number is set to {self.buffer_worker.worker_num}, " f"which will be ignored")
        self.buffer_worker.worker_num = train_node_count

    def train_pool_requirements(self) -> Tuple[float, int]:
        """
        Calculate resource requirements for train pool (train + buffer).

        Returns:
            Tuple of (total_gpus, total_cpus).
        """
        gpus: float = 0
        cpus: int = 0
        if self.train_worker:
            gpus += self.train_worker.total_gpus()
            cpus += self.train_worker.total_cpus()
        if self.buffer_worker:
            gpus += self.buffer_worker.total_gpus()
            cpus += self.buffer_worker.total_cpus()
        return gpus, cpus

    def rollout_pool_requirements(self) -> Tuple[float, int]:
        """
        Calculate resource requirements for rollout pool (eval + env).

        Returns:
            Tuple of (total_gpus, total_cpus).
        """
        gpus: float = 0
        cpus: int = 0
        if self.eval_worker:
            gpus += self.eval_worker.total_gpus()
            cpus += self.eval_worker.total_cpus()
        if self.env_worker:
            for env_sched in self.env_worker:
                # (warning): different env_worker occupies independent computing resources
                # gpus += math.ceil(env_sched.total_gpus())
                gpus += env_sched.total_gpus()
                cpus += env_sched.total_cpus()
        return gpus, cpus

    def get_component_requirements(self, component_type: str) -> Tuple[float, int]:
        """Get resource requirements for all components.

        Returns:
            Tuple of (total_gpus, total_cpus) for the specified component type.

        Raises:
            ValueError: If component type is invalid or the worker is not configured.
        """
        if component_type == "env":
            if not self.env_worker:
                return 0, 0
            gpus, cpus = 0.0, 0
            for env_sched in self.env_worker:
                # (warning): different env_worker occupies independent computing resources
                # gpus += math.ceil(env_sched.total_gpus())
                gpus += env_sched.total_gpus()
                cpus += env_sched.total_cpus()
            return gpus, cpus
        elif component_type == "train":
            if not self.train_worker:
                return 0, 0
            return self.train_worker.total_gpus(), self.train_worker.total_cpus()
        elif component_type == "eval":
            if not self.eval_worker:
                return 0, 0
            return self.eval_worker.total_gpus(), self.eval_worker.total_cpus()
        elif component_type == "buffer":
            if not self.buffer_worker:
                return 0, 0
            return self.buffer_worker.total_gpus(), self.buffer_worker.total_cpus()
        else:
            raise ValueError(f"Invalid component type: {component_type}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "env": [e.to_dict() for e in self.env_worker] if self.env_worker else None,
            "train": self.train_worker.to_dict() if self.train_worker else None,
            "eval": self.eval_worker.to_dict() if self.eval_worker else None,
            "buffer": self.buffer_worker.to_dict() if self.buffer_worker else None,
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = ["Component Scheduling Summary:"]

        if self.train_worker:
            lines.append(
                f"  Train: {self.train_worker.worker_num} workers, " f"{self.train_worker.num_gpus} GPU/worker"
            )

        if self.eval_worker:
            lines.append(f"  Eval: {self.eval_worker.worker_num} workers, " f"{self.eval_worker.num_gpus} GPU/worker")

        if self.buffer_worker:
            lines.append(
                f"  Buffer: {self.buffer_worker.worker_num} workers, " f"{self.buffer_worker.num_gpus} GPU/worker"
            )

        if self.env_worker:
            total_envs = sum(e.worker_num for e in self.env_worker)
            lines.append(f"  Env: {total_envs} total workers across {len(self.env_worker)} groups")

        train_gpus, train_cpus = self.train_pool_requirements()
        rollout_gpus, rollout_cpus = self.rollout_pool_requirements()
        lines.append(f"  Train Pool: {train_gpus} GPUs, {train_cpus} CPUs")
        lines.append(f"  Rollout Pool: {rollout_gpus} GPUs, {rollout_cpus} CPUs")

        return "\n".join(lines)


def setup_component_scheduling(cfg: "MainConfig") -> ComponentScheduling:
    """
    Set up the scheduling configuration for all components from MainConfig.

    This function:
    1. Reads cluster configuration
    2. Validates buffer storage configuration
    3. Creates Scheduling objects for each component type

    Args:
        cfg: The main configuration object.

    Returns:
        ComponentScheduling with all component requirements.

    Raises:
        ValueError: If configuration is invalid.
    """
    cluster_cfg = cfg.cluster
    if cluster_cfg is None:
        raise ValueError("Cluster configuration is required for scheduling setup.")

    # Validate buffer storage configuration
    if cfg.buffer.storage.type == "unified":
        buffer_worker_num = cluster_cfg.buffer_worker_num
        if buffer_worker_num != "auto" and buffer_worker_num != 1:
            warnings.warn(
                f"Unified storage type requires exactly 1 buffer worker, " f"but got {buffer_worker_num}. Setting to 1."
            )
        cluster_cfg.buffer_worker_num = 1
    elif cfg.buffer.storage.type == "sharded":
        if hasattr(cfg.buffer.storage, "num_shards"):
            if cfg.buffer.storage.num_shards != cluster_cfg.buffer_worker_num:
                cluster_cfg.buffer_worker_num = cfg.buffer.storage.num_shards
                warnings.warn(f"Buffer worker count adjusted to match shard count: " f"{cfg.buffer.storage.num_shards}")
    else:
        raise ValueError(f"Invalid storage type: {cfg.buffer.storage.type}")

    # Determine buffer GPU requirements
    buffer_num_gpus = 1 if cfg.buffer.storage.device == "cuda" else 0

    # Build ComponentScheduling
    scheduling = ComponentScheduling(
        env_worker=[
            Scheduling(
                worker_num=env_cfg.num_workers,
                num_cpus=env_cfg.num_cpus,
                num_gpus=env_cfg.num_gpus,
            )
            for env_cfg in cfg.env
        ],
        train_worker=Scheduling(
            worker_num=cluster_cfg.train_worker_num,
            num_cpus=1,
            num_gpus=cfg.policy.train_num_gpus,
        ),
        eval_worker=Scheduling(
            worker_num=cluster_cfg.eval_worker_num,
            num_cpus=1,
            num_gpus=cfg.policy.eval_num_gpus,
        ),
        buffer_worker=Scheduling(
            worker_num=cluster_cfg.buffer_worker_num,
            num_cpus=1,
            num_gpus=buffer_num_gpus,
        ),
    )

    return scheduling
