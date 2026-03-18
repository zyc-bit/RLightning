from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from rlightning.utils.config import LogConfig


class MetricsBackend(ABC):
    """Abstract base class for a metrics backend."""

    @abstractmethod
    def write(self, metrics: Dict[str, Any]) -> None:
        """
        Write metrics to the backend.

        Args:
            metrics: A dictionary of metrics to log.
        """
        pass


class TensorBoardBackend(MetricsBackend):
    """Metrics backend for TensorBoard."""

    def __init__(self, cfg: LogConfig):
        """
        Initialize the TensorBoard backend.

        Args:
            cfg: Configuration object. Expected to have a `log_dir` attribute.
        """
        from torch.utils.tensorboard import SummaryWriter

        log_dir = (
            Path(cfg.log_dir)
            / cfg.project
            / cfg.name
            / "tensorboard"
            / datetime.now().strftime("%Y-%m-%d-%f")
        )
        self.writer = SummaryWriter(log_dir=log_dir)

    def write(self, metrics: Dict[str, Any], step: int):
        """
        Write metrics to TensorBoard.

        Only scalar values (int, float) are logged.

        Args:
            metrics: A dictionary of metrics to log.
            step: The global step used when logging scalar metrics.
        """
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step)

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


class WandBBackend(MetricsBackend):
    """Metrics backend for Weights & Biases."""

    def __init__(self, cfg: LogConfig):
        """
        Initialize the WandB backend.

        Args:
            cfg: Configuration object. Expected to have `project`, `name`,
                 and `config` attributes.
        """
        import wandb

        project = cfg.project
        name = cfg.name
        mode = cfg.mode
        log_dir = (
            Path(cfg.log_dir) / cfg.project / cfg.name / datetime.now().strftime("%Y-%m-%d-%f")
        )
        wandb.init(project=project, dir=log_dir, name=name, mode=mode)

        self.run = wandb

    def write(
        self,
        metrics: Dict[str, Any],
        step: int = None,
    ):
        """
        Write metrics to WandB.

        Args:
            metrics: A dictionary of metrics to log.
            step: The step number to log the metrics at.
        """
        self.run.log(metrics, step=step)

    def close(self):
        """Finish the WandB run."""
        self.run.finish()


class SwanLabBackend(MetricsBackend):
    """Metrics backend for SwanLab."""

    def __init__(self, cfg: LogConfig):
        """
        Initialize the SwanLab backend.

        Args:
            cfg: Configuration object. Expected to have `project`, `name`,
                 and `config` attributes.
        """
        import swanlab

        log_dir = (
            Path(cfg.log_dir)
            / cfg.project
            / cfg.name
            / "swanlab"
            / datetime.now().strftime("%Y-%m-%d-%f")
        )
        project_name = cfg.project
        experiment_name = cfg.name
        swanlab.init(
            project=project_name, experiment_name=experiment_name, logdir=log_dir, mode=cfg.mode
        )

        self.run = swanlab

    def write(self, metrics: Dict[str, Any], step: int = None):
        """
        Write metrics to SwanLab.

        Args:
            metrics: A dictionary of metrics to log.
            step: The step number to log the metrics at.
        """
        self.run.log(metrics, step=step)

    def close(self):
        """Finish the SwanLab run."""
        self.run.finish()
