import json
import logging
import os
from typing import Any, Dict, Optional

import ray
from rich.logging import RichHandler

from rlightning.utils.config import LogConfig
from rlightning.utils.logger.handlers import MetricsHandler, build_metrics_backend

_PROJECT_LOG_NAME = "rlightning"
_project_logger = logging.getLogger(_PROJECT_LOG_NAME)
_loggers: Dict[str, "MetricsLogger"] = {}


def setup_logger(cfg: LogConfig):
    # Always capture all levels at the logger; per-handler levels will control
    # what actually gets emitted.
    _project_logger.setLevel(logging.DEBUG)

    if not is_ray_worker():
        # Reuse Hydra's job file handlers (if any) so our logs also land in
        # `outputs/<timestamp>/<job>.log`. Hydra attaches FileHandlers to the root
        # logger before user code runs; since we set `propagate=False` below, we need
        # to explicitly add those handlers to our logger to avoid losing file logs.
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler) and handler not in _project_logger.handlers:
                # Ensure file handlers capture all messages regardless of cfg.level
                handler.setLevel(logging.DEBUG)
                _project_logger.addHandler(handler)

        # Create experiment handler and add it to the logger
        backend = build_metrics_backend(cfg)
        metrics_handler = MetricsHandler(backend)
        metrics_handler.setLevel(logging.DEBUG)
        _project_logger.addHandler(metrics_handler)
    else:
        # If we're in a Ray worker (or any process) where Hydra's file handlers
        # are not attached, ensure we still log to the same Hydra job log file.
        log_file = os.environ.get("RLIGHTNING_LOG_FILE")
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(logging.DEBUG)  # capture all levels
            file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
            _project_logger.addHandler(file_handler)
        else:
            _project_logger.warning("RLIGHTNING_LOG_FILE is not set, logs will not be saved to file")

    # add rich.logging.RichHandler to make it compatible with rich progress bar
    console_handler = RichHandler(rich_tracebacks=False)
    console_handler.setLevel(cfg.level)  # honor user-visible level
    _project_logger.addHandler(console_handler)

    _project_logger.propagate = False


class MetricsLogger:
    """A logger utility for recording metrics."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize the MetricsLogger.

        It's recommended that logging is configured once at the application's
        entry point, for instance with `logging.basicConfig()`. This logger
        utility will then use the existing logging configuration.

        Args:
            logger: An existing `logging.Logger` instance to wrap.
        """
        self.name = logger.name
        self.logger = logger
        self._metric_prefix = self._get_metric_prefix()

    def _get_metric_prefix(self) -> str:
        """Get prefix for metrics based on Ray runtime context."""
        if is_ray_worker():
            runtime_context = ray.get_runtime_context()
            actor_name = runtime_context.get_actor_name()

            if actor_name:
                return actor_name
            else:
                # Fallback for non-actor workers
                worker_id = runtime_context.get_worker_id()
                return f"worker_{worker_id[:8]}"

        else:
            # if not in Ray context, use the root logger name
            return _PROJECT_LOG_NAME

    def log_metric(
        self,
        payload: Dict[str, Any],
        level: int = logging.INFO,
        step: int = None,
        prefix: str = None,
    ) -> None:
        """
        Log a metric payload with automatic prefixing.

        The payload is formatted into a JSON string with a "[METRIC]" prefix.
        Metric names are automatically prefixed with worker/actor identifiers.

        Args:
            payload: Dictionary containing metric data.
            level: The logging level to use for the metric (default: INFO).
            step: Optional global/local step for metric consumers.
            prefix: Optional override for metric key prefix.
        """
        # Add prefix to all metric keys
        prefixed_payload = {}
        for key, value in payload.items():
            prefixed_key = f"{prefix}/{key}" if prefix else f"{self._metric_prefix}/{key}"
            prefixed_payload[prefixed_key] = value

        metric_message = f"[METRIC] {json.dumps(prefixed_payload, indent=4)}"
        self.logger.log(level, metric_message, extra={"metric_payload": prefixed_payload, "step": step})


def is_ray_worker() -> bool:
    """
    Check if the current context is a Ray worker process.
    """
    if not ray.is_initialized():
        return False
    else:
        from ray._private.worker import LOCAL_MODE, WORKER_MODE, global_worker

        return getattr(global_worker, "mode", None) in [WORKER_MODE, LOCAL_MODE]


def get_logger(name: str) -> logging.Logger:
    """
    Get a logging logger for the given name.
    """
    if _PROJECT_LOG_NAME not in name:
        name = f"{_PROJECT_LOG_NAME}.{name}"
    logger = logging.getLogger(f"{name}")
    return logger


def get_metrics_logger(name: Optional[str] = None) -> "MetricsLogger":
    """
    Get a cached MetricsLogger instance for the given name.

    This function acts as a factory for `MetricsLogger` instances. It wraps the
    standard `logging.getLogger()` to ensure that loggers are uniquely named
    and re-used.

    It's recommended to call this from your modules with `__name__`, like so:
    `logger = get_metrics_logger(__name__)`

    Args:
        name: Name for the logger. If None, an application-wide 'metrics'
              logger is returned.

    Returns:
        A `MetricsLogger` instance.
    """
    if name is None:
        name = f"{_PROJECT_LOG_NAME}.metrics"

    # Create unique logger name per worker to avoid sharing cached instances
    # between different Ray actors/workers
    if is_ray_worker():
        runtime_context = ray.get_runtime_context()
        actor_name = runtime_context.get_actor_name()
        if actor_name:
            unique_name = f"{name}.{actor_name}"
        else:
            worker_id = runtime_context.get_worker_id()
            unique_name = f"{name}.{worker_id[:8]}"
    else:
        unique_name = name

    if unique_name not in _loggers:
        logger = get_logger(unique_name)
        _loggers[unique_name] = MetricsLogger(logger)

    return _loggers[unique_name]


# Convenience function for direct use
def log_metric(payload: Dict[str, Any], level: int = logging.INFO, step: int = None, prefix: str = None) -> None:
    """Log a metric payload using the default metrics logger.

    Args:
        payload: Metrics payload to log.
        level: Logging level.
        step: Optional step index.
        prefix: Optional metric name prefix.
    """
    logger = get_metrics_logger()
    logger.log_metric(payload, level, step, prefix)
