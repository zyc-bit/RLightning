import logging
from queue import Queue
from threading import Thread

from rlightning.utils.config import LogConfig

from .backends import MetricsBackend, SwanLabBackend, TensorBoardBackend, WandBBackend

logger = logging.getLogger(__name__)


class MetricsHandler(logging.Handler):
    """A logging handler that routes metrics to configured backends."""

    def __init__(self, backend: MetricsBackend):
        """
        Initialize the handler with a list of metrics backends.

        A background worker thread is started to process metrics from a queue.

        Args:
            backend: The `MetricsBackend` instance used to consume metrics.
        """
        super().__init__()
        self._backend = backend
        self._queue: Queue = Queue()
        self._worker = Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def emit(self, record: logging.LogRecord):
        """
        Place metric payloads onto the queue for background processing.

        This method is called by the logging system for each log record. It
        checks for a `metric_payload` attribute on the record and, if present,
        adds it to the processing queue.

        Args:
            record: The log record to process.
        """
        # Check if the log record is a metric
        if hasattr(record, "metric_payload"):
            self._queue.put(record)

    def _worker_loop(self):
        """
        The background worker loop that processes metrics.

        This loop runs indefinitely, taking metric payloads from the queue and
        dispatching them to all configured backends.
        """
        while True:
            record = self._queue.get()
            if record is None:
                break  # Exit signal
            self._backend.write(record.metric_payload, step=record.step)

    def close(self):
        try:
            self._queue.put(None)  # Send exit signal to worker
            self._worker.join(timeout=2.0)  # Wait for the worker to finish
        finally:
            try:
                self._backend.close()
            except Exception as e:
                logger.exception(f"Error closing metrics backend.")
            super().close()


def build_metrics_backend(cfg: LogConfig) -> MetricsBackend:
    """
    Build a list of metrics backends based on the provided configuration.

    Args:
        cfg: The application's configuration object.

    Returns:
        A list of initialized `MetricsBackend` instances.
    """
    if cfg.backend == "tensorboard":
        return TensorBoardBackend(cfg)
    elif cfg.backend == "wandb":
        return WandBBackend(cfg)
    elif cfg.backend == "swanlab":
        return SwanLabBackend(cfg)
    else:
        raise ValueError(f"Unsupported logging backend: {cfg.backend}")


# def setup_metrics_routing(cfg: LogConfig):
#     """
#     Set up the logging system to route metrics to configured backends.

#     This function should be called once at the application's entry point. It
#     configures the root logger and adds a `MetricsHandler` to it if any
#     backends are specified in the configuration.

#     Args:
#         cfg: The application's configuration object.
#     """
#     # set the framework root logger
#     logger = logging.getLogger(ROOT_LOGGER_NAME)
#     logger.setLevel(cfg.level)

#     # Create the handler and add it to the logger
#     backend = build_metrics_backend(cfg)
#     handler = MetricsHandler(backend)
#     logger.addHandler(handler)
