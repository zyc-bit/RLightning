import logging
from contextlib import contextmanager
from functools import lru_cache, wraps
from typing import Any, TypeAlias, TypedDict, cast

import torch
from codetiming import Timer

from rlightning.utils.logger import get_logger, log_metric
from rlightning.utils.utils import InternalFlag

logger = get_logger(__name__)


class _TimingStat(TypedDict):
    count: int
    total: float
    avg: float


TimingRaw: TypeAlias = dict[str, _TimingStat]


def _new_timing_stat() -> _TimingStat:
    return {"count": 0, "total": 0.0, "avg": 0.0}


@contextmanager
def timer(
    name: str,
    timing_raw: TimingRaw,
    level: str = "info",
    enable: bool = True,
):
    """
    Context manager for timing code blocks and recording timing statistics.

    Args:
        name (str): Name identifier for the timing measurement.
        timing_raw (TimingRaw): Dictionary to store timing statistics with
            keys as names and values as dicts containing count, total, and avg.
        level (str): Logging level for the metric, either "info" or "debug".
            Defaults to "info".
        enable (bool): When False, the context manager does nothing. Defaults to True.
    """
    if not enable:
        yield
        return

    with Timer(name=name, logger=None) as _timer:
        yield

    # Store timing data
    entry = timing_raw.get(name, _new_timing_stat())
    entry["count"] += 1
    entry["total"] += _timer.last
    entry["avg"] = entry["total"] / entry["count"]
    timing_raw[name] = entry
    # log the time profile
    log_level = logging.INFO if level == "info" else logging.DEBUG
    logger.log(log_level, f"time_profile/{name}: {_timer.last:.4f}s")


def timer_wrap(name: str | None = None, level: str = "info", log_to_metric: bool = False, enable: bool = True):
    """
    Decorator for class methods to automatically time execution and record statistics.

    Args:
        name (str | None): Name identifier for the timing measurement. If None,
            uses the function name. Defaults to None.
        level (str): Logging level for the metric, either "info" or "debug".
            Defaults to "info".
        log_to_metric (bool): When True, logging as metrics, otherwise logging to console.
        enable (bool): When False, the context manager does nothing. Defaults to True.


    Returns:
        function: The decorator function that wraps the original method.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not (InternalFlag.DEBUG or enable):
                return func(self, *args, **kwargs)
            # Initialize timing_raw if it doesn't exist
            if not hasattr(self, "timing_raw"):
                self.timing_raw = {}
            timing_raw = cast(TimingRaw, self.timing_raw)

            func_name = name or func.__name__

            with Timer(name=func_name, logger=None) as _timer:
                result = func(self, *args, **kwargs)

            # Store timing data
            entry = timing_raw.get(func_name, _new_timing_stat())
            entry["count"] += 1
            entry["total"] += _timer.last
            entry["avg"] = entry["total"] / entry["count"]
            timing_raw[func_name] = entry

            # log the time profile
            log_level = logging.INFO if level == "info" else logging.DEBUG

            if log_to_metric:
                log_metric(
                    {f"time_profile/{func_name}": _timer.last},
                    level=log_level,
                    step=entry["count"],
                    prefix="Performance",
                )
            else:
                logger.log(log_level, f"time_profile/{func_name}: {_timer.last:.4f}s")
            return result

        return wrapper

    return decorator


def record_timing(name: str, duration_s: float, timing_raw: TimingRaw, level: str = "info"):
    """
    Record a single timing value into timing statistics storage.

    Args:
        name (str): Name identifier for the timing measurement.
        duration_s (float): Duration in seconds to record. Must be a numeric value.
        timing_raw (TimingRaw): Dictionary to store timing statistics with
            the same structure as used by timer.
        level (str): Logging level for the metric, either "info" or "debug".
            Defaults to "info".
    """
    if not isinstance(duration_s, (int, float)):
        return
    entry = timing_raw.get(name, _new_timing_stat())
    entry["count"] += 1
    entry["total"] += float(duration_s)
    entry["avg"] = entry["total"] / entry["count"]
    timing_raw[name] = entry

    # log the time profile
    log_level = logging.INFO if level == "info" else logging.DEBUG
    logger.log(log_level, f"time_profile/{name}: {float(duration_s):.4f}s")


@lru_cache(maxsize=None)
def _get_torch_device() -> Any:
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.cuda if not found.
    """
    if torch.cuda.is_available():
        return torch.cuda
    else:
        return torch.cpu


def _get_current_mem_info(unit: str = "GB", precision: int = 2) -> tuple[str, str, str, str]:
    """Get current memory usage.

    Note that CPU device memory info is always 0.

    Args:
        unit (str, optional): The unit of memory measurement. Defaults to "GB".
        precision (int, optional): The number of decimal places to round memory values. Defaults to 2.

    Returns:
        tuple[str, str, str, str]: A tuple containing memory allocated, memory reserved, memory used, and memory total
        in the specified unit.
    """
    assert unit in ["GB", "MB", "KB"]
    device = _get_torch_device()
    # torch.cpu.memory_allocated() does not exist
    if device == torch.cpu:
        return "0.00", "0.00", "0.00", "0.00"

    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    mem_allocated = _get_torch_device().memory_allocated()
    mem_reserved = _get_torch_device().memory_reserved()
    # use _get_torch_device().mem_get_info to profile device memory
    mem_free, mem_total = _get_torch_device().mem_get_info()
    mem_used = mem_total - mem_free
    mem_allocated = f"{mem_allocated / divisor:.{precision}f}"
    mem_reserved = f"{mem_reserved / divisor:.{precision}f}"
    mem_used = f"{mem_used / divisor:.{precision}f}"
    mem_total = f"{mem_total / divisor:.{precision}f}"
    return mem_allocated, mem_reserved, mem_used, mem_total


def log_gpu_memory_usage(head: str, level=logging.INFO, rank: int = 0):
    """Log GPU memory usage information.

    Args:
        head (str): A descriptive header for the memory usage log message.
        logger (logging.Logger, optional): Logger instance to use for logging. If None, prints to stdout.
        level: Logging level to use. Defaults to logging.DEBUG.
        rank (int): The rank of the process to log memory for. Defaults to 0.
    """
    mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
    message = (
        f"[GPU Memory] {head}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
        f"device memory used/total (GB): {mem_used}/{mem_total}"
    )
    logger.log(msg=message, level=level)
