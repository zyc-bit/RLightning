"""
Distributed logging utilities for RLightning framework.

This package provides lightweight, efficient logging solutions that leverage
Ray's built-in log aggregation capabilities.
"""

from .logger import get_logger, log_metric, setup_logger

__all__ = ["log_metric", "setup_logger", "get_logger"]
