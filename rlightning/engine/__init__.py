"""Engine module for reinforcement learning training loops.

This module provides various RL engine implementations including synchronous,
asynchronous, and specialized engines for different training paradigms.

Available engines:
    - SyncRLEngine: Synchronous reinforcement learning engine.
    - AsyncRLEngine: Asynchronous reinforcement learning engine.
    - EvaluationEngine: Engine for policy evaluation.
    - RSLRLEngine: Engine for RSL-RL based training.
"""

import importlib
import pkgutil

from .async_rl_engine import AsyncRLEngine
from .async_rsl_rl_engine import AsyncRSLRLEngine
from .base_engine import BaseEngine
from .evaluation_engine import EvaluationEngine
from .rsl_rl_engine import RSLRLEngine
from .sync_rl_engine import SyncRLEngine

__all__ = [
    "BaseEngine",
    "SyncRLEngine",
    "AsyncRLEngine",
    "EvaluationEngine",
    "RSLRLEngine",
    "AsyncRSLRLEngine",
]

EXCLUDE_PACKAGES = {}

# Automatically discover and import all sub-packages within this directory.
for loader, name, is_pkg in pkgutil.iter_modules(__path__):
    if name not in EXCLUDE_PACKAGES:
        try:
            importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError:
            # Skip packages whose optional dependencies are not installed
            pass
