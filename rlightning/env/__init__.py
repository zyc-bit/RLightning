"""Environment module for reinforcement learning.

This module provides various environment implementations and utilities for
managing vectorized and distributed environments in RL training.

Available components:
    - BaseEnv: Abstract base class for all environments.
    - EnvGroup: Group manager for multiple environments.
    - EnvMeta: Metadata container for environment properties.
"""

import importlib
import pkgutil

from .base_env import BaseEnv, EnvMeta
from .env_group import EnvGroup

__all__ = ["BaseEnv", "EnvGroup", "EnvMeta"]

EXCLUDE_PACKAGES = {"env_server", "util"}

# Automatically discover and import all sub-packages within this directory.
for loader, name, is_pkg in pkgutil.iter_modules(__path__):
    if name not in EXCLUDE_PACKAGES:
        try:
            importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as e:
            # Skip packages whose optional dependencies are not installed
            print(f"import module {name} failed: {e}")
            pass
