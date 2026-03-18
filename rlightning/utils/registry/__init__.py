"""Registry module for component registration and discovery.

This module provides registry classes for dynamically registering and
retrieving components like buffers, policies, environments, and weights.

Available registries:
    - BUFFERS: Registry for data buffer classes.
    - WEIGHTS: Registry for weight management classes.
    - POLICIES: Registry for policy classes.
    - ENVS: Registry for environment classes.
"""

from .modules_loader import load_modules_from_config
from .registry import Registry

BUFFERS = Registry("buffer")
WEIGHTS = Registry("weights")
POLICIES = Registry("policy")
ENVS = Registry("env")
ENGINE = Registry("engine")

__all__ = [
    "BUFFERS",
    "WEIGHTS",
    "POLICIES",
    "ENVS",
    "ENGINE",
    "load_modules_from_config",
]
