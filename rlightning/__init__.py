"""RLightning: Large-scale distributed reinforcement learning framework."""

from importlib import metadata

from . import buffer, engine, env, policy, types, utils, weights

try:
    __version__ = metadata.version("rlightning")
except metadata.PackageNotFoundError:  # pragma: no cover - local editable mode
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "buffer",
    "engine",
    "env",
    "policy",
    "types",
    "utils",
    "weights",
]
