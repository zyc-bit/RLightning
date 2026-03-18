import importlib
import pkgutil

from .base_policy import BasePolicy
from .policy_group import PolicyGroup

__all__ = ["BasePolicy", "PolicyGroup"]

# Automatically discover and import all sub-packages within this directory.
for loader, name, is_pkg in pkgutil.iter_modules(__path__):
    if is_pkg:
        importlib.import_module(f".{name}", __name__)
