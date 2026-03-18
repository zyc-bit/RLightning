import importlib
import pkgutil

from .weight_transfer_manager import WeightTransferManager

__all__ = ["WeightTransferManager"]

# Automatically discover and import all sub-packages within this directory.
for loader, name, is_pkg in pkgutil.iter_modules(__path__):
    if not is_pkg:
        importlib.import_module(f".{name}", __name__)
        importlib.import_module(f".{name}", __name__)
