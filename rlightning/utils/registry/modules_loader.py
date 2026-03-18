import importlib

from omegaconf import DictConfig
from rlightning.utils.logger import get_logger

logger = get_logger(__name__)


def load_modules_from_config(config: DictConfig) -> None:
    """
    Dynamically loads Python modules specified in the configuration.

    This function looks for a list under the `imports` key in the provided
    Hydra config. If found, it iterates through the list and imports each
    module. This is useful for registering plugins or other components
    at runtime without modifying the core codebase.

    Args:
        config (DictConfig): The Hydra configuration object.
    """
    modules_to_load = config.get("imports", None)
    if modules_to_load:
        logger.info(f"Dynamically loading modules: {modules_to_load}")
        for module_path in modules_to_load:
            try:
                importlib.import_module(module_path)
                logger.debug(f"Successfully imported module: {module_path}")
            except ImportError as e:
                logger.exception(f"Failed to import module '{module_path}'.")
                raise
