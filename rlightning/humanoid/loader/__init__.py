from rlightning.utils.config import Config

from .base import MotionLoader
from .lafan_loader import LafanLoader
from .actorcore_loader import ActorCoreLoader
from .amass_loader import AmassLoader


MOTION_LOADER_LIB = dict(lafan=LafanLoader, actorcore=ActorCoreLoader, amass=AmassLoader)


def get_loader(loader_type: str, loader_cfg: Config) -> MotionLoader:
    """Create a loader with given loader type and configuration.

    The loader type should be registered and the configuration should follow the use in correlated loader implementation.

    Args:
        loader_type (str): Registered loader type.
        loader_cfg (Config): Loader configuration.

    Returns:
        MotionLoader: An instance of motion loader
    """

    return MOTION_LOADER_LIB[loader_type](loader_cfg)
