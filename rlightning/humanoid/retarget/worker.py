import os
import pickle


from rlightning.humanoid.types import RetargetCfg
from rlightning.humanoid import loader
from rlightning.utils.logger import get_logger
from rlightning.utils.ray import RayActorMixin

logger = get_logger(__name__)


class Retarget(RayActorMixin):

    def __init__(self, cfg: RetargetCfg):
        self.cfg = cfg

        logger.info(f"Target robot: {cfg.robot.humanoid_type}")
        logger.info(f"Loader type: {cfg.loader}")
        logger.info(f"Retarget type: {cfg.retargetting}")
        logger.info(f"Formatter: {cfg.formatter}")

    def run(self):
        """Performing retargeting"""

        motion_loader = loader.get_loader(self.cfg.loader, self.cfg.loader_cfg)
        motion_loader.load(self.cfg.motion_dataset)

        retargeter = None
        formatter = None

        motions = []

        for save_path, frames, extras in motion_loader:
            retargeter.update(extras)
            qpos_list = retargeter.retarget(frames)
            formatted_results = formatter.format(qpos_list, extras)
            motions.append(formatted_results)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(formatted_results, f)
            logger.info(f"Retargeted motion saved to {save_path}")

        retargeter.finish()

        return motions
