import numpy as np
import dataclasses

from rlightning.utils.logger import get_logger
from rlightning.utils.config import Config

logger = get_logger(__name__)


class Formatter:

    class FormatterCfg(Config):
        robot_xml_path: str = ""
        height_adjust: bool = False
        root_offset: bool = False
        quat_order: str = "xyzw"
        kinematic_model_device: str = "cpu"

    @dataclasses.dataclass
    class Motion:
        fps: float
        root_pos: np.ndarray
        root_rot: np.ndarray
        dof_pos: np.ndarray

    def __init__(self):
        self.cfg: Formatter.FormatterCfg = None

    def __call__(self, qpos_list, extras) -> Motion:
        return self.format(qpos_list, extras)

    def format(self, qpos_list, extras) -> Motion:
        root_pos = qpos_list[:, :3]
        root_rot = qpos_list[:, 3:7]  # wxyz
        if self.cfg.quat_order == "xyzw":
            root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
        dof_pos = qpos_list[:, 7:]
        # num_frames = root_pos.shape[0]

        motion_data = self.Motion(
            **{
                "fps": extras["fps"],
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
            }
        )

        return motion_data
