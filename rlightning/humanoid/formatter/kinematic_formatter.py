import torch
import dataclasses
import numpy as np

from rlightning.utils.logger import get_logger

from rlightning.utils.config import Config
from rlightning.humanoid.utils.kinematics_model import KinematicsModel
from .base import Formatter

logger = get_logger(__name__)


class KinematicFormatter(Formatter):

    @dataclasses.dataclass
    class Motion:
        fps: float
        root_pos: np.ndarray
        root_rot: np.ndarray
        dof_pos: np.ndarray

    class FormatterCfg(Config):
        robot_xml_path: str
        height_adjust: bool = False
        root_offset: bool = False
        quat_order: str = "xyzw"
        kinematic_model_device: str = "cpu"

    cfg: FormatterCfg

    def __init__(self, config: FormatterCfg = None, **kwargs):
        super().__init__()
        if config is None:
            config = self.FormatterCfg(**kwargs)

        self.cfg = config
        self.quat_order = config.quat_order
        self.kinematic_model_device = config.kinematic_model_device
        self.kinematic_model = KinematicsModel(
            file_path=config.robot_xml_path, device=self.kinematic_model_device
        )
        self.height_adjust = config.height_adjust
        logger.info(f"[Formatter] height_adjust: {self.height_adjust}")
        self.root_offset = config.root_offset
        logger.info(f"[Formatter] root offset: {config.root_offset}")

    def format(self, qpos_list, extras):
        root_pos = qpos_list[:, :3]
        root_rot = qpos_list[:, 3:7]  # wxyz
        if self.quat_order == "xyzw":
            root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
        dof_pos = qpos_list[:, 7:]
        # num_frames = root_pos.shape[0]

        body_pos, _ = self.kinematic_model.forward_kinematics(
            torch.from_numpy(root_pos).float().to(self.kinematic_model_device),
            torch.from_numpy(root_rot).float().to(self.kinematic_model_device),
            torch.from_numpy(dof_pos).float().to(self.kinematic_model_device),
        )

        if self.height_adjust:
            ground_offset = 0.0
            lowest_height = torch.min(body_pos[..., 2]).item()
            root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset

        if self.root_offset:
            root_pos[:, :2] -= root_pos[0, :2]

        return self.Motion(extras["fps"], root_pos, root_rot, dof_pos)
