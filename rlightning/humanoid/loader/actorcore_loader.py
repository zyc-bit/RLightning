from typing import Any
from rlightning.utils.logger import get_logger

import random

from rlightning.utils.config import Config
from rlightning.humanoid.utils.lafan_vendor.extract import read_bvh
from rlightning.humanoid.utils.lafan_vendor.utils import quat_fk
from rlightning.humanoid.loader.base import MotionLoader, LoaderCfg

from .optimizer import ActorCoreShapeFitting

logger = get_logger(__name__)


class RobotCfg(Config):

    robot_type: str
    robot_reset_height: bool
    robot_xml_path: str


class ActorCoreLoaderCfg(LoaderCfg):

    use_optimized_shape: bool = True
    """Enable optimized shaping"""

    robot: RobotCfg

    optim_joint_matches: Any

    optim_iterations: int


class ActorCoreLoader(MotionLoader):

    config: ActorCoreLoaderCfg

    def load(self, data_path: str):
        super().load(data_path)

        # then ...
        bvh_file = random.choice(self.files)
        self.use_optimized_shape = self.config.use_optimized_shape
        assert (
            self.config.use_optimized_shape
        ), "Currently we only support optimization-based shape attainment!"
        self.scale = ActorCoreShapeFitting.optimize(
            robot_type=self.config.robot.robot_type,
            robot_xml_path=self.config.robot.robot_xml_path,
            robot_rest_height=self.config.robot.robot_rest_height,
            bvh_file=bvh_file,
            optim_joint_matches=self.config.optim_joint_matches,
            optim_iterations=self.config.optim_iterations,
        )

    def _load_sample(self, f_path):
        bvh_data = read_bvh(f_path)

        skip = bvh_data.fps // 30
        if skip > 1:
            bvh_data.pos = bvh_data.pos[::skip]
            bvh_data.quats = bvh_data.quats[::skip]

        if self.use_optimized_shape:
            bvh_data.pos[:, 1:] *= self.scale[..., None]

        global_data = quat_fk(bvh_data.quats, bvh_data.pos, bvh_data.parents)

        frames = []
        for frame in range(bvh_data.pos.shape[0]):
            result = {}
            for i, bone in enumerate(bvh_data.bones):
                orientation = global_data[0][frame, i]
                position = global_data[1][frame, i] / 100  # cm to m
                result[bone] = (position, orientation)

            # Add modified foot pose
            result["LeftFootMod"] = (result["CC_Base_L_Foot"][0], result["CC_Base_L_ToeBase"][1])
            result["RightFootMod"] = (result["CC_Base_R_Foot"][0], result["CC_Base_R_ToeBase"][1])

            frames.append(result)

        extras = {
            "fps": 30,
            "disable_scale_table": self.use_optimized_shape,
        }

        return frames, extras
