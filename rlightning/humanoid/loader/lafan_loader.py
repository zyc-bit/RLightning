"""
This is an implementation for loading BVH Lafan datafiles
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlightning.humanoid.formatter import KinematicFormatter
from rlightning.humanoid.loader.base import LoaderCfg, MotionLoader
from rlightning.humanoid.retarget import GmrRetargeter
from rlightning.humanoid.utils.lafan_vendor.extract import read_bvh
from rlightning.humanoid.utils.lafan_vendor.utils import quat_fk, quat_mul


class LafanLoader(MotionLoader):

    formatter_cls = KinematicFormatter
    retargeter_cls = GmrRetargeter

    def _load_sample(self, sample_path: str):
        bvh_data = read_bvh(sample_path)

        skip = int(bvh_data.fps / 30)
        if skip > 1:
            bvh_data.pos = bvh_data.pos[::skip]
            bvh_data.quats = bvh_data.quats[::skip]

        global_data = quat_fk(bvh_data.quats, bvh_data.pos, bvh_data.parents)

        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

        frames = []
        # each frame is a tuple of joint position and orientation
        for frame in range(bvh_data.pos.shape[0]):
            result = {}
            for i, bone in enumerate(bvh_data.bones):
                orientation = quat_mul(rotation_quat, global_data[0][frame, i])
                position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
                result[bone] = (position, orientation)

            # Add modified foot pose
            result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftToe"][1])
            result["RightFootMod"] = (result["RightFoot"][0], result["RightToe"][1])

            frames.append(result)

        human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
        # human_height = human_height + 0.2  # cm to m
        human_height = 1.75  # cm to m

        extras = {"fps": 30, "actual_human_height": human_height}

        return frames, extras


if __name__ == "__main__":
    import argparse

    from rlightning.humanoid.formatter import Formatter
    from rlightning.humanoid.retarget import Retargeter
    from rlightning.utils.logger import get_logger

    logger = get_logger(__name__)

    parser = argparse.ArgumentParser("Test lafan loader")
    parser.add_argument("--f-path", help="Motion file/directory path.", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite or not")

    args = parser.parse_args()

    robot_xml_path = "./g1_mocap_29dof.xml"
    cfg = LoaderCfg(
        data_format=".bvh",
        data_path=args.f_path,
        overwrite=args.overwrite,
        retargeter=Retargeter.RetargeterCfg(
            robot_xml_path=robot_xml_path,
            solver="daqp",
            damping=0.5,
            use_velocity_limit=False,
            ik_config_path="./examples/wbc_tracking/conf/ik/lafan_to_unitree_g1.yaml",
        ),
        formatter=Formatter.FormatterCfg(
            robot_xml_path=robot_xml_path,
            height_adjust=False,
            root_offset=False,
            quat_order="xyzw",
            kinematic_model_device="cpu",
        ),
    )
    loader = LafanLoader(cfg)
    loader.prepare(args.f_path)
    motion = loader.sample()[0]

    logger.info(f"FPS: {motion.fps}")
    logger.info(f"root_pos shape: {motion.root_pos.shape}")
    logger.info(f"root_rot shape: {motion.root_rot.shape}")
    logger.info(f"dof_pos: {motion.dof_pos.shape}")
