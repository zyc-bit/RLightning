"""
The simple motion loader accept motion file with .npz format, and the included items are:

- mocap_framerate: the frame rate of the motion capture data
- poses: the pose data in axis-angle format, shape (num_frames, 72)
- trans: the root translation data, shape (num_frames, 3)
- betas: the shape parameters, shape (10,)
- gender: the gender of the SMPL model, string

"""

import numpy as np

from rlightning.utils.logger import get_logger

from rlightning.humanoid.types import SMPLXMotion

logger = get_logger(__name__)


def load(motion_path: str) -> SMPLXMotion:
    """Load motion data from a .npz file.

    Args:
        motion_path (str): Path to the .npz file.

    Returns:
        dict: A dictionary containing the motion data, or None if loading failed.
    """

    with open(motion_path, "rb") as f:
        try:
            entry_data = dict(np.load(f, allow_pickle=True))
        except Exception as e:
            logger.error(f"Error loading {motion_path}: {e}")
            exit(1)

    if "mocap_framerate" not in entry_data:
        return None

    frame_rate = entry_data["mocap_framerate"]
    root_trans = entry_data["trans"]
    pose_aa = np.concatenate(
        [entry_data["poses"][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1
    )
    betas = entry_data["betas"]
    gender = entry_data["gender"]

    return SMPLXMotion.from_dict(
        {
            "pose_aa": pose_aa,
            "gender": gender,
            "trans": root_trans,
            "betas": betas,
            "fps": frame_rate,
        }
    )
