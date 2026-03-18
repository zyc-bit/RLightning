import os

import numpy as np
import smplx
import torch
from natsort import natsorted
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from rlightning.utils.logger import get_logger
from rlightning.humanoid.loader.base import Mode, MotionLoader
from .optimizer import SPMLXShapeFitting

logger = get_logger(__name__)


def slerp(rot1, rot2, t):
    """Spherical linear interpolation between two rotations."""
    # Convert to quaternions
    q1 = rot1.as_quat()
    q2 = rot2.as_quat()

    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute dot product
    dot = np.sum(q1 * q2)

    # If the dot product is negative, slerp won't take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # If the inputs are too close, linearly interpolate
    if dot > 0.9995:
        return R.from_quat(q1 + t * (q2 - q1))

    # Perform SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q1 + s1 * q2

    return R.from_quat(q)


class AmassLoader(MotionLoader):

    def __init__(self, config):
        super().__init__(config)
        self.body_model_template = lambda gender: smplx.create(
            self.config.body_model_path,
            "smplx",
            gender=gender,
            use_pca=False,
        )
        self._optimize_shape()

    def load(self, data_path):
        self.data_path = data_path

        if os.path.isdir(self.data_path):
            self.mode = Mode.dir

            self.data_list = []
            for dir_path, _, filenames in os.walk(self.data_path):
                for filename in natsorted(filenames):
                    if filename.endswith("_stagei.npz"):
                        continue
                    if filename.endswith((".pkl", ".npz")):
                        self.data_list.append(os.path.join(dir_path, filename))

            if self.config.exclude_content:
                logger.info(
                    f"[Loader] Removing motions with keywords: {self.config.exclude_content}"
                )
                self.data_list = [
                    path
                    for path in self.data_list
                    if not any(
                        content in self.config.exclude_content
                        for content in self.config.exclude_content
                    )
                ]

            self.data_num = len(self.data_list)
        elif os.path.isfile(self.data_path):
            assert (
                os.path.splitext(self.data_path)[-1] == self.format
            ), f"You are using the data loader for {self.format} format but the data you give is {self.data_path.suffix} format!"
            self.mode = Mode.file
            self.data_list = [self.data_path]
            self.data_num = 1
        else:
            raise Exception(f"Check your data path: {self.data_path}")

        logger.info(f"[Loader] Total number of motions: {self.data_num}")

    def _load_sample(self, sample_path):
        smplx_data = np.load(sample_path, allow_pickle=True)
        num_frames = smplx_data["pose_body"].shape[0]

        if self.use_optimized_shape:
            betas = self.shape
            body_model = self.body_model_template(gender="neutral")
        else:
            betas = torch.tensor(smplx_data["betas"]).float().view(1, -1)
            body_model = self.body_model_template(str(smplx_data["gender"]))
        smplx_output = body_model(
            betas=betas,  # (16,)
            global_orient=torch.tensor(smplx_data["root_orient"]).float(),  # (N, 3)
            body_pose=torch.tensor(smplx_data["pose_body"]).float(),  # (N, 63)
            transl=torch.tensor(smplx_data["trans"]).float(),  # (N, 3)
            left_hand_pose=torch.zeros(num_frames, 45).float(),
            right_hand_pose=torch.zeros(num_frames, 45).float(),
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
            # expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )

        if len(smplx_data["betas"].shape) == 1:
            human_height = 1.66 + 0.1 * smplx_data["betas"][0]
        else:
            human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]

        src_fps = smplx_data["mocap_frame_rate"].item()
        frame_skip = int(src_fps / 30)
        global_orient = smplx_output.global_orient.squeeze()
        full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
        joints = smplx_output.joints.detach().numpy().squeeze()
        if self.use_optimized_shape:
            root_pos = joints[:, :1]
            joints = (joints - joints[:, :1]) * self.scale + root_pos
        joint_names = JOINT_NAMES[: len(body_model.parents)]
        parents = body_model.parents

        if src_fps > 30:

            new_num_frames = num_frames // frame_skip

            original_time = np.arange(num_frames)
            target_time = np.linspace(0, num_frames - 1, new_num_frames)

            global_orient_interp = []
            for i in range(len(target_time)):
                t = target_time[i]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1

                rot1 = R.from_rotvec(global_orient[idx1])
                rot2 = R.from_rotvec(global_orient[idx2])
                interp_rot = slerp(rot1, rot2, alpha)
                global_orient_interp.append(interp_rot.as_rotvec())
            global_orient = np.stack(global_orient_interp, axis=0)

            # Interpolate full body pose using SLERP
            full_body_pose_interp = []
            for i in range(full_body_pose.shape[1]):  # For each joint
                joint_rots = []
                for j in range(len(target_time)):
                    t = target_time[j]
                    idx1 = int(np.floor(t))
                    idx2 = min(idx1 + 1, num_frames - 1)
                    alpha = t - idx1

                    rot1 = R.from_rotvec(full_body_pose[idx1, i])
                    rot2 = R.from_rotvec(full_body_pose[idx2, i])
                    interp_rot = slerp(rot1, rot2, alpha)
                    joint_rots.append(interp_rot.as_rotvec())
                full_body_pose_interp.append(np.stack(joint_rots, axis=0))
            full_body_pose = np.stack(full_body_pose_interp, axis=1)

            # Interpolate joint positions using linear interpolation
            joints_interp = []
            for i in range(joints.shape[1]):  # For each joint
                for j in range(3):  # For each coordinate
                    interp_func = interp1d(original_time, joints[:, i, j], kind="linear")
                    joints_interp.append(interp_func(target_time))
            joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)

            aligned_fps = len(global_orient) / num_frames * src_fps
        else:
            aligned_fps = 30

        frames = []
        for curr_frame in range(len(global_orient)):
            result = {}
            single_global_orient = global_orient[curr_frame]
            single_full_body_pose = full_body_pose[curr_frame]
            single_joints = joints[curr_frame]
            joint_orientations = []
            for i, joint_name in enumerate(joint_names):
                if i == 0:
                    rot = R.from_rotvec(single_global_orient)
                else:
                    rot = joint_orientations[parents[i]] * R.from_rotvec(
                        single_full_body_pose[i].squeeze()
                    )
                joint_orientations.append(rot)
                result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))

            frames.append(result)

        extras = {
            "fps": aligned_fps,
            "actual_human_height": human_height,
            "disable_scale_table": self.use_optimized_shape,
        }

        return frames, extras

    def _optimize_shape(self):
        self.use_optimized_shape = self.config.use_optimized_shape
        if self.use_optimized_shape:
            logger.info("[Loader] Using optimized shape for retargeting.")
            self.shape, self.scale = SPMLXShapeFitting.optimize(
                robot_type=self.config.robot.robot_type,
                robot_xml_path=self.config.robot.robot_xml_path,
                body_model_path=self.config.body_model_path,
                optim_joint_matches=self.config.optim_joint_matches,
                optim_iterations=self.config.optim_iterations,
            )
