from typing import Dict

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from smpl_sim.utils import torch_utils
from smpl_sim.utils.smoothing_utils import gaussian_filter_1d_batch

from rlightning.utils.logger import get_logger
from rlightning.humanoid.types import DataRetrieverCfg, RetargetedMotion
from rlightning.humanoid.loader import smplx_loader
from rlightning.utils.progress import get_progress

from .humanoid_batch import HumanoidBatch

logger = get_logger(__name__)


def retargetting(
    smpl_model_dir: str,
    smpl_parser: SMPL_Parser,
    motion_names,
    motion_path_dict,
    cfg: DataRetrieverCfg,
    device: str = "cpu",
) -> Dict[str, RetargetedMotion]:

    # load forward kinematics model
    robot_fk = HumanoidBatch(cfg.robot)
    num_augment_joint = len(cfg.robot.extend_configs)

    # load SMPL parser
    smpl_parser = SMPL_Parser(model_path=smpl_model_dir, gender=cfg.gender)
    smpl_shape, smpl_scale = joblib.load(f"data/{cfg.robot.humanoid_type}/smpl_shape.pkl")

    # get key joint names and indices for both robot and SMPL
    joint_names_robot = robot_fk.body_names_augment
    key_joint_names_robot = [pair[0] for pair in cfg.robot.joint_matches]
    key_joint_indices_robot = [joint_names_robot.index(name) for name in key_joint_names_robot]

    key_joint_names_smpl = [pair[1] for pair in cfg.robot.joint_matches]
    key_joint_indices_smpl = [SMPL_BONE_ORDER_NAMES.index(name) for name in key_joint_names_smpl]

    retarget_data_dict = {}
    progress = get_progress()
    task = progress.add_task("Retargeting motions...", total=len(motion_names))

    for motion_name in motion_names:
        motion_raw_data = smplx_loader.load(motion_path_dict[motion_name])
        progress.update(task, advance=1)

        if motion_raw_data is None:
            continue

        # sample the motion data to 30 fps
        raw_fps = motion_raw_data.fps
        desired_fps = 30
        skip = int(raw_fps // desired_fps)

        root_pos = motion_raw_data.trans[::skip]
        pose_aa = motion_raw_data.pose_aa[::skip]

        root_pos = torch.from_numpy(root_pos).float().to(device)
        pose_aa = torch.from_numpy(pose_aa).float().to(device)

        num_frames = pose_aa.shape[0]

        if num_frames < 10:
            print(f"Skipping {motion_name} due to insufficient frames: {num_frames}")
            continue

        with torch.no_grad():
            # Use the loaded pose_aa to compute forward kinematics for SMPL
            # with the optimzed shape and scale
            verts_smpl, joint_pos_smpl = smpl_parser.get_joints_verts(
                pose_aa, smpl_shape, root_pos
            )
            root_pos_smpl = joint_pos_smpl[:, 0:1]
            joint_pos_smpl = smpl_scale * (joint_pos_smpl - root_pos_smpl) + root_pos_smpl
            joint_pos_smpl[..., 2] -= verts_smpl[0, :, 2].min().item()  # align the ground plane

            root_pos_smpl = joint_pos_smpl[:, 0].clone()

            root_quat_smpl = torch.from_numpy(
                (
                    sRot.from_rotvec(pose_aa[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                ).as_quat()
            ).float()  # can't directly use this
            root_rot_smpl = torch.from_numpy(
                sRot.from_quat(torch_utils.calc_heading_quat(root_quat_smpl)).as_rotvec()
            ).float()  # so only use the heading.

        # prepare the variables for optimization
        dof_pos_var = torch.autograd.Variable(
            torch.zeros((1, num_frames, robot_fk.num_dof, 1)),
            requires_grad=True,
        )
        root_pos_offset_var = torch.autograd.Variable(
            torch.zeros(1, 3),
            requires_grad=True,
        )
        root_rot_var = torch.autograd.Variable(
            root_rot_smpl.clone(),
            requires_grad=True,
        )
        # optimizer
        optimizer = torch.optim.Adam([dof_pos_var, root_pos_offset_var, root_rot_var], lr=0.02)

        filter_kernel_size = 5
        filter_sigma = 0.75

        for iteration in range(cfg.get("fitting_iterations", 500)):
            # prepare the angle-axis of each joint for robot
            pose_aa_robot = torch.cat(
                [
                    root_rot_var[None, :, None],
                    robot_fk.dof_axis * dof_pos_var,
                    torch.zeros((1, num_frames, num_augment_joint, 3), device=device),
                ],
                axis=2,
            )
            # compute forward kinematics for robot
            fk_return_robot = robot_fk.fk_batch(
                pose_aa_robot, root_pos_smpl[None,] + root_pos_offset_var
            )

            if num_augment_joint > 0:
                key_joint_pos_robot = fk_return_robot.global_translation_extend[
                    :, :, key_joint_indices_robot
                ]
            else:
                key_joint_pos_robot = fk_return_robot.global_translation[
                    :, :, key_joint_indices_robot
                ]
            key_joint_pos_smpl = joint_pos_smpl[:, key_joint_indices_smpl]
            # compute the difference of key joints position between SMPL and robot
            diff = key_joint_pos_robot - key_joint_pos_smpl

            # compute the loss: norm of the difference and a regularization term for dof_pos_var
            loss = diff.norm(dim=-1).mean() + 0.01 * torch.mean(torch.square(dof_pos_var))

            # update the optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # apply dof saturation
            dof_pos_var.data.clamp_(
                min=robot_fk.joints_range[:, 0, None], max=robot_fk.joints_range[:, 1, None]
            )

            # filter the dof positions
            # I don't know why the operation is so complex here
            # refer to the original PHC repository for more details
            dof_pos_var.data = gaussian_filter_1d_batch(
                dof_pos_var.squeeze().transpose(1, 0)[None,], filter_kernel_size, filter_sigma
            ).transpose(2, 1)[..., None]

            progress.update(task, advance=1, iteration=iteration, loss=loss.item())

        # after optimization

        # apply the dof saturation
        dof_pos_var.data.clamp_(
            min=robot_fk.joints_range[:, 0, None], max=robot_fk.joints_range[:, 1, None]
        )

        # optimized angle-axis of each joint for robot
        pose_aa_robot_opt = torch.cat(
            [
                root_rot_var[None, :, None],
                robot_fk.dof_axis * dof_pos_var,
                torch.zeros((1, num_frames, num_augment_joint, 3), device=device),
            ],
            axis=2,
        )

        # optimized root position of robot
        root_pos_opt = (root_pos_smpl + root_pos_offset_var).clone()

        # move to the ground plane
        combined_mesh = robot_fk.mesh_fk(
            pose_aa_robot_opt[:, :1].detach(), root_pos_opt[None, :1].detach()
        )
        height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()
        root_pos_opt[..., 2] -= height_diff

        # save the joint positions of robot:
        if num_augment_joint > 0:
            joint_pos_robot = fk_return_robot.global_translation_extend
        else:
            joint_pos_robot = fk_return_robot.global_translation
        # save the joint positions of robot
        joint_pos_robot_dump = joint_pos_robot.squeeze().detach().cpu().numpy().copy()
        joint_pos_robot_dump[..., 2] -= height_diff

        # also save the smpl joint positions for later use
        joint_pos_smpl_dump = joint_pos_smpl.detach().cpu().numpy().copy()
        joint_pos_smpl_dump[..., 2] -= height_diff

        # save the retargeted data
        retarget_data_dict[motion_name] = RetargetedMotion.from_dict(
            {
                "root_pos": root_pos_opt.squeeze().detach().cpu().numpy(),
                "root_rot": sRot.from_rotvec(root_rot_var.detach().numpy()).as_quat(),
                "pose_aa": pose_aa_robot_opt.squeeze().detach().cpu().numpy(),
                "dof_pos": dof_pos_var.squeeze().detach().cpu().numpy(),
                "fps": desired_fps,
                "joint_pos_robot": joint_pos_robot_dump,
                "joint_pos_smpl": joint_pos_smpl_dump,
            }
        )

    return retarget_data_dict
