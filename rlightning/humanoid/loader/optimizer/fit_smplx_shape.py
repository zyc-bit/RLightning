import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from rich.progress import Progress
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from rlightning.utils.logger import get_logger
from rlightning.humanoid.utils.kinematics_model.kinematics_model import KinematicsModel

logger = get_logger(__name__)


class SPMLXShapeFitting:

    @staticmethod
    def optimize(
        robot_type: str,
        robot_xml_path: str,
        body_model_path: str,
        optim_joint_matches: str,
        optim_iterations: int,
        device: str = "cuda:0",
    ):

        kinematic_model_device = device

        kinematic_model = KinematicsModel(file_path=robot_xml_path, device=kinematic_model_device)

        smplx_neutral_model = smplx.create(
            body_model_path, "smplx", gender="neutral", use_pca=False
        ).to(kinematic_model_device)

        robot_body_rest_pose = torch.zeros(
            kinematic_model.num_dof, dtype=torch.float, device=kinematic_model_device
        )
        robot_body_joint_names = kinematic_model.body_names

        smplx_body_rest_pose = np.zeros(
            (1, 1 + smplx_neutral_model.NUM_BODY_JOINTS, 3), dtype=np.float32
        )
        smplx_body_joint_names = JOINT_NAMES

        match_config = OmegaConf.load(optim_joint_matches)
        joint_match_config = match_config.joint_matches
        robot_body_joint_pick = [i[0] for i in joint_match_config]
        smplx_body_joint_pick = [i[1] for i in joint_match_config]
        robot_body_joint_pick_idx = [
            robot_body_joint_names.index(j) for j in robot_body_joint_pick
        ]
        smplx_body_joint_pick_idx = [
            smplx_body_joint_names.index(j) for j in smplx_body_joint_pick
        ]

        smplx_pose_modifier = match_config.smplx_pose_modifier
        for (
            mod_key,
            mod_value,
        ) in smplx_pose_modifier.items():  # from smpl rest body pose to robot rest pose
            assert mod_key in smplx_body_joint_names, f"{mod_key} is not in SMPLX joint names!"
            smplx_body_rest_pose[:, smplx_body_joint_names.index(mod_key)] = R.from_euler(
                "xyz", mod_value, degrees=True
            ).as_rotvec()
        smplx_body_rest_pose = (
            torch.from_numpy(smplx_body_rest_pose).float().to(kinematic_model_device).view(1, -1)
        )

        smplx_output = smplx_neutral_model(
            betas=torch.zeros(1, 16, dtype=torch.float, device=kinematic_model_device),  # (16,)
            global_orient=smplx_body_rest_pose[:, :3],  # (N, 3)
            body_pose=smplx_body_rest_pose[:, 3:],  # (N, 63)
            transl=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),  # (N, 3)
            left_hand_pose=torch.zeros(1, 45, dtype=torch.float, device=kinematic_model_device),
            right_hand_pose=torch.zeros(1, 45, dtype=torch.float, device=kinematic_model_device),
            jaw_pose=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),
            leye_pose=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),
            reye_pose=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),
            # expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )
        joint_pos = smplx_output.joints  # (1, num_joints, 3)
        root_trans_offset = joint_pos[:, 0]

        robot_body_pos, _ = kinematic_model.forward_kinematics(
            root_pos=root_trans_offset,
            root_rot=torch.tensor(
                [0, 0, 0, 1], dtype=torch.float, device=kinematic_model_device
            ).unsqueeze(0),
            dof_pos=robot_body_rest_pose.unsqueeze(0),
        )

        robot_smplx_shape = torch.zeros(1, 16, requires_grad=True, device=kinematic_model_device)
        scale = torch.ones([1], requires_grad=True, device=kinematic_model_device)
        shape_optimizer = torch.optim.Adam([robot_smplx_shape, scale], lr=0.1)

        num_iterations = optim_iterations
        logger.info(
            f"[Loader] Optimizing the smplx shape parameters. It takes {num_iterations} iterations in total!"
        )

        with Progress() as progress:
            task = progress.add_task("Iteration: 0 / Loss: NaN", total=num_iterations)

            for iter in range(num_iterations):
                joint_pos = smplx_neutral_model(
                    betas=robot_smplx_shape,  # (16,)
                    global_orient=smplx_body_rest_pose[:, :3],  # (N, 3)
                    body_pose=smplx_body_rest_pose[:, 3:],  # (N, 63)
                    transl=torch.zeros(
                        1, 3, dtype=torch.float, device=kinematic_model_device
                    ),  # (N, 3)
                    left_hand_pose=torch.zeros(
                        1, 45, dtype=torch.float, device=kinematic_model_device
                    ),
                    right_hand_pose=torch.zeros(
                        1, 45, dtype=torch.float, device=kinematic_model_device
                    ),
                    jaw_pose=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),
                    leye_pose=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),
                    reye_pose=torch.zeros(1, 3, dtype=torch.float, device=kinematic_model_device),
                    # expression=torch.zeros(num_frames, 10).float(),
                    return_full_pose=True,
                ).joints
                root_pos = joint_pos[:, 0]
                joint_pos = (joint_pos - joint_pos[:, 0]) * scale + root_pos

                diff = (
                    robot_body_pos[:, robot_body_joint_pick_idx].detach()
                    - joint_pos[:, smplx_body_joint_pick_idx]
                )
                loss = diff.norm(dim=-1).square().sum()

                progress.update(
                    task, description=f"Iteration: {iter} / Loss: {loss.item() * 1000}"
                )

                shape_optimizer.zero_grad()
                loss.backward()
                shape_optimizer.step()

                progress.update(task, advance=1)

        shape_vis_path = os.path.join(
            HydraConfig.get().runtime.output_dir, f"{robot_type}_optim_smplx_shape.png"
        )

        robot_body_3d = robot_body_pos[:, robot_body_joint_pick_idx].cpu().detach().numpy()
        robot_body_3d = robot_body_3d - robot_body_3d[:, 0:1]

        smplx_body_3d = joint_pos[:, smplx_body_joint_pick_idx].cpu().detach().numpy()
        smplx_body_3d = smplx_body_3d - smplx_body_3d[:, 0:1]

        idx = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(0, 45)
        ax.scatter(
            robot_body_3d[idx, :, 0],
            robot_body_3d[idx, :, 1],
            robot_body_3d[idx, :, 2],
            label="Humanoid Robot Shape",
            c="blue",
        )
        ax.scatter(
            smplx_body_3d[idx, :, 0],
            smplx_body_3d[idx, :, 1],
            smplx_body_3d[idx, :, 2],
            label="Fitted SMPLX Shape",
            c="red",
        )

        drange = 1
        ax.set_xlim(-drange, drange)
        ax.set_ylim(-drange, drange)
        ax.set_zlim(-drange, drange)
        ax.legend()
        plt.savefig(shape_vis_path)

        shape = robot_smplx_shape.cpu().detach()
        scale = scale.cpu().detach().numpy()

        gc.collect()

        return shape, scale
