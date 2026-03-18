import gc
import os

import matplotlib.pyplot as plt
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from rich.progress import Progress
from scipy.spatial.transform import Rotation as R

from rlightning.utils.logger import get_logger
from rlightning.humanoid.utils.kinematics_model.kinematics_model import KinematicsModel
from rlightning.humanoid.utils.lafan_vendor.extract import read_bvh

logger = get_logger(__name__)


class ActorCoreShapeFitting:

    @staticmethod
    def optimize(
        robot_type,
        robot_xml_path,
        robot_rest_height,
        bvh_file,
        optim_joint_matches,
        optim_iterations,
        device="cuda:0",
    ):
        kinematic_model_device = device

        kinematic_model = KinematicsModel(file_path=robot_xml_path, device=kinematic_model_device)

        robot_body_joint_names = kinematic_model.body_names
        robot_dof_names = kinematic_model.dof_names
        robot_root_pos = torch.tensor(
            [0, 0, robot_rest_height], dtype=torch.float, device=kinematic_model_device
        )

        bvh_data = read_bvh(bvh_file)

        # bvh_pos_merged = bvh_data.pos[:,1:]
        # bvh_pos_merged[:,0:1] += bvh_data.pos[:, 0:1] + quat_mul_vec(bvh_data.quats[:, 0:1], bvh_data.offsets[0:1])
        # bvh_quats_merged = bvh_data.quats[:,1:]
        # bvh_quats_merged[:,0:1] = quat_mul_np(bvh_data.quats[:,0:1], bvh_data.quats[:, 1:2])
        # bvh_bones_merged = bvh_data.bones[1:]
        # bvh_offsets_merged = bvh_data.offsets[1:]
        # bvh_parents_merged = [idx - 1 for idx in bvh_data.parents[1:]]

        # bvh_data.pos = bvh_pos_merged
        # bvh_data.quats = bvh_quats_merged
        # bvh_data.bones = bvh_bones_merged
        # bvh_data.offsets = bvh_offsets_merged
        # bvh_data.parents = bvh_parents_merged

        bvh_offset = torch.from_numpy(bvh_data.offsets).float().to(kinematic_model_device)
        bvh_offset[0, [0, 1]] = 0  # z up for converted BVH
        bvh_joint_names = bvh_data.bones
        bvh_joint_num = len(bvh_joint_names)

        match_config = OmegaConf.load(optim_joint_matches)

        robot_pose_modifier = match_config.robot_pose_modifier
        robot_body_rest_pose = torch.zeros(
            kinematic_model.num_dof, dtype=torch.float, device=kinematic_model_device
        )
        for mod_key, mod_value in robot_pose_modifier.items():
            assert mod_key in robot_dof_names, f"{mod_key} is not in Robot joint names!"
            robot_body_rest_pose[robot_dof_names.index(mod_key)] = mod_value
        robot_body_pos, _ = kinematic_model.forward_kinematics(
            root_pos=robot_root_pos.unsqueeze(0),
            root_rot=torch.from_numpy(R.from_euler("xyz", [0, 0, -90], degrees=True).as_quat())
            .float()
            .to(kinematic_model_device)
            .unsqueeze(0),
            dof_pos=robot_body_rest_pose.unsqueeze(0),
        )

        joint_match_config = match_config.joint_matches
        robot_body_joint_pick = [i[0] for i in joint_match_config]
        bvh_body_joint_pick = [i[1] for i in joint_match_config]
        robot_body_joint_pick_idx = [
            robot_body_joint_names.index(j) for j in robot_body_joint_pick
        ]
        bvh_body_joint_pick_idx = [bvh_joint_names.index(j) for j in bvh_body_joint_pick]

        scale = torch.zeros(
            (bvh_joint_num - 1,),
            dtype=torch.float,
            device=kinematic_model_device,
            requires_grad=True,
        )
        scale_optimizer = torch.optim.Adam([scale], lr=0.01)

        num_iterations = optim_iterations
        logger.info(f"[Loader] Optimizing the bvh scale. It takes {num_iterations} in total!")

        with Progress() as progress:
            task = progress.add_task(f"Iteration: 0 / Loss: NaN", total=num_iterations)

            for iter in range(num_iterations):
                rest_positions = torch.zeros(
                    (1, bvh_joint_num, 3), dtype=torch.float32, device=kinematic_model_device
                )
                for i in range(bvh_joint_num):
                    parent = bvh_data.parents[i]
                    if parent == -1:
                        assert i == 0
                        # rest_positions[:, i] = torch.tensor([0,0,robot_rest_height], dtype=torch.float, device=kinematic_model_device).unsqueeze(0) * 100
                        rest_positions[:, i] = torch.tensor(
                            [0, 0, 0], dtype=torch.float, device=kinematic_model_device
                        ).unsqueeze(0)
                    # elif parent == 0:
                    #     assert i == 1
                    #     rest_positions[:, i] = torch.tensor([0,0, robot_rest_height], dtype=torch.float, device=kinematic_model_device).unsqueeze(0) * 100
                    else:
                        rest_positions[:, i] = rest_positions[:, parent] + bvh_offset[
                            i
                        ] * torch.exp(scale[i - 1])
                rest_positions = rest_positions / 100

                diff = (
                    robot_body_pos[:, robot_body_joint_pick_idx]
                    - rest_positions[:, bvh_body_joint_pick_idx]
                )
                loss = diff.norm(dim=-1).square().sum()

                progress.update(
                    task, description=f"Iteration: {iter} / Loss: {loss.item() * 1000}"
                )

                scale_optimizer.zero_grad()
                loss.backward()
                scale_optimizer.step()

                progress.update(task, advance=1)

        shape_vis_path = os.path.join(
            HydraConfig.get().runtime.output_dir, f"{robot_type}_optim_bvh_shape.png"
        )

        robot_body_3d = robot_body_pos[:, robot_body_joint_pick_idx].cpu().detach().numpy()
        robot_body_3d = robot_body_3d - robot_body_3d[:, 0:1]

        bvh_body_3d = rest_positions[:, bvh_body_joint_pick_idx].cpu().detach().numpy()
        bvh_body_3d = bvh_body_3d - bvh_body_3d[:, 0:1]

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
            bvh_body_3d[idx, :, 0],
            bvh_body_3d[idx, :, 1],
            bvh_body_3d[idx, :, 2],
            label="Fitted BVH Shape",
            c="red",
        )

        drange = 1.5
        ax.set_xlim(-drange, drange)
        ax.set_ylim(-drange, drange)
        ax.set_zlim(-drange, drange)
        ax.legend()
        plt.savefig(shape_vis_path)

        scale = torch.exp(scale).cpu().detach().numpy()
        logger.info(f"[Loader] Optimized BVH scale: {scale}")

        gc.collect()

        return scale
