"""
This script integrates the full stack of downloading, processing, and loading
data for humanoid simulations in a rlightning learning framework.
"""

import glob
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES

from rlightning.humanoid.types import DataRetrieverCfg
from rlightning.utils.logger import get_logger
from rlightning.utils.progress import get_progress

from .types import RetargetedMotion
from .retarget.humanoid_batch import HumanoidBatch
from .retarget.retarget import retargetting

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

logger = get_logger(__name__)


def fit_smpl_shape(cfg: DataRetrieverCfg, device: str = "cpu"):
    import os

    import joblib
    import torch

    from scipy.spatial.transform import Rotation as sRot
    from smpl_sim.smpllib.smpl_parser import SMPL_Parser
    from torch.autograd import Variable

    # load forward kinematics model
    robot_fk = HumanoidBatch(cfg.robot)

    # get key joint names and indices for both robot and SMPL
    joint_names_robot = robot_fk.body_names_augment
    key_joint_names_robot = [pair[0] for pair in cfg.robot.joint_matches]
    key_joint_indices_robot = [joint_names_robot.index(name) for name in key_joint_names_robot]

    key_joint_names_smpl = [pair[1] for pair in cfg.robot.joint_matches]
    key_joint_indices_smpl = [SMPL_BONE_ORDER_NAMES.index(name) for name in key_joint_names_smpl]

    # prepare stand pose of axis-angle for SMPL
    stand_pose_aa_smpl = torch.zeros(
        (1, len(SMPL_BONE_ORDER_NAMES), 3), dtype=torch.float32, device=device
    )
    modifier = cfg.robot.smpl_pose_modifier
    for joint in modifier.keys():
        euler_angle = eval(modifier[joint])
        aa = sRot.from_euler("xyz", euler_angle, degrees=False).as_rotvec()
        stand_pose_aa_smpl[:, SMPL_BONE_ORDER_NAMES.index(joint), :] = torch.tensor(
            aa, dtype=torch.float32, device=device
        ).view(1, 3)
    stand_pose_aa_smpl = stand_pose_aa_smpl.reshape(-1, len(SMPL_BONE_ORDER_NAMES) * 3)

    # load SMPL model
    model_path = os.path.join(DIR_PATH, "smpl_model")
    smpl_parser = SMPL_Parser(model_path=model_path, gender=cfg.gender)

    # compute forward kinematics for SMPL, and get the root translation
    trans = torch.zeros((1, 3), dtype=torch.float32, device=device)
    beta = torch.zeros((1, 10), dtype=torch.float32, device=device)  # 10 shape parameters

    _, joint_pos = smpl_parser.get_joints_verts(stand_pose_aa_smpl, beta, trans)
    root_trans_offset = joint_pos[:, 0]

    # prepare stand pose of axis-angle for robot
    stand_pose_aa_robot = torch.zeros(
        (1, 1, 1, robot_fk.num_bodies, 3), dtype=torch.float32, device=device
    )

    # compute forward kinematics for robot
    fk_return_robot = robot_fk.fk_batch(stand_pose_aa_robot, root_trans_offset[None, 0:1])

    fitting_cfg = cfg.shape_fitting

    # prepare variables for optimization
    shape_var = Variable(
        torch.zeros((1, 10), dtype=torch.float32, device=device), requires_grad=True
    )  # 10 shape parameters
    scale_var = Variable(
        torch.ones([1], dtype=torch.float32, device=device), requires_grad=True
    )  # scale factor

    # optimizer
    optimizer = torch.optim.Adam([shape_var, scale_var], lr=fitting_cfg.learning_rate)

    logger.info("Start fitting SMPL shape...")

    progress = get_progress()
    task = progress.add_task("[red]Fitting SMPL Shape", total=fitting_cfg.train_iterations)

    for i in range(fitting_cfg.train_iterations):
        optimizer.zero_grad()

        # compute forward kinematics for SMPL
        _, joint_pos_smpl = smpl_parser.get_joints_verts(stand_pose_aa_smpl, shape_var, trans[0:1])
        root_pos_smpl = joint_pos_smpl[:, 0]
        joint_pos_smpl = scale_var * (joint_pos_smpl - root_pos_smpl) + root_pos_smpl

        # compute difference of key joints position between SMPL and robot
        if len(cfg.robot.extend_config) > 0:
            key_joint_pos_robot = fk_return_robot.global_translation_extend[
                :, :, key_joint_indices_robot
            ]
        else:
            key_joint_pos_robot = fk_return_robot.global_translation[:, :, key_joint_indices_robot]
        key_joint_pos_smpl = joint_pos_smpl[:, key_joint_indices_smpl]
        diff = key_joint_pos_robot - key_joint_pos_smpl

        loss = diff.norm(dim=-1).square().sum()

        progress.update(task, advance=1, iteration=i, loss=loss.item())

        loss.backward()
        optimizer.step()

    logger.success("Optimization finished.")
    logger.info(f"Final shape parameters: {shape_var.detach().cpu().numpy()}")
    logger.info(f"Final scale factor: {scale_var.detach().cpu().item()}")

    os.makedirs(f"data/{cfg.robot.humanoid_type}", exist_ok=True)
    joblib.dump(
        (shape_var.detach(), scale_var.detach()), f"data/{cfg.robot.humanoid_type}/smpl_shape.pkl"
    )

    if fitting_cfg.visualize:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 24})

        robot_key_joint_pos = (
            fk_return_robot.global_translation_extend[0, :, key_joint_indices_robot, :]
            .detach()
            .cpu()
            .numpy()
        )
        robot_key_joint_pos = robot_key_joint_pos - robot_key_joint_pos[:, 0:1]

        smpl_key_joint_pos = joint_pos_smpl[:, key_joint_indices_smpl].detach().cpu().numpy()
        smpl_key_joint_pos = smpl_key_joint_pos - smpl_key_joint_pos[:, 0:1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            robot_key_joint_pos[0, :, 0],
            robot_key_joint_pos[0, :, 1],
            robot_key_joint_pos[0, :, 2],
            label="Robot Key Joints",
            color="blue",
            s=100,
        )

        ax.scatter(
            smpl_key_joint_pos[0, :, 0],
            smpl_key_joint_pos[0, :, 1],
            smpl_key_joint_pos[0, :, 2],
            label="Fitted SMPL Key Joints",
            color="red",
            s=100,
        )

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-1, 1)

        ax.set_box_aspect([1, 1, 2])

        ax.set_xlabel("X", fontsize=18)
        ax.set_ylabel("Y", fontsize=18)
        ax.set_zlabel("Z", fontsize=18)

        ax.legend(fontsize=20)
        plt.show()


def get_joint_names(cfg: DataRetrieverCfg):
    """get the joint names for both robot and SMPL.
    Please note that the joint names for robot is actually the link names
    """
    robot_fk = HumanoidBatch(cfg.robot)
    joint_names_robot = robot_fk.body_names_augment
    joint_names_smpl = SMPL_BONE_ORDER_NAMES

    body_to_joint = robot_fk.mjcf_data["body_to_joint"]
    dof_names = []
    for body in robot_fk.body_names:
        if body in body_to_joint.keys():
            # if the body has a joint, append the joint name
            dof_names.append(body_to_joint[body])
    dof_names = dof_names[1:]  # remove the root joint

    return dof_names, joint_names_robot, joint_names_smpl


def fit_smpl_motion(cfg: DataRetrieverCfg, device: str = "cpu"):
    if cfg.motion_dataset is None:
        logger.warning("No motion dataset specified for fitting SMPL motion.")
        return

    all_files = glob.glob(f"{cfg.motion_dataset}/**/*.npz", recursive=True)
    logger.success(f"Found {len(all_files)} files in {cfg.motion_dataset}.")

    # mapping from motion_name to motion_file_path
    motion_path_dict = {}

    for f_path in all_files:
        motion_name = Path(f_path).stem
        # replace special characters with underscores
        motion_name = motion_name.replace("/", "_").replace(" ", "_").replace("-", "_")
        motion_path_dict[motion_name] = f_path

    motion_names = list(motion_path_dict.keys())

    num_jobs = 30
    smpl_model_dir = os.path.join(DIR_PATH, "smpl_model")
    chunk = np.ceil(len(motion_names) / num_jobs).astype(int)

    jobs = [motion_names[i : i + chunk] for i in range(0, len(motion_names), chunk)]
    jobs_args = [
        (smpl_model_dir, jobs[i], motion_path_dict, cfg, device) for i in range(len(jobs))
    ]

    if len(jobs_args) == 1:
        retarget_data_dict: Dict[str, RetargetedMotion] = retargetting(*jobs_args[0])
    else:
        try:
            pool = mp.Pool(num_jobs)
            retarget_data_dict_list = pool.starmap(retargetting, jobs_args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        retarget_data_dict = {}
        for retarget_data_dict_chunk in retarget_data_dict_list:
            retarget_data_dict.update(retarget_data_dict_chunk)

    dof_names, joint_names_robot, joint_names_smpl = get_joint_names(cfg)

    output_dict = {
        "dof_names": dof_names,
        "joint_names_robot": joint_names_robot,
        "joint_names_smpl": joint_names_smpl,
        "retarget_data": retarget_data_dict,
    }

    # debugging output
    print(f"Dof names: {dof_names}")
    print(f"Joint names (robot): {joint_names_robot}")
    print(next(iter(retarget_data_dict.values()))["joint_pos_robot"].shape)
    # debugging output
    print(f"Joint names (SMPL): {joint_names_smpl}")
    print(next(iter(retarget_data_dict.values()))["joint_pos_smpl"].shape)

    # save the retargeted data
    os.makedirs(f"data/{cfg.robot.humanoid_type}/retargeted", exist_ok=True)
    output_file_name = cfg.get("output_file_name", "retargeted_motion")
    output_file_path = f"data/{cfg.robot.humanoid_type}/retargeted/{output_file_name}.pkl"
    joblib.dump(output_dict, output_file_path)
    logger.info(
        f"Retargeted motion data saved to {output_file_path}. "
        f"Total {len(retarget_data_dict)} motions processed."
    )


def cli(cfg: DataRetrieverCfg, device: str, task: str):
    if task == "fitting_shape":
        fit_smpl_shape(cfg, device)
    elif task == "fitting_motion":
        fit_smpl_motion(cfg, device)
