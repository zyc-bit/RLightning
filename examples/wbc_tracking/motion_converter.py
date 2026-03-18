import argparse
import dataclasses
import glob
import os
import traceback
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Process retargeted motion files")

parser.add_argument("--input-dir", type=str, required=True, help="source motion dir")
parser.add_argument("--output-fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel environments")
parser.add_argument("--robot-type", type=str, default="g1", help="Type of humanoid robot")

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_slerp

from rlightning.humanoid.loader.lafan_loader import LafanLoader
from rlightning.humanoid.utils.kinematics_model import torch_utils
from rlightning.utils.logger import get_logger
from rlightning.utils.profiler import profiler

logger = get_logger(__name__)


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = dataclasses.MISSING


def load_lafan_motion(motion_file: str, output_dt: float):
    try:
        motion: LafanLoader.formatter_cls.Motion = joblib.load(motion_file)

        fps = motion.fps
        base_pos = torch.tensor(motion.root_pos, dtype=torch.float32)
        base_rot = torch.tensor(motion.root_rot, dtype=torch.float32)[:, [3, 0, 1, 2]]
        dof_pos = torch.tensor(motion.dof_pos, dtype=torch.float32)

        input_frames = base_pos.shape[0]
        input_dt = 1.0 / fps  # Assuming 30 fps input, adjust if needed
        duration = (input_frames - 1) * input_dt

        # Interpolate to output fps
        times = torch.arange(0, duration, output_dt, dtype=torch.float32)
        output_frames = times.shape[0]

        phase = times / duration
        index_0 = (phase * (input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(input_frames - 1))
        blend = phase * (input_frames - 1) - index_0

        motion_base_pos = base_pos[index_0] * (1 - blend.unsqueeze(1)) + base_pos[
            index_1
        ] * blend.unsqueeze(1)
        motion_base_rot = torch.zeros_like(base_rot[index_0])
        for i in range(base_rot[index_0].shape[0]):
            motion_base_rot[i] = quat_slerp(base_rot[index_0][i], base_rot[index_1][i], blend[i])
        motion_dof_pos = dof_pos[index_0] * (1 - blend.unsqueeze(1)) + dof_pos[
            index_1
        ] * blend.unsqueeze(1)

        # Compute velocities
        base_lin_vel = torch.gradient(motion_base_pos, spacing=output_dt, dim=0)[0]
        dof_vel = torch.gradient(motion_dof_pos, spacing=output_dt, dim=0)[0]
        q_prev, q_next = motion_base_rot[:-2], motion_base_rot[2:]
        q_rel = torch_utils.quat_mul(q_next, torch_utils.quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * output_dt)
        base_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

        logger.info(
            f"Motion loaded: {Path(motion_file).stem}, duration: {output_frames * output_dt:.2f}s"
        )

        return {
            "base_pos": motion_base_pos,
            "base_rot": motion_base_rot,
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "dof_pos": motion_dof_pos,
            "dof_vel": dof_vel,
            "output_frames": output_frames,
            "file_name": Path(motion_file).stem,
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        exit(1)


class BatchMotionConverter:

    def __init__(
        self,
        input_dir: str,
        output_fps: float,
        device: torch.device,
        num_envs: int,
        sim: sim_utils.SimulationContext,
        scene: InteractiveScene,
        joint_names: List[str],
    ):
        """Initialize converter for a batch of motions.

        Args:
            input_dir (str): Source motion directory
            output_fps (float): Target fps
            device (torch.device): Device type
            num_envs (int): Number of environments
            sim (sim_utils.SimulationContext): Simulation context
            scene (InteractiveScene): Interactive scene
            joint_names (List[str]): A list of joint names
        """

        self.input_dir = input_dir
        self.output_dir = os.path.join(self.input_dir, "wbc_tracking")

        os.makedirs(self.output_dir, exist_ok=True)

        self.output_fps = output_fps
        self.output_dt = 1.0 / self.output_fps
        self.device = device
        self.num_envs = num_envs
        self.current_batch_idx = 0
        self.batch_motions = []

        # Load all motions
        self._load_all_motions()
        self.num_motions = len(self.batch_motions)
        self.num_batches = (self.num_motions + num_envs - 1) // num_envs
        self.sim = sim
        self.scene = scene
        self.joint_names = joint_names

        logger.info(
            f"Loaded {self.num_motions} motions, will process in {self.num_batches} batches"
        )

    @profiler.timer_wrap("motion_loading", level="info")
    def _load_all_motions(self):
        """Load all motions into memory."""

        motion_files = glob.glob(f"{self.input_dir}/*.pkl", recursive=True)
        assert len(motion_files) > 0

        # XXX(ming): as torch's jitscript raises pickle error when backend is processing, thus we
        #   use threading as the backend, if you wanna speed up, you can choose to use non-jit function
        #   in the implementation of load_lafan_motion
        self.batch_motions = joblib.Parallel(n_jobs=-1, verbose=10, backend="threading")(
            joblib.delayed(load_lafan_motion)(motion_file, self.output_dt)
            for motion_file in motion_files
        )
        self.batch_motions = [
            motion
            for motion in self.batch_motions
            if motion is not None and motion["output_frames"] > 2
        ]
        self.batch_motions = sorted(self.batch_motions, key=lambda x: x["output_frames"])

    def get_current_batch(self):
        """Get the current batch of motions."""

        start_idx = self.current_batch_idx * self.num_envs
        end_idx = min((self.current_batch_idx + 1) * self.num_envs, self.num_motions)

        if start_idx >= self.num_motions:
            return None, None

        batch_motions = self.batch_motions[start_idx:end_idx]
        batch_size = len(batch_motions)

        # Pad batch if necessary
        if batch_size < self.num_envs:
            # Repeat the last motion to fill the batch
            padding_needed = self.num_envs - batch_size
            for _ in range(padding_needed):
                batch_motions.append(batch_motions[-1])

        return batch_motions, batch_size

    def next_batch(self):
        """Move to next batch."""

        self.current_batch_idx += 1
        return self.current_batch_idx < self.num_batches

    def run_simulator(self):
        """Convert original motions and save motion to wbc motion format"""

        robot: ArticulationCfg = self.scene["robot"]
        robot_joint_indexes = robot.find_joints(self.joint_names, preserve_order=True)[0]

        for batch_idx in range(self.num_batches):
            batch_motions, actual_batch_size = self.get_current_batch()
            if batch_motions is None:
                break

            logger.info(
                f"Processing batch {batch_idx + 1}/{self.num_batches} with {actual_batch_size} motions"
            )

            # Initialize data loggers for this batch
            batch_logs = {}
            for env_idx in range(actual_batch_size):
                motion_name = batch_motions[env_idx]["file_name"]
                batch_logs[env_idx] = {
                    "fps": self.output_fps,
                    "joint_pos": [],
                    "joint_vel": [],
                    "body_pos_w": [],
                    "body_quat_w": [],
                    "body_lin_vel_w": [],
                    "body_ang_vel_w": [],
                    "file_name": motion_name,
                }

            max_frames = max(
                motion["output_frames"] for motion in batch_motions[:actual_batch_size]
            )
            frame_counters = [0] * self.num_envs
            completed_envs = set()

            # Simulation loop for current batch
            while simulation_app.is_running() and len(completed_envs) < actual_batch_size:
                # Prepare batch data for current frame
                batch_base_pos = []
                batch_base_rot = []
                batch_base_lin_vel = []
                batch_base_ang_vel = []
                batch_dof_pos = []
                batch_dof_vel = []

                for env_idx in range(self.num_envs):
                    if (
                        env_idx < actual_batch_size
                        and frame_counters[env_idx] < batch_motions[env_idx]["output_frames"]
                    ):
                        motion = batch_motions[env_idx]
                        frame_idx = frame_counters[env_idx]

                        batch_base_pos.append(motion["base_pos"][frame_idx : frame_idx + 1])
                        batch_base_rot.append(motion["base_rot"][frame_idx : frame_idx + 1])
                        batch_base_lin_vel.append(
                            motion["base_lin_vel"][frame_idx : frame_idx + 1]
                        )
                        batch_base_ang_vel.append(
                            motion["base_ang_vel"][frame_idx : frame_idx + 1]
                        )
                        batch_dof_pos.append(motion["dof_pos"][frame_idx : frame_idx + 1])
                        batch_dof_vel.append(motion["dof_vel"][frame_idx : frame_idx + 1])

                        frame_counters[env_idx] += 1
                    else:
                        # Use last frame for completed or padded environments
                        if env_idx < actual_batch_size:
                            motion = batch_motions[env_idx]
                            last_idx = motion["output_frames"] - 1
                            batch_base_pos.append(motion["base_pos"][last_idx : last_idx + 1])
                            batch_base_rot.append(motion["base_rot"][last_idx : last_idx + 1])
                            batch_base_lin_vel.append(
                                motion["base_lin_vel"][last_idx : last_idx + 1]
                            )
                            batch_base_ang_vel.append(
                                motion["base_ang_vel"][last_idx : last_idx + 1]
                            )
                            batch_dof_pos.append(motion["dof_pos"][last_idx : last_idx + 1])
                            batch_dof_vel.append(motion["dof_vel"][last_idx : last_idx + 1])
                        else:
                            # For padded environments, use data from last actual motion
                            motion = batch_motions[actual_batch_size - 1]
                            last_idx = motion["output_frames"] - 1
                            batch_base_pos.append(motion["base_pos"][last_idx : last_idx + 1])
                            batch_base_rot.append(motion["base_rot"][last_idx : last_idx + 1])
                            batch_base_lin_vel.append(
                                motion["base_lin_vel"][last_idx : last_idx + 1]
                            )
                            batch_base_ang_vel.append(
                                motion["base_ang_vel"][last_idx : last_idx + 1]
                            )
                            batch_dof_pos.append(motion["dof_pos"][last_idx : last_idx + 1])
                            batch_dof_vel.append(motion["dof_vel"][last_idx : last_idx + 1])

                # Stack batch data
                base_pos = torch.cat(batch_base_pos, dim=0).to(self.sim.device)
                base_rot = torch.cat(batch_base_rot, dim=0).to(self.sim.device)
                base_lin_vel = torch.cat(batch_base_lin_vel, dim=0).to(self.sim.device)
                base_ang_vel = torch.cat(batch_base_ang_vel, dim=0).to(self.sim.device)
                dof_pos = torch.cat(batch_dof_pos, dim=0).to(self.sim.device)
                dof_vel = torch.cat(batch_dof_vel, dim=0).to(self.sim.device)

                # Set root state
                root_states = robot.data.default_root_state.clone()
                root_states[:, :3] = base_pos
                root_states[:, :2] += self.scene.env_origins[:, :2]
                root_states[:, 3:7] = base_rot
                root_states[:, 7:10] = base_lin_vel
                root_states[:, 10:] = base_ang_vel
                robot.write_root_state_to_sim(root_states)

                # Set joint state
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                joint_pos[:, robot_joint_indexes] = dof_pos
                joint_vel[:, robot_joint_indexes] = dof_vel
                robot.write_joint_state_to_sim(joint_pos, joint_vel)

                self.sim.render()
                self.scene.update(self.sim.get_physics_dt())

                # Log data for active environments
                for env_idx in range(actual_batch_size):
                    if frame_counters[env_idx] <= batch_motions[env_idx]["output_frames"]:
                        batch_logs[env_idx]["joint_pos"].append(
                            robot.data.joint_pos[env_idx, :].cpu().numpy().copy()
                        )
                        batch_logs[env_idx]["joint_vel"].append(
                            robot.data.joint_vel[env_idx, :].cpu().numpy().copy()
                        )
                        batch_logs[env_idx]["body_pos_w"].append(
                            robot.data.body_pos_w[env_idx, :].cpu().numpy().copy()
                        )
                        batch_logs[env_idx]["body_quat_w"].append(
                            robot.data.body_quat_w[env_idx, :].cpu().numpy().copy()
                        )
                        batch_logs[env_idx]["body_lin_vel_w"].append(
                            robot.data.body_lin_vel_w[env_idx, :].cpu().numpy().copy()
                        )
                        batch_logs[env_idx]["body_ang_vel_w"].append(
                            robot.data.body_ang_vel_w[env_idx, :].cpu().numpy().copy()
                        )

                # Check for completed environments
                for env_idx in range(actual_batch_size):
                    if (
                        env_idx not in completed_envs
                        and frame_counters[env_idx] >= batch_motions[env_idx]["output_frames"]
                    ):
                        completed_envs.add(env_idx)
                        # Save completed motion
                        motion_name = batch_logs[env_idx]["file_name"]
                        output_path = os.path.join(self.output_dir, f"{motion_name}.npz")

                        # Convert lists to numpy arrays
                        log_data = {}
                        for k in (
                            "joint_pos",
                            "joint_vel",
                            "body_pos_w",
                            "body_quat_w",
                            "body_lin_vel_w",
                            "body_ang_vel_w",
                        ):
                            log_data[k] = np.stack(batch_logs[env_idx][k], axis=0)
                        log_data["fps"] = batch_logs[env_idx]["fps"]

                        np.savez(output_path, **log_data)
                        logger.info(
                            f"{len(completed_envs)} / {actual_batch_size}; Saved motion: {output_path}."
                        )

            if not self.next_batch():
                break


def load_custom_npz(file_path: str, arr_filter: str = "data") -> Dict[str, np.ndarray]:
    data_dict = {}

    with zipfile.ZipFile(file_path, "r") as zf:
        for file_name in zf.namelist():
            if not file_name.endswith(".npy"):
                continue

            with zf.open(file_name) as f:
                npy_bytes = f.read()

            arr = np.load(BytesIO(npy_bytes))

            arr_name = file_name.rsplit(".npy", 1)[0]
            data_dict[arr_name] = arr

    return data_dict[arr_filter]


if __name__ == "__main__":
    if args_cli.robot_type == "g1":
        from wbc_tracking.robots.g1 import G1_CYLINDER_CFG as ROBOT_CFG
    else:
        raise NotImplementedError

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = sim_utils.SimulationContext(sim_cfg)

    # Design scene with multiple environments
    scene_cfg = ReplayMotionsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene_cfg.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    converter = BatchMotionConverter(
        args_cli.input_dir,
        args_cli.output_fps,
        args_cli.device,
        args_cli.num_envs,
        sim,
        scene,
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )

    converter.run_simulator()
