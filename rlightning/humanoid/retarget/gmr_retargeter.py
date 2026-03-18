from typing import Any, Sequence, Dict

import mink
import numpy as np
import mujoco as mj

from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from rlightning.utils.logger import get_logger
from rlightning.utils.config import Config
from .base import Retargeter

logger = get_logger(__name__)


class GmrRetargeter(Retargeter):

    class RetargeterCfg(Config):
        robot_xml_path: str = ""
        ik_config_path: str = ""
        solver: str = "daqp"
        damping: float = 0.5
        use_velocity_limit: bool = False

    def __init__(self, config: RetargeterCfg = None, **kwargs):
        super().__init__()

        if config is None:
            config = self.RetargeterCfg(**kwargs)

        self.cfg = config
        self.solver = config.solver
        self.damping = config.damping

        logger.debug(f"[Retargeter] Loading robot model from {config.robot_xml_path}")

        self.xml_file = config.robot_xml_path
        self.model = mj.MjModel.from_xml_path(self.xml_file)

        logger.debug(f"[Retargeter] Robot Degrees of Freedom names and their order:")

        self.robot_dof_names = {}
        for i in range(self.model.nv):
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            logger.debug(f"[Retargeter]: Dof {i}: {dof_name}")

        logger.debug("[Retargeter] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            logger.debug(f"[Retargeter] Body ID {i}: {body_name}")

        logger.debug("[Retargeter] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            logger.debug(f"[Retargeter] Motor ID {i}: {motor_name}")

        ik_config = OmegaConf.load(config.ik_config_path)
        logger.debug(f"[Retargeter] Using IK config: {ik_config}")

        self.human_height_assumption = ik_config["human_height_assumption"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_scale_table_orig = ik_config["human_scale_table"]

        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        # self.use_damping = ik_config.get('use_damping', False)
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if config.use_velocity_limit:
            logger.info(f"[Retargeter] Velocity limit activated!")
            VELOCITY_LIMITS = {k: np.pi / 4 for k in self.robot_motor_names.keys()}
            logger.debug(f"[Retargeter] Velocity limits: {VELOCITY_LIMITS}")
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))

        self._setup_retarget_configuration()

        self.ground_offset = 0.0

    def _setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)

        self.tasks1 = []
        self.tasks2 = []

        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(rot_offset, scalar_first=True)
                self.tasks1.append(task)
                self.task_errors1[task] = []

        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(rot_offset, scalar_first=True)
                self.tasks2.append(task)
                self.task_errors2[task] = []

        # self.damping_task = mink.DampingTask(self.model,cost=5.0)

    def retarget(self, frames: Sequence[Any], extras: Dict[str, Any]) -> np.ndarray:
        # adjust the human scale table
        self.human_scale_table = {}

        if "disable_scale_table" in extras and extras["disable_scale_table"]:
            logger.info(
                f"[Retargeter] You have enabled optimized shape; The defined scale table is disabled!"
            )
            for key in self.human_scale_table_orig.keys():
                self.human_scale_table[key] = 1.0

        else:
            # compute the scale ratio based on given human height and the assumption in the IK config
            if "actual_human_height" in extras:
                ratio = extras["actual_human_height"] / self.human_height_assumption
            else:
                ratio = 1.0

            for key in self.human_scale_table_orig.keys():
                self.human_scale_table[key] = self.human_scale_table_orig[key] * ratio

        qpos_list = []
        for frame_idx in range(len(frames)):
            frame = frames[frame_idx]
            qpos = self.retarget_frame(frame)
            if self.enable_viewer:
                self.viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=self.scaled_human_data,
                )
            qpos_list.append(qpos.copy())
            while self.paused:
                pass
        qpos_list = np.array(qpos_list)
        return qpos_list

    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(
            human_data, self.human_root_name, self.human_scale_table
        )
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

    def retarget_frame(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                # if self.use_damping:
                #     vel1 = mink.solve_ik(
                #         self.configuration, [*self.tasks1, self.damping_task], dt, self.solver, self.damping, self.ik_limits
                #     )
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                # if self.use_damping:
                #     vel2 = mink.solve_ik(
                #         self.configuration, [*self.tasks2, self.damping_task], dt, self.solver, self.damping, self.ik_limits
                #     )
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)

                next_error = self.error2()
                num_iter += 1

        return self.configuration.data.qpos.copy()

    def error1(self):
        return np.linalg.norm(
            np.concatenate([task.compute_error(self.configuration) for task in self.tasks1])
        )

    def error2(self):
        return np.linalg.norm(
            np.concatenate([task.compute_error(self.configuration) for task in self.tasks2])
        )

    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [
                np.asarray(human_data[body_name][0]),
                np.asarray(human_data[body_name][1]),
            ]
        return human_data

    def scale_human_data(self, human_data, human_root_name, human_scale_table):

        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]

        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos

        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (
                    human_data[body_name][0] - root_pos
                ) * human_scale_table[body_name]

        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (
                human_data_local[body_name] + scaled_root_pos,
                human_data[body_name][1],
            )

        return human_data_global

    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(
                scalar_first=True
            )
            offset_human_data[body_name][1] = updated_quat

            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)

            offset_human_data[body_name][0] = pos + global_pos_offset

        return offset_human_data

    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = (
                pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
            )
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
