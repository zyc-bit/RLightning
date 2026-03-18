import dataclasses
from typing import List, Literal

import numpy as np

from rlightning.utils.config import Config


@dataclasses.dataclass
class SMPLXMotion:
    pose_aa: np.ndarray
    gender: str
    trans: np.ndarray
    betas: np.ndarray
    fps: float

    @classmethod
    def from_dict(cls, data: dict) -> "SMPLXMotion":
        return cls(
            pose_aa=data["pose_aa"],
            gender=data["gender"],
            trans=data["trans"],
            betas=data["betas"],
            fps=data["fps"],
        )


@dataclasses.dataclass
class RetargetedMotion:
    root_pos: np.ndarray
    root_rot: np.ndarray
    pose_aa: np.ndarray
    dof_pos: np.ndarray
    fps: float
    joint_pos_robot: np.ndarray
    joint_pos_smpl: np.ndarray

    @classmethod
    def from_dict(cls, data: dict) -> "RetargetedMotion":
        return cls(
            root_pos=data["root_pos"],
            root_rot=data["root_rot"],
            pose_aa=data["pose_aa"],
            dof_pos=data["dof_pos"],
            fps=data["fps"],
            joint_pos_robot=data["joint_pos_robot"],
            joint_pos_smpl=data["joint_pos_smpl"],
        )


class ExtendConfig(Config):
    """Configuration for extending the humanoid model with additional joints."""

    joint_name: str
    """Name of the joint to be added."""

    parent_name: str
    """Name of the parent joint to which the new joint will be attached."""

    pos: list[float]
    """Local position of the new joint relative to its parent."""

    rot: list[float]
    """Local rotation of the new joint relative to its parent."""


class HumanoidBatchCfg(Config):
    """Configuration for initializing HumanoidBatch."""

    asset_file_path: str
    """File path to the MJCF asset."""

    humanoid_type: str
    """Type of the humanoid model."""

    joint_matches: list[tuple[str, str]]
    smpl_pose_modifier: dict[str, str]

    extend_configs: list[ExtendConfig] = None
    """List of configurations for extending the humanoid model with additional joints."""


class SMPLShapeFittingCfg(Config):
    train_iterations: int = 3000
    learning_rate: float = 0.1
    visualize: bool = False


class DataRetrieverCfg(Config):
    robot: HumanoidBatchCfg
    gender: Literal["neutral", "male", "female"] = "neutral"
    shape_fitting: SMPLShapeFittingCfg = SMPLShapeFittingCfg()
    motion_dataset: str = None


class RetargetCfg(Config):
    robot: HumanoidBatchCfg
    loader: str
    loader_cfg: Config
    motion_dataset: str
    retargetting: str
    formatter: str


class RetargetGroup(Config):

    retarget_list: List[RetargetCfg]
    """A list of retargetting tasks"""
