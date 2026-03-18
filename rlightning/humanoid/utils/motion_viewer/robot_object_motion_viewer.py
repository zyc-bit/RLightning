import numpy as np
import mujoco as mj

from pathlib import Path
from scipy.spatial.transform import Rotation as R
from easydict import EasyDict

from rlightning.utils.logger import get_logger
from .robot_motion_viewer import RobotMotionViewer

logger = get_logger(__name__)


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1


class RobotObjectMotionViewer(RobotMotionViewer):

    def setup(self):
        self.reset()

        # insert object to robot xml
        self.object_name = self.object_config.name

        orig_xml_path = self.robot_config.robot_xml_path
        with open(orig_xml_path, "r") as f:
            robot_xml = f.read()

        asset_point = robot_xml.find("</asset>")
        scale = float(self.object_config.scale[0])
        insertion_content = f'  <mesh name="{self.object_config.name}_mesh" file="{self.object_config.mesh}" scale="{scale} {scale} {scale}"/>\n '
        robot_xml = robot_xml[:asset_point] + insertion_content + robot_xml[asset_point:]

        world_body_point = robot_xml.find("</worldbody>")
        insertion_content = f"""  <body name="{self.object_config.name}">
    <freejoint name="{self.object_config.name}_joint"/>
    <geom type="mesh" mesh="{self.object_config.name}_mesh" rgba="0.7 0.7 0.7 1" mass="0.1"/>
</body>\n  """
        robot_xml = robot_xml[:world_body_point] + insertion_content + robot_xml[world_body_point:]

        orig_xml_path = Path(orig_xml_path)
        tmp_xml_dir = orig_xml_path.parent
        tmp_xml_name = orig_xml_path.stem
        self.tmp_xml_path = str(tmp_xml_dir / f"{tmp_xml_name}_{self.object_config.name}.xml")

        logger.info(f"[Viewer] Robot-Object description file saved to {self.tmp_xml_path}")

        with open(self.tmp_xml_path, "w") as f:
            f.write(robot_xml)

        self.robot_config.robot_xml_path = self.tmp_xml_path

        super().setup()

    def reset(self, fps=30):
        super().reset(fps)

        if "obj_name" not in self.current_motion_data:
            raise ValueError("No object data found in the motion data!")
        self.object_config = EasyDict(
            {
                "name": self.current_motion_data["obj_name"],
                "mesh": self.current_motion_data["obj_mesh"],
                "scale": self.current_motion_data["obj_scale"],
            }
        )

    def step(
        self,
        human_motion_data=None,
        show_human_body_name=False,
        human_point_scale=0.1,
        human_pos_offset=np.array([0.0, 0.0, 0]),
        obj_pos=None,
        obj_rot=None,
    ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.

        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """

        self.frame_idx += 1

        motion_data = self.current_motion_data
        robot_joint_num = motion_data["dof_pos"].shape[-1]

        self.data.qpos[:3] = motion_data["root_pos"][self.frame_idx]
        self.data.qpos[3:7] = motion_data["root_rot"][
            self.frame_idx
        ]  # quat need to be scalar first! for mujoco
        self.data.qpos[7 : 7 + robot_joint_num] = motion_data["dof_pos"][self.frame_idx]

        if obj_pos is not None:
            obj_joint_id = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_JOINT, f"{self.object_name}_joint"
            )
            if obj_joint_id == -1:
                raise ValueError(f"Object {self.object_name} not found!")

            start_idx = 7 + robot_joint_num + (obj_joint_id - robot_joint_num - 1) * 7

            self.data.qpos[start_idx : start_idx + 3] = obj_pos
            self.data.qpos[start_idx + 3 : start_idx + 7] = obj_rot

        mj.mj_forward(self.model, self.data)

        if self.camera_follow:
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.viewer.cam.distance = self.viewer_cam_distance
            self.viewer.cam.elevation = -10  # face, slightly down upon
            # self.viewer.cam.azimuth = 180    # face forward

        if human_motion_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0
            # Draw the task targets for reference
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None,
                )

        self.viewer.sync()
        if self.rate_limit:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)
