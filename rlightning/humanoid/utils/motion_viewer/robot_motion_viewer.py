import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
import numpy as np
import glob
import random
import pickle

from rich import print
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from rlightning.utils.logger import get_logger

logger = get_logger(__name__)


def load_motion_data(motion_file, quat_order="xyzw"):
    with open(motion_file, "rb") as f:
        motion_dict = pickle.load(f)
        if quat_order == "xyzw":
            motion_dict["root_rot"] = motion_dict["root_rot"][:, [3, 0, 1, 2]]
            if "obj_root_rot" in motion_dict:
                motion_dict["obj_root_rot"] = motion_dict["obj_root_rot"][:, [3, 0, 1, 2]]
    return motion_dict


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


class RobotMotionViewer:

    @property
    def current_motion(self) -> str:
        return self.motion_list[self.current_motion_idx]

    @property
    def is_paused(self) -> bool:
        return self.ispaused

    def set_paused(self, paused: bool):
        self.ispaused = paused

    def __init__(
        self,
        robot_config,
        motion_path: str,
        rate_limit: bool = True,
        camera_follow: bool = True,
        transparent_robot: int = 0,
        # video recording
        record_video: bool = False,
        video_path: str = None,
        video_width: int = 640,
        video_height: int = 480,
        video_fps: float = 30.0,
        log_dir: str = "./logs",
    ):

        if os.path.isfile(motion_path):
            motion_list = [motion_path]
        elif os.path.isdir(motion_path):
            motion_list = glob.glob(f"{motion_path}/**/*.pkl", recursive=True)

        self.motion_list = motion_list
        logger.info(f"[Viewer] Found {len(self.motion_list)} motion files in {motion_path}.")

        self.robot_config = robot_config
        self.log_dir = log_dir
        self.xml_path: str = robot_config.robot_xml_path
        self.camera_follow = camera_follow
        self.record_video = record_video
        self.transparent_robot = transparent_robot
        self.video_fps = video_fps
        self.video_path = video_path
        self.video_width = video_width
        self.video_height = video_height
        self.rate_limit = rate_limit
        self.step_counter = 0

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.setup()

    def setup(self):
        # create a new mjModel with xml model file
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.current_motion_data = None
        self.current_motion_idx = -1
        self.ispaused = True

        # the primary data structure that contains the time-varying state of the simulation
        self.data = mj.MjData(self.model)

        self.robot_base = self.robot_config.robot_base
        self.viewer_cam_distance = self.robot_config.viewer_cam_dist
        mj.mj_step(self.model, self.data)

        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback,
        )

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = self.transparent_robot

        if self.record_video:
            assert self.video_path is not None, "Please provide video path for recording"
            video_dir = os.path.dirname(self.video_path)

            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.video_fps)
            logger.info(f"[Viewer] Recording video to {self.video_path}")

            # Initialize renderer for video recording
            self.renderer = mj.Renderer(
                self.model, height=self.video_height, width=self.video_width
            )

    def reset(self, fps: float = 30.0):
        if self.record_video:
            self.motion_fps = self.video_fps
        else:
            self.motion_fps = fps

        self.frame_idx = 0

        self.current_motion_idx = random.randint(0, len(self.motion_list) - 1)
        self.current_motion_data = load_motion_data(self.motion_list[self.current_motion_idx])
        self.ispaused = False

        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.step_counter = 0

    def key_callback(self, key: str):
        keycode = chr(key).lower()

        if keycode == "r":
            logger.info("[Viewer] Resetting the simulation.")
            mj.mj_resetData(self.model, self.data)
            self.reset()
        elif keycode == "s":
            logger.info("[Viewer] Saving a screenshot.")
            self.save_screenshot()
        elif keycode == "q":
            logger.info("[Viewer] Quitting the viewer.")
            self.close()
        elif keycode == "p":
            self.record_video = not self.record_video
            logger.info(f"[Viewer] Toggled video recording to {self.record_video}.")
        elif keycode == " ":
            self.ispaused = not self.ispaused
            logger.info(f"[Viewer] Toggled pause to {self.ispaused}.")
        elif keycode == "c":
            self.camera_follow = not self.camera_follow
            logger.info(f"[Viewer] Toggled camera follow to {self.camera_follow}.")
        elif keycode == "n":
            self.next_motion()
            logger.info(f"[Viewer] Switch to motion: {self.current_motion}.")

    def save_screenshot(self):
        """Saving screenshot of current frame"""

        # Use renderer for proper offscreen rendering
        self.renderer.update_scene(self.data, camera=self.viewer.cam)
        img = self.renderer.render()

        datetime_str = time.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.log_dir, f"screenshot_{datetime_str}.png")

        imageio.imwrite(file_path, img)
        logger.info(f"[Viewer] Screenshot saved to {file_path}")

    def step(
        self,
        human_motion_data=None,
        show_human_body_name=False,
        human_point_scale=0.1,
        human_pos_offset=np.array([0.0, 0.0, 0]),
    ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.

        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """

        self.frame_idx = (self.frame_idx + 1) % len(self.current_motion_data["dof_pos"])

        motion_data = self.current_motion_data
        self.data.qpos[:3] = motion_data["root_pos"][self.frame_idx]
        self.data.qpos[3:7] = motion_data["root_rot"][
            self.frame_idx
        ]  # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = motion_data["dof_pos"][self.frame_idx]

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

    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
