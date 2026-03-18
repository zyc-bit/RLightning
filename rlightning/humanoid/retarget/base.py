from typing import Dict, Any, Sequence

import numpy as np

from rlightning.utils.config import Config


class Retargeter:

    class RetargeterCfg(Config):
        robot_xml_path: str = ""
        solver: str = ""
        damping: float = 0.0
        use_velocity_limit: bool = False
        ik_config_path: str = ""

    def __init__(self):
        self.cfg: Retargeter.RetargeterCfg = None
        self._init_viewer()
        self.paused = False

    def __call__(self, frames: Sequence[Any], extras: Dict[str, Any]):
        return self.retarget(frames, extras)

    def _init_viewer(self):
        self.enable_viewer = False  # self.config.viewer.enable_viewer
        # if self.enable_viewer:
        #     self.viewer = RobotMotionViewer(
        #         robot_config=self.config.robot,
        #         camera_follow=self.config.viewer.camera_follow,
        #         record_video=self.config.viewer.record_video,
        #         video_path=os.path.join(HydraConfig.get().runtime.output_dir, 'recording.mp4'),
        #         rate_limit=self.config.viewer.rate_limit,
        #         key_callback=self.key_callback
        #     )
        pass

    def retarget(self, frames: Sequence[Any], extras: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    # def update(self, extras):
    #     self.motion_fps = extras['fps']
    #     if self.enable_viewer:
    #         self.viewer.update(self.motion_fps)

    def finish(self):
        if self.enable_viewer:
            self.viewer.close()

    def key_callback(self, keycode):
        if chr(keycode) == " ":
            self.paused = not self.paused
