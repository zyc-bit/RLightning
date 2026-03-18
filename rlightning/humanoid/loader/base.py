import concurrent.futures
import glob
import os
import pickle
import random
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

from rlightning.humanoid.formatter import Formatter
from rlightning.humanoid.retarget import Retargeter
from rlightning.utils.config import Config
from rlightning.utils.logger import get_logger

logger = get_logger(__name__)


class Mode(Enum):
    file = 0
    dir = 1


class LoaderCfg(Config):

    motion_cls: str = ""

    data_path: str = ""

    data_format: Literal[".bvh", ".npz"] = ".npz"
    """Indicating the acceptable file format"""

    overwrite: bool = False
    """When overwrite is True, the loader ignores existing files"""

    retargeter: Retargeter.RetargeterCfg = Retargeter.RetargeterCfg()
    formatter: Formatter.FormatterCfg = Formatter.FormatterCfg()


Frame = List[Any]


class MotionLoader:

    config: LoaderCfg
    formatter_cls = Formatter
    retargeter_cls = Retargeter

    def __init__(self, config: LoaderCfg):
        self.config = config
        self.format = config.data_format
        self.overwrite = self.config.get("overwrite", False)

        self.retargeter = self.retargeter_cls(**config.retargeter.to_dict())
        self.formatter = self.formatter_cls(**config.formatter.to_dict())

        self._files = []
        self._retargted_motion_files = []
        self._num_motions = 0

    @property
    def files(self) -> List[str]:
        """The list of motion file paths."""

        return self._files

    @property
    def retargted_motion_files(self) -> Sequence[Any]:
        return self._retargted_motion_files

    @property
    def num_motions(self) -> int:
        """The amount of motions"""

        return self._num_motions

    def prepare(self, data_path: str = None):
        """Prepare retargeted motions

        Args:
            data_path (str, optional): Raw motion path, chould be a directory. Defaults to None.
        """

        self.load(data_path)
        self.retarget()

    def load(self, data_path: str = None):
        """Load raw motions from a given file/directory.

        Args:
            data_path (str): File or directory path.
        """

        self.data_path = data_path or self.config.data_path

        if os.path.isdir(self.data_path):
            self.mode = Mode.dir
            self._files = glob.glob(f"{self.data_path}/**/*{self.format}", recursive=True)
            self._num_motions = len(self._files)
        elif os.path.isfile(self.data_path):
            assert (
                os.path.splitext(self.data_path)[-1] == self.format
            ), f"You are using the data loader for {self.format} format but the data you give is {self.data_path.suffix} format!"
            self.mode = Mode.file
            self._files = [self.data_path]
            self._num_motions = 1
        else:
            raise Exception(f"Check your data path: {self.data_path}")

        logger.info(f"[Loader] Total number of motions: {self._num_motions}")

    def __iter__(self):
        """Return a tuple of retargeted file path, frames and extras of parsed raw motion

        Yields:
            Tuple[str, list, dict]: A list of ...
        """

        self.current_idx = 0

        if self.mode == Mode.file:
            retargeted_dir_path = os.path.dirname(self._files[0])
        else:
            dir_path = os.path.dirname(self._files[0])
            retargeted_dir_path = os.path.join(dir_path, "retargeted")
            if not os.path.exists(retargeted_dir_path):
                os.makedirs(retargeted_dir_path)

        while self.current_idx < self._num_motions:
            try:
                sample_path = self._files[self.current_idx]
                f_name = os.path.basename(sample_path)
                if self.mode == Mode.file:
                    retargeted_f_path = os.path.join(retargeted_dir_path, f"retargeted_{f_name}").replace(
                        self.format, ".pkl"
                    )
                else:
                    retargeted_f_path = os.path.join(retargeted_dir_path, f_name).replace(self.format, ".pkl")
                if os.path.exists(retargeted_f_path) and not self.overwrite:
                    logger.warning(f"[Loader] {retargeted_f_path} already exists!")
                    self.current_idx += 1
                    self._retargted_motion_files.append(retargeted_f_path)
                    continue

                frames, extras = self._load_sample(sample_path)
                yield retargeted_f_path, frames, extras

                self.current_idx += 1
            except Exception as e:
                logger.warning(
                    f"[Loader] File {self._files[self.current_idx]} is broken! Skip it for error: {traceback.format_exc()}"
                )
                self.current_idx += 1

    def sample(self, n: int = 1) -> List[Formatter.Motion]:
        """Random sample a motion"""

        f_paths = random.sample(self.retargted_motion_files, n)
        motions = []
        for f_path in f_paths:
            with open(f_path, "rb") as f:
                motion = pickle.load(f)
                motions.append(motion)
        return motions

    def _load_sample(self, f_path: Path) -> Tuple[Frame, Dict[str, Any]]:
        """Loading motion from a given file path, return a tuple of extracted frames and the corresponding information.

        Args:
            f_path (Path): Motoin file path

        Raises:
            NotImplementedError: Not implemented error.

        Returns:
            Tuple[Frame, Dict[str, Any]]: A tuple of Frames and a dict of extracting information.
        """

        raise NotImplementedError

    def retarget_single(self, target_path: str, frames: Sequence[Dict[str, Any]], extras: Dict[str, Any]):
        try:
            qpos_list = self.retargeter(frames, extras)
            formatted_results = self.formatter(qpos_list, extras)
            with open(target_path, "wb") as f:
                pickle.dump(formatted_results, f)
            is_ok = True
        except Exception as e:
            is_ok = False
            logger.error(f"failed on path: {target_path}, detailed as: {traceback.format_exc()}")

        return is_ok

    def retarget(self, num_workers: int = None):
        """Execute retargeting in multi-process.

        If there are existing files and overwrite is set to True, we jump it.

        Args:
            num_workers (int, optional): The parallel worker nums. Defaults to None.
        """
        from rlightning.utils.progress import get_progress

        num_workers = num_workers or os.cpu_count() // 2

        progress = get_progress()
        success_num = 0
        fail_num = 0
        progress_task = progress.add_task(
            "[green]Retargeting ...",
            total=self._num_motions,
            success=success_num,
            fail=fail_num,
            waiting=self._num_motions - success_num - fail_num,
        )

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            tasks = {}
            for target_f_path, frames, extras in self:
                future = executor.submit(self.retarget_single, target_path=target_f_path, frames=frames, extras=extras)
                tasks[future] = target_f_path
                progress.update(
                    progress_task,
                    advance=0,
                    success=success_num,
                    fail=fail_num,
                    waiting=self._num_motions - success_num - fail_num,
                )
                dones = []
                try:
                    for future in concurrent.futures.as_completed(tasks, timeout=1):
                        f_path = tasks[future]
                        try:
                            result = future.result()
                            success_num += 1
                            self._retargted_motion_files.append(f_path)
                        except Exception as e:
                            fail_num += 1
                        progress.update(
                            progress_task,
                            advance=1,
                            success=success_num,
                            fail=fail_num,
                            waiting=self._num_motions - success_num - fail_num,
                        )
                        dones.append(future)
                except TimeoutError:
                    pass

                # remove completed tasks
                for k in dones:
                    tasks.pop(k)

            for future in concurrent.futures.as_completed(tasks):
                f_path = tasks[future]
                try:
                    result = future.result()
                    success_num += 1
                except Exception as e:
                    fail_num += 1
                progress.update(
                    progress_task,
                    advance=1,
                    success=success_num,
                    fail=fail_num,
                    waiting=self._num_motions - success_num - fail_num,
                )
            progress.update(progress_task, description="[bold green]Retargeting completed")
