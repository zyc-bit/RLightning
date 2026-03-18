"""Entry for launch RLightning experiments.

This module provides the main entry point for launching RLightning RL experiments,
handling configuration loading, logging setup, and Ray cluster initialization.
"""

import os
from pathlib import Path
from typing import Callable, Union

import hydra
import ray
from omegaconf import DictConfig

from rlightning.utils.config import (
    ClusterConfig,
    MainConfig,
    validate_config_for_placement,
)
from rlightning.utils.logger import get_logger, setup_logger
from rlightning.utils.placement import GlobalResourceManager
from rlightning.utils.placement.scheduling import setup_component_scheduling
from rlightning.utils.registry import load_modules_from_config
from rlightning.utils.utils import InternalFlag


def launch(main_func: Callable[[MainConfig], None], config_path: Union[str, Path]) -> None:
    """Launch a RLightning experiment with the given main function.

    This function handles all the boilerplate for launching experiments:
    - Hydra configuration loading.
    - Logging setup.
    - Ray cluster initialization (for distributed mode).
    - Graceful shutdown.

    Args:
        main_func: The user-defined main function that takes a MainConfig
            and runs the experiment logic.
        config_path: Path to the Hydra configuration directory.

    Example:
        >>> def main(config: MainConfig):
        ...     # Your experiment logic here
        ...     pass
        >>> launch(main_func=main, config_path="./conf")
    """
    config_path = str(config_path)

    @hydra.main(config_path=config_path, version_base=None)
    def entrypoint(cfg: DictConfig) -> None:
        """Hydra entry point for configuration loading.

        This function is decorated by @hydra.main and handles following tasks:
        - Converts the loaded DictConfig to MainConfig, which designed specifically for RLightning.
        - Sets up internal environment variables based on configuration flags.
        - Logger and module initialization.
        - Ray cluster connection (if distributed mode).
        - Calls the user-defined main function with the prepared configuration.
        """
        # capture hydra output dir and job name for downstream workers
        try:
            from hydra.core.hydra_config import HydraConfig

            hydra_cfg = HydraConfig.get()
            hydra_output_dir = Path(hydra_cfg.runtime.output_dir)
            hydra_job_name = hydra_cfg.job.name
            log_file = hydra_output_dir / f"{hydra_job_name}.log"
            os.environ["RLIGHTNING_LOG_FILE"] = str(log_file)
        except Exception:
            # If HydraConfig is unavailable, fall back to not setting the env
            logger.warning("HydraConfig is unavailable, will not set RLIGHTNING_LOG_FILE")

        # convert to built-in Config
        cfg = MainConfig.from_omegaconf(cfg)

        # setup internal env variables
        os.environ["RLIGHTNING_DEBUG"] = "1" if cfg.debug else "0"
        os.environ["RLIGHTNING_VERBOSE"] = "1" if cfg.verbose else "0"

        if cfg.cluster is not None:
            os.environ["RLIGHTNING_REMOTE_TRAIN"] = "1" if cfg.cluster.remote_train else "0"
            os.environ["RLIGHTNING_REMOTE_EVAL"] = "1" if cfg.cluster.remote_eval else "0"
            os.environ["RLIGHTNING_REMOTE_STORAGE"] = "1" if cfg.cluster.remote_storage else "0"
            os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if cfg.cluster.remote_env else "0"

        # setup hook that’s called after workers start and before Tasks and Actors are scheduled
        def setup_func() -> None:
            """Configure logging, module registration, and monkey patches."""
            # set logging handlers
            setup_logger(cfg.log)
            load_modules_from_config(cfg)

        setup_func()

        logger = get_logger(__name__)

        logger.info(f"--- Full Configuration ---\n{cfg.to_yaml()}")

        if cfg.cluster is not None:
            logger.info("Running in distributed mode.")
            # Connect to ray cluster
            if not ray.is_initialized():
                runtime_env = {
                    "env_vars": {
                        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                        "RAY_DEBUG": "1" if cfg.debug else "0",
                        "RLIGHTNING_LOG_FILE": os.environ.get("RLIGHTNING_LOG_FILE", ""),
                        **InternalFlag.get_env_vars(),
                    },
                    "worker_process_setup_hook": setup_func,
                }
                try:
                    if cfg.cluster.ray_address == "auto":
                        logger.info("Try to auto detect ray cluster and connect")
                    else:
                        logger.info(f"Connecting to ray cluster at {cfg.cluster.ray_address}")
                    ray.init(cfg.cluster.ray_address, runtime_env=runtime_env)
                except ConnectionError as e:
                    logger.exception("Failed to connect to ray cluster.")
                    raise e

                logger.info("Connected to Ray Cluster.")

            # determine the scheduling of the workers
            scheduling = setup_component_scheduling(cfg)

            # setup global resource manager
            global_resource_manager = GlobalResourceManager.get_instance()
            global_resource_manager.initialize(cfg.cluster, scheduling, config_path + "/cluster")

            # validate the main config for placement strategy
            cfg = validate_config_for_placement(cfg)
        else:
            logger.info("Running in local mode.")
            cfg.cluster = ClusterConfig(
                train_worker_num=1,
                eval_worker_num=1,
                train_each_gpu_num=1.0,
                eval_each_gpu_num=1.0,
            )
        cfg.policy.train_config = cfg.train

        # main func
        try:
            main_func(cfg)
        except Exception as e:
            raise e
        finally:
            # close progress
            if InternalFlag.VERBOSE:
                from rlightning.utils.progress import get_progress

                progress = get_progress()
                progress.stop()

            if cfg.cluster is not None:
                ray.shutdown()

    entrypoint()  # pylint: disable=E1120
