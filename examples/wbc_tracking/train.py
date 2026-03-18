import logging
from pathlib import Path

from utils import episode_postprocess_fn

from rlightning.utils import builders
from rlightning.utils.config import MainConfig
from rlightning.utils.launch import launch

logger = logging.getLogger(__name__)


def main(config: MainConfig):  # pylint: disable=C0116

    # build env group
    env_group = builders.build_env_group(config.env)

    # build policies group
    policy_group = builders.build_policy_group(
        policy_cls=config.policy.type,
        policy_cfg=config.policy,
        cluster_cfg=config.cluster,
        backend="nccl",
    )

    buffer = builders.build_data_buffer(
        buffer_cls=config.buffer.type,
        buffer_cfg=config.buffer,
        postprocess_fn=episode_postprocess_fn,
    )

    engine = builders.build_engine(config, env_group, policy_group, buffer)

    engine.run()

    try:
        env_group.close()
    except Exception as e:  # pylint: disable=broad-except
        # IsaacLab distributed env workers sometimes die before close() completes
        # (Ray ActorDiedError / SYSTEM_ERROR). Training has already finished at this
        # point, so treat this as a non-fatal cleanup warning.
        logger.warning("env_group.close() raised a non-fatal cleanup error: %s", e)


if __name__ == "__main__":
    launch(main_func=main, config_path=Path(__file__).parent / "conf")
