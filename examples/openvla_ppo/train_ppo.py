"""Training entry point for OpenVLA PPO on ManiSkill environments."""

from pathlib import Path

from examples.openvla_ppo.utils import (
    env_preprocess_fn,
    env_ret_preprocess_fn,
    episode_postprocess_fn,
    policy_resp_preprocess_fn,
)
from rlightning.utils.builders import (
    build_data_buffer,
    build_engine,
    build_env_group,
    build_policy_group,
)
from rlightning.utils.config.config import MainConfig
from rlightning.utils.launch import launch


def main(config: MainConfig) -> None:
    """Run PPO training for OpenVLA.

    Args:
        config: Main configuration object.
    """

    # build env group
    env_group = build_env_group(config.env, preprocess_fn=env_preprocess_fn)

    # build policy group
    policy_group = build_policy_group(
        policy_cls=config.policy.type,
        policy_cfg=config.policy,
        cluster_cfg=config.cluster,
    )

    # build data buffer
    rollout_buffer = build_data_buffer(
        buffer_cls=config.buffer.type,
        buffer_cfg=config.buffer,
        env_ret_preprocess_fn=env_ret_preprocess_fn,
        policy_resp_preprocess_fn=policy_resp_preprocess_fn,
        postprocess_fn=lambda episode: episode_postprocess_fn(config.policy, episode),
    )

    engine = build_engine(
        config=config,
        env_group=env_group,
        policy_group=policy_group,
        buffer=rollout_buffer,
    )
    engine.run()


if __name__ == "__main__":
    launch(main_func=main, config_path=Path(__file__).parent / "conf")
