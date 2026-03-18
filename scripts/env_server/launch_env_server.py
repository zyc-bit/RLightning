import traceback

import hydra
from omegaconf import DictConfig

from rlightning.env import ENV_MAP
from rlightning.env.remote_env.env_client import ZMQEnvServer
from rlightning.utils.logger import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="conf", version_base=None)
def main(cfg: DictConfig):
    """Entry point"""
    hostname = cfg.client.hostname
    port = cfg.client.port

    # build env
    env_cls = ENV_MAP[cfg.backend]
    env = env_cls(cfg)
    env_server = ZMQEnvServer(env, hostname, port)
    env_server.connect()

    try:
        env_server.run()
    except KeyboardInterrupt:
        logger.warning("KeyboardInterruption detected, close connection ...")
    except Exception:
        logger.exception(traceback.format_exc())
    finally:
        env_server.close()

    logger.info(f"{hostname}:{port}")


if __name__ == "__main__":
    main()  # pylint: disable=E1120
