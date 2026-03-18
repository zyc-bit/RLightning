from pathlib import Path

from rlightning.utils.config import MainConfig
from rlightning.utils.launch import launch


def main(config: MainConfig):
    # STEP1: build env, buffer, policy groups
    # STEP2: build engine
    # STEP3: engine.run()
    pass


if __name__ == "__main__":
    launch(main_func=main, config_path=Path(__file__).parent / "conf")
