Build Your Own Project
=========================

``./examples/algorithm_template/`` provides a minimal project skeleton for
implementing a custom algorithm on RLightning. Copy it as the starting point
for your own RL project:

.. code-block:: bash

   cp -r examples/algorithm_template/ /path/to/your/project
   cd /path/to/your/project
   uv sync

The template follows a three-file layout:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File / Directory
     - Purpose
   * - ``train.py``
     - Entry point. Calls builders to assemble components, then runs the engine.
   * - ``conf/``
     - Hydra config directory. ``train_algo.yaml`` selects the engine and wires
       sub-configs for env, policy, buffer, cluster, train, and log.
   * - ``launch_train.sh``
     - Launcher script. Starts a local Ray cluster and invokes ``train.py``.
   * - ``pyproject.toml``
     - Project dependencies managed by ``uv``. You can add dependencies as needed.

A minimal ``train.py`` follows three steps:

.. code-block:: python

   from pathlib import Path
   from rlightning.utils.config import MainConfig
   from rlightning.utils.launch import launch
   from rlightning.utils.builders import (
       build_env_group, build_policy_group, build_data_buffer, build_engine
   )

   def main(config: MainConfig):
       env_group = build_env_group(config.env)
       policy_group = build_policy_group(config.policy.type, config.policy, config.cluster)
       buffer = build_data_buffer(config.buffer.type, config.buffer)
       engine = build_engine(config, env_group, policy_group, buffer)
       engine.run()

   if __name__ == "__main__":
       launch(main_func=main, config_path=Path(__file__).parent / "conf")

Launch training:

.. code-block:: bash

   bash launch_train.sh

See :doc:`../user_guide/build_your_own/customize_policy` for a step-by-step
guide to implementing your policy.
