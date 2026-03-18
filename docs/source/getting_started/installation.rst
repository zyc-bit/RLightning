Installation
============


RLightning uses ``uv`` for virtual environment and dependency management.


Install uv
----------

Please verify that ``uv`` is available on your system. If ``uv`` is not available,
please install it by following the official documentation:
https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

You can verify the installation by running:

.. code-block:: bash

   uv --version

.. note::

   The installation method may vary depending on your system.
   Please follow the official documentation to ensure a correct installation.

   For details on ``uv`` usage in projects, please refer to:
   https://docs.astral.sh/uv/guides/projects/


Download Repository
-------------------

Clone the project repository from Gitee:

.. code-block:: bash

   git clone https://gitee.pilab.org.cn/L2/yangzhenyu/RLightning.git
   cd RLightning


Install
-----------------------

Install RLightning:

.. code-block:: bash

   make install-dev

Or install directly using ``uv``:

.. code-block:: bash

   uv sync --extra dev

.. note::

   RLightning is not designed to be run as a standalone application. It is
   intended to be used as a core dependency within your own RL training project.
   To install RLightning as a dependency for your project, please refer to
   :doc:`build_your_own_rl` 

Simulator Backends
------------------

RLightning supports multiple simulators. Install the backends you need:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Backend
     - Install Command
     - Use Case
   * - ManiSkill
     - ``make install-maniskill``
     - Manipulation tasks (OpenVLA PPO)
   * - IsaacLab
     - ``make install-isaaclab``
     - Locomotion tasks (Humanoid Wholebody Control)
   * - MuJoCo
     - ``make install-mujoco``
     - Classic control and custom MuJoCo environments


Each backend creates an isolated virtual environment under ``.venvs/``.


Examples
----------------

Individual examples under ``examples/`` maintain their own dependencies with ``uv`` .
To set up an example:

.. code-block:: bash

   cd examples/<project_name>/
   uv sync

More details about running examples can be found in :doc:`quickstart`.

