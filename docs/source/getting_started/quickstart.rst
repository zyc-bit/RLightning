Quickstart
==========

Here provides some examples as a quickstart guide to get you up and running with RLightning.

Each example maintains its own virtual environment under ``examples/<project_name>/.venv``.


OpenVLA PPO
-----------

PPO-based fine-tuning of OpenVLA (a 7B vision-language-action model) on
ManiSkill manipulation tasks.

**1. Environment setup**

.. code-block:: bash

   cd examples/openvla_ppo
   uv sync

.. note::

   **Python 3.11 is required** (the ``flash_attn`` pre-built wheel only
   supports cp311). Pin the version in ``pyproject.toml``:

   .. code-block:: toml

      requires-python = "==3.11.*"

**2. Download model weights**

Install the download tool and fetch the checkpoint (China mainland users
can set ``HF_ENDPOINT`` for acceleration):

.. code-block:: bash

   uv pip install huggingface_hub

   export HF_ENDPOINT="https://hf-mirror.com"   # optional, for China mainland

   .venv/bin/huggingface-cli download gen-robot/openvla-7b-rlvla-warmup \
     --local-dir /data/ckpts/gen-robot/openvla-7b-rlvla-warmup

The default config expects the checkpoint at
``/data/ckpts/gen-robot/openvla-7b-rlvla-warmup``. To use a different
path, update ``model_path`` and ``tokenizer_path`` in
``conf/policy/openvla_ppo.yaml``.

**3. Download simulation assets**

ManiSkill built-in assets (bridge table scene and WidowX robot):

.. code-block:: bash

   source .venv/bin/activate
   python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
   python -m mani_skill.utils.download_asset widowx250s -y

Custom scene assets (carrot/plate objects and table overlay backgrounds):

.. code-block:: bash

   cd examples/openvla_ppo/maniskill
   ../.venv/bin/hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

**4. Launch training**

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Script
     - Mode
     - Use Case
   * - ``launch_train_ppo_sync.sh``
     - Single-GPU sync
     - Simplest, quick validation
   * - ``launch_train_ppo_ddp.sh``
     - DDP
     - 2 trainers + 1 eval worker
   * - ``launch_train_ppo_colocate_ddp_x8.sh``
     - Colocated DDP x8
     - Large-scale multi-GPU

Single-GPU quick start:

.. code-block:: bash

   bash launch_train_ppo_sync.sh

The script auto-detects GPU count, starts a Ray cluster, and launches training.

.. For a detailed configuration walkthrough, see
.. :doc:`../algorithms/manipulation/openvla_ppo_training`.


OpenPI PPO (LIBERO)
-------------------

PPO-based fine-tuning of `OpenPI <https://github.com/RLinf/openpi>`_
(π₀/π₀.₅) vision-language-action models on the 
`LIBERO <https://libero-project.github.io/>`_ manipulation benchmark.

**1. Setup LIBERO**

Clone LIBERO to ``.venv/LIBERO`` for editable install (required because
assets are not included when installing from git):

.. code-block:: bash

   cd examples/openpi_ppo
   uv venv .venv
   bash scripts/setup_libero.sh

**2. Environment setup**

.. code-block:: bash

   cd examples/openpi_ppo
   uv sync

**3. Setup OpenPI**

Apply the transformers library patches required for OpenPI PyTorch models:

.. code-block:: bash

   cd examples/openpi_ppo
   bash scripts/setup_openpi.sh

This also downloads OpenPI assets (tokenizer, etc.) and resolves the
``pynvml`` / ``nvidia-ml-py`` conflict.

**4. Download model weights**
Install the download tool and fetch the checkpoint (China mainland users
can set ``HF_ENDPOINT`` for acceleration):

.. code-block:: bash

   uv pip install huggingface_hub

   export HF_ENDPOINT="https://hf-mirror.com"   # optional, for China mainland

   .venv/bin/huggingface-cli download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
     --local-dir /data/ckpts/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

The default config expects the checkpoint at
``/data/ckpts/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT``. To use a
different path, update ``model_path`` and ``tokenizer_path`` in
``conf/policy/openpi_ppo.yaml``.

**5. Launch training**

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Script
     - Mode
     - Use Case
   * - ``launch_train_ppo_sync.sh``
     - Single-GPU sync
     - Simplest, quick validation
   * - ``launch_train_ppo_sync_tiny.sh``
     - Single-GPU tiny
     - Reduced batch size, fast iteration
   * - ``launch_train_ppo_sync_ddp.sh``
     - DDP (8 GPUs)
     - Multi-GPU distributed
   * - ``launch_train_ppo_sync_tiny_ddp.sh``
     - DDP tiny (8 GPUs)
     - Multi-GPU with reduced batch size

Single-GPU quick start:

.. code-block:: bash

   cd RLightning
   bash examples/openpi_ppo/launch_train_ppo_sync.sh

The script auto-detects GPU count, starts a Ray cluster, and launches training.


WBC Tracking (IsaacLab + RSL_RL)
--------------------------------

Humanoid whole-body control (WBC) motion tracking using a Unitree robot
in IsaacLab simulation.

.. note::

   **Prerequisite:** NVIDIA GPU with an Isaac Sim compatible driver.
   This example has the highest environment requirements among all examples.

**1. Download robot assets**

.. code-block:: bash

   cd RLightning
   bash examples/wbc_tracking/setup.sh

This downloads the Unitree robot URDF model to
``examples/wbc_tracking/assets/unitree_description/``.

**2. Initialize git submodules**

.. code-block:: bash

   git submodule update --init --recursive

The training depends on ``third_party/rsl_rl``. This must be completed
**before** ``uv sync``.

**3. Environment setup**

.. code-block:: bash

   cd examples/wbc_tracking
   uv sync

Dependencies are heavy: ``rlightning[dev, isaaclab, mujoco, humanoid]`` + ``rsl-rl``.

**4. Download and process motion data**

.. code-block:: bash

   cd RLightning
   source examples/wbc_tracking/.venv/bin/activate

4.1 Download the lafan motion capture dataset:

.. code-block:: bash

   python -m rlightning.humanoid.utils.download.download_lafan

4.2 Retarget motions to the Unitree robot:

.. code-block:: bash

   PYTHONPATH=$PWD/examples python -m wbc_tracking.retarget_lafan --f-path .data/lafan1

4.3 Convert to WBC tracking task format:

.. code-block:: bash

   PYTHONPATH=$PWD/examples python -m wbc_tracking.motion_converter --input-dir .data/lafan1/retargeted

Processed data is saved under ``.data/lafan1/retargeted/wbc_tracking/``.

**5. Launch training**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Script
     - Description
   * - ``launch.sh``
     - Single-node, multi-process
   * - ``launch_local.sh``
     - Local, no Ray
   * - ``launch_ddp.sh``
     - Single-node DDP
   * - ``launch_multi_node.sh``
     - Multi-node distributed
   * - ``launch_multi_node_ddp_x8.sh``
     - Multi-node DDP x8

.. code-block:: bash

   cd RLightning
   bash examples/wbc_tracking/launch.sh

.. note::

   For multi-node scripts, start the Ray cluster manually on each node first.
   See `Ray documentation <https://docs.ray.io/en/latest/cluster/getting-started.html>`_.
