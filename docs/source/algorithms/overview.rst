Overview
========

RLightning supports two primary algorithmic paradigms — **synchronous
on-policy** and **asynchronous off-policy** — together with specialized
variants for Vision-Language-Action (VLA) fine-tuning and IsaacLab
locomotion. The choice of paradigm determines which engine and buffer type
to use; everything else (policy implementation, reward shaping, model
architecture) is algorithm-specific.


Algorithm Categories
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Category
     - Engine
     - Buffer
     - Typical Use Case
   * - On-policy synchronous
     - ``syncrl``
     - ``RolloutBuffer``
     - PPO, VLA fine-tuning with PPO
   * - On-policy locomotion (RSL-RL)
     - ``rsl`` / ``async_rsl``
     - ``RolloutBuffer``
     - IsaacLab humanoid control, WBC tracking
   * - Off-policy asynchronous
     - ``asyncrl``
     - ``ReplayBuffer``
     - Off-policy algorithms with V-trace correction
   * - VLA fine-tuning
     - ``syncrl``
     - ``RolloutBuffer``
     - OpenVLA PPO, OpenPI PPO (1B–7B models)

The engine is configured via a single ``engine`` field in the top-level
config. Switching algorithmic paradigm requires changing ``engine`` and
``buffer.type``; the policy implementation stays the same.

.. code-block:: yaml

   # On-policy synchronous (PPO)
   engine: syncrl
   buffer:
     type: RolloutBuffer

   # Off-policy asynchronous
   engine: asyncrl
   buffer:
     type: ReplayBuffer


Built-in Algorithms
-------------------

PPO — Synchronous On-Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Engine**: ``syncrl`` — **Buffer**: ``RolloutBuffer`` — **Example**: ``examples/algorithm_template/``

Standard PPO with clipped surrogate objective. Each epoch collects a full
rollout batch, trains on it, discards the data, then syncs weights. The
``RolloutBuffer`` is cleared after every train step (on-policy semantics).

Key properties:

- Strict on-policy guarantee: no staleness in training data.
- Simple to debug — sequential execution, no thread coordination.
- Periodic evaluation and checkpointing via ``eval_interval`` / ``save_interval``.

.. code-block:: yaml

   engine: syncrl
   buffer:
     type: RolloutBuffer

OpenVLA PPO
~~~~~~~~~~~

**Engine**: ``syncrl`` — **Buffer**: ``RolloutBuffer`` — **Example**: ``examples/openvla_ppo/``

PPO applied to a 7B-parameter Vision-Language-Action model on robotic
manipulation tasks (ManiSkill). The policy wraps an OpenVLA backbone; the
``syncrl`` engine drives the on-policy training loop. Large model size
requires either colocated mode (shared GPU) or DDP across multiple GPUs.

Key properties:

- Policy backbone is a 7B VLA model; batch sizes are small.
- Supports DDP training via ``train.parallel: ddp``.
- Colocated mode (``cluster.is_colocated: true``) reduces peak GPU memory
  by offloading eval model weights during training.

.. code-block:: yaml

   engine: syncrl
   cluster:
     is_colocated: true   # share GPU between eval and train policies

OpenPI PPO
~~~~~~~~~~~~

**Engine**: ``syncrl`` — **Buffer**: ``RolloutBuffer`` — **Example**: ``examples/openpi_ppo/``

PPO fine-tuning of an OpenPI vision-language model on manipulation tasks. 
Architecture and training loop are identical to
OpenVLA PPO; the difference is the model backbone and task setup.

RSL-RL (IsaacLab Locomotion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Engine**: ``rsl`` / ``async_rsl`` — **Buffer**: ``RolloutBuffer`` — **Example**: ``examples/rslrl_isaaclab/``, ``examples/wbc_tracking/``

Integrates with the RSL-RL framework for IsaacLab-based robot locomotion
and whole-body control (WBC) tasks. The ``rsl`` engine extends
``SyncRLEngine`` with asynchronous environment stepping (``step_async`` /
``collect_async``) to achieve higher throughput with large numbers of
parallel simulated environments. The ``async_rsl`` variant adds concurrent
training threads for further throughput gains.

Key properties:

- Designed for IsaacLab with thousands of parallel environments.
- MLP or small Transformer policy backbone.
- WBC tracking (``examples/wbc_tracking/``) supports multi-task training
  and multi-node DDP.

.. code-block:: yaml

   engine: rsl    # or async_rsl for multi-threaded training


Supported Examples
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Algorithm
     - Simulator
     - Engine
     - Example Path
   * - PPO (template)
     - Any
     - ``syncrl``
     - ``examples/algorithm_template/``
   * - OpenVLA PPO
     - ManiSkill
     - ``syncrl``
     - ``examples/openvla_ppo/``
   * - OpenPI PPO
     - LIBERO
     - ``syncrl``
     - ``examples/openpi_ppo/``
   * - RSL-RL (IsaacLab)
     - IsaacLab
     - ``rsl``
     - ``examples/rslrl_isaaclab/``
   * - WBC Tracking
     - IsaacLab
     - ``rsl`` / ``async_rsl``
     - ``examples/wbc_tracking/``


Implementing a Custom Algorithm
--------------------------------

Start from ``examples/algorithm_template/``. The template provides a
minimal ``pyproject.toml``, Hydra config structure, and a ``train.py``
entry point with the standard builder pattern.

Steps:

1. Copy ``examples/algorithm_template/`` to a new directory.
2. Subclass ``BasePolicy`` and implement ``rollout()`` and ``learn()``.
   Register the class with ``@POLICY.register("my_algo")``.
3. Set ``policy.type: my_algo`` in the config.
4. Choose ``engine: syncrl`` (on-policy) or ``engine: asyncrl``
   (off-policy) and the matching buffer type.
5. Run with ``uv run python train.py --config-name train_algo``.

.. tip::

   Always develop and debug in single-process mode (remove the ``cluster``
   config key) before enabling Ray actors. See
   :doc:`../user_guide/debug_scaling_up` for the full debugging workflow.


Choosing an Algorithm
---------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Scenario
     - Recommended approach
   * - Standard on-policy training (PPO, A2C)
     - ``syncrl`` + ``RolloutBuffer`` + ``algorithm_template``
   * - High-throughput off-policy (many envs, GPU bottleneck)
     - ``asyncrl`` + ``ReplayBuffer`` + V-trace correction
   * - VLA fine-tuning (1B–7B models)
     - ``syncrl`` + ``RolloutBuffer`` + colocated or DDP mode
   * - IsaacLab locomotion / WBC
     - ``rsl`` or ``async_rsl`` + ``RolloutBuffer``
   * - Evaluation only (no training)
     - ``eval`` engine


Further Reading
---------------

- :doc:`../user_guide/core_components/engine` — engine internals and
  configuration reference.
- :doc:`../user_guide/debug_scaling_up` — single-process debugging,
  Ray cluster setup, and checkpoint handling.
- ``examples/algorithm_template/`` — minimal starting point for a new
  algorithm.
