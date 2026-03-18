Debug & Scaling Up
==================

This guide covers the recommended workflow for developing and debugging a
new algorithm in RLightning, then scaling from local debugging to a
multi-node Ray cluster.

The core principle: **validate algorithm correctness in local debugging mode
first, then distribute**. Full local single-process execution eliminates Ray
actor overhead, serialization, and network I/O, making bugs much easier to
isolate; component-local debugging lets you inspect specific components even
when the full program must still run in distributed mode.


Local Debugging
---------------

RLightning supports two local debugging modes.

**Mode 1: Full local single-process debugging**

This is the simplest and most recommended starting point. There is no
distributed execution at all: environment, eval policy, train policy, and
buffer all run inside the driver process. Set all ``remote_*`` flags to
``false`` in ``ClusterConfig``:

.. code-block:: yaml

   cluster:
     remote_train: false
     remote_eval: false
     remote_storage: false
     remote_env: false

With this configuration, RLightning does not rely on distributed actors and
runs entirely in one local process. You can use a standard Python debugger
(for example, ``pdb`` or the VS Code debugger) and set breakpoints anywhere
in the policy, engine, buffer, or environment code.

To activate this mode without editing the config file, remove the
``cluster`` section at the command line using Hydra's override syntax:

.. code-block:: bash

   python train.py ~cluster

This drops the ``cluster`` key from the resolved config. The engine detects
``config.cluster is None`` and falls back to fully local single-process
execution.

**Mode 2: Component-local debugging in a distributed run**

When a full single-process run does not fit on one GPU, or when you want to
debug the correctness of only part of a distributed system, you can move only
the target component back into the driver process while keeping the other
components remote. This is done by setting the corresponding ``remote_*``
field to ``false`` without changing the others.

For example, to debug the environment logic while keeping policy and buffer
distributed:

.. code-block:: yaml

   cluster:
     remote_train: true
     remote_eval: true
     remote_storage: true
     remote_env: false

Similarly, you can set ``remote_train: false``, ``remote_eval: false``, or
``remote_storage: false`` to bring the corresponding component into the
driver process for breakpoint-based debugging. This is useful when a specific
component is suspected to be incorrect under distributed execution, and you
want to inspect it directly with ``pdb`` or the VS Code debugger while the
rest of the system continues to run remotely.

.. note::

   Full local single-process debugging is limited by the resources of the
   driver process, typically one GPU. Use a small model, small batch size,
   and a short ``max_rollout_steps`` to avoid OOM during iteration. If the
   full program does not fit locally, switch to component-local debugging
   instead of forcing all components into one process.

.. tip::

   Set ``debug: true`` in the top-level config (the default). The engine
   then runs a warm-up phase — a short dummy rollout + train pass — before
   the real training loop. Warm-up catches dataflow mismatches early, before
   you invest time in a long run.



Ray Cluster Setup
-----------------

Single-node distributed
~~~~~~~~~~~~~~~~~~~~~~~

Start the Ray head on the training machine before launching the script:

.. code-block:: bash

   ray stop
   ray start --head --num-gpus=<GPU_COUNT>
   uv run python train.py --config-name train_algo

The training script calls ``ray.init("auto")`` (set by
``cluster.ray_address: auto``) and connects to the local head.

Multi-node cluster
~~~~~~~~~~~~~~~~~~

Start the Ray head on the primary node:

.. code-block:: bash

   ray stop
   ray start --head --num-gpus=<GPU_COUNT>

Note the printed IP address and port (e.g., ``10.0.0.1:6379``). On each
worker node, join the cluster:

.. code-block:: bash

   # Activate the same virtual environment as the head node first
   source /path/to/.venv/bin/activate
   ray start --address='10.0.0.1:6379' --num-gpus=<GPU_COUNT>

Verify the cluster state from the head node:

.. code-block:: bash

   ray status

Launch training from the head node after all workers have joined:

.. code-block:: bash

   uv run python train.py --config-name train_algo

.. warning::

   The virtual environment on every node must be identical. If Ray reports
   import errors or version conflicts on workers, rebuild the environment on
   each node from the same ``pyproject.toml`` and ``uv.lock``. A mismatch
   between the Python versions used to start ``ray start`` and the training
   script will also cause failures.


