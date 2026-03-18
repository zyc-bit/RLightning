Performance Profiling
=====================

RLightning collects per-phase wall-clock timings for every training iteration
and exposes GPU memory snapshots through a lightweight profiler module. This
page explains how to enable detailed profiling, read the output, identify
bottlenecks, and configure logging backends.


Built-in Timing
---------------

The engine instruments four phases automatically:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Phase name
     - What it measures
   * - ``_rollout``
     - Total time from ``_pre_rollout_hook`` to ``_post_rollout_hook``,
       including env stepping, policy inference, and buffer writes.
   * - ``rollout``
     - Inner rollout time only (excludes offload reload/save overhead). Only
       recorded when ``RLIGHTNING_DEBUG=1``.
   * - ``_train``
     - Total time from ``_pre_train_hook`` to ``_post_train_hook``, including
       gradient computation and optimizer step.
   * - ``train``
     - Inner train time only. Only recorded when ``RLIGHTNING_DEBUG=1``.
   * - ``_sync_weights``
     - Total weight-sync time, including any offload operations.
   * - ``sync_weights``
     - Inner sync time only. Only recorded when ``RLIGHTNING_DEBUG=1``.
   * - ``update_dataset``
     - Time to sample from the buffer and transfer data to the train policy.
       Only recorded when ``RLIGHTNING_DEBUG=1``.

The ``timer_wrap`` decorator on ``_rollout``, ``_train``, and
``_sync_weights`` always fires. The inner ``timer`` context managers are
gated on ``InternalFlag.DEBUG`` (i.e., the ``RLIGHTNING_DEBUG`` environment
variable).

Each timing entry accumulates ``count``, ``total``, and ``avg`` across calls
within the same engine instance. The engine's ``print_timing_summary()``
method logs the full table at ``DEBUG`` level and is called automatically at
the end of every epoch when ``RLIGHTNING_DEBUG=1``.


Enabling Profiling
------------------

Two environment variables control profiling depth:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Effect
   * - ``RLIGHTNING_DEBUG=1``
     - Enables inner-phase timers (``rollout``, ``train``, ``sync_weights``,
       ``update_dataset``) and calls ``print_timing_summary()`` after each
       epoch.
   * - ``RLIGHTNING_VERBOSE=1``
     - Enables the rich progress bar. Does not affect timing collection.

Set these before launching:

.. code-block:: bash

   RLIGHTNING_DEBUG=1 RLIGHTNING_VERBOSE=1 uv run train.py --config-name train_algo

In multi-process (Ray) runs, ``launch()`` automatically propagates these
variables to all Ray workers via ``runtime_env.env_vars``.


Reading Profiler Output
-----------------------

With ``RLIGHTNING_DEBUG=1``, the log contains two classes of entries.

**Per-call timing** (emitted at ``INFO`` level after each method call):

.. code-block:: text

   [rlightning] INFO - time_profile/_rollout: 1.2345s
   [rlightning] INFO - time_profile/_train: 0.8901s
   [rlightning] INFO - time_profile/_sync_weights: 0.1234s

**Epoch summary** (emitted at ``DEBUG`` level at the end of each epoch):

.. code-block:: text

   [rlightning] DEBUG - Timing summary:
   [rlightning] DEBUG - SyncRLEngine:
   [rlightning] DEBUG -     _rollout        count=1   total=1.234567s avg=1.234567s
   [rlightning] DEBUG -     _train          count=1   total=0.890123s avg=0.890123s
   [rlightning] DEBUG -     _sync_weights   count=1   total=0.123456s avg=0.123456s

The summary propagates down to ``env_group``, ``policy_group``, and
``buffer`` — each prints its own timing breakdown.

**GPU memory** is logged via ``log_gpu_memory_usage(head, level)``:

.. code-block:: text

   [GPU Memory] before_train, memory allocated (GB): 12.34,
   memory reserved (GB): 14.00, device memory used/total (GB): 15.00/24.00

Call ``log_gpu_memory_usage`` at any point in custom policy or env code to
snapshot memory at that phase.


Common Bottlenecks and Remedies
--------------------------------

Rollout-Bound
~~~~~~~~~~~~~

**Symptom**: ``_rollout`` time dominates; ``_train`` finishes quickly and
waits for new data.

**Remedies**:

- Increase ``eval_worker_num`` to add more inference workers.
- Increase ``env_worker_num`` (or ``num_envs`` per env group) to collect more
  experience in parallel.
- Switch from ``syncrl`` to ``asyncrl`` so training proceeds concurrently
  with rollout.
- Use ``env_strategy: device-colocate`` to avoid idle GPU time between Env
  and Eval workers.

Training-Bound
~~~~~~~~~~~~~~

**Symptom**: ``_train`` time dominates; rollout completes but the buffer
fills before training catches up.

**Remedies**:

- Enable DDP multi-GPU training by increasing ``train_worker_num`` and
  ``train_each_gpu_num``.
- Use ``enable_offload: true`` with ``strategy: colocate`` to reclaim GPU
  memory occupied by the eval model during training.
- Reduce ``batch_size`` or limit gradient accumulation steps if the training
  loop itself is the bottleneck.

Data Transfer Bound
~~~~~~~~~~~~~~~~~~~

**Symptom**: ``update_dataset`` time is disproportionately large; GPU
utilization is low during training.

**Remedies**:

- Use ``strategy: disaggregate`` to place buffer shards on the same nodes
  as train workers, eliminating cross-node data movement.
- Set ``buffer_worker_num: auto`` so GRM automatically aligns shards with
  train nodes.
- Enable node-affinity routing for buffer shards (automatic under resource-
  pool strategies).

Weight Sync Bound
~~~~~~~~~~~~~~~~~

**Symptom**: ``_sync_weights`` time is large relative to ``_train``.

**Remedies**:

- Co-locate train and eval workers (``strategy: colocate``) to reduce
  weight-transfer distance.
- Use ``enable_offload: true`` only when necessary — offload add
  reload/save overhead to each sync cycle.

.. note::

   [Placeholder: representative timing numbers from benchmark runs showing
   the expected ratio of rollout:train:sync_weights for typical PPO
   configurations.]


Logging Backends
----------------

Experiment metrics (rewards, losses, and ``time_profile/*`` entries
when ``log_to_metric=True``) are written to the configured backend.

Configuration is in the ``log`` section:

.. code-block:: yaml

   log:
     level: INFO             # console log level: DEBUG / INFO / WARNING / ERROR
     backend: tensorboard    # tensorboard | wandb | swanlab
     project: my_project
     name: run_001
     log_dir: ./runs
     mode: null              # wandb only: online | offline | shared | disabled

TensorBoard
~~~~~~~~~~~

The default backend. Scalars are written with ``SummaryWriter`` to
``<log_dir>/<project>/<name>/tensorboard/<timestamp>/``. Launch the viewer:

.. code-block:: bash

   tensorboard --logdir ./runs

Weights & Biases
~~~~~~~~~~~~~~~~

Set ``backend: wandb``. Default mode is ``offline`` to avoid network
dependency during training; sync manually with ``wandb sync``:

.. code-block:: yaml

   log:
     backend: wandb
     project: rlightning_runs
     name: ppo_maniskill
     mode: offline

.. warning::

   Do not instantiate TensorBoard loggers inside Ray actor class constructors.
   TensorBoard's internal ``threading.Lock`` cannot be serialized by Ray.
   Use ``get_metrics_logger(__name__)`` and ``log_metric()`` instead.

SwanLab
~~~~~~~

Set ``backend: swanlab``. Supports ``cloud`` and ``local`` modes:

.. code-block:: yaml

   log:
     backend: swanlab
     project: rlightning_runs
     name: ppo_maniskill
     mode: local


Debugging Tips
--------------

Single-process mode
~~~~~~~~~~~~~~~~~~~

Remove the ``cluster`` config block to run everything in a single process
without Ray. This enables standard Python debuggers (pdb, VSCode breakpoints):

.. code-block:: bash

   uv run python train.py ~cluster

Single-process mode is limited to one GPU but eliminates Ray serialization
and actor lifecycle overhead, making it the fastest way to debug data flow
issues.

Multi-process debugging
~~~~~~~~~~~~~~~~~~~~~~~

For distributed debugging, use the **Ray Distributed Debugger** VSCode
extension (``anyscalecompute.ray-distributed-debugger``). It attaches to
individual Ray worker processes via remote attach, allowing step-through
debugging of actor code running in a live cluster.

Install the required VSCode extensions:

- ``ms-python.debugpy`` (Python Debugger)
- ``anyscalecompute.ray-distributed-debugger``

Set ``RAY_DEBUG=1`` in the ``runtime_env.env_vars`` passed to ``ray.init``
to enable the debugger server inside workers.

.. tip::

   Start with a minimal configuration (1 train worker, 1 eval worker, 1 env,
   small model) to reproduce the issue before scaling up to the full cluster.
   This avoids long iteration cycles during diagnosis.

