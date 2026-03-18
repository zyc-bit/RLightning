Placement Strategy
==================

Worker placement controls which physical GPUs each component (Train, Eval,
Env, Buffer) occupies. Poor placement inflates cross-node data transfer
latency, splits model parameters across bandwidth-limited links, and wastes
GPU memory by leaving accelerators underutilized. RLightning's
``GlobalResourceManager`` (GRM) handles placement automatically or lets you
specify it explicitly via YAML.


Why Placement Matters
---------------------

- **Communication overhead** — Weight synchronization and experience data
  travel between Eval and Train workers. Placing them on separate nodes without
  high-speed interconnects creates a bottleneck that grows with model size.
- **GPU utilization** — Env workers and Eval workers can share a GPU during
  rollout. Without co-placement, one GPU sits idle while the other is busy.
- **Buffer routing** — Buffer shards must be collocated with their
  corresponding Train workers to avoid unnecessary cross-node copies during
  ``sample()``.

GRM reads ``cluster.placement``, creates Ray placement groups, and injects a
``PlacementGroupSchedulingStrategy`` into every actor at creation time via
``rlightning/utils/ray/launcher.py::launch_ray_actor()``.


Auto Mode (Recommended)
-----------------------

Set ``placement.mode: "auto"`` (the default). GRM computes component
distribution from worker counts and available cluster resources.

.. code-block:: yaml

   cluster:
     train_worker_num: 2
     eval_worker_num: 2
     buffer_worker_num: auto   # recommended; GRM infers from cluster GPUs

     placement:
       mode: auto
       strategy: disaggregate   # or colocate / default
       env_strategy: default    # or device-colocate

The three ``strategy`` values differ in how they partition the cluster.

default
~~~~~~~

No placement groups are created. Ray schedules actors freely using its own
bin-packing heuristics. Buffer shards with multiple workers use node-affinity
for simple spreading. Choose ``default`` for small single-node experiments or
when you do not need strict resource isolation.

.. code-block:: yaml

   placement:
     mode: auto
     strategy: default

disaggregate
~~~~~~~~~~~~

The cluster is split into two isolated placement groups:

- **train_pool** — Train workers + Buffer shards
- **rollout_pool** — Eval workers + Env workers

Data transfer between pools crosses a well-defined boundary. This prevents
rollout workers from competing with training workers for memory bandwidth and
is the recommended choice for multi-node runs where training and inference have
different hardware requirements.

.. code-block:: yaml

   placement:
     mode: auto
     strategy: disaggregate
     env_strategy: default   # Env and Eval each get dedicated GPUs

colocate
~~~~~~~~

All components share a single **global_pool** placement group. Train, Eval,
and Env workers are co-scheduled on the same set of nodes. Whether Env and
Eval workers share the same GPU device is controlled by ``env_strategy``.

.. code-block:: yaml

   placement:
     mode: auto
     strategy: colocate
     env_strategy: device-colocate   # Env and Eval share GPU memory

.. note::

   When ``env_strategy: device-colocate`` is set, GPU demand for Env and Eval
   is computed as ``max(env_gpus, eval_gpus)`` instead of the sum, enabling
   resource sharing between the two components on the same device.


env_strategy Options
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Behavior
   * - ``default``
     - Env and Eval workers request separate GPU allocations. Each component
       occupies its own GPU slice.
   * - ``device-colocate``
     - Env and Eval workers share GPU memory. GPU demand is resolved as
       ``max(env, eval)`` instead of sum. Requires ``strategy: colocate``.

.. note::

   ``env_strategy: "node-affinity"`` is listed in some internal documents but
   is not yet implemented. Use ``default`` or ``device-colocate``.


Manual Mode
-----------

Set ``placement.mode: "manual"`` and supply an explicit ``resource_pool``
list. Each pool entry names which GPU indices on which nodes each component
type owns. Manual mode gives full control over topology.

.. code-block:: yaml

   cluster:
     train_worker_num: 1
     eval_worker_num: 1
     buffer_worker_num: auto

     placement:
       mode: manual

   resource_pool:
     - name: global_pool
       num_node: 3
       num_gpus: 8
       train: "0-7"
       eval:  "8-15"
       env:   "16-23"

GPU index ranges in ``resource_pool`` are **pool-global** — they count GPUs
sequentially across all nodes in the pool. A pool with ``num_node: 3,
num_gpus: 8`` has indices 0–23.

.. warning::

   When ``node_ids`` is not specified, all nodes in a pool must have the same
   GPU count. Mixed-GPU-count nodes require explicit ``node_ids`` binding with
   a per-node ``num_gpus`` list.

.. warning::

   Each component type (train, eval, env, buffer) may appear in at most one
   resource pool. Splitting a component across pools is not supported.

Example Topologies
~~~~~~~~~~~~~~~~~~

**All co-located** (single pool, all components share resources):

.. code-block:: yaml

   resource_pool:
     - name: global_pool
       num_node: 3
       num_gpus: 8
       train: "0-7, 8-15, 16-23"
       eval:  "0-7, 8-15, 16-23"
       env:   "0-7, 8-15, 16-23"

**Fully disaggregated** (one pool per component):

.. code-block:: yaml

   resource_pool:
     - name: train_pool
       num_node: 1
       num_gpus: 8
       train: "0-7"
     - name: eval_pool
       num_node: 1
       num_gpus: 8
       eval: "0-7"
     - name: env_pool
       num_node: 1
       num_gpus: 8
       env: "0-7"

**Mixed** (Train isolated; Eval and Env share rollout nodes):

.. code-block:: yaml

   resource_pool:
     - name: train_pool
       num_node: 1
       num_gpus: 8
       train: "0-7"
     - name: rollout_pool
       num_node: 2
       num_gpus: 8
       eval: "0-7, 8-15"
       env:  "0-7, 8-15"

.. tip::

   After a successful auto-mode run, GRM writes the computed resource pool to
   ``resource_pool_auto.yaml`` (via ``save_yaml_config``). Use this file as a
   starting point for manual tuning.


Offload Support
---------------

When ``is_colocated: true``, train and eval policies share the same GPU.
The engine automatically offloads model weights between rollout and training
phases to prevent out-of-memory errors.

.. code-block:: yaml

   cluster:
     is_colocated: true
     enable_offload: true   # automatically set to true when is_colocated is true

     placement:
       mode: auto
       strategy: colocate
       env_strategy: device-colocate

``enable_offload`` can also be set independently of ``is_colocated`` when
you want explicit offloading control. The engine's pre/post hooks handle
``reload_eval_model()`` before rollout and ``offload_eval_model()`` after
rollout, and ``reload_model_param_and_grad()`` before training.

Use offloading when:

- GPU memory is tight and you cannot afford to keep both train and eval
  weights resident simultaneously.
- You are running a very large model (e.g., 7B+ VLA) on a limited GPU budget.


Constraints
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Constraint
     - Detail
   * - ``buffer_worker_num`` alignment
     - Under resource-pool strategies (disaggregate, colocate, manual),
       ``buffer_worker_num`` is forced to equal the number of nodes occupied
       by train workers. User-specified values are overridden.
   * - Homogeneous node GPUs (manual, no node_ids)
     - All nodes in a pool must have the same GPU count when ``node_ids`` is
       omitted.
   * - Single pool per component
     - A given component type (train/eval/env/buffer) must reside entirely in
       one pool.
   * - ``default`` strategy buffer sharding
     - With ``strategy: default``, buffer shards use simple node-affinity
       (not placement groups). No train-node alignment is enforced.

.. note::

   [Placeholder: topology diagram showing disaggregate vs colocate layouts
   across nodes with GPU index ranges.]

