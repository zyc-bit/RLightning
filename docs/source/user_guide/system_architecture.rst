System Architecture
===================

Four-Layer Architecture
-----------------------

.. figure:: ../_static/images/system_architecture.png
  :alt: RLightning system architecture
  :width: 40%
  :align: center

Application Layer
~~~~~~~~~~~~~~~~~

The Application Layer is where users write RL applications. It provides:

- **Configuration files** (YAML + Hydra) that declare all component
  settings and compose them into a single ``MainConfig``.
- **Policy subclasses** — users subclass ``BasePolicy`` and implement
  ``construct_network``, ``rollout_step``, and ``train``.
- **Entry-point scripts** that call the builder API to assemble the
  training pipeline.

Controller Layer
~~~~~~~~~~~~~~~~

The Controller Layer implements the control plane: it decides *what* to
do and *when*, but does not perform computation itself.

It is split into two hierarchical sublayers:

**Engine** (coarse-grained orchestration)
  Drives the top-level RL training loop, broken into four stages per
  iteration:

  1. ``rollout`` — collect experience from the environment.
  2. ``update_dataset`` — transfer data from buffer to training workers.
  3. ``train`` — update policy weights.
  4. ``sync_weights`` — push updated weights to eval workers.

  Built-in engines cover common paradigms:

  - ``SyncRLEngine`` — on-policy synchronous (PPO-style)
  - ``AsyncRLEngine`` — off-policy asynchronous

  See :doc:`./core_components/engine` for more details.

**WorkerGroup** (fine-grained task dispatch)
  Each component (Env, Policy, Buffer) is wrapped in a WorkerGroup that:

  - Exposes a batched interface to the Engine.
  - Internally routes requests across multiple Worker instances via a
    ``BatchRouter`` and ``AsyncIOHandler``.
  - Handles load balancing, result aggregation, and node-affinity routing.

.. mermaid::

   sequenceDiagram
      participant E as Engine
      participant PG as PolicyGroup (WorkerGroup)
      participant EG as EnvGroup (WorkerGroup)
      participant BG as BufferGroup (WorkerGroup)

      loop Each Epoch
         E-->>EG: rollout (schedule)
         EG->>E: EnvRet (obs, reward, done)
         E-->>PG: rollout_step (schedule)
         PG->>E: PolicyResponse (actions, log_probs, values)
         E-->>BG: store (schedule)
         E-->>BG: sample (schedule)
         BG->>E: BatchedData
         E-->>PG: train (schedule)
         E-->>PG: sync_weights (schedule)
      end

Worker Layer
~~~~~~~~~~~~

Workers execute actual computation and storage. Each type scales
horizontally and independently:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Worker Type
     - Scales With
     - Responsibility
   * - Env Worker
     - ``num_workers`` in EnvConfig
     - Runs simulator; executes ``reset()`` / ``step()``
   * - Eval Policy Worker (Actor)
     - ``eval_worker_num`` in ClusterConfig
     - Inference-only; serves ``rollout_step`` requests
   * - Train Policy Worker (Learner)
     - ``train_worker_num`` in ClusterConfig
     - Holds optimizer; executes ``train``; owns model gradients
   * - Buffer Worker
     - ``buffer_worker_num`` in ClusterConfig
     - Stores and samples experience; manages storage shards

Runtime and Resource Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Runtime and Resource Layer is the underlying infrastructure that abstracts
distributed execution complexity and is fully transparent to upper-level
applications. It comprises two components:

**Resource Manager** (fine-grained resource orchestration)
  Manages independent scaling across all Worker types. Users declare the
  number of each Worker type in the config; To optimize communication efficiency and
  hardware utilization, the Resource Manager provides placement strategies.

  See :doc:`./advanced/placement_strategy` for configuration details.

**Runtime Adapter** (unified sequential / parallel execution)
  Decouples algorithm logic from the underlying runtime. All Workers can be
  configured to run as local processes for prototyping and debugging. Once
  validated, switching to distributed execution requires only a config change. 
  no algorithm code modifications are needed.


See Also
--------
To Learn More, see the following sections:

- :doc:`core_components/engine` — Engine API and configuration reference.
- :doc:`core_components/data_buffer` — Buffer types and configuration.
- :doc:`core_components/policy_policygroup` — Policy and PolicyGroup.
- :doc:`advanced/placement_strategy` — GPU placement strategies.
