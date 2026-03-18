Policy & PolicyGroup
=====================

Policy encapsulates a complete RL algorithm: the neural network, optimizer,
rollout logic, and training step. PolicyGroup manages collections of Policy
workers and handles weight synchronization between them.


Overview
--------

At runtime, RLightning runs two roles of the same Policy class:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Role
     - Responsibilities
     - Key Methods Called
   * - **EVAL policy**
     - Inference during rollout; pulls updated weights from train policy in background
     - ``init_eval``, ``rollout_step``, ``update_weights``
   * - **TRAIN policy**
     - Gradient updates; signals eval policy after each epoch
     - ``init_train``, ``update_dataset``, ``train``, ``notify_update_weights``

Both roles are instances of the same user-defined subclass of ``BasePolicy``.
The ``role_type`` attribute (``PolicyRole.EVAL`` or ``PolicyRole.TRAIN``)
determines which initialization path and methods are active.

.. mermaid::

   sequenceDiagram
      participant TP as Train Policy
      participant WB as WeightBuffer
      participant EP as Eval Policy

      loop Each Epoch
         EP->>EP: rollout_step (inference)
         TP->>TP: train (gradient update)
         TP->>WB: send_weights() — pushes state_dict to WeightBuffer
         TP->>EP: notify_update_weights() — sets update signal
         Note over EP: Background daemon thread wakes up
         EP->>WB: update_weights_from_buffer() — loads WeightBuffer into model
      end

Weight transfer runs in a background daemon thread on the eval side, so rollout
and weight updates overlap without stalling the main training loop. The updater
thread is only started when ``weight_buffer_strategy`` is not ``"None"``.


BasePolicy Interface
--------------------

``BasePolicy`` (``rlightning/policy/base_policy.py``) is the abstract base
class that all policies must subclass.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Method
     - Override?
     - Description
   * - ``construct_network(env_meta, ...)``
     - **Required**
     - Create model(s) and append to ``self.model_list`` as
       ``(name, module)`` pairs.
   * - ``setup_optimizer(optim_cfg)``
     - **Required**
     - Create optimizer; store as ``self.optimizer`` or similar.
   * - ``rollout_step(env_ret)``
     - **Required**
     - Pure inference: ``EnvRet → PolicyResponse``. Must not modify
       optimizer state. In ``async`` rollout mode, define as an ``async``
       coroutine — the framework will ``await`` it.
   * - ``postprocess(env_ret, policy_resp)``
     - **Required**
     - Post-process environment and policy outputs (e.g., value
       bootstrapping, advantage computation). Called by RSL-RL style
       engines after each rollout step.
   * - ``update_dataset(data)``
     - **Required**
     - Receive a ``TensorDict`` batch from the buffer into
       ``self.dataset`` for use in ``train``.
   * - ``train(...)``
     - **Required**
     - Run one training epoch using ``self.dataset``. Call
       ``self.log_metric(key, value)`` to log metrics.
   * - ``get_trainable_parameters()``
     - **Required**
     - Return ``{name: {param_name: tensor}}`` dict for weight
       transfer. Default uses ``self.model_list``.
   * - ``load_state_dict(state_dict)``
     - **Required**
     - Load weights into model(s). Default handles ``self.model_list``.
   * - ``save_checkpoint(path)``
     - Optional
     - Persist weights to disk. Default saves all models in
       ``self.model_list``.
   * - ``wrap_with_ddp(gpu_id, process_group)``
     - Optional
     - Wrap models with ``DistributedDataParallel``. Only needed when
       models are non-standard ``nn.Module`` types; standard models are
       wrapped automatically when ``train_worker_num > 1``.
   * - ``init_eval(eval_config, env_meta)``
     - Rarely needed
     - Called by the framework on EVAL workers. Default handles model
       creation, eval mode, and weight updater thread.
   * - ``init_train(train_config, env_meta)``
     - Rarely needed
     - Called by the framework on TRAIN workers. Default handles model
       creation, train mode, DDP wrapping, and optimizer setup.

.. note::

   ``self.model_list`` is a list of ``(name: str, model: nn.Module)``
   tuples. Populate it in ``construct_network`` so the framework can
   automatically handle weight extraction, DDP wrapping, and checkpointing.


Weight Synchronization
----------------------

After each training epoch, train policy pushes weights to eval policy via
a ``WeightBuffer``. Three buffer strategies trade memory for latency:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Strategy
     - Behavior
   * - ``Double``
     - Two weight buffers per eval policy; writer alternates between
       them. Reader always gets the latest complete snapshot. Default
       for most cases.
   * - ``Shared``
     - One shared buffer per node, shared by all eval policies on the
       same node. Reduces memory usage in colocated deployments.
   * - ``Sharded``
     - Weights are split across buffer shards; used with
       ``ShardedWeightBuffer`` for very large models.

Configure the strategy in ``policy.weight_buffer``:

.. code-block:: yaml

   policy:
     weight_buffer:
       type: WeightBuffer        # or CPUWeightBuffer, ShardedWeightBuffer
       buffer_strategy: Double   # None, Double, Shared, Sharded


PolicyGroup
-----------

``PolicyGroup`` is the WorkerGroup wrapper around Policy workers. It:

- Maintains separate pools of EVAL and TRAIN workers.
- Routes async rollout requests across EVAL workers via ``BatchRouter``
  for load balancing.
- Broadcasts weight sync from the TRAIN worker to all EVAL workers after
  each epoch.

The number of workers in each pool is set in ``cluster``:

.. code-block:: yaml

   cluster:
     train_worker_num: 2   # number of TRAIN policy actors
     eval_worker_num: 4    # number of EVAL policy actors


Configuration Reference
-----------------------

Key fields in the ``policy`` config section (``PolicyConfig``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Description
   * - ``type``
     - (required)
     - Registered name of the Policy class, e.g. ``"VLAPPOPolicy"``.
   * - ``rollout_mode``
     - ``"sync"``
     - Rollout mode: ``"sync"`` or ``"async"``.
   * - ``weight_buffer.type``
     - ``"WeightBuffer"``
     - Weight buffer class.
   * - ``weight_buffer.buffer_strategy``
     - ``"Double"``
     - Weight transfer strategy.
   * - ``model_cfg``
     - (varies)
     - Passed to ``construct_network``; user-defined structure.

See Also
--------

- :doc:`../build_your_own/customize_policy` — Step-by-step guide to implementing a custom policy.
- :doc:`engine` — How engines call Policy methods in the training loop.
- :doc:`../advanced/placement_strategy` — How TRAIN and EVAL workers are placed.
