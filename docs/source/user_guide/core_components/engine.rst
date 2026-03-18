Engine
======

The Engine is the top-level component that orchestrates the complete
reinforcement learning training loop. It coordinates EnvGroup (environment
interaction), PolicyGroup (policy inference and training), and DataBuffer
(experience storage) into a unified workflow. Users select an engine type
via a single configuration field and call ``run()`` to start training.

A typical training cycle for Engine works as follows:

1. **Initialization** — The engine initializes components:
   EnvGroup, PolicyGroup, DataBuffer.
   An initial ``sync_weights()`` call copies train-policy weights to the
   eval policy before training begins.
2. **Rollout** — The eval policy performs inference on observations from
   EnvGroup. Transitions (``EnvRet`` + ``PolicyResponse``) are stored in
   DataBuffer.
3. **Update dataset** — Sampled data is transferred from DataBuffer into
   the train policy.
4. **Train** — The train policy updates its parameters on the prepared dataset.
5. **Sync weights** — Updated weights are pushed from the train policy to
   the eval policy.
6. Steps 2–5 repeat for the configured number of epochs. ``SyncRLEngine``
   executes them sequentially in order, while ``AsyncRLEngine`` runs them
   as an overlapping pipeline across four concurrent threads.


Engine Types
------------

RLightning provides several engine implementations for different training
paradigms. All engines share the same interface — ``run()`` starts
loops — but differ in how rollout, training, and weight updates are
scheduled.

.. container:: mermaid-width-50

  .. mermaid::

    flowchart TB
       Base["BaseEngine"] --> Sync["SyncRLEngine<br/>(syncrl)"]
       Base --> Async["AsyncRLEngine<br/>(asyncrl)"]
       Base --> Eval["EvaluationEngine<br/>(eval)"]
       Sync --> RSL["RSLRLEngine<br/>(rsl)"]
       Async --> AsyncRSL["AsyncRSLRLEngine<br/>(async_rsl)"]

SyncRLEngine
~~~~~~~~~~~~

Sequential execution: rollout → update dataset → train → sync weights,
repeated each epoch. Straightforward and easy to debug.

.. mermaid::

   sequenceDiagram
       participant Engine as SyncRLEngine
       participant EG as EnvGroup
       participant PGE as PolicyGroup (Eval)
       participant BF as DataBuffer
       participant PGT as PolicyGroup (Train)

       loop each epoch
           rect rgb(227, 242, 253)
               Note over Engine: Rollout
               Engine->>EG: reset() / step()
               EG-->>Engine: batched_env_ret
               Engine->>PGE: rollout_batch(batched_env_ret)
               PGE-->>Engine: batched_policy_resp
               Engine->>BF: add_batched_transition()
           end
           rect rgb(255, 243, 224)
               Note over Engine: Update Dataset
               Engine->>BF: sample()
               BF-->>PGT: training data
               Engine->>PGT: update_dataset()
           end
           rect rgb(255, 224, 224)
               Note over Engine: Train
               Engine->>PGT: train()
           end
           rect rgb(232, 245, 233)
               Note over Engine: Sync Weights
               Engine->>PGT: sync_weights()
               PGT-->>PGE: updated weights
           end
       end

AsyncRLEngine
~~~~~~~~~~~~~

Rollout, dataset update, training, and weight sync run in separate
threads. Suitable for off-policy algorithms and scenarios
where you want maximum throughput.

.. container:: mermaid-width-50

  .. mermaid::

    flowchart TB
      RT["Rollout Thread<br/>EnvGroup & PolicyGroup(Eval) & DataBuffer"]

        subgraph Pipeline["Epoch Dependency Pipeline"]
            direction LR
        DT["Dataset Thread<br/>DataBuffer & PolicyGroup(Train)"]
        TT["Train Thread<br/>PolicyGroup(Train)"]
        ST["Sync Thread<br/>PolicyGroup(Train) & PolicyGroup(Eval)"]

        TT -->|"ready_for_update_dataset<br/>ready_for_sync_weights"| DT
            TT -->|"ready_for_sync_weights"| ST
            DT -->|"dataset_ready"| TT
            ST -->|"weights_updated"| TT
        end

      RT -.->|"continuously"| Pipeline

The four threads coordinate automatically — you just call ``engine.run()``.

RSLRLEngine
~~~~~~~~~~~

**Registration name**: ``rsl``

Extends SyncRLEngine with asynchronous environment stepping
(``step_async()`` / ``collect_async()``) and policy post-processing.
Designed for on-policy robot learning with multiple parallel environments.
Follows the same epoch structure as SyncRLEngine but achieves higher
throughput through async environment interaction.

AsyncRSLRLEngine
~~~~~~~~~~~~~~~~

**Registration name**: ``async_rsl``

Combines the async threading model of AsyncRLEngine with RSL-RL style
post-processing. Suitable for high-throughput on-policy training with
vectorized environments.

EvaluationEngine
~~~~~~~~~~~~~~~~

**Registration name**: ``eval``

Runs policy evaluation without training. It performs rollout with eval policy;
training and weight sync are no-ops. Checkpoint loading is not handled by the
engine itself (prepare checkpoint loading via config/user code). 

Comparison
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 20 15 35

   * - Engine
     - Execution
     - Algorithm Type
     - Env Stepping
     - Use Case
   * - ``syncrl``
     - Sequential
     - On-policy
     - Synchronous
     - General RL training
   * - ``asyncrl``
     - Multi-threaded
     - Off-policy
     - Asynchronous
     - High-throughput training
   * - ``rsl``
     - Sequential
     - On-policy
     - Asynchronous
     - Robot learning
   * - ``async_rsl``
     - Multi-threaded
     - Off-policy
     - Asynchronous
     - High-throughput robot learning
   * - ``eval``
     - Sequential
     - —
     - Synchronous
     - Policy evaluation


Usage
-----

Building and Running an Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended entry point uses ``launch()`` to handle Hydra configuration
loading, logging setup, and Ray cluster initialization, while the user only
needs to build the core components and call ``engine.run()``:

.. code-block:: python

   from pathlib import Path
   from rlightning.utils.launch import launch
   from rlightning.utils.config import MainConfig
   from rlightning.utils.builders import (
       build_engine,
       build_env_group,
       build_policy_group,
       build_data_buffer,
   )

   def main(config: MainConfig):
     # Build components
     env_group = build_env_group(config.env)
     policy_group = build_policy_group(
       config.policy.type, config.policy, config.cluster
     )
     buffer = build_data_buffer(config.buffer.type, config.buffer)

     # Build and run engine
     engine = build_engine(config, env_group, policy_group, buffer)
     engine.run()

   if __name__ == "__main__":
       launch(main_func=main, config_path=Path(__file__).parent / "conf")

``build_engine`` reads ``config.engine`` to select the engine class, then
creates an instance with all components. The constructor initializes runtime
components according to engine type (for example, ``eval`` does not initialize
train policy / buffer). The user-defined ``main`` function focuses only on
building the pipeline and calling ``run()``, while ``launch()`` handles the
boilerplate around configuration and cluster setup.

