Data Buffer
===========

The Data Buffer is the component responsible for storing experience data
collected during rollout and providing it to training workers for policy
optimization. RLightning provides two buffer implementations that cover the
most common reinforcement learning paradigms: **RolloutBuffer** for
on-policy algorithms and **ReplayBuffer** for off-policy algorithms.

.. container:: mermaid-height-auto

   .. mermaid::

    flowchart LR
      subgraph Collect["Rollout"]
        Env["Env Workers"]
        Policy["Policy Workers"]
      end

      subgraph Buffer["Data Buffer"]
        Preprocess["Preprocess"]
        Episode["Episode<br/>Accumulation"]
        Postprocess["Postprocess"]
        Storage["Storage"]
        Sampler["Sampler"]
      end

      Train["Train Workers"]

      Env -- "EnvRet" --> Preprocess
      Policy -- "PolicyResponse" --> Preprocess
      Preprocess --> Episode
      Episode --> Postprocess
      Postprocess --> Storage
      Storage --> Sampler
      Sampler -- "sample()" --> Train

A typical data flow looks like this:

1. Environment workers produce observations and rewards (``EnvRet``).
2. Policy workers produce actions and auxiliary outputs (``PolicyResponse``).
3. The buffer receives both, preprocesses them, and accumulates transitions
   into episodes.
4. When an episode is complete, the buffer applies post-processing (e.g., GAE
   computation) and stores the result.
5. Training workers sample data from the buffer to update the policy.


Buffer Types
------------

RLightning ships two concrete buffer types. Both share the same interface for
data ingestion, episode management, preprocessing, and sampling — they differ
only in how they handle data after it has been sampled.

RolloutBuffer
~~~~~~~~~~~~~

``RolloutBuffer`` is designed for **on-policy** workflows (for example, PPO,
A2C, or TRPO). After each sampling call, the buffer is **automatically
cleared** so
that only fresh on-policy data is used for the next training step.

By default, ``RolloutBuffer`` uses the ``AllDataSampler``, which returns every
stored transition in each sampling call.

ReplayBuffer
~~~~~~~~~~~~

``ReplayBuffer`` is designed for **off-policy** workflows (for example, DQN,
SAC, TD3, or DDPG). Data **persists** after sampling and is overwritten only
when
the buffer reaches its capacity, following a circular (FIFO) replacement
strategy.

By default, ``ReplayBuffer`` uses the ``UniformSampler``, which draws
transitions uniformly at random with replacement.

.. note::

   The algorithm names above are illustrative examples of on-policy/off-policy
   usage. Actual algorithm availability depends on the policy and engine
   implementations configured in your experiment.

Comparison
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * - Property
     - RolloutBuffer
     - ReplayBuffer
   * - Policy type
     - On-policy
     - Off-policy
   * - Data after sampling
     - Automatically cleared
     - Retained
   * - Default sampler
     - ``AllDataSampler``
     - ``UniformSampler``
   * - Data reuse
     - Single use
     - Multiple reuses


Design Highlights
-----------------

- **Router-Worker architecture** — ``DataBuffer`` acts as a Router
  (similar to ``EnvGroup`` and ``PolicyGroup``): it manages the preprocessing
  pipeline and coordinates data flow, but delegates physical storage to one
  or more ``Storage`` worker instances. Each ``Storage`` worker handles its
  own shard of the data independently.

- **Automatic episode management** — The buffer auto-detects episode
  boundaries from environment signals (``last_terminated`` /
  ``last_truncated``), handles transition accumulation, and triggers
  post-processing when an episode is finalized. You can also control episode
  boundaries manually when needed.

- **Flexible data ingestion** — Both synchronous and asynchronous rollout
  patterns are supported through the same buffer interface. In practice you
  need to call different add APIs: sync paths use ``add_transition`` /
  ``add_batched_transition``, while async paths use ``add_data_async`` /
  ``add_batched_data_async``.

- **Built-in preprocessing** — Default preprocessing handles common
  transformations including reward processing and observation shifting
  (creating ``next_observation`` from consecutive steps). By default,
  observations are kept as-is (not flattened). You can customize preprocessing
  at the granularity you need — from individual fields up to the entire
  transition.

- **Scalable distributed storage** — Seamlessly scale from single-node
  unified storage to multi-node sharded storage by changing one configuration
  field (``storage.type``). Sharded storage distributes memory pressure and
  enables local sampling on each node.

- **Node affinity** — Co-locates storage shards with environment and training
  workers on the same physical node, minimizing cross-node network overhead
  in multi-node clusters.


Usage
-----

Creating a Buffer
~~~~~~~~~~~~~~~~~

Buffers can be created programmatically or loaded from a YAML configuration
file:

.. code-block:: python

   from rlightning.buffer import RolloutBuffer, ReplayBuffer
   from rlightning.utils.config import BufferConfig, StorageConfig, SamplerConfig

   # From a configuration object
   config = BufferConfig(
       type="RolloutBuffer",
       capacity=10000,
       storage=StorageConfig(type="unified", mode="fixed"),
       sampler=SamplerConfig(type="all"),
       auto_truncate_episode=True,
   )
   buffer = RolloutBuffer(config=config)

   # Or from a YAML file
   config = BufferConfig.load_yaml("path/to/buffer_config.yaml")
   buffer = RolloutBuffer(config=config)

Before adding data, the buffer must be initialized with environment metadata:

.. code-block:: python

   buffer.init(env_meta_list=env_metas, env_ids=env_ids)

Adding Data
~~~~~~~~~~~

The most common pattern is **synchronous batched addition**, where each call
provides a matched pair of environment returns and policy responses from
``EnvGroup`` and ``PolicyGroup``:

.. code-block:: python

   buffer.add_batched_transition(
       batched_env_ret=batched_env_ret,
       batched_policy_resp=batched_policy_resp,
   )

For sync APIs, ``add_transition`` / ``add_batched_transition`` accept
``is_eval`` (default ``False``). When ``is_eval=True``, transitions are marked
as evaluation data and are not added to training storage; ``truncated`` is
also ignored for storage-finalization in this mode (only ``episode_info`` stats
are recorded). Async APIs (``add_data_async`` / ``add_batched_data_async``)
do not expose an ``is_eval`` parameter.

When using asynchronous environments (e.g., ``EnvGroup.step_async``),
``EnvRet`` and ``PolicyResponse`` may arrive at different times. Use
**asynchronous batched addition** instead — the buffer automatically matches
the latest pending item by ``env_id``:

.. code-block:: python

   buffer.add_batched_data_async(batched_data=batched_env_ret)
   buffer.add_batched_data_async(batched_data=batched_policy_resp)

For single-environment cases, the corresponding methods are
``add_transition`` (sync) and ``add_data_async`` (async). If you have a
complete, already-processed episode, you can add it directly with
``add_episode``, bypassing the internal preprocessing pipeline.

Async matching assumes per-``env_id`` order is preserved and only one unmatched
step is pending per ``env_id``. Out-of-order arrival (or multiple unmatched
items for the same ``env_id``) is not guaranteed to be paired correctly.

Episode Management
~~~~~~~~~~~~~~~~~~

Episodes can be finalized (truncated) in two ways:

1. **Automatic truncation**: When ``auto_truncate_episode`` is ``True``, the
   buffer detects ``last_terminated`` or ``last_truncated`` flags in the
   environment return and automatically finalizes the episode.

2. **Manual truncation**: When ``auto_truncate_episode`` is ``False`` (the
   default), you control when episodes end. This is also useful for forcing
   truncation at a rollout boundary (e.g., after a fixed number of steps).

The streaming ``add_*`` methods accept an optional ``truncated`` parameter
(or ``truncations`` for batched variants) to signal episode boundaries inline:

.. code-block:: python

   # Signal truncation during addition
   buffer.add_transition(
       env_id=env_id,
       env_ret=env_ret,
       policy_resp=policy_resp,
       truncated=True,  # finalize this episode
   )

   # Batched: truncations is a list of booleans, one per environment
   buffer.add_batched_transition(
       batched_env_ret=batched_env_ret,
       batched_policy_resp=batched_policy_resp,
       truncations=[False, True, False],  # only env at index 1 is truncated
   )

You can also finalize episodes explicitly after adding data using
``truncate_one_episode(item)`` or ``truncate_episodes([item_1, ...])``,
where ``item`` is either an ``env_id`` string or any object with an
``env_id`` attribute (such as ``EnvRet`` or ``PolicyResponse``).
Truncating an episode triggers any registered post-processing (such as GAE
or return computation) before the data is pushed into storage.

Sampling Data
~~~~~~~~~~~~~

Call ``sample`` to draw a batch of data for training:

.. code-block:: python

   sample_data = buffer.sample(batch_size=256, shuffle=True, drop_last=True)

``sample`` returns sampled training data directly (not index dictionaries). The
return value is a list where each element is one worker's sampled mini-batch.

In ``sharded`` mode, sampling currently requires all shards to have equal data
size, and each shard samples ``batch_size // num_shards`` items.

.. note::

   When using ``RolloutBuffer``, calling ``sample`` automatically clears
   the buffer. Make sure all data has been consumed before calling
   ``sample``, as unsampled transitions will be discarded.

Custom Preprocessing
~~~~~~~~~~~~~~~~~~~~

The buffer uses a **nested preprocessing hierarchy** that processes each
transition before it is accumulated into an episode, and each completed
episode before it is pushed into storage:

.. code-block:: text

   preprocess_fn                          (per-transition, highest level)
   ├── env_ret_preprocess_fn              (EnvRet fields)
   │   ├── obs_preprocessor               (observation)
   │   └── reward_preprocessor            (reward)
   └── policy_resp_preprocess_fn          (PolicyResponse fields)

   postprocess_fn                         (per-episode, after truncation)

.. important::

   Specifying a higher-level preprocessor **bypasses** all inner functions
   beneath it. For example, passing a custom ``env_ret_preprocess_fn``
   replaces the default handling of all ``EnvRet`` fields — neither
   ``obs_preprocessor`` nor ``reward_preprocessor`` will run.

By default, the buffer handles standard preprocessing automatically:
reward processing and observation shifting (creating ``observation`` and
``next_observation`` from consecutive steps). Observation flattening is not
enabled by default.

The most common customization is providing a **post-processing function** that
runs when an episode is finalized — for example, to compute GAE:

.. code-block:: python

   def my_postprocess_fn(episode):
       """Compute GAE and discounted returns."""
       rewards = episode["last_reward"]
       values = episode["value"]
       advantages = compute_gae(rewards, values, gamma=0.99, lam=0.95)
       episode["advantages"] = advantages
       episode["returns"] = advantages + values[:-1]
       return episode

   buffer = RolloutBuffer(
       config=config,
       postprocess_fn=my_postprocess_fn,
   )

.. tip::

   RLightning provides ``default_compute_gae`` and ``default_gae_no_loop``
   utility functions in ``rlightning.buffer.utils`` for common GAE
   computation patterns.

For finer control, you can also pass custom ``obs_preprocessor`` or
``reward_preprocessor`` functions to override how individual fields are
transformed at each step.

Inspecting the Buffer
~~~~~~~~~~~~~~~~~~~~~

Several methods are available for debugging and monitoring:

.. code-block:: python

   # Current number of stored transitions
   buffer.size()
   len(buffer)

   # Retrieve all stored data (use with caution on large buffers)
   all_data = buffer.get_all()

   # Aggregate environment statistics (e.g., mean episode reward)
   stats = buffer.get_env_stats(reset=True)

   # Print timing and profiling information
   buffer.print_timing_summary()

   # Clear all stored data
   buffer.clear()


Distributed Storage
-------------------

Unified Storage
~~~~~~~~~~~~~~~

In ``unified`` mode (the default), all data is stored in a single storage
instance within the same process. This is the simplest setup and works well
for single-node experiments.

.. code-block:: yaml

   buffer:
     type: RolloutBuffer
     capacity: 10000
     storage:
       type: unified

Sharded Storage
~~~~~~~~~~~~~~~

In ``sharded`` mode, data is distributed across multiple storage shards, each
running as a separate Ray actor. It requires remote storage
(``remote_storage=True``). This is useful when:

- **Storage pressure is high**: For off-policy algorithms and mixed SFT+RL
  training, the replay buffer can grow very large. Sharding distributes the
  memory footprint across multiple processes and nodes.
- **Local sampling is preferred**: Storage shards can be placed near the
  training workers that consume their data, so each trainer samples locally
  from its own shard without cross-node communication.

.. code-block:: yaml

   buffer:
     type: ReplayBuffer
     capacity: 100000
     storage:
       type: sharded
       device: cpu

Node Affinity
~~~~~~~~~~~~~

For multi-node clusters, you can enable **node affinity** to ensure that
environment workers, storage shards, and training workers on the same physical
node communicate locally, reducing cross-node network overhead.

.. mermaid::

   flowchart TB
       subgraph Node1["Node 1"]
           direction TB
           E1["Env Worker 0"]
           E2["Env Worker 1"]
           S1["Storage 0"]
           T1["Train Worker 0"]
           T2["Train Worker 1"]
           E1 -- "write" --> S1
           E2 -- "write" --> S1
           S1 -- "sample" --> T1
           S1 -- "sample" --> T2
       end

       subgraph Node2["Node 2"]
           direction TB
           E3["Env Worker 2"]
           E4["Env Worker 3"]
           S2["Storage 1"]
           T3["Train Worker 2"]
           T4["Train Worker 3"]
           E3 -- "write" --> S2
           E4 -- "write" --> S2
           S2 -- "sample" --> T3
           S2 -- "sample" --> T4
       end

.. code-block:: yaml

   buffer:
     type: RolloutBuffer
     capacity: 100000
     storage:
       type: sharded
     node_affinity_env: true
     node_affinity_train: true

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Effect
   * - ``node_affinity_env: true``
     - Environment workers write to the storage shard on the same node.
   * - ``node_affinity_train: true``
     - Training workers sample from the storage shard on the same node.
