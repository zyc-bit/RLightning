Env & EnvGroup
==============

Env and EnvGroup are the components responsible for managing environment
interactions in RLightning. **Env** provides a unified interface to different
simulation backends (MuJoCo, ManiSkill, Isaac Sim, real robots, etc.),
while **EnvGroup** manages multiple Env instances as a single unit for
batched, parallel data collection.


A typical interaction cycle works as follows:

1. Environments are initialized and reset to produce initial observations
   (``EnvRet``).
2. Policy workers receive observations and produce actions
   (``PolicyResponse``).
3. EnvGroup distributes actions to each Env, which steps the underlying
   simulator.
4. EnvGroup collects the results and returns them as batched data
   (``BatchedData``).
5. The cycle repeats; collected data flows to the Data Buffer for training.


Env
---

Env is a unified wrapper around different simulation backends. All
implementations inherit from ``BaseEnv`` and expose the same ``step()`` /
``reset()`` interface regardless of the underlying simulator.

.. container:: mermaid-width-80

  .. mermaid::

    flowchart TB
       Base["BaseEnv"] --> Mujoco["MujocoEnv"]
       Base --> Maniskill["ManiskillEnv"]
       Base --> ALE["ALEEnv"]
       Base --> Isaac["IsaacEnv"]
       Base --> Custom["YourCustomEnv"]

**Data Interface**

The ``step()`` method follows a consistent flow across all backends:

1. **Input**: receives a ``PolicyResponse`` object.
2. **Preprocessing**: extracts the ``action`` from the ``PolicyResponse``
   via a configurable preprocessing function.
3. **Backend execution**: passes the action to the simulator's native
   ``step()`` method.
4. **Output**: wraps the results (observation, reward, terminated,
   truncated, info) as an ``EnvRet`` object and returns it.

Users can implement custom environments by subclassing ``BaseEnv`` and
implementing ``reset()`` and ``step()``.

**Built-in Backends**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Backend
     - Description
   * - ``mujoco``
     - MuJoCo physics environments (Ant, HalfCheetah, Hopper, Humanoid, etc.)
   * - ``maniskill``
     - ManiSkill manipulation environments (supports vectorized envs)
   * - ``ale``
     - Arcade Learning Environment (Atari games)
   * - ``isaac_manager_based``
     - NVIDIA Isaac Sim environments (supports vectorized envs)
   * - ``isaac_marl``
     - NVIDIA Isaac Lab MARL-style environments (supports vectorized envs)
   * - ``piper``
     - Piper real robot environments
   * - ``env_server``
     - Remote environment via ZMQ (for distributed setups)


EnvGroup
--------

EnvGroup manages multiple Env instances as a single unit. Users interact
with EnvGroup rather than individual Envs — it handles batching,
dispatching, and result collection.

.. container:: mermaid-width-50

  .. mermaid::

    flowchart TB
        EG["EnvGroup"] --> E1["Env 1"]
        EG --> E2["Env 2"]
        EG --> E3["..."]
        EG --> EN["Env n"]

EnvGroup supports two stepping modes:

- **Synchronous**: ``step()`` — blocks until all environments return.
- **Asynchronous**: ``step_async()`` + ``collect_async()`` — non-blocking,
  returns results as they become available.


Design Highlights
-----------------

- **Standardized interface** — All environments expose the same ``step()``
  and ``reset()`` interface with unified input (``PolicyResponse``) and
  output (``EnvRet``), regardless of whether the underlying backend is
  MuJoCo, ManiSkill, Isaac Sim, or a real robot.

- **Backend abstraction** — Switching between simulation backends, or
  between simulation and real hardware, requires only a configuration
  change. Your training loop code stays the same.

- **Batched execution** — EnvGroup provides a unified ``step`` interface
  for parallel interaction with multiple environments. You interact with
  EnvGroup as if it were a single environment.

- **Asynchronous stepping** — ``step_async`` and ``collect_async`` separate
  environment computation from result collection, enabling overlap between
  environment steps and policy inference to maximize throughput.

- **Automatic reset** — The ``auto_reset`` context manager automatically
  resets environments when they reach a maximum step count, simplifying
  control flow in multi-environment rollout loops.


Configuration
-------------

The environment is configured through the ``envs`` section of the
experiment configuration file. Each item in the list defines one type of
environment.

.. code-block:: yaml

   envs:
     - name: "halfcheetah"
       backend: "mujoco"
       task: "HalfCheetah-v5"
       num_workers: 4

The full set of configuration fields is described below.

``name`` (str) **[required]**
  A user-defined name for this environment entry. Recommended format:
  ``backend_task``.

``backend`` (str) **[required]**
  Simulator backend type. Supported values: ``"mujoco"``,
  ``"maniskill"``, ``"ale"``, ``"isaac_manager_based"``,
  ``"isaac_marl"``, ``"piper"``, ``"env_server"``.

``task`` (str) **[required]**
  Task or environment name within the backend (e.g.,
  ``"HalfCheetah-v5"``, ``"StackCube-v1"``).

``num_workers`` (int)
  Number of independent environment worker instances to create. In
  distributed mode each worker runs as a separate Ray actor with its
  own process and resource allocation. Defaults to ``1``.

``num_envs`` (int)
  Number of vectorized sub-environments within each worker instance.
  Only supported by ``maniskill``, ``isaac_manager_based``, and
  ``isaac_marl``. Defaults to ``1``.

``max_episode_steps`` (int | None)
  Maximum steps per episode. Used by the ``auto_reset`` context manager.
  Defaults to ``None`` (no limit).

``num_cpus`` (int)
  CPUs allocated per environment worker (remote execution only). Defaults
  to ``1``.

``num_gpus`` (float)
  GPUs allocated per environment worker (remote execution only). Defaults
  to ``0.0``.

``env_kwargs`` (dict)
  Backend-specific configuration passed to the environment constructor.

**Multi-task example**

Multiple environment types can be combined for multi-task training:

.. code-block:: yaml

   envs:
     - name: "stack_cube_task"
       backend: "maniskill"
       task: "StackCube-v1"
       num_workers: 2
       num_envs: 1
       max_episode_steps: 100
     - name: "pick_cube_task"
       backend: "maniskill"
       task: "PickCube-v1"
       num_workers: 1
       num_envs: 2
       max_episode_steps: 50
       num_gpus: 0.1

.. note::

   In this configuration, EnvGroup creates 3 independent worker instances
   in total (``num_workers`` sum: 2 + 1). Each instance is assigned a unique
   identifier at initialization. ``num_envs`` defines the number of
   vectorized sub-environments within each worker instance.


Usage
-----

Creating an EnvGroup
~~~~~~~~~~~~~~~~~~~~

Use ``build_env_group`` to create an EnvGroup from configuration:

.. code-block:: python

   from rlightning.env import build_env_group

   env_group = build_env_group(config.envs, preprocess_fn=env_preprocess_fn)

The ``preprocess_fn`` parameter controls how ``PolicyResponse`` is
converted to an action before being passed to the simulator:

- **Default** (``default_env_preprocess_fn``): extracts the ``action``
  field from the ``PolicyResponse``.
- **Single function**: applied to all Env instances.
- **List of functions**: one per Env instance (must match the total
  number of environments).

Initializing
~~~~~~~~~~~~

.. code-block:: python

   env_metas = env_group.init()

``init()`` initializes all Env instances and returns a list of ``EnvMeta``
objects, each containing ``env_id``, ``observation_space``,
``action_space``, and ``num_envs``. For consistency checks, the current
implementation validates only ``action_space`` shape compatibility.

Synchronous Step & Reset
~~~~~~~~~~~~~~~~~~~~~~~~

The standard synchronous loop resets all environments, then alternates
between policy rollout and environment stepping:

.. code-block:: python

   batched_env_ret, _ = env_group.reset(seed=0)

   for _ in range(max_rollout_steps):
       batched_policy_resp = policy_group.rollout_batch(batched_env_ret)
       buffer.add_batched_transition(batched_env_ret, batched_policy_resp)
       batched_env_ret, _ = env_group.step(batched_policy_resp)

- ``reset()`` resets all environments and returns
  ``(BatchedData, truncations)``.
- ``step()`` dispatches each ``PolicyResponse`` to its corresponding Env
  by ``env_id`` and returns ``(BatchedData, truncations)``. Each Env
  instance is assigned a unique identifier (UUID + worker index) at
  initialization.
- ``reset(seed=...)`` accepts an ``int`` seed (for example, ``seed=0``).
- ``reset()`` always returns ``List[bool]`` for ``truncations``
  (default all ``False``).
- ``step()`` returns ``None`` for ``truncations`` outside ``auto_reset``;
  inside ``auto_reset`` it returns ``List[bool]``.

Asynchronous Step & Reset
~~~~~~~~~~~~~~~~~~~~~~~~~

Asynchronous stepping separates triggering environment computation from
collecting results:

.. code-block:: python

   env_group.step_async(batched_policy_resp)
   batched_env_ret, truncations = env_group.collect_async()

- ``step_async()`` triggers environment steps without blocking.
- ``collect_async()`` returns results from environments that have finished, 
  blocks until at least one result is ready, then returns available results.
- ``collect_async(timeout=0.1)`` uses ``timeout`` as an additional
  collection window *after* the first ready result arrives, to gather more
  completed results.
- ``collect_async(wait_all=True)`` blocks until all environments finish.

.. tip::

   Async stepping avoids idle time. Policy inference can begin as soon as
   any environment returns, rather than waiting for all environments to
   finish.

Auto Reset
~~~~~~~~~~

.. code-block:: python

   with env_group.auto_reset(max_episode_steps=100):
       for _ in range(total_steps):
           env_group.step_async(batched_policy_resp)
           batched_env_ret, truncations = env_group.collect_async()
           # truncations indicates which environments were auto-reset

- Context manager that auto-resets environments reaching the maximum step
  count.
- ``max_episode_steps`` accepts a positive ``int`` applied uniformly to
  all environments. If omitted (``None``), each environment uses its own
  ``max_episode_steps`` from config (raises ``ValueError`` if any env has
  it unset).
- Inside the context, ``step`` / ``step_async`` automatically check and
  reset environments at the step limit.
- In this context, ``step()`` returns ``List[bool]`` for ``truncations``
  (instead of ``None``), and ``collect_async()`` returns per-result
  truncation flags (instead of all ``False``).

Inspecting the EnvGroup
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   len(env_group)                       # Number of environments
   env_group.size()                     # Number of local env instances
   env = env_group[env_id]              # Get a specific Env by ID
   env_group.get_observation_spaces()   # List of observation spaces
   env_group.get_action_spaces()        # List of action spaces
   env_group.get_stats()                # Throughput statistics
   env_group.print_timing_summary()     # Print timing info

- ``len(env_group)`` counts local env instances + ``env_server`` entries.
- ``env_group.size()`` counts only local env instances (excludes
  ``env_server`` entries).
