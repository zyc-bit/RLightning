Customize Env
=============

Env wraps simulation backends behind a mostly unified ``reset()`` /
``step()`` interface, so that the rest of the training pipeline
(EnvGroup, Policy, Buffer, Engine) does not need to know which simulator
is running. Some remote backends (for example ``env_server``) are
async-first and use ``step_async()`` / ``collect_async()`` instead of
synchronous ``step()``.
Customization ranges from a one-line preprocess function to a full
custom Env subclass with GPU memory management.

This guide covers the three levels of customization and shows how to
wire custom environments into the training pipeline.


Customization Overview
----------------------

.. list-table::
   :header-rows: 1
   :widths: 10 30 30 30

   * - Level
     - Approach
     - When to Use
     - What You Change
   * - 1
     - Custom preprocess function
     - Scale, clip, or reshape actions before the simulator steps
     - ``preprocess_fn`` argument to ``build_env_group()``
   * - 2
     - Custom Env subclass
     - Integrate a new simulator or robot backend
     - Subclass ``BaseEnv``, implement ``reset()`` / ``step()``, register
   * - 3
     - Advanced action processing inside ``step()``
     - Complex action transformations (chunking, coordinate conversion)
     - Override action extraction logic within ``step()``


Level 1: Custom Preprocess Function
------------------------------------

The preprocess function controls how a ``PolicyResponse`` is converted
to an action before the simulator steps. For standard backends, it is
called inside ``step()`` via ``self._preprocess_fn(policy_resp)``.

**Default behavior**

The built-in ``default_env_preprocess_fn`` simply extracts the
``action`` field:

.. code-block:: python

   def default_env_preprocess_fn(policy_resp: PolicyResponse):
       return policy_resp.action

**Writing a custom preprocess function**

Here presents an example: A custom preprocess function receives a ``PolicyResponse`` and returns
the action (or transformed action) to pass to the simulator:

.. code-block:: python

   import numpy as np
   from rlightning.types import PolicyResponse

   def scaled_preprocess_fn(policy_resp: PolicyResponse):
       """Scale continuous actions to [-1, 1] range."""
       action = policy_resp.action
       return np.clip(action * 0.5, -1.0, 1.0)

**Wiring it in**

Pass the function to ``build_env_group()``:

.. code-block:: python

   from rlightning.utils.builders import build_env_group

   env_group = build_env_group(
       env_cfgs=config.env,
       preprocess_fn=scaled_preprocess_fn,
   )

**Per-env preprocess functions**

When different environments need different preprocessing, pass a list of
functions — one per environment config entry:

.. code-block:: python

   env_group = build_env_group(
       env_cfgs=[env_cfg_a, env_cfg_b],
       preprocess_fn=[preprocess_fn_a, preprocess_fn_b],
   )

Each function is replicated across the ``num_workers`` instances of its
corresponding config entry.

.. note::

   The preprocess function is called inside the Env actor (or process).
   It must be serializable by Ray if environments run remotely. Avoid
   closures that capture non-serializable objects.
   ``env_server`` is an exception: server-side ``preprocess_fn`` is
   ignored, and preprocessing should be handled on the client side.


Level 2: Custom Env Subclass
-----------------------------

To integrate a new simulator or robot backend, subclass ``BaseEnv`` and
implement the two abstract methods: ``reset()`` and ``step()``.
For async-first remote-server patterns, ``step()`` may intentionally
raise ``NotImplementedError`` while ``step_async()`` /
``collect_async()`` provide the actual interaction path.

Core Pattern
~~~~~~~~~~~~

.. code-block:: python

   from rlightning.env.base_env import BaseEnv
   from rlightning.types import EnvRet, PolicyResponse
   from rlightning.utils.registry import ENVS

   @ENVS.register("my_backend")
   class MyEnv(BaseEnv):

       def __init__(self, config, worker_index=0, preprocess_fn=None, **kwargs):
           super().__init__(config, worker_index, preprocess_fn)
           # Create the underlying simulator
           self.env = ...  # your simulator initialization

       def reset(self, *args, **kwargs) -> EnvRet | list[EnvRet]:
           observation, info = self.env.reset(*args, **kwargs)
           return EnvRet(
               env_id=self.env_id,
               observation=observation,
               info=info,
           )

       def step(self, policy_resp: PolicyResponse) -> EnvRet:
           action = self._preprocess_fn(policy_resp)
           observation, reward, terminated, truncated, info = self.env.step(action)
           return EnvRet(
               env_id=self.env_id,
               observation=observation,
               last_reward=reward,
               last_terminated=terminated,
               last_truncated=truncated,
               info=info,
           )

Key points:

- ``@ENVS.register("my_backend")`` registers the class so it can be
  referenced in YAML config with ``backend: my_backend``.
- The constructor receives ``config`` (``EnvConfig``), ``worker_index``
  (``int``), and ``preprocess_fn`` (``Callable``). Always call
  ``super().__init__()`` first.
- ``self.env_id`` is auto-generated by ``BaseEnv.__init__()`` and
  **must** be included in every ``EnvRet`` you return.
- Use ``self._preprocess_fn(policy_resp)`` in ``step()`` to extract the
  action from the ``PolicyResponse``.
- ``reset()`` may return ``EnvRet`` or ``List[EnvRet]`` depending on
  backend/runtime mode (e.g., some remote backends return batched lists).

Constructor Details
~~~~~~~~~~~~~~~~~~~

``BaseEnv.__init__()`` handles:

- Deep-copying the config (``self.config``).
- Generating a unique ``self.env_id`` (UUID + worker index).
- Storing ``self.num_envs`` and ``self.max_episode_steps`` from config.
- Storing the preprocess function as ``self._preprocess_fn``.

Your subclass constructor should create the underlying simulator and
store observation/action spaces:

.. code-block:: python

   def __init__(self, config, worker_index=0, preprocess_fn=None, **kwargs):
       super().__init__(config, worker_index, preprocess_fn)
       self.env = gym.make(self.config.task, max_episode_steps=self.config.max_episode_steps)
       self.observation_space = self.env.observation_space
       self.action_space = self.env.action_space

EnvRet Fields
~~~~~~~~~~~~~

``EnvRet`` is a dataclass with the following fields:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``env_id`` (str)
     - **required**
     - Unique environment identifier (use ``self.env_id``)
   * - ``observation`` (Any)
     - **required**
     - Observation after step/reset
   * - ``last_reward`` (float | Tensor | ndarray)
     - ``0.0``
     - Reward received after the last step
   * - ``last_terminated`` (bool | Tensor | ndarray)
     - ``False``
     - Whether the episode terminated
   * - ``last_truncated`` (bool | Tensor | ndarray)
     - ``False``
     - Whether the episode was truncated
   * - ``info`` (dict)
     - ``{}``
     - Additional info from the environment
   * - ``_extra`` (dict)
     - ``{}``
     - Extra fields for extensibility
   * - ``ts_env_sent_ns`` (int)
     - auto-filled
     - Internal timestamp (ns) set when EnvRet is produced/sent

For ``reset()``, you typically only set ``env_id``, ``observation``, and
optionally ``info``. For ``step()``, you should set all relevant fields.

In vectorized environments (``num_envs > 1``), reward and done fields are
often batched tensors/arrays rather than Python scalars.

Real Example: MujocoEnv
~~~~~~~~~~~~~~~~~~~~~~~~

``MujocoEnv`` is the simplest built-in backend. It wraps a Gymnasium
MuJoCo environment:

.. code-block:: python

   from rlightning.env.base_env import BaseEnv
   from rlightning.env.utils.utils import default_env_preprocess_fn
   from rlightning.types import EnvRet, PolicyResponse
   from rlightning.utils.registry import ENVS

   import gymnasium as gym
   import numpy as np

   @ENVS.register("mujoco")
   class MujocoEnv(BaseEnv):

       def __init__(self, config, worker_index=0,
                    preprocess_fn=default_env_preprocess_fn, **kwargs):
           super().__init__(config, worker_index, preprocess_fn)
           self.env = gym.make(
               self.config.task,
               max_episode_steps=self.config.max_episode_steps,
           )
           self.observation_space = self.env.observation_space
           self.action_space = self.env.action_space

       def reset(self, *args, **kwargs) -> EnvRet:
           observation, info = self.env.reset(*args, **kwargs)
           return EnvRet(env_id=self.env_id, observation=observation, info=info)

       def step(self, policy_resp: PolicyResponse) -> EnvRet:
           action = self._preprocess_fn(policy_resp)
           action = np.asarray(action)
           observation, reward, terminated, truncated, info = self.env.step(action)
           return EnvRet(
               env_id=self.env_id,
               observation=observation,
               last_reward=reward,
               last_terminated=terminated,
               last_truncated=truncated,
               info=info,
           )


Optional Overrides
------------------

``BaseEnv`` provides several methods you can override for advanced use
cases. Most have sensible default implementations; ``offload()`` and
``reload()`` raise ``NotImplementedError`` in the base class and must be
implemented if your environment needs GPU memory management.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - ``get_observation_space()``
     - Return custom observation space (default: ``self.env.observation_space``)
   * - ``get_action_space()``
     - Return custom action space (default: ``self.env.action_space``)
   * - ``init()``
     - Custom initialization logic (default: returns ``get_metadata()``)
   * - ``close()``
     - Cleanup when the environment is shut down (default: no-op)
   * - ``offload()``
     - Free GPU memory for colocated mode (default: raises ``NotImplementedError``)
   * - ``reload()``
     - Restore GPU memory after offload (default: raises ``NotImplementedError``)
   * - ``is_finish()``
     - Signal when a remote env client loop should stop (default: ``False``)

Observation and Action Spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Override ``get_observation_space()`` and ``get_action_space()`` when your
environment's spaces differ from ``self.env.observation_space`` (for
example, when you extract image observations from a dict-space):

.. code-block:: python

   import gymnasium as gym

   def get_observation_space(self) -> gym.Space:
       # Return only the image part of the observation space
       return gym.spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8)

   def get_action_space(self) -> gym.Space:
       return self.env.action_space

GPU Memory Management: offload / reload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In colocated mode (policy and environment share the same GPU), RLightning
calls ``offload()`` to free GPU memory before training and ``reload()``
to restore it before rollout. Override these methods when your
environment holds GPU resources.

The ``ManiskillEnv`` implementation is a real-world example:

.. code-block:: python

   def offload(self):
       """Free GPU memory by closing and deleting the environment."""
       if self._is_offloaded:
           return
       if hasattr(self.env, "close"):
           try:
               self.env.close()
           except Exception:
               pass
       del self.env
       self.env = None
       self.clear_memory()  # gc.collect() + torch.cuda.empty_cache()
       self._is_offloaded = True

   def reload(self):
       """Recreate the environment to restore GPU resources."""
       if not self._is_offloaded:
           return
       self.env = gym.make(**self.env_config)
       self._is_offloaded = False

.. tip::

   Track the offload state with a boolean flag (``self._is_offloaded``)
   and guard ``reset()`` / ``step()`` to raise an error if called while
   offloaded. ``ManiskillEnv`` raises ``RuntimeError`` in both methods
   if ``self._is_offloaded`` is ``True``.


Wiring a Custom Env
--------------------

Step 1: Register
~~~~~~~~~~~~~~~~

Use the ``@ENVS.register()`` decorator on your class:

.. code-block:: python

   from rlightning.utils.registry import ENVS
   from rlightning.env.base_env import BaseEnv

   @ENVS.register("my_backend")
   class MyEnv(BaseEnv):
       ...

Step 2: Configure
~~~~~~~~~~~~~~~~~

Reference the registered name in your YAML config:

.. code-block:: yaml

   env:
     - name: "my_task"
       backend: "my_backend"
       task: "MyTask-v1"
       num_workers: 2
       max_episode_steps: 200

Step 3: Import
~~~~~~~~~~~~~~

Ensure the module containing your registered class is imported before
``build_env_group()`` runs. Use the ``imports`` field in your entry
config:

.. code-block:: yaml

   imports:
     - my_project.envs  # Module containing @ENVS.register(...)

   env:
     - name: "my_task"
       backend: "my_backend"
       task: "MyTask-v1"
       num_workers: 2

