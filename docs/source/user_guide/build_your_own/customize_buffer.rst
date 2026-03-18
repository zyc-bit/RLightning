Customize Buffer
================

The Data Buffer supports customization at multiple levels. You can
customize how data is preprocessed at ingestion, how completed episodes
are post-processed for training, and how data is sampled. This guide
covers the most common customization patterns and shows how to wire them
into the training pipeline.

.. container:: mermaid-height-auto

  .. mermaid::

    flowchart LR
      subgraph Preprocess["① Preprocessing"]
        ER["EnvRet"] --> ERP["env_ret_preprocess_fn"]
        PR["PolicyResponse"] --> PRP["policy_resp_preprocess_fn"]
      end
      subgraph EP["② Episode Postprocessing"]
        Raw["Raw Episode"] --> Post["postprocess_fn"]
      end
      subgraph Samp["③ Sampling"]
        Storage["Storage"] --> Sampler["Sampler"]
      end
      ERP --> Raw
      PRP --> Raw
      Post --> Storage
      Sampler --> Train["Train Workers"]

Most customization points are passed as constructor parameters to
``build_data_buffer()`` (or directly to the buffer class). One exception is
custom samplers: these are typically assigned to ``buffer.sampler`` after
``buffer.init(...)``. You only need to provide the parts you want to
override — everything else uses sensible defaults.


Customization Overview
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Customization Point
     - When to Use
     - Interface
   * - ``postprocess_fn``
     - Compute advantages, returns, or reward shaping per episode
     - ``fn(episode_dict) -> dict``
   * - ``obs_preprocessor``
     - Flatten, normalize, or one-hot encode observations
     - Callable ``fn(obs) -> obs`` (``Preprocessor`` instances also work)
   * - ``reward_preprocessor``
     - Scale, clip, or transform rewards
     - Callable ``fn(reward) -> reward`` (``Preprocessor`` instances also work)
   * - ``env_ret_preprocess_fn``
     - Change which fields are extracted from ``EnvRet``
     - ``fn(dict, EnvRet, obs_pre, rew_pre) -> dict``
   * - ``policy_resp_preprocess_fn``
     - Change which fields are extracted from ``PolicyResponse``
     - ``fn(dict, PolicyResponse) -> dict``
   * - Custom Sampler
     - Prioritized replay, stratified sampling, etc.
     - Subclass ``BaseSampler``
   * - Custom Buffer
     - Entirely new buffer behavior
     - Subclass ``DataBuffer`` + register


Custom Postprocessing
---------------------

Custom postprocessing is the most common customization. A postprocess
function is called when an episode is finalized (via
``truncate_one_episode``, ``truncate_episodes``, inline ``truncated=True``
signals, or ``auto_truncate_episode``). It receives a dict of lists — one
list per field, one element per timestep — and returns a dict of tensors
ready for storage and training.

What the Default Postprocess Function Does
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built-in ``default_postprocess_fn`` performs these steps:

1. Stacks each field's list into a tensor.
2. Shifts keys prefixed with ``last_`` by one timestep (e.g.,
   ``last_reward[1:]`` becomes ``reward``).
3. Creates ``next_observation`` from consecutive ``observation`` entries.
4. Trims policy fields (``action``, ``log_prob``, etc.) to align with
   the environment fields.
5. Filters out ``info`` keys.

This covers many standard use cases. When you need additional
computation — for example, Generalized Advantage Estimation (GAE) — you
write a custom postprocess function.

Writing a Custom Postprocess Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function signature is:

.. code-block:: python

   def my_postprocess_fn(raw_episode: dict[str, list]) -> dict[str, torch.Tensor]:
       ...

The ``raw_episode`` argument is a dictionary where each key maps to a
list of per-timestep values. For example, after a 100-step episode you
might see:

- ``raw_episode["observation"]`` — list of 101 observations (one extra
  from the initial reset)
- ``raw_episode["action"]`` — list of 101 actions (last one is unused)
- ``raw_episode["last_reward"]`` — list of 101 rewards (first one is
  a dummy from reset)
- ``raw_episode["value"]`` — list of 101 value estimates

Your function must return a dictionary of tensors with consistent batch
dimensions.

The following example shows a complete postprocess function for an algorithm
that uses Generalized Advantage Estimation (GAE):

.. code-block:: python

   import torch

   def compute_gae(rewards, values, dones, gamma, gae_lambda):
       """Compute Generalized Advantage Estimation (GAE)."""
       advantages = torch.zeros_like(rewards)
       lastgaelam = torch.zeros_like(rewards[0])

       for t in reversed(range(rewards.shape[0])):
           next_not_done = 1.0 - dones[t].float()
           current_value = values[t]
           next_value = values[t + 1]
           delta = rewards[t] + gamma * next_value * next_not_done - current_value
           lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
           advantages[t] = lastgaelam

       returns = advantages + values[:-1]
       return advantages, returns


   def episode_postprocess_fn(raw_episode: dict[str, list]) -> dict[str, torch.Tensor]:
       """Post-process raw episode data with GAE computation."""
       # 1. Convert lists to tensors
       episode = {}
       for key, value in raw_episode.items():
           if key in ["info"]:
               continue
           if isinstance(value[0], torch.Tensor):
               episode[key] = torch.stack(value).squeeze()
           else:
               episode[key] = torch.tensor(value, device="cuda")

       # 2. Align timestep offsets
       episode["last_reward"] = episode["last_reward"][1:]
       dones = torch.logical_or(episode["last_terminated"], episode["last_truncated"])
       episode["done"] = dones[1:]
       episode["action"] = episode["action"][:-1]
       episode["last_terminated"] = episode["last_terminated"][1:]
       episode["last_truncated"] = episode["last_truncated"][1:]
       episode["log_prob"] = episode["log_prob"][:-1]
       episode["entropy"] = episode["entropy"][:-1]
       episode["observation"] = episode["observation"][:-1]

       # 3. Compute GAE
       rewards = episode["last_reward"]
       values = episode["value"]
       episode["value"] = episode["value"][:-1]
       dones = episode["done"]

       GAMMA = 0.8
       GAE_LAMBDA = 0.9
       advantages, returns = compute_gae(rewards, values, dones, GAMMA, GAE_LAMBDA)
       advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

       episode["advantages"] = advantages
       episode["returns"] = returns

       return episode

Wiring It into the Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass the custom function to ``build_data_buffer()``:

.. code-block:: python

   from my_project.utils import episode_postprocess_fn
   from rlightning.utils.builders import build_data_buffer

   data_buffer = build_data_buffer(
       buffer_cls=config.buffer.type,
       buffer_cfg=config.buffer,
       postprocess_fn=episode_postprocess_fn,
   )

Or pass it directly when constructing the buffer class:

.. code-block:: python

   from rlightning.buffer import RolloutBuffer

   buffer = RolloutBuffer(
       config=config,
       postprocess_fn=episode_postprocess_fn,
   )


Custom Preprocessing
--------------------

Observation and Reward Preprocessors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preprocessors are applied at ingestion time — each ``add_transition``
call transforms observations and rewards before they enter the episode
buffer. This is useful for normalizing or reshaping raw data from the
environment.

RLightning ships three built-in preprocessor classes:

- ``NonPreprocessor`` — returns data unchanged (class-based no-op).
- ``BoxFlattenPreprocessor`` — flattens multi-dimensional ``Box``
  observations into 1D vectors.
- ``DiscretePreprocessor`` — one-hot encodes ``Discrete`` values.

By default, ``build_data_buffer()`` uses
``default_obs_preprocessor`` / ``default_reward_preprocessor``
(identity callables), not ``NonPreprocessor``.

To write a custom preprocessor, subclass ``Preprocessor`` and implement
``transform``, ``batch_transform``, and the ``shape`` property:

.. code-block:: python

   from rlightning.buffer.utils.preprocessors import Preprocessor

   class NormalizePreprocessor(Preprocessor):
       """Normalize observations by subtracting mean and dividing by std."""

       def __init__(self, space, mean, std):
           super().__init__(space)
           self.mean = mean
           self.std = std

       def transform(self, data):
           return (data - self.mean) / (self.std + 1e-8)

       def batch_transform(self, batched_data):
           return (batched_data - self.mean) / (self.std + 1e-8)

       @property
       def shape(self):
           return self.original_space.shape

Wire it into the builder:

.. code-block:: python

   data_buffer = build_data_buffer(
       buffer_cls=config.buffer.type,
       buffer_cfg=config.buffer,
       obs_preprocessor=NormalizePreprocessor(obs_space, mean=0.0, std=255.0),
   )

Custom env_ret / policy_resp Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For deeper control over field extraction, you can replace the functions
that convert ``EnvRet`` and ``PolicyResponse`` objects into the
transition dictionary. This is useful when you need to add derived fields
or change the default field mapping.

The function signatures are:

.. code-block:: python

   def custom_env_ret_preprocess_fn(
       transition_dict: dict,
       env_ret: EnvRet,
       obs_preprocessor: Preprocessor,
       reward_preprocessor: Preprocessor,
   ) -> dict:
       """Extract fields from EnvRet into the transition dict."""
       ...

   def custom_policy_resp_preprocess_fn(
       transition_dict: dict,
       policy_resp: PolicyResponse,
   ) -> dict:
       """Extract fields from PolicyResponse into the transition dict."""
       ...

.. note::

   The buffer enforces mutual exclusion rules to prevent conflicting
   customizations:

   - You **cannot** combine a custom ``obs_preprocessor`` (or
     ``reward_preprocessor``) with a custom ``env_ret_preprocess_fn``,
     because the custom ``env_ret_preprocess_fn`` would bypass the
     default logic that calls the preprocessors.
   - You **cannot** combine a custom ``obs_preprocessor`` (or
     ``reward_preprocessor``) with a custom ``preprocess_fn``, because
     the custom ``preprocess_fn`` replaces the default pipeline that calls
     those preprocessors.
   - You **cannot** combine a custom ``env_ret_preprocess_fn`` (or
     ``policy_resp_preprocess_fn``) with a custom ``preprocess_fn``,
     because the custom ``preprocess_fn`` replaces the entire pipeline
     that calls the inner functions.

   If you need full control over the per-step preprocessing pipeline,
   provide a single ``preprocess_fn`` that handles everything:

   .. code-block:: python

      def custom_preprocess_fn(
          transition_dict: dict,
          env_ret=None,
          policy_resp=None,
          obs_preprocessor=None,
          reward_preprocessor=None,
          env_ret_preprocess_fn=None,
          policy_resp_preprocess_fn=None,
      ) -> dict:
          """Full control over per-step preprocessing."""
          ...


Custom Buffer Subclass
----------------------

When function-level customizations are not enough, you can create a new
buffer type by subclassing ``DataBuffer`` (or one of its subclasses) and
registering it with the ``BUFFERS`` registry:

1. Subclass ``DataBuffer`` (or ``RolloutBuffer`` / ``ReplayBuffer``).
2. Register with ``@BUFFERS.register("MyBuffer")``.
3. Reference in your config with ``type: "MyBuffer"``.

Here is an example — a buffer that logs sampling statistics:

.. code-block:: python

   from rlightning.utils.registry import BUFFERS
   from rlightning.buffer.base_buffer import DataBuffer
   from rlightning.utils.logger import get_logger

   logger = get_logger(__name__)

   @BUFFERS.register("LoggingReplayBuffer")
   class LoggingReplayBuffer(DataBuffer):
       """Replay buffer that logs sampling statistics."""

       def sample(self, batch_size=None, shuffle=True, drop_last=True):
           sample_data = super().sample(batch_size, shuffle=shuffle, drop_last=drop_last)
           logger.info(f"Sampled {len(sample_data)} batches, buffer size: {self.size()}")
           return sample_data

To use a custom buffer, make sure the module is imported before the
buffer is constructed. If you are using RLightning's config system, add
the module to the ``imports`` list in the entry config so that it is
registered at startup:

.. code-block:: yaml

   imports:
     - my_project.buffers

   buffer:
     type: LoggingReplayBuffer
     capacity: 100000


