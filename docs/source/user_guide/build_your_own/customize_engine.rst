Customize Engine
================

The Engine orchestrates the reinforcement learning training loop:
rollout |rarr| update dataset |rarr| train |rarr| sync weights. RLightning
provides several built-in engines (see :doc:`../core_components/engine`),
but you can customize the loop at three levels — from lightweight
lifecycle hooks to full custom engine subclasses.

.. |rarr| unicode:: U+2192

.. mermaid::

   flowchart LR
      subgraph Epoch["Each Epoch"]
         direction LR
         R["Rollout"] --> U["Update Dataset"] --> T["Train"] --> S["Sync Weights"]
      end

      PreR["_pre_rollout_hook"] -.-> R
      R -.-> PostR["_post_rollout_hook"]
      PreU["_pre_update_dataset_hook"] -.-> U
      U -.-> PostU["_post_update_dataset_hook"]
      PreT["_pre_train_hook"] -.-> T
      T -.-> PostT["_post_train_hook"]
      PreS["_pre_sync_weights_hook"] -.-> S
      S -.-> PostS["_post_sync_weights_hook"]

This guide covers the three customization levels and shows how to wire
custom engines into the training pipeline.


Customization Overview
----------------------

.. list-table::
   :header-rows: 1
   :widths: 10 25 30 35

   * - Level
     - Approach
     - When to Use
     - What You Override
   * - 1
     - Lifecycle hooks
     - Add logging, LR scheduling, or other side effects around existing phases
     - ``_pre_*_hook()`` / ``_post_*_hook()`` methods
   * - 2
     - Override core methods
     - Change how rollout, training, or dataset update works while reusing the rest
     - ``rollout()``, ``train()``, ``update_dataset()``, ``warm_up()``, or ``run()``
   * - 3
     - Full custom engine
     - Entirely new training loop or coordination pattern
     - Subclass ``BaseEngine``, implement all abstract methods


Level 1: Lifecycle Hooks
------------------------

The easiest way to customize engine behavior is to override hook methods.
Each phase of the training loop is wrapped by an internal method that
calls a pre-hook and post-hook around the user-facing method:

.. code-block:: text

   _rollout()          = _pre_rollout_hook()  → rollout()  → _post_rollout_hook()
   _update_dataset()   = _pre_update_dataset_hook() → update_dataset() → _post_update_dataset_hook()
   _train()            = _pre_train_hook()    → train()    → _post_train_hook()
   _sync_weights()     = _pre_sync_weights_hook() → sync_weights() → _post_sync_weights_hook()

Override any hook to inject custom logic before or after a phase. The
default hook implementations handle colocated-mode offload/reload; if
you don't need that behavior, you can skip calling ``super()``.

Available Hooks
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Hook Method
     - Default Behavior
   * - ``_pre_rollout_hook()``
     - Reloads eval model and environments when colocated offload is enabled
   * - ``_post_rollout_hook()``
     - Offloads eval model and environments when colocated offload is enabled
   * - ``_pre_update_dataset_hook()``
     - No-op (``AsyncRLEngine`` overrides this to wait for enough data)
   * - ``_post_update_dataset_hook()``
     - No-op
   * - ``_pre_train_hook()``
     - Reloads model parameters, gradients, and optimizer when colocated offload is enabled
   * - ``_post_train_hook()``
     - No-op
   * - ``_pre_sync_weights_hook()``
     - Offloads optimizer when colocated offload is enabled
   * - ``_post_sync_weights_hook()``
     - Offloads model parameters and gradients when colocated offload is enabled

.. note::

   If you use colocated mode (``cluster.enable_offload: true``), call
   ``super()`` in your hook override so the offload/reload logic is
   preserved.

Example: Logging After Each Training Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rlightning.engine.sync_rl_engine import SyncRLEngine
   from rlightning.utils.registry import ENGINE
   from rlightning.utils.logger import get_logger

   logger = get_logger(__name__)

   @ENGINE.register("syncrl_with_logging")
   class LoggingSyncRLEngine(SyncRLEngine):
       """SyncRLEngine with custom post-train logging."""

       def _post_train_hook(self) -> None:
           super()._post_train_hook()
           logger.info(f"Epoch {self.epoch}: training step completed")
           logger.info(f"Buffer size: {len(self.buffer)}")

Example: Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rlightning.engine.sync_rl_engine import SyncRLEngine
   from rlightning.utils.registry import ENGINE
   from rlightning.utils.logger import get_logger

   logger = get_logger(__name__)

   @ENGINE.register("syncrl_lr_schedule")
   class LRScheduleSyncRLEngine(SyncRLEngine):
       """SyncRLEngine with linear learning rate warmup."""

       def _pre_train_hook(self) -> None:
           super()._pre_train_hook()

           warmup_epochs = 10
           if self.epoch < warmup_epochs:
               scale = (self.epoch + 1) / warmup_epochs
               for policy in self.policy_group.train_list:
                   for param_group in policy.optimizer.param_groups:
                       param_group["lr"] = self.config.train.lr * scale

               logger.info(f"Epoch {self.epoch}: LR warmup scale = {scale:.2f}")


Level 2: Overriding Core Methods
---------------------------------

For deeper changes, subclass an existing engine and override one or more
core methods. The training loop in ``SyncRLEngine.run()`` calls the
internal wrapper methods in sequence:

.. code-block:: python

   def run(self) -> None:
       for self.epoch in self.iter_epochs(num_epochs=self.config.train.max_epochs):
           self._rollout(obj_set="train", prefix="rollout")
           self._update_dataset()
           self._train()
           self._sync_weights()
           # ... periodic evaluation and checkpointing

Override the **unwrapped** versions (``rollout()``, ``train()``,
``update_dataset()``) — the internal wrappers (``_rollout()``,
``_train()``, etc.) handle profiling and hook invocation automatically.

.. tip::

   You can also override ``run()`` itself to change the epoch structure
   (e.g., add extra rollout phases, change evaluation frequency, or
   implement curriculum learning).

Real-World Example: RSLRLEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RSLRLEngine`` extends ``SyncRLEngine`` by overriding ``rollout()``,
``update_dataset()``, ``train()``, and ``warm_up()``. It adds
asynchronous environment stepping (``step_async()`` /
``collect_async()``) and policy post-processing to the rollout loop while
keeping the same epoch structure.

Key differences from ``SyncRLEngine``:

- **rollout()**: Uses ``step_async()`` / ``collect_async()`` for
  non-blocking environment interaction, and calls
  ``policy_group.postprocess()`` after each inference step.
- **update_dataset()**: For on-policy updates, batch size depends on sampler
  type: if ``buffer.sampler.type == "all"``, use full-buffer sampling;
  otherwise use ``num_envs * num_train_policies``.
- **train()**: Adds metric aggregation and logging.
- **warm_up()**: Simplified — just syncs weights without a dummy run.

.. code-block:: python

   from rlightning.engine.sync_rl_engine import SyncRLEngine
   from rlightning.utils.registry import ENGINE
   from rlightning.utils.logger import get_logger

   logger = get_logger(__name__)

   @ENGINE.register("rsl")
   class RSLRLEngine(SyncRLEngine):

       def warm_up(self):
           self.sync_weights()
           logger.info("Warm up done, ready to run.")

       def rollout(self, obj_set, prefix="", is_eval=False):
           # Use async env stepping + policy postprocessing
           ...
           batched_policy_resp = self.policy_group.rollout_batch(batched_env_ret)
           processed = self.policy_group.postprocess(batched_env_ret, batched_policy_resp)
           self.buffer.add_batched_data_async(processed)
           ...

       def update_dataset(self):
           # Adjust batch size for on-policy training
           if self.config.buffer.sampler.type == "all":
               batch_size = len(self.buffer)
           else:
               batch_size = self.num_envs * len(self.policy_group.train_list)
           data = self.buffer.sample(batch_size=batch_size)
           self.policy_group.update_dataset(data)

       def train(self):
           training_info = self.policy_group.train()
           # Aggregate and log metrics
           ...

Example: Adding Evaluation After Each Training Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rlightning.engine.sync_rl_engine import SyncRLEngine
   from rlightning.utils.registry import ENGINE
   from rlightning.utils.logger import get_logger

   logger = get_logger(__name__)

   @ENGINE.register("syncrl_eval_every_epoch")
   class EvalEverySyncRLEngine(SyncRLEngine):
       """SyncRLEngine that evaluates after every training step."""

       def run(self) -> None:
           for self.epoch in self.iter_epochs(num_epochs=self.config.train.max_epochs):
               self._rollout(obj_set="train", prefix="rollout")
               self._update_dataset()
               self._train()
               self._sync_weights()

               # Evaluate every epoch
               logger.info(f"Evaluating at epoch {self.epoch}")
               self._rollout(obj_set="train", prefix="eval", is_eval=True)

               if (self.epoch + 1) % self.config.train.save_interval == 0:
                   ckpt_path = f"{self.config.train.save_dir}/epoch_{self.epoch}.pt"
                   self.policy_group.save_checkpoint(path=ckpt_path)

           # Final checkpoint
           ckpt_path = f"{self.config.train.save_dir}/epoch_last.pt"
           self.policy_group.save_checkpoint(path=ckpt_path)
           logger.info("Done.")


Level 3: Custom Engine from BaseEngine
---------------------------------------

For completely custom training loops, subclass ``BaseEngine`` directly.
You must implement all five abstract methods:

- **warm_up()** — initialization and optional dummy run
- **run()** — the top-level training loop
- **rollout()** — collect experience from environments
- **update_dataset()** — move data from buffer to policy
- **train()** — perform a training step

Register your engine with ``@ENGINE.register("name")`` so it can be
selected via configuration.

.. note::

   ``launch()`` converts YAML into ``MainConfig`` and currently validates
   ``engine`` against built-in literals only (``syncrl``, ``asyncrl``,
   ``rsl``, ``async_rsl``, ``eval``). To use a custom engine name in YAML,
   extend the ``MainConfig.engine`` type in your project first.

.. code-block:: python

   from rlightning.engine.base_engine import BaseEngine
   from rlightning.utils.registry import ENGINE
   from rlightning.utils.logger import get_logger, log_metric

   logger = get_logger(__name__)

   @ENGINE.register("curriculum")
   class CurriculumEngine(BaseEngine):
       """Custom engine with curriculum-based rollout."""

       def __init__(self, config, env_group=None, policy_group=None, buffer=None):
           super().__init__(config, env_group, policy_group, buffer)

           # Initialize components
           env_meta_list = self.env_group.init()
           self.policy_group.init_train(self.config.train, env_meta_list[0])
           self.policy_group.init_eval(env_meta=env_meta_list[0])
           env_ids = self.env_group.env_ids
           self.buffer.init(env_meta_list, env_ids)
           self.sync_weights()

           # Curriculum: define difficulty stages
           self.stages = [
               {"obj_set": "easy", "epochs": 50},
               {"obj_set": "medium", "epochs": 30},
               {"obj_set": "hard", "epochs": 20},
           ]

       def warm_up(self) -> None:
           self.sync_weights()
           logger.info("Curriculum engine warmed up.")

       def rollout(self, obj_set: str) -> None:
           batched_policy_resp = None
           max_steps = self.config.train.max_rollout_steps
           with self.env_group.auto_reset(max_episode_steps=max_steps):
               for _ in range(max_steps + 1):
                   if batched_policy_resp is None:
                       batched_env_ret, truncations = self.env_group.reset(
                           options={"obj_set": obj_set},
                       )
                   else:
                       batched_env_ret, truncations = self.env_group.step(
                           batched_policy_resp,
                       )
                   batched_policy_resp = self.policy_group.rollout_batch(
                       batched_env_ret,
                   )
                   self.buffer.add_batched_transition(
                       batched_env_ret, batched_policy_resp, truncations,
                   )

           env_stats = self.buffer.get_env_stats(reset=True)
           log_metric(env_stats, step=self.epoch, prefix="rollout")

       def update_dataset(self) -> None:
           data = self.buffer.sample(batch_size=self.config.train.batch_size)
           self.policy_group.update_dataset(data)

       def train(self) -> None:
           training_info = self.policy_group.train()
           if training_info is not None:
               log_metric(training_info, step=self.epoch, prefix="train")

       def run(self) -> None:
           global_epoch = 0
           for stage in self.stages:
               obj_set = stage["obj_set"]
               num_epochs = stage["epochs"]
               logger.info(f"Starting curriculum stage: {obj_set} ({num_epochs} epochs)")

               for _ in range(num_epochs):
                   self.epoch = global_epoch
                   self._rollout(obj_set=obj_set)
                   self._update_dataset()
                   self._train()
                   self._sync_weights()

                   if (global_epoch + 1) % self.config.train.save_interval == 0:
                       ckpt_path = f"{self.config.train.save_dir}/epoch_{global_epoch}.pt"
                       self.policy_group.save_checkpoint(path=ckpt_path)

                   global_epoch += 1

           ckpt_path = f"{self.config.train.save_dir}/epoch_last.pt"
           self.policy_group.save_checkpoint(path=ckpt_path)
           logger.info("Curriculum training done.")

Wiring a Custom Engine
~~~~~~~~~~~~~~~~~~~~~~

To use a custom engine, configure the ``engine`` field and make sure the
module is imported at startup:

.. note::

   If you use ``launch()`` + ``MainConfig``, custom engine names require
   extending ``MainConfig.engine`` first. Otherwise, config validation fails
   before registry lookup.

.. code-block:: yaml

   imports:
     - my_project.engines  # Module containing @ENGINE.register(...)

   engine: curriculum

   train:
     max_epochs: 100
     max_rollout_steps: 200
     batch_size: 256

The ``imports`` field ensures your module is imported before
``build_engine()`` looks up the registry, so that the
``@ENGINE.register(...)`` decorator has run.


Extending AsyncRLEngine
------------------------

``AsyncRLEngine`` runs four threads coordinated by ``AsyncCoordinator``:

.. mermaid::

   flowchart LR
      subgraph Threads
         R["Rollout Thread<br/>(continuous)"]
         D["Dataset Thread<br/>(waits for data)"]
         T["Train Thread<br/>(epoch loop)"]
         W["Sync Thread<br/>(waits for train)"]
      end
      R -- "buffer" --> D -- "dataset ready" --> T -- "train done" --> W
      W -- "weights updated" --> T

Key points for async customization:

- **rollout()** runs in its own thread and loops until the coordinator
  signals stop (``coordinator.is_running()``). It does **not** follow the
  epoch structure.
- **_pre_update_dataset_hook()** is overridden to wait until the buffer
  has enough data (``buffer.size() >= batch_size``).
- **_train_loop()** is the epoch driver — it waits for dataset-ready and
  weights-updated signals before each training step.
- **_sync_weights_loop()** waits for the train step to finish before
  syncing.

Real-World Example: AsyncRSLRLEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AsyncRSLRLEngine`` extends ``AsyncRLEngine`` with RSL-style
customizations:

- **rollout()**: Adds ``policy_group.postprocess()`` after inference, same
  as the synchronous ``RSLRLEngine``.
- **train()**: Adds metric logging (training info, rollout stats,
  performance metrics) after each training step.
- **warm_up()**: Simplified — just syncs weights.

.. code-block:: python

   from rlightning.engine import AsyncRLEngine
   from rlightning.utils.logger import log_metric
   from rlightning.utils.registry import ENGINE

   @ENGINE.register("async_rsl")
   class AsyncRSLRLEngine(AsyncRLEngine):

       def rollout(self, *args, **kwargs):
           batched_policy_resp = None
           with self.env_group.auto_reset():
               while self.coordinator.is_running():
                   if batched_policy_resp is None:
                       batched_env_ret, truncations = self.env_group.reset(seed=0)
                   else:
                       self.env_group.step_async(batched_policy_resp)
                       batched_env_ret, truncations = self.env_group.collect_async()
                   batched_policy_resp = self.policy_group.rollout_batch(batched_env_ret)
                   self.buffer.add_batched_data_async(batched_env_ret)
                   # RSL-style post-processing
                   processed = self.policy_group.postprocess(
                       batched_env_ret, batched_policy_resp,
                   )
                   self.buffer.add_batched_data_async(processed, truncations)

       def train(self) -> None:
           training_info = self.policy_group.train()
           # Log training, rollout, and performance metrics
           log_metric(training_info, step=self.epoch, prefix="Train")
           rollout_metrics = self.buffer.get_env_stats()
           if len(rollout_metrics):
               log_metric(rollout_metrics, step=self.epoch, prefix="Rollout")

.. note::

   When customizing async engines, be mindful of thread safety. The
   buffer is shared across threads, and the coordinator handles
   synchronization. Avoid accessing shared state (e.g., ``self.epoch``,
   ``self.policy_group``) from the rollout thread without proper
   coordination.


Full Example
------------

This end-to-end example shows a custom engine that adds per-epoch
evaluation and custom metric logging. It extends ``SyncRLEngine`` to
keep the standard rollout and training logic.

**1. Custom engine** (``my_project/engines.py``):

.. code-block:: python

   from rlightning.engine.sync_rl_engine import SyncRLEngine
   from rlightning.utils.registry import ENGINE
   from rlightning.utils.logger import get_logger, log_metric

   logger = get_logger(__name__)

   @ENGINE.register("custom_eval_engine")
   class CustomEvalEngine(SyncRLEngine):
       """Engine with per-epoch evaluation and custom metric logging."""

       def _post_train_hook(self) -> None:
           super()._post_train_hook()
           logger.info(f"Epoch {self.epoch}: train step done, buffer size = {len(self.buffer)}")

       def run(self) -> None:
           for self.epoch in self.iter_epochs(num_epochs=self.config.train.max_epochs):
               # Standard training cycle
               self._rollout(obj_set="train", prefix="rollout")
               self._update_dataset()
               self._train()
               self._sync_weights()

               # Per-epoch evaluation
               logger.info(f"Epoch {self.epoch}: running evaluation")
               self._rollout(obj_set="train", prefix="eval", is_eval=True)

               # Periodic OOD evaluation and checkpointing
               if (self.epoch + 1) % self.config.train.eval_interval == 0:
                   self._rollout(obj_set="test", prefix="eval_ood", is_eval=True)

               if (self.epoch + 1) % self.config.train.save_interval == 0:
                   ckpt_path = f"{self.config.train.save_dir}/epoch_{self.epoch}.pt"
                   self.policy_group.save_checkpoint(path=ckpt_path)

           # Final evaluation and checkpoint
           self._rollout(obj_set="train", prefix="eval", is_eval=True)
           self._rollout(obj_set="test", prefix="eval_ood", is_eval=True)
           ckpt_path = f"{self.config.train.save_dir}/epoch_last.pt"
           self.policy_group.save_checkpoint(path=ckpt_path)
           logger.info("Done.")

**2. Training script** (``train.py``):

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
       env_group = build_env_group(config.env)
       policy_group = build_policy_group(
           config.policy.type, config.policy, config.cluster,
       )
       buffer = build_data_buffer(config.buffer.type, config.buffer)
       engine = build_engine(config, env_group, policy_group, buffer)
       engine.run()

   if __name__ == "__main__":
       launch(main_func=main, config_path=Path(__file__).parent / "conf")

**3. Configuration** (``conf/train.yaml``):

.. code-block:: yaml

   defaults:
     - buffer: rollout_buffer
     - env: my_env
     - train: train
     - policy: my_policy
     - log: tensorboard
     - cluster: default
     - _self_

   imports:
     - my_project.engines

   # requires MainConfig.engine to include "custom_eval_engine"
   engine: custom_eval_engine

   train:
     max_epochs: 100
     max_rollout_steps: 200
     batch_size: 256
     eval_interval: 10
     save_interval: 50
     save_dir: ./checkpoints
