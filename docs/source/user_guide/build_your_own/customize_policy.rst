Customize Policy
================

This guide shows how to implement a custom RL algorithm by subclassing
``BasePolicy``. The same class is used for both EVAL (inference) and TRAIN
(gradient update) roles — the framework calls the appropriate methods
depending on the worker's role.


Overview
--------

The required steps are:

1. Subclass ``BasePolicy`` and implement the eight abstract methods.
2. Register the class with ``@POLICIES.register``.
3. Point the ``policy.type`` config field to the registered name.

.. code-block:: python

   from rlightning.policy import BasePolicy
   from rlightning.utils.registry import POLICIES

   @POLICIES.register("MyPolicy")
   class MyPolicy(BasePolicy):
       # --- Network & optimizer ---
       def construct_network(self, env_meta, *args, **kwargs): ...
       def setup_optimizer(self, optim_cfg): ...

       # --- Rollout (EVAL workers) ---
       def rollout_step(self, env_ret, **kwargs): ...
       def postprocess(self, env_ret=None, policy_resp=None): ...

       # --- Training (TRAIN workers) ---
       def update_dataset(self, data): ...
       def train(self, *args, **kwargs): ...

       # --- Weight management ---
       def get_trainable_parameters(self): ...
       def load_state_dict(self, state_dict, *args, **kwargs): ...


Step 1: construct_network
-------------------------

Create all neural network modules here. Assign each model as an attribute
of ``self`` (i.e. register it as an ``nn.Module`` sub-module). The framework
automatically discovers trainable modules via ``_find_model()`` and populates
``self.model_list`` — you do **not** need to set it manually.

``model_list`` is used for DDP wrapping, weight synchronization, and
checkpointing.

.. code-block:: python

   def construct_network(self, env_meta, *args, **kwargs):
       action_space = env_meta.action_space
       action_dim = action_space.shape[0]

       self.encoder   = NatureCNN()
       self.actor     = nn.Linear(self.encoder.out_feature_dim, action_dim)
       self.critic    = nn.Linear(self.encoder.out_feature_dim, 1)

       if torch.cuda.is_available():
           self.encoder.cuda()
           self.actor.cuda()
           self.critic.cuda()

.. note::

   ``env_meta`` is populated by the framework from the first environment
   reset. It exposes ``observation_space``, ``action_space``, and other
   environment metadata.

.. tip::

   If you need explicit control over which modules are tracked, set
   ``self.model_list`` before ``init_eval()`` / ``init_train()`` is called.
   Otherwise the framework auto-discovers all ``nn.Module`` attributes with
   trainable parameters.


Step 2: setup_optimizer
-----------------------

Create the optimizer. Called only on TRAIN workers, after ``construct_network``
and ``_find_model()``.

.. code-block:: python

   def setup_optimizer(self, optim_cfg):
       params = (
           list(self.encoder.parameters())
           + list(self.actor.parameters())
           + list(self.critic.parameters())
       )
       self.optimizer = torch.optim.AdamW(params, lr=optim_cfg.lr)


Step 3: rollout_step
--------------------

Pure inference: convert an ``EnvRet`` to a ``PolicyResponse``. This runs
on EVAL workers. The base class already applies ``@torch.inference_mode()``,
so you do **not** need ``torch.no_grad()`` inside this method.

.. code-block:: python

   from rlightning.types import EnvRet, PolicyResponse

   def rollout_step(self, env_ret: EnvRet, **kwargs) -> PolicyResponse:
       obs = env_ret.observation.float().unsqueeze(0).cuda()

       action, log_prob, entropy = self.get_action(obs)
       value = self.get_value(obs)

       return PolicyResponse(
           env_id=env_ret.env_id,
           action=action,
           log_prob=log_prob,
           entropy=entropy,
           value=value,
       )

.. important::

   ``PolicyResponse`` requires ``env_id`` as its first argument. Additional
   fields are passed as keyword arguments and stored dynamically.

.. list-table:: PolicyResponse fields
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``env_id`` *(required)*
     - Environment identifier (from ``env_ret.env_id``).
   * - ``action``
     - Action to send to the environment.
   * - ``log_prob``
     - Log probability of the sampled action (for on-policy algorithms).
   * - ``value``
     - Critic value estimate (for actor-critic algorithms).
   * - Additional fields
     - Any extra data stored in ``PolicyResponse`` will be
       passed through the buffer and available in ``update_dataset``.

.. note::

   The framework handles device transfer (CPU/CUDA/numpy conversion)
   automatically in the rollout hooks. You can return tensors directly
   without calling ``.cpu().numpy()``.


Step 4: postprocess
-------------------

Post-processes environment and policy outputs. This method is abstract
and must be implemented, but if your algorithm does not require
post-processing (e.g. for simple synchronous PPO), you can just pass:

.. code-block:: python

   def postprocess(self, env_ret=None, policy_resp=None):
       pass

This method is used by some algorithms for
additional processing after rollout, such as computing advantages
on the eval worker side.


Step 5: update_dataset
----------------------

Called on TRAIN workers after the buffer is sampled. Store the data for
use in ``train()``.

Step 6: train
-------------

Define the training logic here to do gradient updates.


Step 7: get_trainable_parameters / load_state_dict
---------------------------------------------------

These two methods handle weight serialization for distributed weight
synchronization between TRAIN and EVAL workers. Here is an simple example.

.. code-block:: python

   from torch.nn.parallel import DistributedDataParallel as DDP

   def get_trainable_parameters(self):
       state_dict = {}
       for name, model in self.model_list:
           module = model.module if isinstance(model, DDP) else model
           state_dict[name] = module.state_dict()
       return state_dict

   def load_state_dict(self, state_dict, strict=True, assign=False):
       for name, model in self.model_list:
           model.load_state_dict(state_dict[name], strict=strict)


Optional: Checkpointing
-----------------------

The default ``save_checkpoint`` saves all models in ``self.model_list``
(unwrapping DDP if needed). Override it if you need custom checkpoint
logic (e.g. saving optimizer state or extra parameters):



Optional: DDP Wrapping
----------------------

Multi-GPU training wraps models with ``DistributedDataParallel``
automatically for all modules in ``self.model_list``. To customize which
modules are wrapped, define a ``wrap_with_ddp`` method:

.. code-block:: python

   def wrap_with_ddp(self, logic_gpu_id, process_group):
       self.actor = DDP(
           self.actor.cuda(),
           device_ids=[logic_gpu_id],
           process_group=process_group,
       )


Registration and Config Wiring
-------------------------------

Decorate the class with ``@POLICIES.register("name")`` and set
``policy.type`` in the config:

.. code-block:: yaml

   policy:
     type: "MyPolicy"
     model_cfg:
       obs_dim: 64
       action_dim: 7
     weight_buffer:
       type: WeightBuffer
       buffer_strategy: Double

.. tip::

   Keep ``model_cfg`` as a ``Config`` object (arbitrary extra fields).
   Access it in ``construct_network`` via ``self.config.model_cfg``.


See Also
--------

- :doc:`../core_components/policy_policygroup` — BasePolicy interface reference.
- :doc:`../core_components/engine` — How engines call policy methods.
- :doc:`customize_buffer` — Customize experience storage and sampling.
