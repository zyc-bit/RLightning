Configuration
=============

This page is a field-by-field reference for all configuration groups in
RLightning. Each section corresponds to one configuration group and lists
every available field with its type, default, and description.

For an explanation of the configuration system design — how config groups are
organized, how YAML files are composed, and how to run experiments — see
:doc:`core_components/config`.


Environment Configuration
-------------------------

The ``env`` group defines the environments used for data collection.
Multiple environments may be specified as a list.

``name`` (str) **[required]**
  User-defined name used to identify the environment configuration.

``backend`` (str) **[required]**
  Backend used to create the environment instance.

``task`` (str) **[required]**
  Task identifier passed to the environment creation interface.

``num_workers`` (int)
  Number of environment workers with identical configuration.
  Defaults to ``1``.

``num_envs`` (int)
  Number of vectorized environments within a single environment instance.
  Defaults to ``1``.

``max_episode_steps`` (int or ``None``)
  Maximum number of steps per episode.
  Defaults to ``None`` (no limit).

``num_cpus`` (int)
  Number of CPUs allocated per environment worker (remote mode only).
  Defaults to ``1``.

``num_gpus`` (float)
  Number of GPUs allocated per environment worker (remote mode only).
  Defaults to ``0.0``.

``env_kwargs`` (Config)
  Additional keyword arguments passed to the environment constructor.

``init_params`` (Config or ``None``)
  Initialization parameters for the environment backend.

``policy_setup`` (str)
  Policy setup identifier used by certain backends.
  Defaults to ``"widowx"``.



Buffer Configuration
--------------------

The ``buffer`` group controls how collected experience is stored and sampled
during training.

``type`` (str) **[required]**
  Buffer implementation type.
  Supported values are ``ReplayBuffer`` and ``RolloutBuffer``.

``capacity`` (int) **[required]**
  Maximum number of transitions stored in the buffer.

``auto_truncate_episode`` (bool)
  Whether to automatically truncate episodes based on ``last_terminated`` or ``last_truncated`` signals
  returned by the environment.
  Defaults to ``False``.

``node_affinity_env`` (bool)
  Whether to enable node affinity for environment workers.
  Defaults to ``False``.

``node_affinity_train`` (bool)
  Whether to enable node affinity for training workers.
  Defaults to ``False``.

Storage Configuration
~~~~~~~~~~~~~~~~~~~~~

The ``storage`` sub-configuration defines how buffer data is physically stored.

``mode`` (str)
  Storage behavior when capacity is reached.

  - ``circular``: overwrite old data
  - ``fixed``: stop accepting new data

  Defaults to ``"circular"``.

``type`` (str)
  Storage type. Supported values are ``"unified"`` and ``"sharded"``.

  - ``unified``: single-location storage (default)
  - ``sharded``: sharded storage

``unit`` (str)
  Storage granularity. ``"transition"`` or ``"episode"``.
  
  - ``transition``: post-processed transitions (default)
  - ``episode``: complete episodes, possibly without post-processing

``device`` (str)
  Storage device. ``"cpu"`` or ``"cuda"``.
  Defaults to ``"cpu"``.

Sampler Configuration
~~~~~~~~~~~~~~~~~~~~~

The ``sampler`` sub-configuration controls how data is sampled from the buffer.

``type`` (str) **[required]**
  Sampling strategy.

  Supported values:

  - ``all``: AllDataSampler, sample all stored data
  - ``batch``: BatchSampler, sequential batch sampling
  - ``uniform``: UniformSampler, uniform random sampling

  ReplayBuffer: ``UniformSampler`` by default; RolloutBuffer: ``AllDataSampler``



Policy Configuration
--------------------

The ``policy`` group defines policy execution, inference behavior, and resource
allocation for policy workers.

``type`` (str) **[required]**
  Policy implementation identifier.

``backend`` (dict or ``None``)
  Inference backend configuration for evaluation policies, such as ``transformers`` (PyTorch-based inference)
  or ``vllm``

``train_num_gpus`` (float)
  Number of GPUs allocated per training policy worker (remote mode only).
  In single-process mode or when ``cluster.remote_train`` is ``False``, this
  parameter is ignored. 
  Defaults to ``1.0``.

``eval_num_gpus`` (float)
  Number of GPUs allocated per evaluation policy worker (remote mode only).
  In single-process mode or when ``cluster.remote_train`` is ``False``, this
  parameter is ignored. 
  Defaults to ``1.0``.

``model_cfg`` (dict or ``None``)
  Model-specific configuration.

``optim_cfg`` (dict or ``None``)
  Optimizer-specific configuration.

``rollout_mode`` (str)
  Policy inference concurrency mode.
  Supported values are ``"sync"`` and ``"async"``.
  Defaults to ``"sync"``.

``router_type`` (str)
  Routing strategy for asynchronous rollout.

  - ``simple``: load-balanced routing
  - ``node_affinity``: route to policies on the same node

  Defaults to ``"simple"``.

``weight_buffer`` (dict)
  Configuration for policy weight synchronization.

  - ``type`` (str): weight buffer implementation type.
    Supported values: ``"WeightBuffer"``, ``"CPUWeightBuffer"``,
    ``"ShardedWeightBuffer"``. Defaults to ``"WeightBuffer"``.

  - ``buffer_strategy`` (str): weight synchronization strategy.
    Supported values: ``"None"``, ``"Double"``, ``"Shared"``, ``"Sharded"``.
    Defaults to ``"Double"``. Note: ``"Shared"`` requires
    ``type="CPUWeightBuffer"``; ``"Sharded"`` requires
    ``type="ShardedWeightBuffer"``.

``policy_kwargs`` (dict or ``None``)
  Additional keyword arguments passed to the policy constructor.


Training Configuration
----------------------

The ``train`` group controls the overall training procedure.

``max_epochs`` (int) **[required]**
  Maximum number of training epochs.

``batch_size`` (int)
  Training batch size.
  Defaults to ``64``.

``max_rollout_steps`` (int)
  Maximum number of rollout steps per rollout stage.
  A value of ``-1`` disables step-based truncation.
  Defaults to ``-1``.

``lr`` (float)
  Learning rate used by the optimizer.
  Defaults to ``0.0003``.

``parallel`` (str or ``None``)
  Training parallelization mode.
  Currently supports ``"ddp"`` or ``None``. ``None`` means no parallel.

``eval_interval`` (int)
  Evaluation interval measured in training epochs.
  Defaults to ``10``.

``save_interval`` (int)
  Checkpoint saving interval measured in training epochs.
  Defaults to ``50``.

``save_dir`` (str or ``None``)
  Directory used to store checkpoints.
  If not specified, a subdirectory under the logging directory is used.



Logging Configuration
---------------------

The ``log`` group controls experiment logging and tracking.

``level`` (str)
  Logging level.
  Supported values: ``"DEBUG"``, ``"INFO"``, ``"WARINING"``, ``"ERROR"``,
  ``"CRITICAL"``. Note: ``"WARINING"`` is the current spelling in the code.
  Defaults to ``"DEBUG"``.

``backend`` (str)
  Logging backend.
  Supported values include ``tensorboard``, ``wandb``, and ``swanlab``.
  Defaults to ``tensorboard``.

``project`` (str)
  Project name used by the logging backend.
  Defaults to ``"default_project"``.

``name`` (str)
  Experiment name used by the logging backend.
  Defaults to ``"default_exp"``.

``log_dir`` (str)
  Directory used to store experiment logs.
  Defaults to ``"./runs"``.

``mode`` (str or ``None``)
  Backend-specific logging mode.

  - ``wandb``: ``"online"``, ``"offline"``, ``"shared"``, ``"disabled"``.
    Defaults to ``"offline"`` when not set.
  - ``swanlab``: ``"cloud"``, ``"local"``, ``"disabled"``.
    Defaults to ``"local"`` when not set.
  - ``tensorboard``: this field is ignored.



Cluster Configuration
---------------------

The ``cluster`` group defines how training components are launched in
distributed or multi-process execution.

``ray_address`` (str)
  Address of the Ray head node.
  Defaults to ``"auto"``.

``train_worker_num`` (int)
  Number of training policy workers.
  Defaults to ``1``.

``eval_worker_num`` (int)
  Number of evaluation policy workers.
  Defaults to ``1``.

``train_each_gpu_num`` (float)
  Number of GPUs allocated per training policy worker.
  Defaults to ``1.0``.

``eval_each_gpu_num`` (float)
  Number of GPUs allocated per evaluation policy worker.
  Defaults to ``1.0``.

``buffer_worker_num`` (int or ``"auto"``)
  Number of buffer storage workers.
  Defaults to ``1``.

``remote_train`` / ``remote_eval`` / ``remote_storage`` / ``remote_env`` (bool)
  Whether the corresponding component runs as a Ray actor.
  Defaults to ``True``.

``is_colocated`` (bool)
  Whether train and eval policies share the same GPU.
  When ``True``, ``enable_offload`` is automatically set to ``True`` and
  weight buffer strategy is set to ``"None"``.
  Defaults to ``False``.

``enable_offload`` (bool)
  Whether to enable offloading for policy and environment workers.
  Defaults to ``False``.

``placement`` (dict)
  Resource placement configuration. Sub-fields:

  - ``mode`` (str): ``"auto"`` (default) or ``"manual"``.
  - ``strategy`` (str): ``"default"`` (default), ``"disaggregate"``,
    or ``"colocate"``.
  - ``env_strategy`` (str): ``"default"`` (default) or
    ``"device-colocate"``.


Global Flags
------------

``debug`` (bool)
  Enables debug mode.
  Defaults to ``True``.

``verbose`` (bool)
  Enables verbose output.
  Defaults to ``True``.
