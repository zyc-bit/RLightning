Config
======

The configuration system is the interface through which users define and run
experiments. It is built on **Hydra** for modular YAML composition and
**Pydantic** for type validation and default value handling. Users organize
configuration into modular group files, compose them in an entry config, and
launch experiments with a single command.

**Design goals**:

- **Experiment-driven**: Run different experiments by switching entry configs
  — config is the minimal programming interface.
- **Modular composition**: Config is split into functional groups (env, buffer,
  policy, train, log, cluster) that can be mixed and matched.
- **Quick identification**: Config file names map directly to experiment intent.
- **Automatic validation**: Built-in type checking, default values, and
  cross-field validation via Pydantic.


Config Groups
-------------

Configuration is organized into a two-level hierarchy. The top-level
``MainConfig`` links functional group configs, each stored in its own YAML
file. On the left, groups are organized by directory; on the right, the
corresponding Pydantic config classes.


.. container:: mermaid-height-auto

  .. mermaid::

    flowchart TB
      MC["MainConfig"] --> EC["EnvConfig"]
      MC --> BC["BufferConfig"]
      BC --> StC["StorageConfig"]
      BC --> SmC["SamplerConfig"]
      MC --> PC["PolicyConfig"]
      MC --> TC["TrainConfig"]
      MC --> CC["ClusterConfig"]
      MC --> LC["LogConfig"]

The groups fall into two categories:

- **Algorithm groups** (``buffer``, ``env``, ``policy``, ``train``): Define the
  RL algorithm, environment, and training behavior. These are the groups users
  change most often.
- **System groups** (``log``, ``cluster``): Control logging, distributed
  execution, and resource allocation. The ``cluster`` group is optional — omit
  it for single-process single-GPU mode.
- **Extra fields** (``debug``, ``verbose``): Top-level flags for profiling and
  progress display.


Config Directory Structure
--------------------------

A standard experiment organizes config files in a ``conf/`` directory. Each
subdirectory corresponds to one config group. The root-level YAML files are
entry configs that compose group configs via Hydra's ``defaults`` list.

.. code-block:: text

   examples/openvla_ppo/conf/
   |-- buffer/
   |   `-- rollout_buffer.yaml
   |-- cluster/
   |   `-- 1t1e.yaml
   |-- env/
   |   `-- maniskill.yaml
   |-- log/
   |   `-- wandb.yaml
   |-- policy/
   |   |-- openvla_ppo.yaml
   |   `-- backend/
   |       `-- transformers.yaml
   |-- train/
   |   `-- train.yaml
   `-- train_ppo.yaml          <-- entry config


Entry Config
------------

The entry config is the top-level file that composes group configs into a
single experiment definition. It consists of a ``defaults`` list plus local
field overrides.

.. code-block:: yaml

   defaults:
     - buffer: rollout_buffer
     - env: maniskill
     - train: train
     - policy: openvla_ppo
     - log: wandb
     - cluster: 1t1e
     - _self_

   engine: syncrl

- ``defaults``: Specifies which group config file to use for each group.
  For example, ``policy: openvla_ppo`` maps to ``conf/policy/openvla_ppo.yaml``.
- ``_self_``: A Hydra placeholder that controls merge priority. Place it at
  the end so the current file's fields take precedence over defaults.
- Fields below ``defaults`` (like ``engine``) are merged into the final config.

After Hydra resolves the defaults, the resulting config is equivalent to
merging all referenced group files plus the local fields.


Group Configs
-------------

Each group config file defines the settings for one functional component.
Below are examples.

Environment
~~~~~~~~~~~

``conf/env/maniskill.yaml``:

.. code-block:: yaml

   - name: "maniskill_for_openvla-put_on_plate"
     task: "PutOnPlateInScene25Main-v3"
     backend: "maniskill"
     num_workers: 1
     num_envs: 32
     num_cpus: 1
     num_gpus: 1
     max_episode_steps: 80

The env group is a **list** of environment configs. Each item defines one type
of environment; ``num_workers`` controls how many instances of that type are
created.

Buffer
~~~~~~

``conf/buffer/rollout_buffer.yaml``:

.. code-block:: yaml

   type: "RolloutBuffer"
   capacity: 10240
   storage:
     type: "unified"
     device: "cpu"

Policy
~~~~~~

``conf/policy/openvla_ppo.yaml``:

.. code-block:: yaml

   defaults:
     - backend: transformers

   type: "VLAPPOPolicy"
   device: "cuda"
   rollout_mode: "sync"
   weight_buffer:
     type: "WeightBuffer"
     buffer_strategy: "Double"

   model_cfg:
     model_name: "openvla"
     model_path: "/data/ckpts/gen-robot/openvla-7b-rlvla-warmup"
     tokenizer_path: "/data/ckpts/gen-robot/openvla-7b-rlvla-warmup"

   optim_cfg:
     lr: 1.0e-4

.. note::

   Policy configs can have their own nested defaults (e.g.,
   ``backend: transformers``). Hydra resolves nested defaults the same way
   as top-level ones.

Train
~~~~~

``conf/train/train.yaml``:

.. code-block:: yaml

   max_epochs: 400
   max_rollout_steps: 160
   batch_size: 5120
   eval_interval: 5
   save_interval: 10
   save_dir: ${log.log_dir}/${log.project}/${log.name}/weights

Log
~~~

``conf/log/wandb.yaml``:

.. code-block:: yaml

   level: "INFO"
   backend: "wandb"
   project: "openvla_ppo"
   mode: "offline"
   log_dir: "runs"
   name: "default_exp"

Cluster
~~~~~~~

``conf/cluster/one_train_one_eval.yaml``:

.. code-block:: yaml

   ray_address: auto
   train_worker_num: 1
   eval_worker_num: 1

.. tip::

   Omit the ``cluster`` group entirely to run in single-process single-GPU
   mode. When no ``cluster`` config is provided, ``launch()`` automatically
   creates a local-mode cluster with one train worker and one eval worker.


.. seealso::

   For the complete list of configuration fields for each group, see
   :doc:`../configuration`.


Usage
-----

Running Experiments
~~~~~~~~~~~~~~~~~~~

Run an experiment by pointing the training script to an entry config:

.. code-block:: bash

   python examples/xxx/train.py --config-name train_exp1

Multiple experiments are multiple entry configs:

.. code-block:: bash

   python examples/xxx/train.py --config-name train_exp1
   python examples/xxx/train.py --config-name train_exp2

Command-Line Overrides
~~~~~~~~~~~~~~~~~~~~~~

Hydra allows overriding any config field from the command line. Use
``key=value`` for existing fields or ``+key=value`` for new fields:

.. code-block:: bash

   uv run python train_ppo.py --config-name train_ppo \
       train.batch_size=2560 log.mode=online

Dynamic Module Loading
~~~~~~~~~~~~~~~~~~~~~~

The ``imports`` field lists Python module paths that are loaded at startup.
These modules typically register custom environments, policies, or other
components using the framework's registry decorators. This allows plugin-style
extension without modifying the core codebase.

.. code-block:: yaml

   imports:
     - examples.openvla_ppo.maniskill
