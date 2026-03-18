OpenVLA PPO Training
====================

This tutorial walks through training OpenVLA with PPO on ManiSkill using
RLightning. OpenVLA is a 7B vision-language-action model fine-tuned with
LoRA; PPO provides the on-policy RL objective.


Prerequisites
-------------

**Environment setup**

Navigate to the example directory and create the virtual environment:

.. code-block:: bash

   cd examples/openvla_ppo/
   uv sync

.. note::

   **Python 3.11 is required** (the ``flash_attn`` pre-built wheel only
   supports cp311). Pin the version in ``pyproject.toml``:

   .. code-block:: toml

      requires-python = "==3.11.*"

**Hardware requirements**

- Minimum: 1 × A100 80 GB (single-process mode, small batch)
- Recommended: 8 × H200 / A100 GPUs for multi-process training
- The default config uses 32 parallel ManiSkill environments; reduce
  ``num_envs`` if GPU memory is limited.

**Model checkpoint**

Install the download tool and fetch the pretrained checkpoint:

.. code-block:: bash

   uv pip install huggingface_hub

   # Optional: use hf-mirror for China mainland users
   export HF_ENDPOINT="https://hf-mirror.com"

   .venv/bin/huggingface-cli download gen-robot/openvla-7b-rlvla-warmup \
     --local-dir /data/ckpts/gen-robot/openvla-7b-rlvla-warmup

The default config expects the checkpoint at:

.. code-block:: yaml

   model_cfg:
     model_path: "/data/ckpts/gen-robot/openvla-7b-rlvla-warmup"
     tokenizer_path: "/data/ckpts/gen-robot/openvla-7b-rlvla-warmup"

To use a different path, update these fields in
``examples/openvla_ppo/conf/policy/openvla_ppo.yaml``.

**Simulation assets**

Download ManiSkill built-in assets (bridge table scene and WidowX robot):

.. code-block:: bash

   source .venv/bin/activate
   python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
   python -m mani_skill.utils.download_asset widowx250s -y

Download custom scene assets (carrot/plate objects and table overlay backgrounds):

.. code-block:: bash

   cd examples/openvla_ppo/maniskill
   ../.venv/bin/hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets


Training Commands
-----------------

All commands assume you are in ``examples/openvla_ppo/`` with the
virtual environment activated.

Sync multi-process (single node)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd examples/openvla_ppo/
   uv sync
   bash launch_train_ppo_sync.sh

This starts a Ray cluster on the local node and launches one Train +
one Eval worker with 32 parallel environments.

Sync DDP (single node, 8 GPUs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd examples/openvla_ppo/
   uv sync
   bash launch_train_ppo_ddp.sh

Uses 8 Train workers + 8 Eval workers in colocated DDP mode with
``train_dp8.yaml`` train settings.


Config Walkthrough
------------------

The entry config is ``conf/train_ppo.yaml``:

.. code-block:: yaml

   defaults:
     - buffer: rollout_buffer
     - env: maniskill
     - train: train
     - policy: openvla_ppo

**Training settings** (``conf/train/train.yaml``):

.. code-block:: yaml

   max_epochs: 400
   max_rollout_steps: 160      # env steps per rollout phase
   batch_size: 5120            # total transitions per training step
   mini_batch_size: 160        # minibatch for PPO gradient update
   micro_batch_size: 8         # gradient accumulation step size
   eval_interval: 5            # evaluate every N epochs
   save_interval: 10           # checkpoint every N epochs
   save_dir: ./outputs/weights

**Environment settings** (``conf/env/maniskill.yaml``):

.. code-block:: yaml

   - name: "maniskill_for_openvla-put_on_plate"
     task: "PutOnPlateInScene25Main-v3"
     backend: "maniskill"
     num_workers: 1
     num_envs: 32

**Policy settings** (``conf/policy/openvla_ppo.yaml``):

.. code-block:: yaml

   type: "VLAPPOPolicy"
   device: "cuda"
   rollout_mode: "sync"
   weight_buffer:
     type: "WeightBuffer"
     buffer_strategy: "Double"
   model_cfg:
     model_name: "openvla"
     action_dim: 7
     is_lora: true
     lora_rank: 32
     lora_alpha: 32
     precision: "bf16"


Architecture Overview
---------------------

OpenVLA PPO uses a ``SyncRLEngine`` with ``RolloutBuffer``:

.. mermaid::

   flowchart LR
      E["SyncRLEngine"] --> EG["EnvGroup\n(ManiSkill)"]
      E --> PG["PolicyGroup\n(VLAPPOPolicy)"]
      E --> B["RolloutBuffer"]

      EG -->|EnvRet\n(image + proprio + reward)| PG
      PG -->|PolicyResponse\n(action, log_prob, value)| B
      B -->|BatchedData| PG
      PG -->|updated LoRA weights| PG

The EVAL policy runs VLA inference (image → action) using the
Transformers backend. The TRAIN policy computes PPO loss over stored
rollouts and updates LoRA parameters.


Expected Training Behavior
--------------------------

- **Metric**: task success rate on ``PutOnPlateInScene25Main-v3``.
- **Convergence**: success rate typically rises during the first 50–100
  epochs as the policy learns basic manipulation skills; full convergence
  occurs within 200–400 epochs depending on hardware and batch size.
- **Throughput**: on 8 × H200, the default config achieves
  approximately [TBD] rollout steps/s.

.. note::

   [Placeholder: convergence curve figure — success rate vs. wall-clock time]


Tips
----

.. tip::

   **Start with single-process debugging** before launching distributed:

   .. code-block:: bash

      # In conf/cluster/1t1e.yaml, set remote flags to false
      # Then run: uv run python train_ppo.py cluster=1t1e

   This runs the entire pipeline in one process, making Python debuggers
   and ``print`` statements work normally.

.. tip::

   **Reduce ``num_envs``** if you hit GPU OOM during rollout. Env
   workers and eval policy share the same GPU by default; reducing
   environment parallelism lowers peak memory.

.. tip::

   **Adjust ``micro_batch_size``** to fit the model into GPU memory
   during the training phase. Smaller values use gradient accumulation
   with more steps.

.. tip::

   Set ``log.backend: wandb`` and ``log.mode: online`` to stream metrics
   to Weights & Biases during training.


Porting from RLinf
------------------

OpenVLA PPO in RLightning is a direct port of the OpenVLA-RL implementation
from RLinf. The PPO objective, reward function, and VLA model architecture
are identical. The key differences are:

- Ray-based distributed execution (vs. RLinf's custom distributed runtime).
- YAML-driven configuration via Hydra (vs. RLinf's script-level config).
- Modular Policy/Buffer/Engine components that can be reused in other algorithms.

RLightning achieves comparable algorithmic convergence to RLinf with
approximately 1.3× faster wall-clock training on equivalent hardware.

See Also
--------

- :doc:`../../getting_started/quickstart` — Quick training commands for all examples.
- :doc:`../../user_guide/build_your_own/customize_policy` — Implement a custom policy.
- :doc:`../../benchmark/index` — Performance comparison with RLinf.
