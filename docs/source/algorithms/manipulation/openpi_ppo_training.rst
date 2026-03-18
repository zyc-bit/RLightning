OpenPI PPO Training
===================

This tutorial walks through training OpenPI (π₀/π₀.₅) with PPO on LIBERO
using RLightning. OpenPI is a vision-language-action model for robotic
manipulation; PPO provides the on-policy RL objective.


Environment
-----------

**LIBERO benchmark**

LIBERO is a robotic manipulation benchmark built on robosuite (MuJoCo).
The agent commands a 7-DoF robot arm to complete household manipulation
tasks such as pick-and-place, stacking, drawer opening, and spatial
rearrangement.

- **Observation**: RGB images from workspace cameras (commonly 256×256),
  including a main view and a wrist-mounted view.
- **Proprioception**: End-effector pose (position + orientation) and
  gripper state.
- **Action space**: 7-dimensional continuous — 3D position (x, y, z),
  3D rotation (roll, pitch, yaw), and gripper open/close.
- **Reward**: Sparse reward based on task success/failure.
- **Task description**: Natural language instruction describing the goal.


Algorithm
---------

PPO (Proximal Policy Optimization) with the following components:

- **GAE** (Generalized Advantage Estimation) for advantage computation
  (``gamma=0.99``, ``gae_lambda=0.95``).
- **Ratio-based policy clipping** (``clip_ratio=0.2``).
- **Value function clipping** (``value_clip_ratio=0.2``).
- **Entropy regularization** (configurable via ``entropy_bonus``).
- **Chunk-level rewards and log-probabilities**: rewards and log-probs
  are aggregated at the action-chunk level rather than per-token,
  matching the multi-step action prediction of OpenPI.
- **Flow-SDE noise**: OpenPI uses a flow-based stochastic differential
  equation noise method (``noise_method: flow_sde``) for action
  denoising during rollout. 
  (πRL: `ONLINE RL FINE-TUNING FOR FLOW-BASED VISION-LANGUAGE-ACTION MODELS. <https://arxiv.org/abs/2510.25889>`_)


Prerequisites
-------------

**Environment setup**

LIBERO must be cloned locally for editable install (assets are not
included in a normal pip install). Then install dependencies and apply
OpenPI patches:

.. code-block:: bash

   cd examples/openpi_ppo
   uv venv .venv
   bash scripts/setup_libero.sh
   uv sync
   bash scripts/setup_openpi.sh

.. note::

   **Python 3.11 is required** (the ``flash_attn`` pre-built wheel only
   supports cp311).

**Hardware requirements**

- Minimum: 1 × A100 80 GB (single-process mode, small batch)
- Recommended: 8 × H200 / A100 GPUs for multi-process DDP training

**Model checkpoint**

Install the download tool and fetch the pretrained checkpoint:

.. code-block:: bash

   uv pip install huggingface_hub

   # Optional: use hf-mirror for China mainland users
   export HF_ENDPOINT="https://hf-mirror.com"

   .venv/bin/huggingface-cli download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
     --local-dir /data/ckpts/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

The default config expects the checkpoint at:

.. code-block:: yaml

   model_cfg:
     model_path: "/data/ckpts/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT"
     tokenizer_path: "/data/ckpts/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT"

To use a different path, update these fields in
``examples/openpi_ppo/conf/policy/openpi_ppo.yaml``.


Training Commands
-----------------

All commands assume you are in the project root (``RLightning/``).

Sync single-GPU
~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash examples/openpi_ppo/launch_train_ppo_sync.sh

This starts a Ray cluster on the local node and launches one Train +
one Eval worker with 64 parallel environments.

Sync single-GPU (tiny)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash examples/openpi_ppo/launch_train_ppo_sync_tiny.sh

A reduced-resource variant for quick validation or when GPU memory is
limited. Compared to the standard config, the main changes are:

.. code-block:: yaml

   policy:
     optim_cfg:
       lr: 1.0e-6          # 5.0e-6 → 1.0e-6
   train:
     mini_batch_size: 256   # 2048 → 256
     micro_batch_size: 32   # 128 → 32
     rollout_epoch: 1       # 8 → 1

Sync DDP (8 GPUs)
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash examples/openpi_ppo/launch_train_ppo_sync_ddp.sh

Uses 8 Train workers + 8 Eval workers in colocated DDP mode with 8
parallel environments per worker (64 total).

Sync DDP tiny (8 GPUs)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash examples/openpi_ppo/launch_train_ppo_sync_tiny_ddp.sh

DDP mode with reduced batch sizes for faster iteration on 8 GPUs.


Config Walkthrough
------------------

The entry config is ``conf/train_ppo.yaml``:

.. code-block:: yaml

   defaults:
     - buffer: rollout_buffer
     - env: libero
     - train: train
     - policy: openpi_ppo
     - log: wandb
     - cluster: 1t1e

   engine: syncrl
   buffer:
     capacity: 24576

**Training settings** (``conf/train/train.yaml``):

.. code-block:: yaml

   max_epochs: 1000
   max_rollout_steps: 48        # env steps per rollout phase
   warm_up_rollout_steps: 32    # warm-up steps before training
   mini_batch_size: 2048        # minibatch for PPO gradient update
   micro_batch_size: 128        # gradient accumulation step size
   update_epoch: 4              # PPO update passes per rollout
   rollout_epoch: 8             # rollout batches per epoch
   eval_interval: -1            # -1 disables periodic eval
   save_interval: -1            # -1 disables periodic checkpoints

**Environment settings** (``conf/env/libero.yaml``):

.. code-block:: yaml

   - name: "libero_spatial_for_openpi-ppo"
     task: "libero_spatial"
     backend: "libero"
     num_workers: 1
     num_envs: 64
     max_episode_steps: 240
     num_action_chunks: 5       # multi-step action prediction

**Policy settings** (``conf/policy/openpi_ppo.yaml``):

.. code-block:: yaml

   type: "VLAPPOPolicy"
   device: "cuda"
   rollout_mode: "sync"
   weight_buffer:
     type: "WeightBuffer"
     buffer_strategy: "Double"
   model_cfg:
     model_name: "openpi"
     action_dim: 7
     num_action_chunks: 5
     num_steps: 4               # denoising steps
     add_value_head: True
     use_proprio: True
     openpi:
       config_name: "pi0_libero"
       noise_method: "flow_sde"

**Cluster settings** (``conf/cluster/1t1e.yaml``):

.. code-block:: yaml

   train_worker_num: 1
   eval_worker_num: 1
   placement:
     mode: "auto"
     strategy: "disaggregate"


Architecture Overview
---------------------

OpenPI PPO uses a ``SyncRLEngine`` with ``RolloutBuffer``:

.. mermaid::

   flowchart LR
      E["SyncRLEngine"] --> EG["EnvGroup\n(LIBERO)"]
      E --> PG["PolicyGroup\n(VLAPPOPolicy)"]
      E --> B["RolloutBuffer"]

      EG -->|EnvRet\n(image + proprio + reward)| PG
      PG -->|PolicyResponse\n(action, log_prob, value)| B
      B -->|BatchedData| PG
      PG -->|updated weights| PG

The EVAL policy runs OpenPI inference (image → denoised action chunks)
using the Transformers backend. The TRAIN policy computes PPO loss over
stored rollouts and updates model parameters.

**Data flow**:

1. The environment produces observations (RGB images, proprioceptive
   state, language instruction) and rewards.
2. The eval policy predicts action chunks via iterative denoising
   (flow-SDE, 4 steps) and estimates values.
3. Transitions are stored in the ``RolloutBuffer``.
4. After each rollout phase, GAE advantages are computed over episodes.
5. The train policy performs multiple PPO update passes over minibatches
   sampled from the buffer.


Tips
----

.. tip::

   **Start with the tiny config** for debugging before launching
   full-scale training:

   .. code-block:: bash

      bash examples/openpi_ppo/launch_train_ppo_sync_tiny.sh

   This uses smaller batch sizes and a single rollout epoch, making it
   much faster to iterate.

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
   to Weights & Biases during training. TensorBoard is also supported
   via ``log=tensorboard``.


See Also
--------

- :doc:`../../getting_started/quickstart` — Quick training commands for all examples.
- :doc:`../../user_guide/build_your_own/customize_policy` — Implement a custom policy.
- :doc:`openvla_ppo_training` — OpenVLA PPO training on ManiSkill (similar architecture).
