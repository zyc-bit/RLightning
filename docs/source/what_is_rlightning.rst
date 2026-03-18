What is RLightning?
==================

Core Properties
---------------

RLightning is a distributed reinforcement learning framework for embodied
intelligence, built around two core properties: ease of use and efficiency.

**Ease of use** — Algorithms are written and debugged in a familiar
single-process style. RLightning then transparently distributes
execution across nodes and GPUs without changes to code, making
the path from local prototype to large-scale training seamless.

**Efficiency** — Various built-in methods, such as placement scheduling 
and asynchronous task scheduling, to improve training throughput, 
maximize GPU utilization, and preserve algorithm accuracy at distributed scale.

System Challenges
----------------

Embodied RL poses three main challenges:

- **Algorithmic diversity** — Model scales span from compact MLPs to 7B+
  vision-language-action models (VLAs); specialized architectures (dual-system,
  tri-system) require flexible algorithm prototyping.
- **Data-intensive interaction** — While not identical to LLM-based RL, embodied RL is 
  mostly bottlenecked by high-frequency online environment interaction, imposing
  strict data throughput requirements.
- **Heterogeneous ecosystem** — Training pipelines must integrate diverse
  simulators, robot morphologies, and task distributions that existing frameworks
  handle poorly.

RLightning addresses these challenges through the following system design principles:

Design Principles
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Principle
     - Description
   * - Flexible Prototyping
     - Write and debug algorithms in single-process style; a runtime adapter
       layer transparently distributes execution at scale with no code changes.
   * - Scalable Distributed Execution
     - Env workers, Policy workers, and Buffers scale independently.
       Asynchronous scheduling overlaps rollout and training to maximize
       throughput.
   * - Extensible Modular Design
     - Loosely-coupled components with well-defined extension points. Integrate
       new simulators, algorithm libraries, or real-robot backends by subclassing
       a base class and registering it — no framework changes required.
   * - Embodiment-Oriented Optimization
     - Asynchronous I/O, data routing, fine-grained resource scheduling, and
       flexible task orchestration minimize communication overhead and maximize
       GPU utilization for high-frequency embodied workloads.


Supported Features
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Category
     - Component
     - Description
   * - **RL Components**
     - DataBuffer
     - ``RolloutBuffer`` for on-policy; ``ReplayBuffer`` for off-policy
   * -
     - Policy
     - Interface for implementing policy models and training / inference algorithms
   * -
     - Env
     - ManiSkill, MuJoCo, IsaacLab, Libero, Remote Env (such as real-world robots)
   * - **Multi-dimensional Scaling**
     - Env
     - Vector env count, env instance count, heterogeneous simulators
   * -
     - Task
     - Multiple tasks within a single training run
   * -
     - Eval Policy (Actor)
     - Multiple eval workers with stateful and load-balancing routing
   * -
     - Train Policy (Learner)
     - Single-process or DDP distributed training
   * -
     - Buffer
     - Unified or Sharded buffer storage with global sampling and data routing
   * - **Task Scheduling**
     - Synchronous
     - ``SyncRLEngine`` for on-policy algorithms (e.g. PPO)
   * -
     - Asynchronous
     - ``AsyncRLEngine`` for off-policy algorithms
   * - **Execution Mode**
     - Single-process, single-GPU
     - Prototype and debug algorithms
   * -
     - Distributed multi-process, multi-GPU and multi-node
     - Scale training and throughput via data-parallel training
   * - **Resource Scheduling**
     - Default
     - Ray default scheduling; node-affinity strategy for buffer workers
   * -
     - Disaggregate
     - Separate resource pools: train + buffer on one pool, eval + env on another
   * -
     - Colocate
     - All components share a single global pool across nodes
   * -
     - Manual
     - Explicit per-node resource pools defined in YAML config
   * - **Weight Synchronization**
     - Double buffer
     - Two GPU weight snapshots; writer alternates, reader always gets latest
   * -
     - CPU buffer
     - Weights stored on CPU; loaded to GPU on demand to reduce peak memory
   * -
     - Sharded buffer
     - Weights split across eval GPUs; all-gather to reconstruct on update
   * - **Observability**
     - Logging
     - Structured metrics; Experiment logger backends: TensorBoard, Wandb, SwanLab
   * -
     - Profiling
     - Built-in timing profiler for rollout, training, and weight-sync stages
