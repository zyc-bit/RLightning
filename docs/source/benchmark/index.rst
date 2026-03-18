Benchmark
=========

RLightning is evaluated on two representative embodied RL tasks covering
both small-model high-frequency control and large-model inference settings.


Experimental Setup
------------------

**Hardware**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Cluster
     - Spec
     - Algorithm
   * - H200 cluster
     - 2 nodes × 8 × H200 141 GB GPUs;
     - OpenVLA PPO (large-model manipulation)
   * - RTX 4090 cluster
     - 8 nodes × 8 × RTX 4090 GPUs;
     - Humanoid WBC (small-model locomotion)

**Software**: Ray 2.46.0, PyTorch 2.6, CUDA 12.4, Isaac Lab 2.2.0,
mani-skill 3.0.0b21.

**Baselines**

- **BeyondMimic** — state-of-the-art open-source implementation for
  humanoid whole-body control. Single-process only; compared at 1 GPU.
- **RLinf** — distributed RL framework; OpenVLA-RL ported to RLightning
  and compared at 8 GPUs.


OpenVLA PPO
----------------------

.. figure:: ../_static/images/openvla-performance.png
  :alt: OpenVLA PPO performance
  :width: 80%
  :align: center



**Key findings**:

- RLightning achieves **comparable convergence accuracy** with benchmark.
- RLightning converges to equivalent accuracy approximately
  ~1.3× faster in wall-clock time compared to RLinf.



Humanoid Whole-Body Control
-----------

Scalability is measured on the humanoid WBC task across intra-node and
inter-node configurations. 

.. figure:: ../_static/images/humanoid_throughput.PNG
  :alt: Humanoid whole body control throughput
  :width: 60%
  :align: center


In the throughput-intensive humanoid whole-body control task, RLightning
maintains throughput efficiency on par with the baseline in the single-GPU
setting. Furthermore, with configuration-only changes and no code rewriting,
the same training pipeline scales smoothly to 2, 4, and 8 nodes, reaching up
to 15× the data throughput of the single-process setup.

