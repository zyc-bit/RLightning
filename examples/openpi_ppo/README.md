## OpenPI PPO on LIBERO

PPO fine-tuning of [OpenPI](https://github.com/RLinf/openpi) (π₀/π₀.₅) models on the [LIBERO](https://libero-project.github.io/) benchmark.

### Installation

Besides basic installation for rlightning, you need to run setup scripts as follows.

1. Setup LIBERO (clone to `.venv/LIBERO` for editable install; required because assets are not included when installing from git)

    ```bash
    cd RLightning/examples/openpi_ppo
    uv venv .venv
    bash scripts/setup_libero.sh
    ```

    It clones LIBERO to `examples/openpi_ppo/.venv/LIBERO` so that assets (XML scenes, etc.) are accessible.

2. Setup the Python package environment

    ```bash
    cd RLightning/examples/openpi_ppo
    uv sync
    ```

3. Setup OpenPI (apply transformers patches)

    ```bash
    cd RLightning/examples/openpi_ppo
    bash scripts/setup_openpi.sh
    ```

    It applies the transformers library patches from `openpi/models_pytorch/transformers_replace` into `transformers`, which are required for OpenPI PyTorch models (AdaRMS, activation precision, KV cache).

### Training

To launch PPO training:

```bash
cd RLightning
bash examples/openpi_ppo/launch_train_ppo_sync.sh
```
