## Beyond Mimic

### Installation

Besides basic installation for rlightning, you need to run setup scripts as

```bash
cd RLightning
bash examples/wbc_tracking/setup.sh
```

It aims to download a sub-package `unitree_description.tar.gz` from remote site, and then decompress to `examples/wbc_tracking/assets/unitree_description`.

Then, initialize the submodules:

```
git submodule update --init --recursive
```

After, setup the python package environment by 

```bash
cd examples/wbc_tracking
uv sync
```

### Single-task training

Here gives an example for sing-task/dataset training with rsl_rl engine

1. Download lafan dataset

    ```bash
    cd RLightning
    source ./examples/wbc_tracking/.venv/bin/activate
    python -m rlightning.humanoid.utils.download.download_lafan
    ```

2. Retargeting lafan to specified robotics

    ```bash
    cd RLightning
    source ./examples/wbc_tracking/.venv/bin/activate
    PYTHONPATH=$PWD/examples python -m wbc_tracking.retarget_lafan --f-path .data/lafan1
    ```

    Then, the retargeted motions will be stored under `.data/lafan1/retargeted`

3. Convert retargeted motions to task motions that satisfies BeyondMimic tasks

    ```bash
    cd RLightning
    source ./examples/wbc_tracking/.venv/bin/activate
    PYTHONPATH=$PWD/examples python -m wbc_tracking.motion_converter --input-dir .data/lafan1/retargeted
    ```

    Then the converted motions will be stored under `.data/lafan1/retargeted/wbc_tracking`

3. Run visualization to validate the legacy of retargeted motions

    [not ready yet]

4. Run training

    ```bash
    cd RLightning
    bash examples/wbc_tracking/launch.sh
    ```