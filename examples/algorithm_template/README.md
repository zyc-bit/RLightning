## RLightning RL Template

A slim template for building and running reinforcement learning workloads on RLightning. Use it both as a starter for your own algorithm and as a quick start for already integrated ones.

### Environment setup

It’s strongly recommended to use uv to set up the Python environment.

```bash
cd examples/algorithm_template
uv sync
```

### Implementing your own algorithm

- Implement the policy and other algorithm components. Follow RLightning’s `README.md` for guidance.
- Reuse this layout: configs in `conf/`, main loop in `train.py`, launcher script.
- Add any additional dependencies to `pyproject.toml` as needed.

### Launch training

```bash
bash launch_train.sh
```
