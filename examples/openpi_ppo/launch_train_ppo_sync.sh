#!/bin/bash
# Launch sync PPO training with Ray for OpenVLA.
# Expects: a local `.venv` under `examples/openvla_ppo`.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# Set WORK_DIR to project root directory (two levels up from script directory)
WORK_DIR=$(dirname $(dirname "$SCRIPT_DIR"))

# auto detect GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
GPU_COUNT=${GPU_COUNT:-0}

. $SCRIPT_DIR/.venv/bin/activate
export PYTHONPATH=$WORK_DIR:$PYTHONPATH

ray stop
ray start --head --num-gpus=$GPU_COUNT

cd $WORK_DIR

# launch training
python $SCRIPT_DIR/train_ppo.py --config-name train_ppo
