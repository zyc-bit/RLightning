#!/bin/bash
# You should launch ray cluster handcraftedly for multi-node training before running this script

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# Set WORK_DIR to project root directory (two levels up from script directory)
WORK_DIR=$(dirname $(dirname "$SCRIPT_DIR"))

# auto detect GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
GPU_COUNT=${GPU_COUNT:-0}

. $SCRIPT_DIR/.venv/bin/activate
export PYTHONPATH=$WORK_DIR:$WORK_DIR/examples:$PYTHONPATH


cd $WORK_DIR

TRAINING_CONFIG="launch_multinode_ddp.yaml"
if [[ $# -gt 0 && "$1" != *=* && "$1" != ~* && "$1" != +* && "$1" != -* ]]; then
  TRAINING_CONFIG="$1"
  shift
fi
python $SCRIPT_DIR/train.py --config-name "${TRAINING_CONFIG}" "$@"
