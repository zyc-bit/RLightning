#!/bin/bash
WORK_DIR=$PWD

. $WORK_DIR/.venv/bin/activate
export PYTHONPATH=$WORK_DIR:$PYTHONPATH
export PYTHONPATH=$WORK_DIR/third_party:$PYTHONPATH
export PYTHONPATH=$WORK_DIR/third_party/rw_rl:$PYTHONPATH
export PYTHONPATH=$WORK_DIR/third_party/rw_rl/src:$PYTHONPATH

uv run scripts/env_server/launch_env_server.py --config-name maniskill
