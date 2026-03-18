#!/bin/bash
WORK_DIR=$PWD
ASSET_DIR=$WORK_DIR/examples/wbc_tracking/assets

if [ ! -d "$ASSET_DIR" ] || \
   [ ! -f "$ASSET_DIR/__init__.py" ] || \
   [ ! -d "$ASSET_DIR/unitree_description" ]; then
  mkdir -p "$ASSET_DIR"
  cat > "$ASSET_DIR/__init__.py" <<'PY'
import os

# Conveniences to other module directories via relative paths
ASSET_DIR = os.path.abspath(os.path.dirname(__file__))
PY

  curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
  tar -xzf unitree_description.tar.gz -C $ASSET_DIR/ && \
  rm unitree_description.tar.gz
else
  echo "Assets already be set!"
fi
