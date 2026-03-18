#!/bin/bash
# Setup LIBERO for openpi_ppo example.
# LIBERO must be installed in editable mode to ensure assets (XML, etc.) are accessible.
# When installed from git via uv sync, package data may not be included in the build.
#
# Usage: bash scripts/setup_libero.sh
# Run this before 'uv sync' in examples/openpi_ppo/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_PPO_DIR="$(dirname "$SCRIPT_DIR")"
LIBERO_DIR="${OPENPI_PPO_DIR}/.venv/LIBERO"
LIBERO_REPO="https://github.com/RLinf/LIBERO.git"

if [ -d "$LIBERO_DIR" ]; then
    echo "[setup_libero] LIBERO already exists at $LIBERO_DIR"
    echo "[setup_libero] Run 'uv sync' to install dependencies (libero will use editable install from local path)"
    exit 0
fi

echo "[setup_libero] Cloning LIBERO to $LIBERO_DIR ..."
mkdir -p "$(dirname "$LIBERO_DIR")"
git clone "$LIBERO_REPO" "$LIBERO_DIR"

echo "[setup_libero] Done. You can now run 'uv sync' in $OPENPI_PPO_DIR"
echo "[setup_libero] libero will be installed in editable mode from $LIBERO_DIR"
