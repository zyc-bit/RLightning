#!/bin/bash
# Setup OpenPI for openpi_ppo example.
# Run this AFTER 'uv sync' in examples/openpi_ppo/
#
# 1. Applies transformers patches (openpi/models_pytorch/transformers_replace -> transformers)
# 2. Downloads OpenPI assets (tokenizer, etc.)
# 3. Uninstalls pynvml (conflict with nvidia-ml-py)
#
# Usage: bash scripts/setup_openpi.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_PPO_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$OPENPI_PPO_DIR/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${OPENPI_PPO_DIR}/.venv}"
export PATH="${VENV_DIR}/bin:$PATH"

# Get Python version (e.g., 3.11) - use venv python
py_major_minor=$("${VENV_DIR}/bin/python" - <<'EOF'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
EOF
)

SITE_PACKAGES="${VENV_DIR}/lib/python${py_major_minor}/site-packages"
OPENPI_TRANSFORMERS_REPLACE="${SITE_PACKAGES}/openpi/models_pytorch/transformers_replace"
TRANSFORMERS_DIR="${SITE_PACKAGES}/transformers"

echo "[setup_openpi] VENV_DIR=$VENV_DIR"
echo "[setup_openpi] Python ${py_major_minor}"

# 1. Apply transformers patches (required for openpi PyTorch models)
if [ -d "$OPENPI_TRANSFORMERS_REPLACE" ]; then
    echo "[setup_openpi] Applying transformers patches..."
    cp -r "${OPENPI_TRANSFORMERS_REPLACE}/"* "$TRANSFORMERS_DIR/"
    echo "[setup_openpi] Transformers patches applied."
else
    echo "[setup_openpi] WARNING: openpi/models_pytorch/transformers_replace not found."
    echo "[setup_openpi] Run 'uv sync' first to install openpi."
    exit 1
fi

# 2. Download OpenPI assets (tokenizer, etc.)
DOWNLOAD_ASSETS="${OPENPI_PPO_DIR}/scripts/download_assets.sh"
if [ -f "$DOWNLOAD_ASSETS" ]; then
    echo "[setup_openpi] Downloading OpenPI assets..."
    bash "$DOWNLOAD_ASSETS" --assets openpi
else
    echo "[setup_openpi] WARNING: download_assets.sh not found at $DOWNLOAD_ASSETS"
    echo "[setup_openpi] Skipping OpenPI asset download."
fi

# # 3. Uninstall pynvml (conflicts with nvidia-ml-py)
echo "[setup_openpi] Uninstalling pynvml if present..."
(cd "$OPENPI_PPO_DIR" && uv pip uninstall pynvml) || true

echo "[setup_openpi] Done."
