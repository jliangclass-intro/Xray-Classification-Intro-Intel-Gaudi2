#!/usr/bin/env bash

# Generic run script that activates the cluster/local environment via setup_env.sh
# (supports Intel Gaudi 2 on SOL, NVIDIA A100 on SOL, NVIDIA V100 on Phoenix, and local venv)
# and launches training with passthrough of all Hydra CLI overrides.

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source setup_env.sh

python "${MY_PYTHON_CODE:-src/train_v2_lightning.py}" "$@"

