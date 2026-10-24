#!/usr/bin/env bash

# Generic run script that works with submit-all-slurm.sh
# Calls train_v2_lightning.py with passthrough of all arguments
# Handles conda environment activation and module loading

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# For SOL/Phoenix Cluster, check if we are already in environment: $CONDA_PREFIX
CLUSTER_NAME=$(scontrol show config 2>/dev/null | grep "ClusterName" | awk -F= '{print $2}' | tr -d ' ' || echo "")

if [ -z "$CONDA_PREFIX" ]; then
    echo "Activating conda environment..."
    if [ "$CLUSTER_NAME" == "sol" ]; then
        module load mamba gcc-13.2.0-gcc-12.1.0
        source activate pytorch-gpu-2.3.1-cuda-12.1
    elif [ "$CLUSTER_NAME" == "phoenix" ]; then
        module load mamba gcc-12.3.0-vw
        source activate pytorch-gpu-2.3.1-cuda-12.1
    else
        # Fallback: try to activate if module system is available
        if command -v module &> /dev/null; then
            module load mamba 2>/dev/null || true
            source activate pytorch-gpu-2.3.1-cuda-12.1 2>/dev/null || true
        fi
    fi
fi

python src/train_v1_classic.py "$@"
