#!/bin/bash

# 1. Check if a Conda environment is already active
if [ -z "$CONDA_PREFIX" ] && command -v scontrol &> /dev/null; then
    # Get the Cluster Name from Slurm
    CLUSTER=$(scontrol show config | grep -i "ClusterName" | awk '{print $3}')
    
    if [ "$CLUSTER" == "phoenix" ]; then
        echo "Activating conda environment for $CLUSTER..."
        module load mamba gcc-12.3.0-vw
        source activate pytorch-gpu-2.3.1-cuda-12.1
        
    elif [ "$CLUSTER" == "sol" ]; then
        echo "Activating conda environment for $CLUSTER..."
        if [[ "$(hostname)" == gaudi* ]]; then
            module load mamba
            source activate gaudi-pytorch-vllm
        else
            module load mamba gcc-13.2.0-gcc-12.1.0
            source activate pytorch-gpu-2.3.1-cuda-12.1
        fi
    else
        echo "Slurm detected, but no specific config for cluster: $CLUSTER"
    fi

# 2. If NOT in a cluster and NO environment is active, use local venv
elif [ -z "$CONDA_PREFIX" ] && [ -z "$VIRTUAL_ENV" ]; then
    VENV_PATH="$HOME/.venvs/torch-cuda/bin/activate"
    
    if [ -f "$VENV_PATH" ]; then
        echo "Not in cluster. Activating local venv: $VENV_PATH"
        source "$VENV_PATH"
    else
        echo "Manual Check: No Slurm cluster found and no local venv at $VENV_PATH"
    fi

else
    # 3. Inform the user if an environment was already loaded
    ACTIVE_ENV="${CONDA_PREFIX:-$VIRTUAL_ENV}"
    echo "Environment already active: $ACTIVE_ENV"
fi
