#!/usr/bin/env bash
set -euo pipefail

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log

mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

# Detener si quedó algo previo
echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true

# Arrancar daemon
nvidia-cuda-mps-control -d

echo "MPS iniciado"