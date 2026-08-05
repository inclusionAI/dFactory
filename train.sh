#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

NNODES=${NNODES:=1}
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
  elif command -v npu-smi >/dev/null 2>&1; then
    NPROC_PER_NODE=$(npu-smi info 2>/dev/null | awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ {print $1}' | sort -u | wc -l | tr -d ' ')
    [[ "$NPROC_PER_NODE" == "0" ]] && NPROC_PER_NODE=1
  else
    NPROC_PER_NODE=1
  fi
fi
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}

if [[ "$NNODES" == "1" ]]; then
  additional_args="$additional_args --standalone"
else
  additional_args="--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  $additional_args $@ 2>&1 | tee log.txt
