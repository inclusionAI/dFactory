#!/usr/bin/env bash
set -euo pipefail

: "${SMOKE_CONFIG:?Set SMOKE_CONFIG to a real training configuration on shared storage}"
: "${SMOKE_OUTPUT_ROOT:?Set SMOKE_OUTPUT_ROOT to a new shared output directory}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
COORDINATION_TIMEOUT_SECONDS="${COORDINATION_TIMEOUT_SECONDS:-600}"

for value_name in NNODES NODE_RANK NPROC_PER_NODE COORDINATION_TIMEOUT_SECONDS; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$value_name must be a non-negative integer" >&2
    exit 2
  fi
done

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

if ((NNODES > 1)); then
  : "${MASTER_ADDR:?Set MASTER_ADDR for a multi-node qualification}"
  : "${MASTER_PORT:?Set MASTER_PORT for a multi-node qualification}"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  MASTER_PORT="${MASTER_PORT:-29500}"
fi

if ((NPROC_PER_NODE < 2)); then
  echo "NPROC_PER_NODE must be at least 2" >&2
  exit 2
fi
if ((NNODES < 1 || NODE_RANK < 0 || NODE_RANK >= NNODES)); then
  echo "NNODES and NODE_RANK do not describe a valid topology" >&2
  exit 2
fi
if ((COORDINATION_TIMEOUT_SECONDS < 1)); then
  echo "COORDINATION_TIMEOUT_SECONDS must be positive" >&2
  exit 2
fi
if [[ ! -f "$SMOKE_CONFIG" ]]; then
  echo "SMOKE_CONFIG does not exist: $SMOKE_CONFIG" >&2
  exit 2
fi

python - <<'PY'
import platform
import sys

import torch

if sys.platform != "linux" or platform.machine() != "x86_64":
    raise SystemExit("release qualification requires x86-64 Linux")
if torch.version.cuda != "12.8" or not torch.cuda.is_available():
    raise SystemExit("release qualification requires a CUDA 12.8 PyTorch build")
if torch.cuda.device_count() < 2:
    raise SystemExit("release qualification requires at least two visible GPUs per node")
if not torch.distributed.is_nccl_available():
    raise SystemExit("release qualification requires NCCL")
PY

code_revision="$(git rev-parse HEAD)"
coordination_file="$SMOKE_OUTPUT_ROOT/.qualification-ready"
coordination_record=$(printf '%s\n' \
  "code_revision=$code_revision" \
  "config=$SMOKE_CONFIG" \
  "nnodes=$NNODES" \
  "nproc_per_node=$NPROC_PER_NODE" \
  "master_addr=$MASTER_ADDR" \
  "master_port=$MASTER_PORT")

if ((NODE_RANK == 0)); then
  if ! mkdir "$SMOKE_OUTPUT_ROOT"; then
    echo "SMOKE_OUTPUT_ROOT must be a new directory: $SMOKE_OUTPUT_ROOT" >&2
    exit 2
  fi
  printf '%s\n' "$coordination_record" >"$coordination_file.tmp"
  mv "$coordination_file.tmp" "$coordination_file"
else
  deadline=$((SECONDS + COORDINATION_TIMEOUT_SECONDS))
  while [[ ! -f "$coordination_file" ]]; do
    if ((SECONDS >= deadline)); then
      echo "timed out waiting for node 0 to initialize $SMOKE_OUTPUT_ROOT" >&2
      exit 2
    fi
    sleep 1
  done
  if [[ "$(<"$coordination_file")" != "$coordination_record" ]]; then
    echo "qualification settings do not match node 0: $coordination_file" >&2
    exit 2
  fi
fi

shared_barrier() {
  local phase="$1"
  local barrier_dir="$SMOKE_OUTPUT_ROOT/.barriers/$phase"
  local marker="$barrier_dir/node-$NODE_RANK"
  local marker_tmp="$marker.tmp"
  local barrier_deadline
  local rank

  mkdir -p "$barrier_dir"
  printf 'completed\n' >"$marker_tmp"
  mv "$marker_tmp" "$marker"

  barrier_deadline=$((SECONDS + COORDINATION_TIMEOUT_SECONDS))
  for ((rank = 0; rank < NNODES; rank++)); do
    while [[ ! -f "$barrier_dir/node-$rank" ]]; do
      if ((SECONDS >= barrier_deadline)); then
        echo "timed out waiting for node $rank at phase $phase" >&2
        exit 2
      fi
      sleep 1
    done
  done
}

launcher=(
  --nnodes="$NNODES"
  --node-rank="$NODE_RANK"
  --nproc-per-node="$NPROC_PER_NODE"
  --master-addr="$MASTER_ADDR"
  --master-port="$MASTER_PORT"
)

base_output="$SMOKE_OUTPUT_ROOT/base"
resume_output="$SMOKE_OUTPUT_ROOT/resume"
accumulation_output="$SMOKE_OUTPUT_ROOT/accumulation"
base_checkpoint="$base_output/checkpoints/global_step_2"

python -m torch.distributed.run "${launcher[@]}" -m dfactory train \
  --config "$SMOKE_CONFIG" \
  --set run.output_dir="$base_output" \
  --set run.max_steps=2 \
  --set run.num_train_epochs=100 \
  --set distributed.dp="$WORLD_SIZE" \
  --set optimization.global_batch_size="$WORLD_SIZE" \
  --set optimization.micro_batch_size=1 \
  --set checkpoint.save_every_steps=1 \
  --set checkpoint.save_hf_weights=true
shared_barrier base

python -m torch.distributed.run "${launcher[@]}" -m dfactory train \
  --config "$SMOKE_CONFIG" \
  --set run.output_dir="$resume_output" \
  --set run.max_steps=3 \
  --set run.num_train_epochs=100 \
  --set distributed.dp="$WORLD_SIZE" \
  --set optimization.global_batch_size="$WORLD_SIZE" \
  --set optimization.micro_batch_size=1 \
  --set checkpoint.resume_from="$base_checkpoint" \
  --set checkpoint.save_every_steps=1 \
  --set checkpoint.save_hf_weights=true
shared_barrier resume

python -m torch.distributed.run "${launcher[@]}" -m dfactory train \
  --config "$SMOKE_CONFIG" \
  --set run.output_dir="$accumulation_output" \
  --set run.max_steps=2 \
  --set run.num_train_epochs=100 \
  --set distributed.dp="$WORLD_SIZE" \
  --set optimization.global_batch_size="$((WORLD_SIZE * 2))" \
  --set optimization.micro_batch_size=1 \
  --set checkpoint.save_every_steps=1 \
  --set checkpoint.save_hf_weights=true
shared_barrier accumulation

node_evidence="$SMOKE_OUTPUT_ROOT/node-${NODE_RANK}-evidence.txt"
{
  date -u '+completed_at=%Y-%m-%dT%H:%M:%SZ'
  echo "hostname=$(hostname)"
  echo "node_rank=$NODE_RANK"
  echo "code_revision=$code_revision"
  python -c 'import torch; print(f"torch={torch.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"nccl={torch.cuda.nccl.version()}")'
  nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
} >"$node_evidence.tmp"
mv "$node_evidence.tmp" "$node_evidence"

if ((NODE_RANK == 0)); then
  if [[ -e "$resume_output/checkpoints/global_step_1" || -e "$resume_output/checkpoints/global_step_2" ]]; then
    echo "resume qualification restarted before global step 3" >&2
    exit 2
  fi

  for checkpoint in \
    "$base_output/checkpoints/global_step_2" \
    "$resume_output/checkpoints/global_step_3" \
    "$accumulation_output/checkpoints/global_step_2"; do
    test -f "$checkpoint/.metadata"
    test -f "$checkpoint/resolved.yaml"
    test -f "$checkpoint/model_assets/config.json"
    test -f "$checkpoint/model_assets/tokenizer_config.json"
    test -f "$checkpoint/hf_ckpt/config.json"
    test -f "$checkpoint/hf_ckpt/tokenizer_config.json"
    test -f "$checkpoint/hf_ckpt/configuration_dfactory_llada2.py"
    test -f "$checkpoint/hf_ckpt/modeling_dfactory_llada2.py"
    find "$checkpoint/hf_ckpt" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .
  done

  EXPORT_PATH="$resume_output/checkpoints/global_step_3/hf_ckpt" python - <<'PY'
import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dfactory.models.llada2 import LLaDA2MoeModelLM

path = os.environ["EXPORT_PATH"]
model = LLaDA2MoeModelLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    local_files_only=True,
)
del model
gc.collect()
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    local_files_only=True,
)
del model
gc.collect()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
del tokenizer
PY

  deadline=$((SECONDS + COORDINATION_TIMEOUT_SECONDS))
  for ((rank = 0; rank < NNODES; rank++)); do
    evidence_file="$SMOKE_OUTPUT_ROOT/node-${rank}-evidence.txt"
    while [[ ! -f "$evidence_file" ]]; do
      if ((SECONDS >= deadline)); then
        echo "timed out waiting for node evidence: $evidence_file" >&2
        exit 2
      fi
      sleep 1
    done
  done

  release_evidence="$SMOKE_OUTPUT_ROOT/release-evidence.txt"
  {
    date -u '+completed_at=%Y-%m-%dT%H:%M:%SZ'
    echo "code_revision=$code_revision"
    echo "config=$SMOKE_CONFIG"
    echo "nnodes=$NNODES"
    echo "nproc_per_node=$NPROC_PER_NODE"
    echo "world_size=$WORLD_SIZE"
    echo "normal_optimizer_steps=2"
    echo "resumed_global_step=3"
    echo "accumulation_optimizer_steps=2"
    echo "gradient_accumulation_steps=2"
    echo "hf_load=dfactory,transformers_auto,tokenizer"
  } >"$release_evidence.tmp"
  mv "$release_evidence.tmp" "$release_evidence"
fi
