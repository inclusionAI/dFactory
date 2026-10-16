#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "Usage: bash scripts/train_single_node.sh CONFIG [--set PATH=VALUE ...]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
config_path="$1"
shift

if [[ "$config_path" != /* ]]; then
  config_path="$PWD/$config_path"
fi

cd "$project_root"

python_bin="${DFACTORY_PYTHON:-python}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
fi
if [[ ! "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || ((NPROC_PER_NODE < 2)); then
  echo "NPROC_PER_NODE must be an integer of at least 2" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

dfactory_args=(
  --config "$config_path"
  "$@"
  --set "distributed.dp=$NPROC_PER_NODE"
)

"$python_bin" -m dfactory config validate "${dfactory_args[@]}" >/dev/null

exec "$python_bin" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$NPROC_PER_NODE" \
  -m dfactory train "${dfactory_args[@]}"
