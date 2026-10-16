<div align="center">
  <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*9yAZT61NXRkAAAAARBAAAAgAemJ7AQ/original" width="40%" alt="dFactory" />
</div>

# dFactory

> **Development preview:** the standalone 0.2 development line currently lives
> on the `develop` branch while the public repository migration is completed.

dFactory is an extensible training framework for discrete diffusion language
models. It provides SFT, autoregressive SFT, multi-stage fine-tuning, and
continued-pretraining recipes for the LLaDA2 family, with PyTorch FSDP2 and
VeOmni providing the distributed runtime.

dFactory owns its model, data, recipe, checkpoint, and configuration semantics.
VeOmni is an installed dependency, not a source submodule and not part of the
public extension API.

## Installation

dFactory supports x86-64 Linux servers with NVIDIA GPUs and CUDA 12.8. It does
not support macOS, MPS, or CPU training. Python 3.11, 3.12, or 3.13 is required.
CUDA dependencies, including VeOmni's GPU runtime, are part of the default
installation rather than an optional execution mode.

```bash
git clone --branch develop https://github.com/inclusionAI/dFactory.git
cd dFactory

# CUDA training and development environment
uv sync --locked --extra dev
source .venv/bin/activate
```

With pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel packaging ninja
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  --no-build-isolation -e .
```

The two-stage pip flow is required because `flash-attn` imports Torch while
building its CUDA extension. `uv sync` handles that build dependency through
the repository configuration.

The distribution, import package, and command are all named `dfactory`.

## Prepare a run

The current SFT path accepts conversation JSONL. Every line is one object:

```json
{"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
```

Download LLaDA2 weights and convert the experts to the merged representation
used by the VeOmni kernels:

```bash
python scripts/download_hf_model.py \
  --repo-id inclusionAI/LLaDA2.0-mini-preview \
  --local-dir /path/to/original-model

python scripts/moe_convertor.py \
  --input-path /path/to/original-model \
  --output-path /path/to/merged-model \
  --mode merge
```

Copy `configs/recipes/sft/llada2_mini.yaml` and set:

- `model.weights_path` to the merged model;
- `model.tokenizer_path` to the original model/tokenizer;
- `data.path` to the training JSONL;
- `distributed.dp` to the number of data-parallel processes.

For LLaDA2 Flash, start from
`configs/recipes/sft/llada2_flash.yaml` and use a validated topology with
enough aggregate GPU memory for model, gradients, and optimizer state.

Validate the complete configuration before launching:

```bash
dfactory config validate --config configs/recipes/sft/llada2_mini.yaml
```

## Train

For normal single-node multi-GPU development, use the repository launcher. It
uses every visible GPU by default, validates the resolved config once, and then
starts `torchrun --standalone`:

```bash
bash scripts/train_single_node.sh \
  configs/recipes/sft/llada2_mini.yaml \
  --set model.weights_path=/models/llada2-mini-merged \
  --set model.tokenizer_path=/models/llada2-mini-original \
  --set data.path=/datasets/train.jsonl \
  --set run.output_dir=/outputs/sft-run \
  --set optimization.global_batch_size=8
```

Set `CUDA_VISIBLE_DEVICES` to select GPUs or `NPROC_PER_NODE` to override the
auto-detected process count. `NPROC_PER_NODE` must be at least two, and the
launcher always sets `distributed.dp` to the resulting local world size. Pass
any other config change as an additional `--set PATH=VALUE` argument.

The same launcher directly supports every checked-in training profile:

| Training target | Config |
|---|---|
| LLaDA2 Mini SFT | `configs/recipes/sft/llada2_mini.yaml` |
| LLaDA2 Mini SFT + SWA | `configs/recipes/sft/llada2_mini_swa.yaml` |
| LLaDA2 Flash SFT | `configs/recipes/sft/llada2_flash.yaml` |
| AR SFT | `configs/recipes/ar_sft/llada2_mini.yaml` |
| MTF | `configs/recipes/mtf/llada2_mini.yaml` |
| EMTF | `configs/recipes/emtf/llada2_mini.yaml` |
| EMTF random-T2T | `configs/recipes/emtf_random_t2t/llada2_mini.yaml` |
| AR2BD continued pretraining | `configs/recipes/ar2bd/llada2_mini.yaml` |
| ESTF continued pretraining | `configs/recipes/estf/llada2_mini.yaml` |

To resume, use the same command with
`--set checkpoint.resume_from=/path/to/checkpoints/global_step_N`.

For multiple nodes, launch at least two processes per node and run the same
command on every node with its own `NODE_RANK`:

```bash
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" \
  --nproc-per-node="$NPROC_PER_NODE" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m dfactory train \
  --config configs/recipes/sft/llada2_mini.yaml \
  --set distributed.dp="$WORLD_SIZE" \
  --set optimization.global_batch_size="$WORLD_SIZE"
```

Resolved configuration is written to `run.output_dir/resolved.yaml`.
Distributed checkpoints are written below `run.output_dir/checkpoints`; every
`global_step_N` directory also contains its resolved configuration and
model/tokenizer assets. Set `checkpoint.resume_from` to that directory to
resume its model, optimizer, scheduler, RNG, and dataloader progress.

## Architecture

```text
src/dfactory/
  api.py                         stable Python API
  cli.py                         dfactory command
  config/                        strict, dFactory-owned configuration
  recipes/                       self-contained recipe config, data, trainer
  trainers/                      shared trainer lifecycle and context
  components/block_diffusion/    shared Block Diffusion primitives
  data/                          sources, transforms, packing, collators
  models/                        dFactory-owned model providers
  runtime/veomni/                sole VeOmni integration boundary
```

`dfactory.trainers.BaseTrainer` owns the complete training lifecycle: runtime,
model and data construction, gradient accumulation, optimization, checkpoint
resume/save, metrics, and model export. Concrete trainers such as
`recipes/sft/trainer.py` and `recipes/emtf/trainer.py` implement only their
algorithm-specific `training_step`. Each recipe directory owns its
configuration, validation, data preparation, trainer, and the `RecipeSpec`
that ties them together.

The Block Diffusion package is also intentionally narrow. It contains reusable
layout, attention, noise, and objective primitives; complete diffusion training
logic belongs to a recipe.

## Extend dFactory

Recipes, dataset sources, and model providers are discovered through Python
entry points:

```toml
[project.entry-points."dfactory.recipes"]
my_recipe = "my_package.recipe:recipe_spec"

[project.entry-points."dfactory.datasets"]
my_source = "my_package.data:dataset_spec"

[project.entry-points."dfactory.models"]
my_model = "my_package.model:model_spec"
```

An extension owns its implementation and declares a compatible dFactory
version. Its `RecipeSpec` owns run validation and dataset construction, and
selects a `BaseTrainer` subclass. The shared lifecycle never assumes SFT
parameters such as `block_size`. An extension does not need to modify the
central CLI or registry.

Runnable packaging examples for all three extension points live under
[`examples/plugins`](examples/plugins). After installation they appear in
`dfactory recipes list`, `dfactory datasets list`, and `dfactory models list`.

## Development and validation

Run development checks on the same x86-64 Linux CUDA 12.8 server class used by
the project. A macOS, MPS, CPU-only, single-process, or single-GPU environment
is not a supported development or qualification target.

```bash
uv run ruff format src tests
uv run ruff check src tests
uv run pytest tests/components/block_diffusion/test_attention.py -v
uv run pytest tests/recipes/sft/test_trainer.py -v
uv build
```

Unit checks do not qualify a release. Training validation requires both Linux
CUDA multi-GPU and multi-node release gates documented in the release guide.

See the [documentation](docs/source/index.rst) for configuration and extension
details, including the migration guide and installable plugin examples.

## License

Apache License 2.0. See [LICENSE](LICENSE).
