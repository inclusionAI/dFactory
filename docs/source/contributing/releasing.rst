=================
Release checklist
=================

Run the following fast quality checks from a clean checkout on a supported
x86-64 Linux CUDA 12.8 server:

.. code-block:: bash

   uv lock --check
   uv sync --locked --extra dev --extra docs
   uv run ruff format --check .
   uv run ruff check .
   uv run pytest -q
   uv run sphinx-build -W --keep-going -b html docs/source docs/_build/html
   uv build

Install the built wheel, plus each package under ``examples/plugins``, into a
fresh environment. Confirm ``dfactory --help`` and all three registry list
commands work without a ``VeOmni/`` checkout.

The checks above must run on the supported Linux CUDA server class. They cover
algorithms, configuration, documentation, packaging boundaries, and helper
contracts, but do not qualify a training release by themselves. Results from
macOS, MPS, CPU-only, single-process, or single-GPU environments must not be
used as evidence.

CUDA distributed release gates
==============================

Copy ``configs/recipes/sft/llada2_mini.yaml`` to a shared path named by
``SMOKE_CONFIG`` and replace the model, tokenizer, and dataset paths with small
real assets. ``SMOKE_OUTPUT_ROOT`` must name a new path on storage shared by
all nodes; do not create it in advance. Node 0 creates the directory and the
other nodes wait for its launch record, which also rejects mismatched topology
or rendezvous settings. The qualification script performs a two-step run,
resumes it into a separate output directory that may contain only the third
step, runs a second two-step job with exactly two-way gradient accumulation,
validates every checkpoint payload, and loads the final Hugging Face export
through both the dFactory class and Transformers auto classes.

First run it on one node with at least two GPUs:

.. code-block:: bash

   SMOKE_CONFIG=/shared/path/smoke.yaml \
   SMOKE_OUTPUT_ROOT=/shared/path/release-smoke-single-node \
   NNODES=1 NODE_RANK=0 NPROC_PER_NODE=2 \
   uv run --no-sync bash scripts/qualify_release.sh

Then run the same smoke configuration across two nodes, with at least two GPUs
per node. Launch this command on every node with a distinct ``NODE_RANK`` and
the same rendezvous settings. The script derives ``WORLD_SIZE`` as
``NNODES * NPROC_PER_NODE``:

.. code-block:: bash

   SMOKE_CONFIG=/shared/path/smoke.yaml \
   SMOKE_OUTPUT_ROOT=/shared/path/release-smoke-two-node \
   NNODES=2 NODE_RANK="$NODE_RANK" NPROC_PER_NODE=2 \
   MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
   uv run --no-sync bash scripts/qualify_release.sh

Run the two-node command concurrently on both nodes with a distinct
``NODE_RANK``. Each training phase waits for every node before the rendezvous
port is reused. The script atomically publishes ``release-evidence.txt`` and
one ``node-N-evidence.txt`` file per node only after every required operation
succeeds.
Add those files together with the model revision, exact launch commands, and
any scheduler metadata to the release record. Both gates are mandatory; macOS,
CPU-only, single-GPU, and unit-test results cannot replace either one.
