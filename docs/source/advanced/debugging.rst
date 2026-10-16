=======================
Debugging training runs
=======================

Validate configuration before allocating GPUs:

.. code-block:: bash

   dfactory config validate --config config.yaml
   dfactory recipes describe sft

For a training failure on a CUDA server, start with a short two-GPU run before
expanding to the full node or multi-node topology:

.. code-block:: bash

   torchrun --standalone --nproc-per-node=2 -m dfactory train \
     --config config.yaml \
     --set run.max_steps=2 \
     --set distributed.dp=2

The resolved configuration is written to ``run.output_dir/resolved.yaml``.
Metrics are JSON objects on stdout when ``logging.report_to`` contains
``stdout``. Checkpoints live below ``run.output_dir/checkpoints``. Each step
directory includes the resolved configuration and model/tokenizer assets next
to the distributed training state.

When a two-process run succeeds but a larger distributed run fails, verify
that ``WORLD_SIZE`` matches ``distributed.dp`` and that
``optimization.global_batch_size`` is divisible by
``optimization.micro_batch_size * distributed.dp``. The initial public release
does not expose tensor or sequence parallelism. Expert parallelism partitions
the data-parallel world and does not multiply ``WORLD_SIZE``.

Recipe bugs should be reproduced against pure functions in
``components.block_diffusion`` or the recipe package before debugging the
runtime gateway. Runtime-specific failures belong under
``dfactory.runtime.veomni``; recipes must not work around them with direct
VeOmni imports.
