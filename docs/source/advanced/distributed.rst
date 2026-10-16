====================
Distributed Training
====================

dFactory uses ``torchrun`` for process launch and delegates FSDP2 and parallel
topology primitives to the installed VeOmni runtime.

Single node
===========

For eight GPUs, set ``distributed.dp: 8`` and launch eight processes:

.. code-block:: bash

   torchrun --standalone --nproc-per-node=8 \
     -m dfactory train \
     --config configs/recipes/sft/llada2_mini.yaml

The data-parallel size must match the ``torchrun`` world size. Expert
parallelism may partition experts within that world and must divide the data
parallel size. Tensor and sequence parallelism are intentionally not exposed
in this first release because the LLaDA2 provider does not yet define those
model transformations.

Multiple nodes
==============

Set ``distributed.dp`` to ``NNODES * NPROC_PER_NODE`` with
``NPROC_PER_NODE >= 2``, then run this command on every node with a distinct
``NODE_RANK``:

.. code-block:: bash

   WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
   torchrun \
     --nnodes="$NNODES" \
     --node-rank="$NODE_RANK" \
     --nproc-per-node="$NPROC_PER_NODE" \
     --master-addr="$MASTER_ADDR" \
     --master-port="$MASTER_PORT" \
     -m dfactory train \
     --config configs/recipes/sft/llada2_mini.yaml \
     --set distributed.dp="$WORLD_SIZE" \
     --set optimization.global_batch_size="$WORLD_SIZE"

All nodes must see the same model, dataset, and output paths. Checkpoints are
written collectively through VeOmni's distributed checkpoint implementation.
At runtime, ``WORLD_SIZE`` must equal ``distributed.dp``. ``distributed.ep``
partitions that data-parallel world; it does not multiply ``WORLD_SIZE``.
