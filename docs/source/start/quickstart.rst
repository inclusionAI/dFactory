==========
Quickstart
==========

Prepare assets
==============

Download a LLaDA2 checkpoint and convert its experts to the merged training
representation:

.. code-block:: bash

   python scripts/download_hf_model.py \
     --repo-id inclusionAI/LLaDA2.0-mini-preview \
     --local-dir /path/to/original-model

   python scripts/moe_convertor.py \
     --input-path /path/to/original-model \
     --output-path /path/to/merged-model \
     --mode merge

Training data is JSON Lines with a ``messages`` array on every record:

.. code-block:: json

   {"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}

Configure
=========

Copy ``configs/recipes/sft/llada2_mini.yaml`` and set the model, tokenizer,
dataset, and output paths. Validate it before launching:

.. code-block:: bash

   dfactory config validate --config configs/recipes/sft/llada2_mini.yaml

Train
=====

The supported training environment is an x86-64 Linux CUDA server. For normal
single-node development, use the repository launcher. It auto-detects all
visible GPUs, requires at least two, validates the resolved config, and starts
``torchrun --standalone``:

.. code-block:: bash

   bash scripts/train_single_node.sh \
     configs/recipes/sft/llada2_mini.yaml \
     --set model.weights_path=/models/llada2-mini-merged \
     --set model.tokenizer_path=/models/llada2-mini-original \
     --set data.path=/datasets/train.jsonl \
     --set run.output_dir=/outputs/sft-run

Use ``CUDA_VISIBLE_DEVICES`` to select GPUs or ``NPROC_PER_NODE`` to override
the detected process count. Additional values can be changed without editing
the file:

.. code-block:: bash

   NPROC_PER_NODE=4 bash scripts/train_single_node.sh \
     configs/recipes/sft/llada2_mini.yaml \
     --set model.weights_path=/models/llada2-mini-merged \
     --set model.tokenizer_path=/models/llada2-mini-original \
     --set data.path=/datasets/train.jsonl \
     --set run.max_steps=10 \
     --set optimization.global_batch_size=8 \
     --set data.packing=false

Resume
======

Set ``checkpoint.resume_from`` to the exact checkpoint directory:

.. code-block:: bash

   bash scripts/train_single_node.sh \
     configs/recipes/sft/llada2_mini.yaml \
     --set model.weights_path=/models/llada2-mini-merged \
     --set model.tokenizer_path=/models/llada2-mini-original \
     --set data.path=/datasets/train.jsonl \
     --set checkpoint.resume_from=/path/to/checkpoints/global_step_1000

The resumed run restores model, optimizer, scheduler, training progress, and
PyTorch random state.
