.. _quickstart:

===============================
Quickstart: SFT upon dFactory
===============================

Last updated: 2025-11-04

This quickstart guide provides comprehensive VeOmni best practices for training, including installation, configuration, and advanced usage patterns.

VeOmni Best Practices
=====================

Usage
-----

Run Example Script
------------------

Verify training startup (need to download the dataset first):

.. code-block:: bash

   sh train.sh tasks/train_llada2_bd.py configs/sft/llada2_mini_bd_sft.yaml

Create Custom Task Directory
----------------------------

`train_torch.py <../../tasks/train_torch.py>`_ can be used for most pre-training and post-training tasks. You can just modify the train config to complete your task. However, if you want to create a new task, you can copy the ``train_torch.py`` file from the ``tasks`` directory and modify it, like `tasks/omni/train_qwen2_vl.py <../../tasks/omni/train_qwen2_vl.py>`_.

.. code-block:: bash

   mkdir tasks/your_task
   cp tasks/train_torch.py tasks/your_task/train.py

Launch Custom Training
----------------------

You can overwrite the default arguments in train yaml by passing them to the script.

.. code-block:: bash

   bash train.sh tasks/your_task/train.py \
       $CONFIG.yaml \
       --model.model_path your_path_to_model \
       --data.train_path your_path_to_dataset \
       --train.output_dir your_path_to_save_checkpoints \
       --train.wandb_project your_project_name \
       --train.wandb_name your_experiment_name

Arguments
---------

Default Parameter Access
~~~~~~~~~~~~~~~~~~~~~~~~

VeOmni offers a unified argument management system that can be easily extended to support custom arguments. For default arguments explanation, refer to `Config arguments Explanation <../config/config.md>`_.

Source code: `veomni/utils/arguments.py <../../VeOmni/veomni/utils/arguments.py>`_.

.. code-block:: python

   from dataclasses import dataclass, field
   from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args

   @dataclass
   class Arguments:
       model: "ModelArguments" = field(default_factory=ModelArguments)
       data: "DataArguments" = field(default_factory=DataArguments)
       train: "TrainingArguments" = field(default_factory=TrainingArguments)

   if __name__ == "__main__":
       args = parse_args(Arguments)
       print(args.train.lr)  # Access default arguments

Custom Parameter Extension
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can extend the default arguments by creating a new class that inherits from the existing class.

.. code-block:: python

   @dataclass
   class CustomTrainingArguments(TrainingArguments):
       enable_xxx: bool = field(
           default=False,
           metadata={"help": "Enable me if necessary."},
       )

   @dataclass
   class Arguments:
       model: "ModelArguments" = field(default_factory=ModelArguments)
       data: "DataArguments" = field(default_factory=DataArguments)
       train: "CustomTrainingArguments" = field(default_factory=CustomTrainingArguments)

Parallel State
--------------

VeOmni uses torch device mesh to manage all parallel states, which is useful for multi-dimensional parallelism (i.e., 3-D parallel) where parallelism composability is required. You can create the parallel state by calling the ``init_parallel_state`` function and get the parallel state by calling the ``get_parallel_state`` function.

For more details about torch device mesh, refer to `Getting Started with DeviceMesh <https://pytorch.org/tutorials/recipes/distributed_device_mesh.html>`_.

Source code: `veomni/distributed/parallel_state.py <../../VeOmni/veomni/distributed/parallel_state.py>`_.

.. note::
   
   The parallel state system provides a unified interface for managing different types of parallelism including data parallel, tensor parallel, expert parallel, and pipeline parallel.

.. code-block:: python

   from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state

   init_parallel_state(
       dp_size=args.train.data_parallel_size,  # data parallel size
       dp_replicate_size=args.train.data_parallel_replicate_size,  # data parallel replicate size
       dp_shard_size=args.train.data_parallel_shard_size,  # data parallel shard degree
       tp_size=args.train.tensor_parallel_size,  # tensor parallel size
       ep_size=args.train.expert_parallel_size,  # expert parallel size
       pp_size=args.train.pipeline_parallel_size,  # pipeline parallel size, not supported now
       cp_size=args.train.context_parallel_size,  # context parallel size, not supported now
       ulysses_size=args.train.ulysses_parallel_size,  # ulysses parallel size
       dp_mode=args.train.data_parallel_mode,  # data parallel mode, can be "ddp", "fsdp1", "fsdp2"
   )

   parallel_state = get_parallel_state()

   # Access dp state
   dp_mesh = parallel_state.dp_mesh
   dp_group = parallel_state.dp_group

   # Access sp state
   sp_group = parallel_state.sp_group
   sp_rank = parallel_state.sp_rank

   # Access tp state
   tp_group = parallel_state.tp_group
   tp_mesh = parallel_state.tp_mesh

Dataset
-------

VeOmni supports two types of datasets by default:

Source code: `veomni/data/dataset.py <../../VeOmni/veomni/data/dataset.py>`_

Dataset Types
~~~~~~~~~~~~~

1. **IterativeDataset** (recommended for large datasets)
2. **MappingDataset** (default for small datasets)

.. code-block:: python

   from veomni.data import (
       build_iterative_dataset,
       build_mapping_dataset,
   )

   if args.data.datasets_type == "iterable":
       train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
       args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
   elif args.data.datasets_type == "mapping":
       train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
       args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))

.. important::

   **Training Steps Calculation**
   
   - **Iterable datasets**: Add ``data.train_size`` (tokens to consume) to config. Train steps ≈ ``train_size / (global_batch_size * max_seq_len)``
   - **Mapping datasets**: Pass ``len(train_dataset)`` to compute correct train steps

Custom Datasets
~~~~~~~~~~~~~~~

VeOmni is a flexible framework that supports custom datasets. You can implement your own dataset function and use it with VeOmni.

.. code-block:: python

   def build_custom_dataset(data_path, transform) -> Dataset:
       # Implement your custom dataset logic
       pass

   elif args.data.datasets_type == "custom":
       logger.info_rank0("Start building custom dataset")
       train_dataset = build_custom_dataset(args.data.train_path, transform=transform)
       # For iterable datasets, remove len(train_dataset)
       args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))

Data Transform (Preprocess)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

VeOmni supports two types of transforms by default:

Source code: `veomni/data/data_transform.py <../../VeOmni/veomni/data/data_transform.py>`_

Transform Types
^^^^^^^^^^^^^^^

1. **process_pretrain_example** (recommended for pretrain task)
2. **process_sft_example** (recommended for sft task)

Pretrain Example
^^^^^^^^^^^^^^^^

.. code-block:: python

   from functools import partial
   from veomni.data.data_transform import process_pretrain_example
   from veomni.models import build_tokenizer

   tokenizer = build_tokenizer(args.model.tokenizer_path)
   # To use AutoTokenizer, replace the line above with the following:
   # tokenizer = AutoTokenizer.from_pretrained(args.model.tokenizer_path)

   transform = partial(
       process_pretrain_example,
       tokenizer=tokenizer,
       max_seq_len=args.data.max_seq_len,
   )

SFT Example
^^^^^^^^^^^

.. code-block:: python

   from functools import partial
   from veomni.data.chat_template import build_chat_template
   from veomni.data.data_transform import process_sft_example

   chat_template = build_chat_template(args.data.chat_template, tokenizer)
   transform = partial(
       process_sft_example,
       chat_template=chat_template,
       max_seq_len=args.data.max_seq_len,
   )

Chat Template
~~~~~~~~~~~~~

VeOmni supports several chat templates by default and you can add your custom chat template by implementing the ``ChatTemplate`` class.

Source code: `veomni/data/chat_template.py <../../VeOmni/veomni/data/chat_template.py>`_

Custom Template Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from collections.abc import Sequence
   from veomni.data.chat_template import ChatTemplate

   class CustomTemplate(ChatTemplate):
       def encode_messages(self, messages: Sequence[dict[str, str]], max_seq_len: int = 8192) -> dict[str, list[int]]:
           # Implement encoding logic
           pass

       def get_jinja_template(self) -> str:
           return ""  # Jinja template string

DataLoader
----------

VeOmni offers a flexible and powerful dataloader implementation that supports:

- Both padding and remove padding (packing) strategy
- Dynamic batching strategy

Source code: `veomni/data/data_loader.py <../../VeOmni/veomni/data/data_loader.py>`_

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from veomni.data import build_dataloader, build_mapping_dataset

   transform = YOUR_TRANSFORM_FUNCTION

   train_dataset = build_mapping_dataset(
       data_path=args.data.train_path,
       transform=transform,
   )

   args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))

   train_dataloader = build_dataloader(
       dataset=train_dataset,
       micro_batch_size=args.train.micro_batch_size,
       global_batch_size=args.train.global_batch_size,
       dataloader_batch_size=args.train.dataloader_batch_size,
       seed=args.train.seed,
       max_seq_len=args.data.max_seq_len,
       collate_fn=None,
       train_steps=args.train.train_steps,
       rmpad=args.train.rmpad,
       rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
       bsz_warmup_ratio=args.train.bsz_warmup_ratio,
       bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
       dyn_bsz_margin=args.train.dyn_bsz_margin,
       dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
       num_workers=args.data.num_workers,
       drop_last=args.data.drop_last,
       pin_memory=args.data.pin_memory,
       prefetch_factor=args.data.prefetch_factor,
   )

Collate Function
~~~~~~~~~~~~~~~~

VeOmni supports three types of collate functions for text tasks by default:

**Text Tasks:**

- ``DataCollatorWithPadding`` (enabled when ``rmpad`` is False and ``rmpad_with_pos_ids`` is False)
- ``DataCollatorWithPacking`` (enabled when ``rmpad`` is True and ``rmpad_with_pos_ids`` is False)
- ``DataCollatorWithPositionIDs`` (enabled when ``rmpad`` is False and ``rmpad_with_pos_ids`` is True)

**Omni Model Tasks:**

- ``OmniDataCollatorWithPacking`` (for when ``rmpad_with_pos_ids`` is True)
- ``OmniDataCollatorWithPadding`` (for when ``rmpad`` is False and ``rmpad_with_pos_ids`` is False)

Source code: `veomni/data/data_collator.py <../../VeOmni/veomni/data/data_collator.py>`_

Omni model details: `veomni/data/multimodal/data_collator.py <../../VeOmni/veomni/data/multimodal/data_collator.py>`_ and usage in `train_omni_model.py <../../tasks/omni/train_omni_model.py>`_"

Model and Optimizer
===================

Model Initialization
--------------------

``build_foundation_model`` implements model initialization with config and weights path:

- Meta device initialization
- Initialize model from model config or weights path

Source code: `veomni/models/auto.py <../../VeOmni/veomni/models/auto.py>`_

.. code-block:: python

   from veomni.models import build_foundation_model

   model = build_foundation_model(
       config_path=args.model.config_path,  # model config path, can be None if weights_path is not None
       weights_path=args.model.model_path,  # model weights path, can be None if config_path is not None
       init_device=args.train.init_device,  # model init device
   )

   # You can replace with the following code if you want to use AutoModelForCausalLM from transformers
   # model = AutoModelForCausalLM.from_pretrained(args.model.model_path)

Parallelize Your Model
----------------------

Source code: `veomni/distributed/torch_parallelize.py <../../VeOmni/veomni/distributed/torch_parallelize.py>`_

.. code-block:: python

   from veomni.distributed.torch_parallelize import build_parallelize_model

   model = build_foundation_model(...)

   model = build_parallelize_model(
       model,
       enable_full_shard=args.train.enable_full_shard,
       enable_mixed_precision=args.train.enable_mixed_precision,
       enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
       init_device=args.train.init_device,
       enable_fsdp_offload=args.train.enable_fsdp_offload,
       basic_modules=model._no_split_modules + args.model.basic_modules,
   )

Optimizer and LR Scheduler
--------------------------

Source code: `veomni/optim <../../VeOmni/veomni/optim>`_

.. code-block:: python

   from veomni.optim import build_lr_scheduler, build_optimizer

   optimizer = build_optimizer(
       model,
       lr=args.train.lr,
       weight_decay=args.train.weight_decay,
       # ... other parameters
   )

   lr_scheduler = build_lr_scheduler(
       optimizer,
       train_steps=args.train.train_steps * args.train.num_train_epochs,
       # ... other parameters
   )

Train Loop
==========

After the parallel_state, model, optimizer, and dataloader are initialized, you can start the training loop.

Basic Training Loop
-------------------

.. code-block:: python

   for epoch in range(args.train.num_train_epochs):
       data_iterator = iter(train_dataloader)
       for _ in range(args.train.train_steps):
           micro_batches = next(data_iterator)
           for micro_batch in micro_batches:
               loss = model(**micro_batch).loss / len(micro_batches)
               loss.backward()

           optimizer.step()
           lr_scheduler.step()
           optimizer.zero_grad()

Custom Loss Function
--------------------

.. code-block:: python

   import torch

   loss_fct = torch.nn.CrossEntropyLoss()

   def loss_func(logits, labels):
       return loss_fct(logits, labels)

   # In train loop:
   output = model(**micro_batch)
   logits = output.logits
   loss = loss_func(logits, labels) / len(micro_batches)

Prerequisites
-------------

- The latest version of ``veomni`` and its dependencies installed following the installation guide
- A compatible GPU with sufficient memory (e.g., NVIDIA A100 with 40GB or higher)

Dataset Introduction
--------------------


