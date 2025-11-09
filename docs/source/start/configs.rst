Configuration Design Guide
**************************

YAML Configuration Structure
============================

Configuration files use hierarchical YAML structure with three main sections:

.. code-block:: yaml

   model:
     # Model-specific settings
   data:
     # Data loading and preprocessing
   train:
     # Training hyperparameters and setup

Model Configuration
===================

Model Section
-------------

**Required fields:**

.. code-block:: yaml

   model:
     config_path: "./configs/model_configs/[model_name]"
     model_path: "./path/to/model"
     tokenizer_path: "./path/to/tokenizer"
     attn_implementation: "sdpa"        # sdpa|eager|flex_attention
     moe_implementation: "fused"        # fused|standard

.. warning::

   **Flash Attention Limitation**: The flash attention backend can only be used with ``full_attention`` or ``causal_attention`` modes. It **cannot** adapt to custom attention types used in LLaDA2.0 models. **Do not use FlashAttention (flash_attn2/flash_attn) for Block Diffusion Mode models**. See :doc:`/algo/block_diffusion` for detailed explanation of block diffusion training.

Data Configuration
==================

Data Section
------------

**Template for conversation data:**

.. code-block:: yaml

   data:
     train_path: "./datasets/train.jsonl"
     data_type: "conversation"          # conversation|plain|instruction
     datasets_type: "mapping"           # mapping|streaming
     dataloader_type: "native"          # native|custom
     max_seq_len: 2048
     text_keys: "messages"              # field name in JSON
     noise_range_low: 0.3               # diffusion noise lower bound
     noise_range_high: 0.8              # diffusion noise upper bound
     num_workers: 16

- Support multiple data formats (JSONL, Parquet)
- Configurable noise ranges for diffusion training
- Flexible text field mapping
- Worker count based on CPU cores

Training Configuration
======================

Training Section
----------------

**Distributed training setup:**

.. code-block:: yaml

   train:
     output_dir: "./outputs/experiment_name"
     
     # Parallel configuration
     data_parallel_mode: "fsdp2"        # fsdp2
     tensor_parallel_size: 1            # model parallel
     ulysses_parallel_size: 1           # sequence parallel
     expert_parallel_size: 1            # MoE parallel
     
     # Batch configuration
     global_batch_size: 16              # total batch across all GPUs
     micro_batch_size: 1                # batch per GPU
     
     # Training schedule
     num_train_epochs: 1
     save_epochs: 1                     # checkpoint frequency
     log_steps: 1                       # logging frequency

**Optimization parameters:**

.. code-block:: yaml

   train:
     optimizer: "adamw"
     beta1: 0.9
     beta2: 0.999
     lr: 1.0e-5                        # learning rate
     lr_warmup_ratio: 0.03             # warmup steps ratio
     lr_decay_style: "cosine"          # cosine|linear|constant
     weight_decay: 0.1
     max_grad_norm: 1.0

**Memory optimization:**

.. code-block:: yaml

   train:
     enable_mixed_precision: true
     enable_gradient_checkpointing: true
     enable_full_shard: true           # FSDP parameter sharding
     enable_fsdp_offload: true         # CPU offloading
     empty_cache_steps: 500            # GPU memory cleanup

Configuration Patterns
======================

Model Scaling
-------------

**Small model template:**

.. code-block:: yaml

   train:
     global_batch_size: 8               # Reduce for smaller models
     micro_batch_size: 1

**Large model template:**

.. code-block:: yaml

   train:
     global_batch_size: 64              # Increase for larger models
     micro_batch_size: 1
     tensor_parallel_size: 2            # Enable model parallelism
     expert_parallel_size: 2            # Distribute experts

Dataset Adaptation
------------------

**For large datasets:**

.. code-block:: yaml

   data:
     datasets_type: "streaming"        # Memory-efficient loading
     num_workers: 32                   # Increase workers

**For small datasets:**

.. code-block:: yaml

   data:
     datasets_type: "mapping"          # Full dataset in memory
     num_workers: 8                    # Reduce overhead

Hardware Adaptation
===================

Single GPU Setup
----------------

.. code-block:: yaml

   train:
     data_parallel_mode: "fsdp2"
     tensor_parallel_size: 1
     expert_parallel_size: 1
     global_batch_size: 4            # Fit single GPU
     micro_batch_size: 1
     enable_fsdp_offload: false      # Disable offloading

Multi-GPU Setup
---------------

.. code-block:: yaml

   train:
     data_parallel_mode: "fsdp2"
     tensor_parallel_size: 1
     expert_parallel_size: 2         # Distribute experts
     global_batch_size: 32           # Scale with GPU count
     micro_batch_size: 1
     enable_fsdp_offload: false      # Faster training

Memory-Constrained Setup
------------------------

.. code-block:: yaml

   train:
     enable_gradient_checkpointing: true
     enable_full_shard: true
     enable_fsdp_offload: true       # Enable CPU offloading
     enable_activation_offload: true # Reduce GPU memory
     micro_batch_size: 1             # Minimal per-GPU batch

Best Practices
==============

**Path Management**
   Use relative paths for configs
      Store absolute paths in environment variables
      Create separate output directories per experiment

**Parameter Tuning**
   Start with conservative batch sizes
      Increase learning rate for larger batches
      Adjust warmup ratio based on dataset size

**Monitoring**
   Enable W&B for experiment tracking
      Set appropriate logging frequency
      Monitor gradient norms and loss curves

**Reproducibility**
   Fix random seeds in training scripts
      Document configuration changes
      Version control configuration files