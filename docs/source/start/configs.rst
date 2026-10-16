====================
Configuration
====================

dFactory uses a strict YAML schema. Unknown sections and fields are errors;
command-line overrides use ``--set path=value``.

.. code-block:: yaml

   run:
     output_dir: outputs/sft
     seed: 42
     max_steps: 1000
     num_train_epochs: 1

   recipe:
     name: sft
     params:
       block_size: 32
       noise_range: [0.3, 0.8]
       same_token_labels: true
       # Optional LLaDA2 sliding-window capability:
       # sliding_window:
       #   window_blocks: 16
       #   sink_tokens: 4

   model:
     provider: llada2
     config_path: configs/model_configs/llada2_mini
     weights_path: /path/to/merged-model
     tokenizer_path: /path/to/original-model
     attention: flex_attention
     moe: fused
     init_device: meta
     mask_token_id: 156895
     eos_token_id: 156892
     pad_token_id: 126081

   data:
     source: jsonl
     path: /path/to/train.jsonl
     format: conversation
     text_key: messages
     max_seq_len: 2048
     packing: true
     num_workers: 8

   distributed:
     mode: fsdp2
     dp: 8
     dp_replicate: 1
     ep: 1
     gradient_checkpointing: true

   optimization:
     optimizer: adamw
     lr: 1.0e-5
     scheduler: cosine
     global_batch_size: 8
     micro_batch_size: 1
     max_grad_norm: 1.0

   checkpoint:
     resume_from: null
     save_every_steps: 1000
     save_every_epochs: 1
     save_hf_weights: true

   logging:
     report_to: [stdout]
     log_every_steps: 10

``global_batch_size`` must be divisible by
``micro_batch_size * distributed.dp``. ``max_seq_len`` must be divisible by the
recipe block size. At training startup, the ``torchrun`` ``WORLD_SIZE`` must
equal ``distributed.dp``; expert parallelism partitions that world rather than
multiplying it. FSDP2 model construction requires ``model.init_device: meta``;
CPU initialization and FSDP CPU offload are not supported. Token IDs are
resolved in the order explicit run config, tokenizer, then model config.

``model.moe: fused`` consumes the merged expert representation produced by
``scripts/moe_convertor.py --mode merge`` and supports expert parallelism.
``model.moe: eager`` consumes split Hugging Face expert weights and is limited
to ``distributed.ep: 1``.

Use ``dfactory config validate --config PATH`` to print the resolved YAML.
