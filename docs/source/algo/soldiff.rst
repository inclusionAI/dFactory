SolDiff
========

SolDiff (Scalable Order Learning for Diffusion LLMs) learns data-dependent
unmasking orders for masked diffusion language models. A posterior q guides
order sampling during training, while a learned p head predicts order scores
from masked states. The learned-order loss ``L_LO`` is supplemented by
rank-weighted reconstruction ``L_rec`` and suffix-order ``L_ord`` losses to
support parallel decoding. Generation uses the main model and p head only.

Models
------

* `SolDiff-8B-Base <https://huggingface.co/GSAI-ML/SolDiff-8B-Base>`_:
  base model for text completion.
* `SolDiff-8B-Instruct <https://huggingface.co/GSAI-ML/SolDiff-8B-Instruct>`_:
  instruction-tuned model for instruction following and chat.

Training
--------

The supplied recipes support Qwen3 1.7B and 8B continued pretraining. Run from
the repository root in an installed dFactory GPU environment:

.. code-block:: bash

   NPROC_PER_NODE=8 bash scripts/train_single_node.sh \
     configs/recipes/soldiff/qwen3_1.7b.yaml \
     --set model.weights_path=/path/to/main-model \
     --set data.path=/path/to/token-documents.jsonl \
     --set run.output_dir=outputs/soldiff \
     --set run.max_steps=1000 \
     --set checkpoint.save_hf_weights=true

For 8B, use ``configs/recipes/soldiff/qwen3_8b.yaml``. Each profile selects its
matching model, tokenizer, q configuration and p-head configuration.
``NPROC_PER_NODE`` sets the GPU count and data-parallel size; use at least two
GPUs. Override YAML fields with ``--set field=value``.

Provide pretokenized documents without padding, using the selected tokenizer:

.. code-block:: json

   {"input_ids": [100, 200, 151645]}
   {"input_ids": [300, 400, 500, 151645]}

The loader appends a missing terminal EOS, concatenates documents, and slices
complete rows of ``data.max_seq_len`` tokens. It discards the final incomplete
row. Keep ``data.format=token_records`` and ``data.packing=false``. Training
uses EOS ``151645`` for document boundaries and mask token ``151669``.
For a dataset saved with ``datasets.save_to_disk``, set
``data.source=hf_dataset`` and point ``data.path`` to its directory.

.. list-table:: Key training settings
   :header-rows: 1
   :widths: 31 17 52

   * - YAML field
     - Default / source
     - Meaning
   * - ``model.weights_path``
     - Required
     - Initial main-model checkpoint.
   * - ``model.config_path``, ``model.tokenizer_path``
     - Selected profile
     - Architecture and tokenizer; match them to the checkpoint.
   * - ``recipe.params.q_config_path``
     - Selected profile
     - q architecture JSON or package directory. A fresh run initializes q randomly; no q weights are required.
   * - ``recipe.params.p_head_config_path``
     - Selected profile
     - p-head structure. It can be omitted when the checkpoint already provides p metadata.
   * - ``data.path``, ``run.output_dir``
     - User paths
     - Input token documents and training output directory.
   * - ``run.max_steps``, ``run.num_train_epochs``
     - ``1000``, ``1``
     - Training stops at the earlier limit. Set both for the intended training duration.
   * - ``data.max_seq_len``
     - ``4096``
     - Tokens per training row; must be divisible by ``recipe.params.block_size``.
   * - ``recipe.params.block_size``
     - ``32``
     - Block size for learned ordering and attention. Use the same block length for generation.
   * - ``optimization.lr``
     - ``2e-5``
     - Learning rate for the main model, p and q.
   * - ``optimization.micro_batch_size``, ``optimization.global_batch_size``
     - ``4``, ``1024``
     - Rows per GPU micro batch and total effective batch. Global batch must be divisible by micro batch times GPU count.
   * - ``recipe.params.loss_backward_divisor``
     - ``4096``
     - Fixed loss divisor before backward, independent of gradient accumulation.
   * - ``recipe.params.suffix_reconstruction_tau``
     - ``8``
     - Rank-decay temperature shared by ``L_rec`` and ``L_ord``.
   * - ``recipe.params.lambda_align``, ``recipe.params.lambda_recon``
     - ``1``, ``1``
     - Weights of ``L_ord`` and ``L_rec`` in ``L_LO + lambda_align * L_ord + lambda_recon * L_rec``.
   * - ``checkpoint.save_every_steps``
     - ``1000``
     - Checkpoint interval in optimizer steps.

To resume training, add
``--set checkpoint.resume_from=/path/to/checkpoints/global_step_N``.
The example enables ``checkpoint.save_hf_weights`` (disabled in the YAML) to
export the final model for generation at
``run.output_dir/checkpoints/global_step_N/hf_ckpt``.

Generation
----------

HF exports include the model, p head and SolDiff decoding code. They can be
loaded with PyTorch and Transformers 4.57.3 without installing dFactory or
VeOmni. q is not needed for generation.

.. code-block:: python

   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer

   path = "GSAI-ML/SolDiff-8B-Base"
   model = AutoModelForCausalLM.from_pretrained(
       path, trust_remote_code=True, dtype=torch.bfloat16,
       attn_implementation="sdpa",
   ).to("cuda")
   tokenizer = AutoTokenizer.from_pretrained(path)
   inputs = tokenizer("Explain how a rainbow forms.", return_tensors="pt").to(model.device)
   output = model.generate(
       **inputs, strategy="prior_static", max_new_tokens=256,
       block_length=32, steps_per_block=32, use_cache=True,
       eos_token_id=[151643, 151645],
   )
   print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))

``model.generate()`` runs SolDiff decoding and returns the prompt plus completion
for one unpadded input. Defaults are saved in ``config.soldiff_generation_config``;
keyword arguments override them. Use the parameter names below with underscores
instead of hyphens in Python.

Within dFactory, the command-line entry point also accepts checkpoints whose p
structure is provided by a matching ``p_head_config.json`` beside ``config.json``:

.. code-block:: bash

   python scripts/soldiff/generate.py \
     --model-path /path/to/hf_ckpt \
     --prompt "Explain how a rainbow forms." \
     --strategy prior_static \
     --max-new-tokens 256 \
     --block-length 32 \
     --steps-per-block 32 \
     --prior-temperature 0 \
     --temperature 0 \
     --use-cache \
     --mask-token-id 151669 \
     --eos-token-id 151643 151645 \
     --no-eos-early-stop \
     --seed 1234

For parallel decoding, replace the strategy argument with
``--strategy prior_confidence_dynamic --confidence-threshold 0.9``.
For an SFT checkpoint, add ``--chat-template --no-enable-thinking`` and
replace ``--no-eos-early-stop`` with ``--eos-early-stop``. Increase
``--max-new-tokens`` as needed, for example to ``8192``.
Use the tokenizer saved with that checkpoint, or provide
``--tokenizer-path /path/to/tokenizer``.

.. list-table:: Key generation settings
   :header-rows: 1
   :widths: 31 17 52

   * - CLI option
     - Default
     - Meaning
   * - ``--model-path``, ``--prompt``
     - Required
     - HF checkpoint directory and input text. The entry point processes one prompt at a time.
   * - ``--strategy``
     - ``prior_static``
     - ``prior_static`` fills a fixed quota of p-selected positions per step; ``prior_confidence_dynamic`` fills a high-confidence prefix in p order.
   * - ``--max-new-tokens``
     - ``1024``
     - Maximum completion length, excluding the prompt.
   * - ``--block-length``, ``--steps-per-block``
     - ``32``, ``32``
     - Working block length and denoising-step budget per block. The defaults give one position per step with ``prior_static``.
   * - ``--prior-temperature``
     - ``0``
     - Zero ranks positions deterministically by p; positive values sample position orders. Supported by both strategies.
   * - ``--confidence-threshold``
     - ``0.9``
     - Dynamic decoding accepts the consecutive p-ordered positions above this token-confidence threshold, with at least the scheduled quota.
   * - ``--use-cache``, ``--no-use-cache``
     - Enabled
     - Reuse completed-block KV caches or recompute the full prefix.
   * - ``--mask-token-id``
     - ``151669``
     - Mask token ID; match it to the checkpoint's tokenizer.
   * - ``--eos-token-id``
     - ``151643 151645``
     - Generation stop IDs; only the generated answer can trigger EOS stopping.
   * - ``--eos-early-stop``, ``--no-eos-early-stop``
     - Disabled
     - Stop inside a block once the generated prefix through EOS is complete; otherwise check at block boundaries.
   * - ``--chat-template``
     - Disabled
     - Format the prompt as a user chat message for an SFT/chat checkpoint.
   * - ``--enable-thinking``, ``--no-enable-thinking``
     - Unspecified
     - With ``--chat-template``, explicitly enable or disable its thinking format. Omit both to use the tokenizer default.
   * - ``--seed``
     - ``1234``
     - Random seed for position sampling.

Token selection always uses argmax. ``--temperature``, ``--top-k`` and
``--top-p`` adjust token probabilities used for confidence. ``--stop TEXT``
adds a text stop sequence and may be repeated. ``--output result.json`` saves
text, completion IDs and generation settings. The default execution uses one
GPU, bfloat16 and SDPA attention; see
``python scripts/soldiff/generate.py --help`` for all options.

The Qwen3 implementation and tokenizer build on public
`Transformers <https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/qwen3/modeling_qwen3.py>`_
and the `Qwen3-1.7B-Base <https://huggingface.co/Qwen/Qwen3-1.7B-Base>`_ and
`Qwen3-8B-Base <https://huggingface.co/Qwen/Qwen3-8B-Base>`_ assets under Apache 2.0.
