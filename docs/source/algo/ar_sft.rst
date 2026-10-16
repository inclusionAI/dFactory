==================
Autoregressive SFT
==================

The ``ar_sft`` recipe trains a causal language-model objective without Block
Diffusion noise or noisy/clean slot duplication. It labels every assistant span
in conversation data and computes shifted next-token cross entropy only where
the shifted label is trainable.

When packing is enabled, documents are concatenated once. Position IDs restart
at zero for every document, and ``ARSFTTrainer`` builds a document-isolated causal
attention mask so tokens cannot attend across packing boundaries. Padding does
not participate in attention or loss.

Start from the public example:

.. code-block:: bash

   dfactory config validate --config configs/recipes/ar_sft/llada2_mini.yaml
   torchrun --nproc-per-node=8 -m dfactory train \
     --config configs/recipes/ar_sft/llada2_mini.yaml \
     --set distributed.dp=8

Behavior and scope
------------------

``ARSFTTrainer`` owns direct inputs, document-causal attention, shifted loss, and the
absence of diffusion noise. Its data pipeline owns conversation preparation,
single-copy packing, per-document position reset, one-based document IDs, and
zero-valued padding IDs.

The first public migration accepts conversation data only, matching the current
dFactory data contract. The former conversation branch reused a diffusion
preprocessor that randomly restricted labels even though the AR trainer discarded
its noised inputs; the public recipe instead keeps every assistant label, which
matches the trainer's stated autoregressive contract. Token-record inputs remain
deferred.

The trainer always constructs an explicit causal mask because dFactory's owned LLaDA2
attention does not infer causality from a missing mask. It emits a sparse
``BlockMask`` for FlexAttention and a dense additive mask for eager or SDPA
attention. Optional SWA and automatic Liger-kernel selection are not carried
because neither has a public recipe/model configuration contract yet.
Gradient accumulation and backward contexts are delegated to the public
``StepContext`` and ``BaseTrainer``.
