AR2BD continued pretraining
===========================

``ar2bd`` continues training an autoregressive checkpoint with the paired
Block Diffusion objective. It consumes pre-tokenized documents, samples noise
at training time, and keeps packed documents isolated in attention.

Data contract
-------------

Set ``data.format`` to ``token_records``. Each source record represents one
document and contains a one-dimensional integer sequence under
``data.text_key`` (``input_ids`` in the example). The transform truncates the
document, aligns its capacity to ``block_size``, and records its valid length.
Packing combines documents without merging their attention domains. Padding is
excluded from attention, noise, and loss. Noise is never stored in the dataset.

Algorithm
---------

For each document ``AR2BDTrainer`` constructs a noisy slot and a clean slot. One
runtime noise rate is sampled for the original document from ``noise_range``
and shared by all of that document's logical blocks. Noisy blocks see
themselves and preceding clean blocks; clean blocks use block-causal attention.
Packed documents cannot attend to one another.

With ``same_token_labels: true``, a masked token is predicted at its own
position and loss is normalized per batch row before rows are averaged. With
``same_token_labels: false``, logits are shifted by one position before the
masked-token objective is evaluated.

Public scope
------------

The recipe owns token-record data preparation; ``AR2BDTrainer`` owns runtime noise sampling, block/document
attention, and the same-position or shifted objective. It deliberately excludes
conversation SFT, complementary-mask datasets, precomputed noise, and the old
``block_diffusion_mode`` switch. Runtime, optimizer, and checkpoint ownership
remain with ``BaseTrainer``.

See ``configs/recipes/ar2bd/llada2_mini.yaml`` for a complete configuration.
