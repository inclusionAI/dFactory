===============
Block Diffusion
===============

Block Diffusion is autoregressive between blocks and denoising-based within a
block. dFactory keeps complete training algorithms in concrete trainers. The
``dfactory.components.block_diffusion`` package contains only primitives shared
by multiple recipes.

Training layout
===============

Each logical sequence is represented by a noised and a clean slot. Packed
sequences are interleaved so their boundaries remain explicit:

.. code-block:: text

   [noisy_1, clean_1, noisy_2, clean_2, ...]

Both slots are block-aligned. Position IDs restart for every slot, and padding
is excluded from attention and loss.

Attention semantics
===================

For each packed pair:

* a noisy block sees itself;
* a noisy block sees preceding clean blocks from the same pair;
* a clean block never sees noisy blocks;
* a clean block sees itself and preceding clean blocks;
* distinct packed pairs cannot attend to one another.

The implementation can construct a dense boolean oracle for tests and
SDPA/eager execution, or a PyTorch ``BlockMask`` for FlexAttention. The sparse
mask is built on the training device, after collation.

Sliding-window capability
=========================

LLaDA2 can select a different Block Diffusion attention mask for each layer
without using a second model implementation. Trainers own the attention policy
and build each distinct window once. ``window_blocks`` counts preceding
blocks; the query's current block is also visible. ``sink_tokens`` keeps the
first clean-slot tokens of each packed sequence visible outside the window.

Enable one fixed window on every layer with:

.. code-block:: yaml

   recipe:
     name: sft
     params:
       block_size: 32
       sliding_window:
         window_blocks: 16
         sink_tokens: 4

For per-layer windows, use one list entry per LLaDA2 layer and set unrestricted
layers to ``null``. Optional ``activation_order`` and ``activation_interval``
progressively enable layers from step zero without mutating model state. See
``configs/recipes/sft/llada2_mini_swa.yaml`` for a complete run configuration.

Noise and objective
===================

Noise is sampled during every training step, not cached in the dataset. The
trainer chooses the sampling group: SFT samples per logical block, while AR2BD
shares one rate across all blocks from the same original document. Only
trainable tokens in the noisy slot are eligible. Multi-turn boundaries are
protected so a user span following an assistant span in the same block cannot
leak through block-local attention.

The default SFT objective predicts the original token at masked positions.
``same_token_labels: false`` enables shifted next-token compatibility mode.
An empty loss mask returns a differentiable zero.

Ownership boundary
==================

The component package does not select recipes, call a model, run backward,
update an optimizer, or save checkpoints. ``SFTTrainer`` owns the algorithm,
while ``BaseTrainer`` owns the shared lifecycle.
