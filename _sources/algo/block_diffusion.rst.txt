Block Diffusion
==================

Last updated: 2025-10-28

`Arriola et al. (2025) <https://arxiv.org/abs/2503.09573>`_ introduce Block Diffusion, a novel class of language models designed to interpolate between discrete denoising diffusion and autoregressive paradigms.

Core Design Rationale
---------------------

The core design of Block Diffusion stems from a careful analysis of the trade-offs between the two dominant language modeling paradigms:

*   **Autoregressive Models (e.g., GPT series)**

    *   **Advantages**: Capable of generating sequences of any length, produce high-quality output, and can leverage KV-caching for efficient inference.
    *   **Disadvantages**: Generation is sequential (token-by-token), which makes it slow and inherently non-parallelizable.

*   **Discrete Diffusion Models**

    *   **Advantages**: Generation is highly parallelizable (e.g., denoising an entire sequence at once) and offers excellent controllability.
    *   **Disadvantages**: Typically limited to fixed-length generation, their quality (as measured by metrics like perplexity) often lags behind autoregressive models, and they cannot use KV-caching.

**Core Hypothesis**: By structuring generation to be autoregressive *between* blocks of tokens and parallel *within* each block, a hybrid model can combine the best of both worlds. This "autoregressive at the macro level, parallel diffusion at the micro level" approach allows the model to:

1.  Support **arbitrary-length generation** and **KV-caching**, like autoregressive models.
2.  Enable **parallel sampling** within blocks, boosting speed and quality, similar to diffusion models.


Methodology
-----------

Block Diffusion's effectiveness relies on its unique probabilistic modeling and an efficient training strategy.

**Noising Process**

The model is built upon the D3PM (Discrete Denoising Diffusion Probabilistic Models) framework. It defines a forward noising process where a clean data sequence :math:`\boldsymbol{x}^{0}` is progressively corrupted over a continuous time step :math:`t \in [0, 1]` to produce a noisier version :math:`\boldsymbol{x}^{t}`. This transition is defined as:

.. math::
    q(\boldsymbol{x}^{t}_{\ell}|\boldsymbol{x}^{s}_{\ell}) = \text{Cat}(\boldsymbol{x}^{t}_{\ell};\boldsymbol{Q}_{t}\boldsymbol{x}^{s}_{\ell})

Here, :math:`\boldsymbol{x}^{t}_{\ell}` is the state of the :math:`\ell`-th token at time :math:`t`. The transition matrix :math:`\boldsymbol{Q}_{t}\in \mathbb{R}^{V\times V}` (where V is the vocabulary size) models various transformations, such as random token replacement or masking.

**Block Diffusion Attention Mask**

A crucial component of Block Diffusion is its specialized attention mask, which dictates how tokens interact during both training and inference. The mask is designed to facilitate the "macro autoregressive, micro parallel" generation strategy.

.. figure:: ../../_static/images/Block_Diffusion_Attention_Mask.png
   :alt: Block Diffusion Attention Mask
   :align: center
   :width: 50%

   Block Diffusion Attention Mask (``block_size=4`` example)

This figure illustrates the attention mask for a `block_size=4` scenario. The mask combines different attention patterns to enable efficient block-wise generation while maintaining contextual awareness:

*   **Block-local attention**: Within each block, tokens can attend to all other tokens in that same block. This is essential for the parallel denoising steps.
*   **Causal attention to preceding blocks**: Each token can attend to all tokens in previously generated blocks. This maintains the autoregressive property at the block level, allowing the model to build coherent sequences.
*   **No future attention**: Tokens cannot attend to tokens in future blocks, upholding the causal nature of sequence generation.

**Decoding Pipeline**

The decoding (or sampling) process clearly illustrates the model's hybrid nature, proceeding one block at a time:

.. figure:: ../../_static/images/Block_Diffusion_Decoding.png
   :alt: Block Diffusion Decoding Pipeline
   :align: center
   :width: 80%

   The Block Diffusion Decoding Pipeline

1.  **Initialization**: The process starts with an initial prompt or a start-of-sequence ``[BOS]`` token.
2.  **Block Generation**: Using all previously generated text as a condition, a new block of tokens is generated in parallel via the reverse denoising process of the diffusion model.
3.  **KV-Caching**: The Key and Value states for the newly generated block are computed and cached.
4.  **Iteration**: The model uses the full sequence of generated text (including the newest block) as the condition for the next block generation, repeating steps 2 and 3 until an end-of-sequence ``[EOS]`` token is produced or the desired length is reached.


**Efficient Training: The Unified Attention Mask**

To train the model efficiently, Block Diffusion employs a clever unified attention mechanism that avoids multiple forward passes. The core idea is to concatenate the noised sequence :math:`\boldsymbol{x}_t` and the original clean sequence :math:`\boldsymbol{x}_0` into a single input. A specially designed attention mask then governs the flow of information within this combined sequence during a single forward pass.

.. figure:: ../../_static/images/Block_Diffusion_Training_Attention_Mask.png
   :alt: Block Diffusion Training Attention Mask
   :align: center
   :width: 80%

   The Block Diffusion Training Attention Mask (for ``block_size=2``)

This specialized mask consists of three distinct components that control the attention patterns:

*   **Block Diagonal Mask** :math:`\mathcal{M}_{BD}`:
    Allows each token in the noised block :math:`\boldsymbol{x}_t` to attend only to other tokens *within the same block*. This constitutes the intra-block self-attention for the denoising task.

*   **Offset Block Causal Mask** :math:`\mathcal{M}_{OBC}`:
    Allows tokens in a noised block in :math:`\boldsymbol{x}_t` to attend to all preceding *clean* blocks in :math:`\boldsymbol{x}_0`. This provides the essential conditional context required for denoising.

*   **Block Causal Mask** :math:`\mathcal{M}_{BC}`:
    Applies a standard causal mask to the clean sequence :math:`\boldsymbol{x}_0`, ensuring each token can only attend to itself and preceding tokens. This part is responsible for computing the KV-cache.

Helper Function to Create Block Diffusion Mask
-----------------------------------------------

The Python function below precisely implements the three-part attention mask logic described above. It is designed for integration with modern deep learning frameworks that support sparse attention (like PyTorch's FlexAttention) to achieve maximum training efficiency.

.. code:: python

    def block_diff_mask(b, h, q_idx, kv_idx, block_size, n):
        """
        Constructs the specialized block diffusion attention mask composed of
        three masks:
            - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
            - **Offset Block Causal Mask (M_OBC)**: Cross-attention for
                conditional context
            - **Block Causal Mask (M_BC)**: Attention to update x0
        Args:
            b, h: Batch and head indices (ignored for mask logic).
            q_idx, kv_idx: Query and Key indices.
            block_size: Defines the block structure.
            n: Sequence length of x_0 and x_t
            Returns:
            A boolean attention mask.
        """
        # Indicate whether token belongs to xt (0) or x0 (1)
        x0_flag_q = (q_idx >= n)
        x0_flag_kv = (kv_idx >= n)

        # Compute block indices
        block_q = torch.where(x0_flag_q == 1,
                              (q_idx - n) // block_size,
                              q_idx // block_size)
        block_kv = torch.where(x0_flag_kv == 1,
                               (kv_idx - n) // block_size,
                               kv_idx // block_size)

        # **1. Block Diagonal Mask (M_BD) **
        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

        # **2. Offset Block-Causal Mask (M_OBC) **
        offset_block_causal = (
            (block_q > block_kv)
            & (x0_flag_q == 0)
            & (x0_flag_kv == 1)
        )
        
        # **3. Block-Causal Mask (M_BC) **
        block_causal = (
            (block_q >= block_kv)
            & (x0_flag_q == 1)
            & (x0_flag_kv == 1)
        )
        
        # **4. Combine Masks **
        return block_diagonal | offset_block_causal | block_causal



.. code:: python

    # Attention computation using FlexAttention with our proposed custom mask.
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    from functools import partial

    # Define block-wise attention mask
    my_block_diff_mask = partial(block_diff_mask, seq_len=seq_len, block_size=block_size)

    # Generate optimized sparse block mask
    block_mask = create_block_mask(
        my_block_diff_mask, 
        None,         # batch_size dim
        None,         # num_heads dim
        seq_len*2,    # query length
        seq_len*2,    # key/value length
        device=device
    )

    # Compute attention using FlexAttention
    # Use no-cudagraphs to avoid an extra copy on small compile graphs.
    # Use max-autotune if compiling a larger model all at once.
    @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
    def single_pass_block_diff_attn(q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask)
