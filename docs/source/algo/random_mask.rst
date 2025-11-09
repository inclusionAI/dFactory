.. meta::
   :description: An overview of the Discrete Diffusion Language Model (dLLM), including its masking process, training objectives, and inference methods.
   :keywords: dLLM, Masked Diffusion, Language Model, Denoising, Generative AI

Discrete Diffusion Model
************************

Last updated: 2025-10-28

The Discrete Language Diffusion Model (dLLM), specifically the masked diffusion variant, presents a non-autoregressive approach to generative language modeling. Unlike traditional models that predict tokens one by one, dLLM learns to restore a complete, clean sequence from a partially masked version of it.

Random Masking Process (Forward Process)
========================================

The core of the diffusion model is its forward process, which systematically introduces noise into the data. For discrete text data, this "noise" is the ``[MASK]`` token. We start with a clean, complete sequence :math:`Y^{0}`. After adding noise for a given "time" :math:`t`, it becomes a masked sequence :math:`Y^{t}`, where :math:`t \in [0, 1]`. When :math:`t=1`, :math:`Y^{1}` represents a sequence composed entirely of ``[MASK]`` tokens.

This process can be visualized as gradually "erasing" tokens from a clean sentence. The higher the value of :math:`t`, the more tokens are erased.

- A complete sequence with :math:`L` tokens is written as:

  .. math::
     Y^{0} = [y^{0}_{1}, \dots, y^{0}_{L}]

The conditional probability of this masking process is defined as a product of independent probabilities for each token:

.. math::
   q(Y^t|Y^0) = \prod_{i=1}^{L} q(y_i^t|y_i^0)

where the probability for an individual token :math:`y_i` to be masked at time :math:`t` is:

.. math::
   q(y_i^t|y_i^0) = 
   \begin{cases}
   1-t, & \text{if } y_i^t = y_i^0 \text{ (token remains unchanged)} \\
   t, & \text{if } y_i^t = \text{[MASK]} \text{ (token is masked)}
   \end{cases}

The model is trained on pairs of :math:`(Y^t, Y^0)` with randomly sampled :math:`t` values, learning to recover the original sequence from various levels of corruption.

Pre-training Objective
======================

The model's goal is to learn the reverse process: predicting the original data :math:`Y^0` given the masked version :math:`Y^t`. This is achieved by training a mask predictor, :math:`p_{\theta}`, which is typically a Transformer architecture without a causal mask, allowing it to see the entire input sequence bidirectionally.

In simple terms, the model is trained to be a universal "fill-in-the-blanks" expert. The pre-training loss function is defined as:

.. math::
   \mathcal{L}_{\text{Pretrain}}(\theta) = -\mathbb{E}_{Y^{0}\sim p_{\text{data}}} \mathbb{E}_{t\sim \mathcal{U}[0,1]} \mathbb{E}_{Y^{t}\sim q(Y^{t}|Y^{0})} \left[ \frac{1}{t} \sum^{L}_{i=1} \mathbb{I}[y^{t}_{i}=\text{[MASK]}] \log p_{\theta}(y^{0}_{i}|Y^{t}) \right]

- :math:`\mathbb{I}[y^{t}_{i}=\text{[MASK]}]` is an indicator function that ensures the loss is only calculated for the tokens that were actually masked.
- The :math:`\frac{1}{t}` term acts as a weighting factor. Intuitively, it can be seen as normalizing the loss by the expected number of masked tokens.
- This loss function, :math:`\mathcal{L}_{\text{Pretrain}}`, serves as an upper bound on the negative log-likelihood of the data, providing a principled objective for generative modeling.

Supervised Fine-tuning (SFT)
============================

To adapt the model for instruction-following, it is fine-tuned on a dataset of prompt-response pairs, denoted as :math:`(X, Y^0)`. A key distinction from pre-training is that the prompt :math:`X` is **always kept intact and is never masked**. Only the tokens in the response :math:`Y^0` are subject to the random masking process.

The SFT objective is to teach the model to predict the masked tokens in the response, conditioned on both the unmasked response tokens and the full prompt. The objective function is defined as:

.. math::
   \mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(X,Y^{0})\sim p_{\text{data}}} \mathbb{E}_{t\sim \mathcal{U}[0,1]} \mathbb{E}_{Y^{t}\sim q(Y^{t}|Y^{0})} \left[ \frac{1}{t} \sum^{|Y^{0}|}_{i=1} \mathbb{I}[y^{t}_{i}=\text{[MASK]}] \log p_{\theta}(y^{0}_{i}|X,Y^{t}) \right]

Inference: The Reverse Denoising Process
========================================

Generation is performed by simulating the reverse process. Starting from a fully masked sequence, the model iteratively predicts and refines the text over a series of discrete steps, moving from :math:`t=1` down to :math:`t=0`.

The process for a given prompt :math:`X` is as follows:

1. **Initialization**: Start with a fully masked response sequence :math:`Y^1` of a desired length.
2. **Iterative Denoising**: For a set number of steps (e.g., from :math:`t=1` down to :math:`s=0.9`, then to :math:`s=0.8`, etc.):

   (a) **Predict**: Feed the current masked sequence :math:`Y^t` (along with the prompt :math:`X`) into the model :math:`p_{\theta}` to get a prediction for the complete, clean sequence.
   (b) **Remask**: Based on the prediction, generate the sequence for the next, slightly less noisy step, :math:`Y^s`. This is crucial for aligning the sampling process with what the model learned during training. A common strategy is "low-confidence remasking," where tokens predicted with the lowest confidence are chosen to be re-masked to ``[MASK]`` for the next step.

3. **Final Output**: After the final step (at or near :math:`t=0`), the resulting sequence is the generated text.

This iterative process allows for a trade-off between generation quality and speed: more steps typically yield higher quality results but require more computation.

Perplexity (PPL) Calculation
============================

To evaluate the model, perplexity (PPL) can be calculated using a Monte Carlo estimation based on the SFT objective. It measures how well the model predicts masked tokens from a held-out test set.

.. math::
   PPL = \exp\left(-\mathbb{E}_{t,(X,Y^{0}),Y^{t}} \left[ \frac{1}{t} \sum^{L'}_{i=1} \mathbb{I}[y^{t}_{i}=\text{[MASK]}] \log p_{\theta}(y^{0}_{i}|X,Y^{t}) \right]\right)

where :math:`L'` corresponds to the dynamic sequence length of the response.
