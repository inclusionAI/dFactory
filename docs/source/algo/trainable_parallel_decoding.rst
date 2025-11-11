Trainable Parallel Decoding
============================

Last updated: 2025-11-10

Trainable Parallel Decoding is a novel approach to accelerate Diffusion Large Language Models (DLLMs) by learning to decode multiple tokens simultaneously during training, thereby reducing inference latency while maintaining generation quality.

Overview
--------

Traditional DLLMs suffer from high inference latency due to their iterative, multi-step sampling process. Trainable Parallel Decoding addresses this limitation by introducing a second-stage fine-tuning paradigm that teaches the model to predict multiple future tokens in a single forward pass. This approach transforms the sequential generation process into a more parallelizable one, significantly reducing the number of required sampling steps.

The framework currently supports two complementary techniques:

1. **Path Distillation (Trajectory Compression)**: Learning to jump between non-consecutive states in optimal generation trajectories
2. **DPARALLEL**: Entropy-based loss regularization to accelerate parallel decoding learning

Path Distillation (Trajectory Compression)
------------------------------------------

Path Distillation is motivated by the key observation from `Song et al., 2025 <http://arxiv.org/abs/2508.02193>`_ that training on high-quality generation paths can significantly improve model efficiency. The method consists of two main stages:

High-Quality Trajectory Distillation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first stage involves creating a dataset of "golden" trajectories through the following process:

1. **Trajectory Generation**: Use a pre-trained DLLM to sample generation paths on a domain-specific dataset (e.g., 200,000 math problems)
2. **Quality Filtering**: Apply an external verifier to filter trajectories that produce correct outputs
3. **Dataset Construction**: Retain only high-quality trajectories that pass verification

Mathematically, given a trajectory :math:`\tau = (s_N, s_{N-1}, \dots, s_0)` representing states from fully masked to final output, we filter:

.. math::
    \mathcal{T}_{\text{gold}} = \{ \tau \in \mathcal{T} \,|\, V(s_0^{\tau}) = \text{True} \}

where :math:`V(\cdot)` is the external verifier function.

Compressed Transition Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second stage fine-tunes the model to predict multi-step transitions instead of single-step ones:

1. **Training Instance Construction**: For each trajectory, randomly sample timestamps :math:`i` and :math:`j` where :math:`N \ge i > j \ge 0`
2. **Target Identification**: The model learns to predict tokens that are [MASK] in :math:`s_i` but revealed in :math:`s_j`
3. **Loss Optimization**: Minimize the negative log-likelihood of compressed transitions

The fine-tuning objective is:

.. math::
    \mathcal{L}_{\text{compress}}(\theta) = - \mathbb{E}_{\tau \in \mathcal{T}_{\text{gold}}, \, i,j \sim U(\tau)} \left[ \sum_{k \in \Delta_{i \to j}} \log p_\theta(x_k = s_j[k] \,|\, s_i) \right]

where :math:`\Delta_{i \to j} = M_i \setminus M_j` represents the indices of tokens to be predicted.

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

The data preparation process involves:

1. **Offline Dataset Creation**: Generate and filter trajectories offline
2. **Data Format**: Prepare input_ids, noisy_input_ids, and labels for training
3. **Training Configuration**: Use standard SFT training with the compressed transition objective

The training data format should include:

- ``input_ids``: The starting state :math:`s_i` with appropriate masking
- ``noisy_input_ids``: The noised version of :math:`s_i` 
- ``labels``: The target tokens to predict (tokens in :math:`s_j` that differ from :math:`s_i`)

DPARALLEL: Learnable Parallel Decoding
--------------------------------------

`Chen et al., 2025 <http://arxiv.org/abs/2509.26488>`_ introduce dParallel, a novel approach that incorporates an entropy-based regularization term into the training loss to encourage parallel decoding capabilities.

Methodology
~~~~~~~~~~~

The key insight is that by adding a confidence-based loss term during supervised fine-tuning, we can guide the model toward making confident, parallel predictions. This is achieved through:

1. **Entropy Regularization**: Add a loss term based on the entropy of the model's predictions
2. **Confidence Scoring**: Use prediction confidence as a signal for parallel decoding quality
3. **Loss Balancing**: Combine the standard cross-entropy loss with the confidence-based term

Configuration
~~~~~~~~~~~~~

To enable DPARALLEL, use the following training configuration:

.. code:: bash

    sh train.sh tasks/train_llada2_bd_with_dparallel.py configs/sft/llada2_mini_bd_sft.yaml --train.confidence_beta {confidence_beta}

Where:

- ``confidence_beta`` controls the strength of the entropy regularization (recommended value: 2.0)
- Higher values encourage more aggressive parallel decoding
- The parameter balances between generation quality and speed-up

Training Process
^^^^^^^^^^^^^^^^

The DPARALLEL training process:

1. **Standard SFT Setup**: Begin with standard supervised fine-tuning
2. **Loss Modification**: Add the confidence-based regularization term
3. **Hyperparameter Tuning**: Adjust ``confidence_beta`` based on desired speed-quality trade-off
4. **Evaluation**: Monitor both generation quality and inference speed metrics