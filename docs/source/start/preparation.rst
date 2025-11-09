.. _preparation:

===========
Preparation
===========

Last updated: 2025-11-08


With the environment and dependencies installed, the final step is to prepare the necessary assets for training. This guide covers the two main prerequisites: downloading and converting the model weights, and formatting the training dataset.

Download and Merge Model Weights
================================

Our training scripts require model weights in a “merged-expert” format
for optimal performance. Before starting, you must download the standard
weights and convert them.

Step 1: Download Original Model
-------------------------------

We provide a helper script to download the weights from Hugging Face:

.. code-block:: bash

   # Choose a destination for the original model files
   python ./scripts/download_hf_model.py \
     --repo_id inclusionAI/LLaDA2.0-mini-preview \
     --local_dir /path/to/separate_expert_model

Step 2: Convert to Merged Format
--------------------------------

Run the following script to create the merged checkpoint required for training:

.. code-block:: bash

   # Use the path from the previous step as the source
   python scripts/moe_convertor.py \
     --input-path /path/to/separate_expert_model \
     --output-path /path/to/save/merged_model \
     --mode merge

The directory ``/path/to/save/merged_model`` is what you will use for
the training script. For more details, see `MoE Expert Merging and
Splitting Utilities <#moe-expert-merging-and-splitting-utilities>`__

Prepare Training Data
=====================

This tutorial uses the ``openai/gsm8k`` dataset and demonstrates how to convert it into the conversational format.

Provided Script
---------------

We provide an example script ``./scripts/build_gsm8k_dataset.py`` for this purpose. You can adapt this script or write your own to process other datasets.

The script converts the "question" and "answer" fields into a conversational messages field. The processed dataset is saved to the ``./gsm8k_datasets/`` directory, split into:

- ``train.jsonl`` - Training data
- ``test.jsonl`` - Evaluation data

Run the script:

.. code-block:: bash

   python ./scripts/build_gsm8k_dataset.py
