<div align="center">
  <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*9yAZT61NXRkAAAAARBAAAAgAemJ7AQ/original" width="40%" alt="dFactory" />
</div>

<h4 align="center">

[![License: MIT](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![HuggingFace: Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview)
[![Static Badge](https://img.shields.io/badge/Tutorial-Get_Start-red)](https://inclusionai.github.io/dFactory/)
[![Static Badge](https://img.shields.io/badge/DeepWiki-Explore-green)](https://deepwiki.com/inclusionAI/dFactory)
<!-- [![arXiv][arxiv-image]][arxiv-url] -->

</h4>


# dFactory: Easy and Efficient dLLM Fine-Tuning

## Features

- **Various models:** LLaDA2.0-mini (16B), LLaDA2.0-flash (100B)
- **Integrated methods:** (Continous) supervised-finetuning (block-diffusion, full attention), etc.


## Supported Models

| Model ID | Description | Size | Config Path | Hugging Face Link |
| --- | --- | --- | --- | --- |
| `inclusionAI/LLaDA2.0-mini-preview` | Instruction-tuned model, ready for downstream applications. | 16B | `configs/model_configs/llada2_mini/` | [🤗 Model Card](https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview) |
| `inclusionAI/LLaDA2.0-mini` | Instruction-tuned model, ready for downstream applications. | 16B | `configs/model_configs/llada2_mini/` | [🤗 Model Card](https://huggingface.co/inclusionAI/LLaDA2.0-mini) |
| `inclusionAI/LLaDA2.0-flash-preview` | Instruction-tuned model, ready for downstream applications. | 100B | `configs/model_configs/llada2_flash/` | [🤗 Model Card](https://huggingface.co/inclusionAI/LLaDA2.0-flash-preview) |
| `inclusionAI/LLaDA2.0-flash` | Instruction-tuned model, ready for downstream applications. | 100B | `configs/model_configs/llada2_flash/` | [🤗 Model Card](https://huggingface.co/inclusionAI/LLaDA2.0-flash) |

## TODO

We are actively working on enhancing the project with new features and improvements. Our roadmap for the near future includes:

- [☑️] **Comprehensive Documentation**: A full documentation site is underway, which will feature in-depth tutorials, API references, and best practices.
- [☑️] **Trainable Parallel Decoding**: Integration of support for trainable parallel decoding to enable more advanced use cases.

Stay tuned for these updates!

## Getting Started

### 0. Environment Setup

#### Option A: Use uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/inclusionAI/dFactory.git --recursive
cd dFactory/VeOmni

# Install dependencies
uv sync --extra gpu

# Activate environment
source .venv/bin/activate

# Back to our workdir
cd ..
```

#### Option B: Use pip

```bash
git clone https://github.com/inclusionAI/dFactory.git --recursive
cd dFactory
pip install -e VeOmni/
```

### 1. Download and Merge Model Weights

Our training scripts require model weights in a "merged-expert" format for optimal performance. Before starting, you must download the standard weights and convert them.

**1. Download the original model:** We provide a helper script to download the weights from the Hugging Face Hub.

```bash
# Choose a destination for the original model files
python ./scripts/download_hf_model.py \
  --repo_id inclusionAI/LLaDA2.0-mini-preview \
  --local_dir /path/to/separate_expert_model
```

**2. Convert to the merged format:** Run the following script to create the merged checkpoint required for training.

```bash
# Use the path from the previous step as the source
python scripts/moe_convertor.py \
  --input-path /path/to/separate_expert_model \
  --output-path /path/to/save/merged_model \
  --mode merge
```

The directory `/path/to/save/merged_model` is what you will use for the training script. For more details, see [MoE Expert Merging and Splitting Utilities](#moe-expert-merging-and-splitting-utilities)

### 2. Prepare Training Data

Before training, the dataset must be prepared. This tutorial uses the `openai/gsm8k` dataset and demonstrates how to convert it into the conversational format.

We provide an example script, `./scripts/build_gsm8k_dataset.py`, for this purpose. You can adapt this script or write your own to process other datasets.

Running the following command executes the script. It converts the "question" and "answer" fields into a conversational messages field. The processed dataset is then saved to the ./gsm8k_datasets/ directory, split into two separate files: `train.jsonl` for training and `test.jsonl` for evaluation.

```bash
python ./scripts/build_gsm8k_dataset.py
```

### 3. Modify Training Configs

Edit `configs/sft/llada2_mini_bd_sft.yaml`:
```yaml
model:
  model_path: "/your/model/path"
data:
  train_path: "/your/data/path"
train:
  output_dir: "/your/output/path"
```

### 4. Run Training

With all preparations complete, you can now start the fine-tuning process with a single command:

```bash
PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH sh train.sh tasks/train_llada2_bd.py configs/sft/llada2_mini_bd_sft.yaml
```

### 5. Interacting with the Fine-Tuned Model

To interact with your fine-tuned model, you must complete two main steps: converting the checkpoint and copying the modeling file.

**Step 1: Convert the Checkpoint**

First, you need to convert the checkpoint from the merged format used during training back to the standard Mixture-of-Experts (MoE) structure.

> **Important: Finding the Correct Input Path**
>
> The --input-path for the conversion script is the path to the saved Hugging Face checkpoint, not the root output directory you specified during training. The checkpoint is typically located in a subdirectory like:
>
> TRAIN_OUTPUT_DIR/checkpoints/global_step_XXX/hf_ckpt/

Run the following command to perform the conversion:

```bash
python scripts/moe_convertor.py \
  --input-path /path/to/merged_model \
  --output-path /path/to/save/separate_expert_model \
  --mode split
```

**Step 2: Copy the Modeling File**

After the conversion, a final manual step is required. You must copy the model's architecture file (e.g., `modeling_llada2_moe.py`) into the newly created separate_expert_model directory.

This file must come from the directory of your original base model — the one you started with before any merge or training operations. The training and conversion processes only update the model weights, not the architecture file, which is why the original version is needed.

```bash
# Example: Copying from the initial, pre-merge model directory
cp /path/to/original_base_model/modeling_llada2_moe.py /path/to/save/separate_expert_model/
```

With the model converted and the modeling file in place, you are now ready to chat! Follow the instructions on the [official model card](https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview#%F0%9F%A4%97-hugging-face-transformers) to start a conversation with your model.

## MoE Expert Merging and Splitting Utilities

We provide a utility script, `./scripts/moe_convertor.py`, to convert MoE model weights between two formats:

1. Separate-Expert Format: The default format used by frameworks like Hugging Face transformers, where each expert's weights are stored as individual tensors.
2. Merged-Expert Format: A consolidated format where weights for all experts in a layer are stacked into a single, higher-dimensional tensor.

### Merging Experts

Convert a model with separate expert weights into the consolidated "merged" format. By merging expert weights, we can leverage highly efficient batched matrix multiplication on GPUs, significantly speeding up computation.

**How it Works:**

The script iterates through each MoE layer and stacks the weights of all experts (e.g., gate_proj, up_proj, down_proj) into a single tensor.

- Before Merging (Separate Experts):

  ```
  model.layers.15.mlp.experts.0.gate_proj.weight  (shape: [4096, 14336])
  model.layers.15.mlp.experts.1.gate_proj.weight  (shape: [4096, 14336])
  ... (and so on for all 8 experts)
  ```

- After Merging (Merged Experts):

  ```
  model.layers.15.mlp.experts.gate_proj.weight  (shape: [8, 4096, 14336])
  ```

**Usage:**

```bash
python scripts/moe_convertor.py \
  --input-path /path/to/separate_expert_model \
  --output-path /path/to/save/merged_model \
  --mode merge
```

### Splitting Experts

This process performs the reverse operation: it takes a model in the "merged" format and splits the expert weights back into separate tensors for each expert. This conversion is useful for:

- Fine-tuning: Converting a merged model back to the standard format for fine-tuning with frameworks like Hugging Face transformers.
- Analysis: Inspecting or modifying the weights of individual experts.
- Compatibility: Ensuring the model can be loaded by tools that expect separate expert weights.

**How it Works:**

The script identifies the merged weight tensors and slices them along the expert dimension to create individual weight files for each expert.

- Before Splitting (Merged Experts):

  ```
  model.layers.15.mlp.experts.gate_proj.weight  (shape: [8, 4096, 14336])
  ```

- After Splitting (Separate Experts):

  ```
  model.layers.15.mlp.experts.0.gate_proj.weight  (shape: [4096, 14336])
  model.layers.15.mlp.experts.1.gate_proj.weight  (shape: [4096, 14336])
  ... (and so on for all 8 experts)
  ```

**Usage:**

```bash
python scripts/moe_convertor.py \
  --input-path /path/to/merged_model \
  --output-path /path/to/save/separate_expert_model \
  --mode split
```

## License

This project is licensed under the Apache 2.0 license - see the LICENSE file for details.
