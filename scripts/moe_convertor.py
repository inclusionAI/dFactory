import os
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from glob import glob
from typing import Generator

import torch
from safetensors.torch import safe_open
from tqdm import tqdm
from transformers import AutoConfig

from veomni.models import build_tokenizer, save_model_weights


@dataclass
class StateDictIterator:
    filepath: str

    def __iter__(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        if self.filepath.endswith(".safetensors"):
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

        else:
            state_dict = torch.load(self.filepath, map_location="cpu", weights_only=True, mmap=True)
            for key in state_dict.keys():
                yield key, state_dict[key]

def moe_merge(state_dict: dict[str, torch.Tensor], config) -> dict[str, torch.Tensor]:
    new_state_dict: dict[str, torch.Tensor] = dict()
    processed_keys: set[str] = set()        

    num_layers = config.num_hidden_layers
    num_experts = config.num_experts
    first_k_dense_replace = config.first_k_dense_replace

    print(f"Merging {num_layers} layers with {num_experts} experts each")
    proj_types = ["gate_proj", "up_proj", "down_proj"]

    for layer_id in range(first_k_dense_replace, num_layers):
        for proj_type in proj_types:
            expert_weights = []
            current_expert_keys = []                

            for expert_id in range(num_experts):
                expert_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{proj_type}.weight"
                assert expert_key in state_dict, f"Missing key: {expert_key}"
                expert_weights.append(state_dict[expert_key])
                current_expert_keys.append(expert_key)

            assert len(expert_weights) == num_experts
            merged_weight = torch.stack(expert_weights, dim=0)
            new_key = f"model.layers.{layer_id}.mlp.experts.{proj_type}"
            new_state_dict[new_key] = merged_weight
            processed_keys.update(current_expert_keys)
            for key in current_expert_keys:
                del state_dict[key]
            print(f"✓ Layer {layer_id}.{proj_type}: {expert_weights[0].shape} -> {merged_weight.shape}")                
            del expert_weights


    for key, tensor in state_dict.items():
        if key not in processed_keys:
            new_state_dict[key] = tensor
    return new_state_dict

def split_moe_experts(
    merged_state_dict: dict[str, torch.Tensor], config
) -> dict[str, torch.Tensor]:
    split_state_dict: dict[str, torch.Tensor] = dict()

    num_experts = getattr(config, "num_experts", None)
    if num_experts is None:
        raise ValueError(
            "Could not find the 'num_experts' attribute in the configuration. "
            "Please ensure the correct model configuration is passed."
        )

    proj_types = ["gate_proj", "up_proj", "down_proj"]

    merged_key_pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(" + "|".join(proj_types) + r")$"
    )

    for key, merged_tensor in merged_state_dict.items():
        match = merged_key_pattern.match(key)

        if match:
            layer_id = match.group(1)
            proj_type = match.group(2)

            if not (merged_tensor.dim() > 1 and merged_tensor.shape[0] == num_experts):
                raise ValueError(
                    f"Tensor '{key}' has an unexpected shape {merged_tensor.shape}. "
                    f"Its first dimension should be equal to the number of experts ({num_experts})."
                )

            for expert_id in range(num_experts):
                expert_tensor = merged_tensor[expert_id]
                original_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{proj_type}.weight"
                split_state_dict[original_key] = expert_tensor

            print(f"Unmerged {key} -> {num_experts} individual expert weights")
        else:
            split_state_dict[key] = merged_tensor

    return split_state_dict


def main(input_path, output_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)

    config = AutoConfig.from_pretrained(input_path, trust_remote_code=True)
    tokenizer = build_tokenizer(input_path)

    safetensor_files = list(glob(os.path.join(input_path, "*.safetensors")))
    safetensor_files.sort()
    state_dict_iterators = [StateDictIterator(shard_file) for shard_file in safetensor_files]

    state_dict = {}
    for state_dict_iterator in tqdm(state_dict_iterators, desc="Loading checkpoint shards"):
        for name, tensor in state_dict_iterator:
            state_dict[name] = tensor.cpu()

    if args.mode == "merge":
        new_state_dict = moe_merge(state_dict, config)
    elif args.mode == "split":
        new_state_dict = split_moe_experts(state_dict, config)
    else:
        raise ValueError("unsupport mode")
    
    state_dict.clear()
    model_assets = [config, tokenizer]
    save_model_weights(output_path, new_state_dict, model_assets=model_assets)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-path", type=str, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument("-m", "--mode", type=str, default="merge", choices=["merge", "split"])
    args = parser.parse_args()
    main(args.input_path, args.output_path)
