"""Hugging Face generation interface for exported SolDiff models."""

from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass

import torch
from transformers import GenerationConfig
from transformers.generation import GenerationMixin

from .generation import SolDiffGenerationConfig, generate_soldiff


def causallm_loss_function(hidden_states, weight, labels):
    """Project inference hidden states without a training-runtime dependency."""
    if labels is not None:
        raise ValueError("This model package supports inference; use the SolDiff training recipe for training losses")
    return None, torch.nn.functional.linear(hidden_states, weight)


class SolDiffGenerationMixin(GenerationMixin):
    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        generation_config=None,
        *,
        input_ids=None,
        attention_mask=None,
        generator=None,
        **kwargs,
    ):
        """Decode one unpadded prompt using SolDiff, returning prompt and completion.

        Keyword arguments override the saved ``config.soldiff_generation_config``.
        Both ``prior_static`` and ``prior_confidence_dynamic`` use greedy token
        choices; ``prior_temperature`` controls position sampling only.
        """
        if inputs is not None and input_ids is not None:
            raise ValueError("Pass either inputs or input_ids, not both")
        input_ids = inputs if input_ids is None else input_ids
        if input_ids is None:
            raise ValueError("input_ids is required")
        if attention_mask is not None and (
            attention_mask.shape != input_ids.shape or not torch.all(attention_mask == 1).item()
        ):
            raise ValueError("SolDiff generate accepts one unpadded prompt; attention_mask must contain only ones")

        names = {field.name for field in fields(SolDiffGenerationConfig)}
        values = dict(getattr(self.config, "soldiff_generation_config", {}) or {})
        if generation_config is not None:
            if isinstance(generation_config, GenerationConfig):
                if generation_config.do_sample or generation_config.num_beams != 1:
                    raise ValueError("SolDiff uses greedy token choices and does not support beams or token sampling")
                if generation_config.num_return_sequences != 1 or generation_config.return_dict_in_generate:
                    raise ValueError("SolDiff generate returns one token sequence tensor")
                metadata = {
                    "transformers_version",
                    "_from_model_config",
                    "bos_token_id",
                    "pad_token_id",
                    "decoder_start_token_id",
                    "do_sample",
                    "num_beams",
                    "num_return_sequences",
                    "return_dict_in_generate",
                }
                unsupported = generation_config.to_diff_dict().keys() - names - metadata
                if unsupported:
                    raise ValueError(f"Unsupported SolDiff generation settings: {', '.join(sorted(unsupported))}")
                provided = {
                    name: value
                    for name, value in generation_config.to_dict().items()
                    if name in names and value is not None
                }
            elif is_dataclass(generation_config) and not isinstance(generation_config, type):
                provided = asdict(generation_config)
            elif isinstance(generation_config, Mapping):
                provided = dict(generation_config)
            else:
                raise TypeError("generation_config must be a SolDiff config, GenerationConfig, or mapping")
            values.update(provided)
        values.update(kwargs)
        unknown = values.keys() - names
        if unknown:
            raise ValueError(f"Unsupported SolDiff generation settings: {', '.join(sorted(unknown))}")
        return generate_soldiff(self, input_ids, SolDiffGenerationConfig(**values), generator=generator)
