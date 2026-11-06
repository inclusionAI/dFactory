"""Generate a text completion with a saved SolDiff model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dfactory.recipes.soldiff.generation import SolDiffGenerationConfig, generate_soldiff, load_soldiff_model

DEFAULT_GENERATION = SolDiffGenerationConfig(
    strategy="prior_static",
    max_new_tokens=1024,
    block_length=32,
    steps_per_block=32,
    prior_temperature=0.0,
    confidence_threshold=0.9,
    temperature=0.0,
    use_cache=True,
    eos_early_stop=False,
    eos_token_id=(151643, 151645),
    mask_token_id=151669,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = DEFAULT_GENERATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF checkpoint directory containing the trained p head")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer-path", help="Tokenizer directory; defaults to the model directory")
    parser.add_argument("--strategy", choices=["prior_static", "prior_confidence_dynamic"], default=defaults.strategy)
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens)
    parser.add_argument("--block-length", type=int, default=defaults.block_length)
    parser.add_argument("--steps-per-block", type=int, default=defaults.steps_per_block)
    parser.add_argument("--prior-temperature", type=float, default=defaults.prior_temperature)
    parser.add_argument("--confidence-threshold", type=float, default=defaults.confidence_threshold)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--top-p", type=float, default=defaults.top_p)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=defaults.use_cache)
    parser.add_argument("--eos-early-stop", action=argparse.BooleanOptionalAction, default=defaults.eos_early_stop)
    parser.add_argument("--eos-token-id", type=int, nargs="+", default=list(defaults.eos_ids))
    parser.add_argument("--mask-token-id", type=int, default=defaults.mask_token_id)
    parser.add_argument("--attention", choices=["sdpa", "eager", "flex_attention"], default="sdpa")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stop", action="append", default=[], help="Text stop sequence; may be repeated")
    parser.add_argument("--chat-template", action="store_true", help="Format the prompt as a user chat message")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set the chat template's thinking mode; omitted uses the tokenizer default",
    )
    parser.add_argument(
        "--output", type=Path, help="Write one JSON result with text, token IDs and generation settings"
    )
    args = parser.parse_args(argv)
    if args.enable_thinking is not None and not args.chat_template:
        parser.error("--enable-thinking/--no-enable-thinking requires --chat-template")
    return args


def decode_completion(tokenizer, completion: torch.Tensor, stop_strings: list[str]) -> str:
    text = tokenizer.decode(completion.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for stop in stop_strings:
        if stop and stop in text:
            text = text.split(stop, 1)[0]
    encoded = tokenizer(text)["input_ids"]
    return tokenizer.decode(encoded, skip_special_tokens=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = SolDiffGenerationConfig(
        strategy=args.strategy,
        max_new_tokens=args.max_new_tokens,
        block_length=args.block_length,
        steps_per_block=args.steps_per_block,
        prior_temperature=args.prior_temperature,
        confidence_threshold=args.confidence_threshold,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=args.use_cache,
        eos_early_stop=args.eos_early_stop,
        eos_token_id=args.eos_token_id,
        mask_token_id=args.mask_token_id,
    )
    config.validate()
    model = load_soldiff_model(
        args.model_path, device=args.device, torch_dtype=getattr(torch, args.dtype), attn_implementation=args.attention
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    prompt = args.prompt
    if args.chat_template:
        if not tokenizer.chat_template:
            raise ValueError("the tokenizer has no chat template")
        chat_kwargs = {}
        if args.enable_thinking is not None:
            chat_kwargs["enable_thinking"] = args.enable_thinking
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)
    torch.manual_seed(args.seed)
    sequences = generate_soldiff(model, input_ids, config)
    completion = sequences[0, input_ids.shape[1] :].cpu()
    text = decode_completion(tokenizer, completion, args.stop)
    print(text, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "prompt": args.prompt,
                    "text": text,
                    "completion_ids": completion.tolist(),
                    "generation_config": asdict(config),
                    "attention": args.attention,
                    "dtype": args.dtype,
                    "seed": args.seed,
                    "stop_strings": args.stop,
                    "chat_template": args.chat_template,
                    "enable_thinking": args.enable_thinking,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
