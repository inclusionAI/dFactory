import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.utils.checkpoint import set_checkpoint_debug_enabled
from tqdm import trange

import models.llada2_moe  # noqa: F401 - registers LLaDA2 MoE with the VeOmni loader.
from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments, parse_args, save_args
from veomni.checkpoint import build_checkpointer
from veomni.data import build_dataloader, build_dataset
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    is_nccl_backend,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.utils.save_safetensor_utils import save_hf_safetensor

try:
    from dataset import build_local_dataset
    from dataset.data_transform import process_mdm_sft_example, process_mdm_tokenized_example
except ImportError:
    from tasks.dataset import build_local_dataset
    from tasks.dataset.data_transform import process_mdm_sft_example, process_mdm_tokenized_example


logger = helper.create_logger(__name__)


@dataclass
class LLaDA2ModelArguments(ModelArguments):
    attn_implementation: Optional[Literal["eager", "sdpa", "flex_attention"]] = field(
        default=None,
        metadata={"help": "Deprecated. Use model.ops_implementation.attn_implementation."},
    )
    moe_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Deprecated. Use model.ops_implementation.moe_implementation."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.attn_implementation is not None:
            self.ops_implementation.attn_implementation = self.attn_implementation
        if self.moe_implementation is not None:
            self.ops_implementation.moe_implementation = self.moe_implementation


@dataclass
class LLaDA2DataArguments(DataArguments):
    data_type: Literal["conversation", "tokenid"] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )
    datasets_type: Literal["mapping", "iterable", "local"] = field(
        default="mapping",
        metadata={"help": "Type of the datasets."},
    )
    text_keys: Optional[str] = field(
        default=None,
        metadata={"help": "Key to get text or token ids from the training data."},
    )
    noise_range_low: float = field(
        default=0.3,
        metadata={"help": "Lower bound of random mask noise ratio."},
    )
    noise_range_high: float = field(
        default=0.8,
        metadata={"help": "Upper bound of random mask noise ratio."},
    )
    mask_token_id: int = field(
        default=156895,
        metadata={"help": "LLaDA2 mask token id."},
    )

    def __post_init__(self):
        if self.text_keys is None:
            self.text_keys = "input_ids" if self.data_type == "tokenid" else "messages"
        super().__post_init__()
        if self.noise_range_low > self.noise_range_high:
            raise ValueError(
                f"noise_range_low ({self.noise_range_low}) cannot be greater than "
                f"noise_range_high ({self.noise_range_high})."
            )
        if not (0.0 <= self.noise_range_low <= 1.0):
            raise ValueError(f"noise_range_low must be between 0.0 and 1.0, but got {self.noise_range_low}.")
        if not (0.0 <= self.noise_range_high <= 1.0):
            raise ValueError(f"noise_range_high must be between 0.0 and 1.0, but got {self.noise_range_high}.")


@dataclass
class LLaDA2TrainingArguments(TrainingArguments):
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW optimizer beta1."},
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "AdamW optimizer beta2."},
    )
    confidence_beta: float = field(
        default=0.0,
        metadata={"help": "Weight for the confidence loss entropy of correct predictions. Set to 0 to disable."},
    )
    block_diffusion_mode: bool = field(
        default=False,
        metadata={"help": "Train MDM in block diffusion mode."},
    )
    block_size: int = field(
        default=32,
        metadata={"help": "Block size for block diffusion."},
    )
    same_token_labels: bool = field(
        default=False,
        metadata={"help": "Use same token labels instead of next-token shifted labels."},
    )


@dataclass
class LLaDA2Arguments(VeOmniArguments):
    model: LLaDA2ModelArguments = field(default_factory=LLaDA2ModelArguments)
    data: LLaDA2DataArguments = field(default_factory=LLaDA2DataArguments)
    train: LLaDA2TrainingArguments = field(default_factory=LLaDA2TrainingArguments)


def block_diffusion_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    del b, h
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)

    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
    return block_diagonal | offset_block_causal | block_causal


def compute_confidence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(logits.device)
    valid_mask = labels != -100
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device)

    predicted_tokens = torch.argmax(logits, dim=-1)
    correct_mask = (predicted_tokens == labels) & valid_mask
    if correct_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy_per_token = -torch.sum(probs * log_probs, dim=-1)
    return entropy_per_token[correct_mask].mean()


def _build_transform(args: LLaDA2Arguments, tokenizer):
    noise_range = (args.data.noise_range_low, args.data.noise_range_high)
    if args.data.data_type == "conversation":
        if not tokenizer.chat_template:
            raise ValueError("No chat template found in the tokenizer.")
        return partial(
            process_mdm_sft_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=noise_range,
            mask_token_id=args.data.mask_token_id,
        )
    if args.data.data_type == "tokenid":
        return partial(
            process_mdm_tokenized_example,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=noise_range,
            mask_token_id=args.data.mask_token_id,
        )
    raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")


def _build_train_dataset(args: LLaDA2Arguments, transform):
    if args.data.datasets_type == "local":
        return build_local_dataset(args.data.train_path, transform=transform, seed=args.train.seed)

    return build_dataset(
        dataset_name=args.data.dataset_name,
        transform=transform,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        **asdict(args.data),
    )


def _build_block_diffusion_mask(args: LLaDA2Arguments) -> Optional[torch.Tensor]:
    if not args.train.block_diffusion_mode:
        return None

    full_len = args.data.max_seq_len * 2
    mask_flag = block_diffusion_mask(
        b=None,
        h=None,
        q_idx=torch.arange(full_len)[:, None],
        kv_idx=torch.arange(full_len)[None, :],
        block_size=args.train.block_size,
        n=args.data.max_seq_len,
    ).unsqueeze(0).unsqueeze(0)

    mask_dtype = torch.float32 if args.train.accelerator.fsdp_config.mixed_precision.enable else torch.bfloat16
    mask = torch.zeros_like(mask_flag, dtype=mask_dtype)
    mask.masked_fill_(mask_flag.logical_not(), float("-inf"))
    return mask


def _prepare_micro_batch(
    args: LLaDA2Arguments,
    micro_batch: Dict[str, Any],
    block_diffusion_attn_mask: Optional[torch.Tensor],
) -> Tuple[Dict[str, Any], int]:
    if args.train.block_diffusion_mode:
        noisy_input_ids = micro_batch.pop("noisy_input_ids")
        clean_input_ids = micro_batch["input_ids"]
        batch_size = noisy_input_ids.shape[0]
        noisy_seq_len = noisy_input_ids.shape[1]

        full_input_ids = torch.cat([noisy_input_ids, clean_input_ids], dim=1)
        noisy_position_ids = torch.arange(noisy_seq_len, device=full_input_ids.device, dtype=torch.long)
        clean_position_ids = torch.arange(clean_input_ids.shape[1], device=full_input_ids.device, dtype=torch.long)
        position_ids = torch.cat([noisy_position_ids, clean_position_ids], dim=0).unsqueeze(0)

        micro_batch["input_ids"] = full_input_ids
        micro_batch["position_ids"] = position_ids.expand(batch_size, -1).clone()
        micro_batch["attention_mask"] = block_diffusion_attn_mask.expand(batch_size, -1, -1, -1)
    else:
        noisy_seq_len = micro_batch["input_ids"].shape[1]
        micro_batch.pop("noisy_input_ids", None)
        micro_batch["attention_mask"] = None

    micro_batch = {
        k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in micro_batch.items()
    }
    return micro_batch, noisy_seq_len


def _compute_llada2_loss(
    args: LLaDA2Arguments,
    noisy_logits: torch.Tensor,
    labels: torch.Tensor,
    num_micro_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    confidence_loss = torch.tensor(0.0, device=noisy_logits.device)
    if args.train.confidence_beta > 0:
        confidence_loss = compute_confidence_loss(logits=noisy_logits, labels=labels)

    if args.train.same_token_labels:
        unscaled_loss = F.cross_entropy(
            noisy_logits.view(-1, noisy_logits.shape[-1]),
            labels.view(-1),
            reduction="none",
        )
        denom = (labels != -100).sum().clamp_min(1)
        consistency_loss = unscaled_loss.sum() / denom
    else:
        shifted_noisy_logits = noisy_logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        unscaled_loss = F.cross_entropy(
            shifted_noisy_logits.view(-1, shifted_noisy_logits.shape[-1]),
            shifted_labels.view(-1),
            reduction="none",
        )
        denom = (shifted_labels != -100).sum().clamp_min(1)
        consistency_loss = unscaled_loss.sum() / denom

    combined_loss = consistency_loss + confidence_loss * args.train.confidence_beta
    return combined_loss / num_micro_steps, consistency_loss, confidence_loss


def run_llada2_training(arguments_cls=LLaDA2Arguments):
    nccl_timeout = os.getenv("NCCL_TIMEOUT", None)
    pg_nccl_timeout = None
    if nccl_timeout is not None and is_nccl_backend():
        pg_nccl_timeout = timedelta(seconds=int(nccl_timeout))
    logger.info(f"Process_group timeout: {nccl_timeout}")
    dist.init_process_group(backend=get_dist_comm_backend(), timeout=pg_nccl_timeout)

    args = parse_args(arguments_cls)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    helper.enable_high_precision_for_bf16()
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.checkpoint.output_dir)

    set_checkpoint_debug_enabled(args.train.gradient_checkpointing.debug)

    Checkpointer = build_checkpointer(
        dist_backend=args.train.accelerator.fsdp_config.fsdp_mode,
        ckpt_manager=args.train.checkpoint.manager,
    )

    init_parallel_state(
        dp_size=args.train.accelerator.dp_size,
        dp_replicate_size=args.train.accelerator.dp_replicate_size,
        dp_shard_size=args.train.accelerator.dp_shard_size,
        tp_size=args.train.accelerator.tp_size,
        pp_size=args.train.accelerator.pp_size,
        cp_size=args.train.accelerator.cp_size,
        extra_parallel_sizes=args.train.accelerator.extra_parallel_sizes,
        extra_parallel_placement_innermost=args.train.accelerator.extra_parallel_placement_innermost,
        extra_parallel_names=args.train.accelerator.extra_parallel_names,
        ulysses_size=args.train.accelerator.ulysses_size,
        dp_mode=args.train.accelerator.fsdp_config.fsdp_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    transform = _build_transform(args, tokenizer)
    train_dataset = _build_train_dataset(args, transform)
    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type in ("mapping", "local") and dataset_length is not None:
        dataset_length = dataset_length / args.train.accelerator.dp_size
    args.compute_train_steps(dataset_length)

    train_dataloader = build_dataloader(
        dataloader_type=args.data.dataloader.type,
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train_steps,
        dyn_bsz=args.train.dyn_bsz,
        dyn_bsz_runtime=args.train.dyn_bsz_runtime,
        dyn_bsz_count_mode=args.train.dyn_bsz_count_mode,
        dyn_bsz_physical_overflow_ratio=args.train.dyn_bsz_physical_overflow_ratio,
        dyn_bsz_buffer_size=args.data.dyn_bsz_buffer_size,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
        num_workers=args.data.dataloader.num_workers,
        worker_num_threads=args.data.dataloader.worker_num_threads,
        drop_last=args.data.dataloader.drop_last,
        pin_memory=args.data.dataloader.pin_memory,
        prefetch_factor=args.data.dataloader.prefetch_factor,
        seed=args.train.seed,
        collate_fn_kwargs={"pad_to_length": args.train.pad_to_length},
        save_steps=args.train.checkpoint.save_steps,
    )

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
        init_device=args.train.init_device,
        ops_implementation=args.model.ops_implementation,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    basic_modules = list(set(getattr(model, "_no_split_modules", None) or []) | set(args.model.basic_modules))
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_reshard_after_forward=args.train.accelerator.fsdp_config.reshard_after_forward,
        mixed_precision=args.train.accelerator.fsdp_config.mixed_precision,
        enable_gradient_checkpointing=args.train.gradient_checkpointing.enable,
        basic_modules=basic_modules,
        enable_reentrant=args.train.gradient_checkpointing.enable_reentrant,
        enable_forward_prefetch=args.train.accelerator.fsdp_config.forward_prefetch,
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.optimizer.lr,
        betas=(args.train.beta1, args.train.beta2),
        weight_decay=args.train.optimizer.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer.type,
        no_decay_modules=args.train.optimizer.no_decay_modules,
        no_decay_params=args.train.optimizer.no_decay_params,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.accelerator.fsdp_config.fsdp_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train_steps * args.train.num_train_epochs,
        lr=args.train.optimizer.lr,
        lr_min=args.train.optimizer.lr_min,
        lr_decay_style=args.train.optimizer.lr_decay_style,
        lr_decay_ratio=args.train.optimizer.lr_decay_ratio,
        lr_warmup_ratio=args.train.optimizer.lr_warmup_ratio,
        lr_start=args.train.optimizer.lr_start,
    )

    model_assets = None
    if args.train.global_rank == 0:
        if args.train.wandb.enable:
            wandb.init(
                project=args.train.wandb.project,
                name=args.train.wandb.name,
                id=args.train.wandb.id,
                resume="allow" if args.train.wandb.id else None,
                settings=wandb.Settings(console="off"),
                config={**vars(args.model), **vars(args.data), **vars(args.train)},
            )

        model_assets = [model_config, tokenizer]
        save_model_assets(args.train.checkpoint.model_assets_dir, model_assets)

    if args.train.profile.this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile.start_step,
            end_step=args.train.profile.end_step,
            trace_dir=args.train.profile.trace_dir,
            record_shapes=args.train.profile.record_shapes,
            profile_memory=args.train.profile.profile_memory,
            with_stack=args.train.profile.with_stack,
            with_modules=args.train.profile.with_modules,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.checkpoint.load_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}
        Checkpointer.load(args.train.checkpoint.load_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train_steps
        start_step = global_step % args.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:
            iter(train_dataloader)

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.checkpoint.load_path} successfully!")

    block_diffusion_attn_mask = _build_block_diffusion_mask(args)
    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.accelerator.offload_config.enable_activation,
        args.train.gradient_checkpointing.enable,
        args.train.accelerator.offload_config.activation_gpu_limit,
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train_steps}, "
        f"epochs: {args.train.num_train_epochs}"
    )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0.0
            total_consistency_loss = 0.0
            total_confidence_loss = 0.0
            synchronize()
            start_time = time.time()
            num_micro_steps = len(micro_batches)

            for micro_step, micro_batch in enumerate(micro_batches):
                if (
                    args.train.accelerator.fsdp_config.fsdp_mode == "fsdp2"
                    and not args.train.accelerator.fsdp_config.reshard_after_backward
                    and num_micro_steps > 1
                ):
                    if micro_step == 0:
                        model.set_reshard_after_backward(False)
                    elif micro_step == num_micro_steps - 1:
                        model.set_reshard_after_backward(True)

                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("cur_token_num", None)
                    micro_batch.pop("source_name", None)

                micro_batch, noisy_seq_len = _prepare_micro_batch(args, micro_batch, block_diffusion_attn_mask)
                labels = micro_batch.pop("labels", None)

                with model_fwd_context:
                    logits = model(**micro_batch, use_cache=False, output_router_logits=False).logits
                    noisy_logits = logits[:, :noisy_seq_len].contiguous() if args.train.block_diffusion_mode else logits
                    loss, consistency_loss, confidence_loss = _compute_llada2_loss(
                        args,
                        noisy_logits=noisy_logits,
                        labels=labels,
                        num_micro_steps=num_micro_steps,
                    )

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                total_consistency_loss += consistency_loss.item() / num_micro_steps
                total_confidence_loss += confidence_loss.item() / num_micro_steps
                del micro_batch

            grad_norm = veomni_clip_grad_norm(model, args.train.optimizer.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if args.train.confidence_beta > 0:
                total_loss, total_consistency_loss, total_confidence_loss, grad_norm = all_reduce(
                    (total_loss, total_consistency_loss, total_confidence_loss, grad_norm),
                    group=get_parallel_state().fsdp_group,
                )
            else:
                total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()

            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            postfix = f"loss: {total_loss:.4f}, grad_norm: {grad_norm:.4f}, lr: {lr:.2e}"
            if args.train.confidence_beta > 0:
                postfix = (
                    f"loss: {total_loss:.4f}, cons: {total_consistency_loss:.4f}, "
                    f"conf: {total_confidence_loss:.4f}, grad_norm: {grad_norm:.4f}, lr: {lr:.2e}"
                )
            data_loader_tqdm.set_postfix_str(postfix, refresh=False)
            data_loader_tqdm.update()

            if args.train.global_rank == 0 and args.train.wandb.enable:
                train_metrics.update(
                    {
                        "training/loss": total_loss,
                        "training/grad_norm": grad_norm,
                        "training/lr": lr,
                    }
                )
                if args.train.confidence_beta > 0:
                    train_metrics.update(
                        {
                            "training/cons_loss": total_consistency_loss,
                            "training/conf_loss": total_confidence_loss,
                        }
                    )
                wandb.log(train_metrics, step=global_step)

            if args.train.profile.this_rank and global_step <= args.train.profile.end_step:
                profiler.step()
                if global_step == args.train.profile.end_step:
                    profiler.stop()

            if args.train.checkpoint.save_steps and global_step % args.train.checkpoint.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.checkpoint.save_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.checkpoint.save_path, state, global_steps=global_step)

                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.checkpoint.save_epochs and (epoch + 1) % args.train.checkpoint.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.checkpoint.save_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.checkpoint.save_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    del optimizer, lr_scheduler
    helper.empty_cache()
    if args.train.checkpoint.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        save_hf_safetensor(
            save_hf_safetensor_path=hf_weights_path,
            ckpt_manager=args.train.checkpoint.manager,
            model_assets=model_assets,
            save_checkpoint_path=save_checkpoint_path,
            is_rank_0=args.train.global_rank == 0,
            model=model,
            fqn_to_index_mapping=args.model.fqn_to_index_mapping,
        )

    dist.barrier()
    dist.destroy_process_group()
