"""Readable console output and full-precision metric records for SolDiff."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

_PROGRESS_FIELDS = (("loss", "loss"), ("grad_norm", "grad_norm"), ("lr", "lr"))
_LOSS_FIELDS = (
    ("local_rloo_loss", "L_LO"),
    ("prior_permutation_alignment_loss", "L_ord"),
    ("suffix_reconstruction_loss", "L_rec"),
)
_PUBLIC_LOSS_NAMES = dict(_LOSS_FIELDS)
_OBJECTIVE_FIELDS = (
    *_LOSS_FIELDS,
    ("ce_per_token", "ce/token"),
    ("q_weighted_ce_per_token", "q_weighted_ce/token"),
)
_NORM_FIELDS = (
    ("q_model_norm", "q_norm"),
    ("p_head_norm", "p_norm"),
    ("order_stats/inversion_norm", "inversion"),
)
_LOGIT_FIELDS = (
    ("q_logits_std", "q_logits_std"),
    ("q_logits_abs_max", "q_logits_abs_max"),
    ("p_logits_std", "p_logits_std"),
    ("p_logits_abs_max", "p_logits_abs_max"),
)
_RANK_PREFIX = "order_stats/avg_rank_by_position/"


def _format_fields(values: Mapping[str, float], fields: tuple[tuple[str, str], ...]) -> str:
    return " | ".join(
        f"{label}={values[name]:.2e}" if name == "lr" else f"{label}={values[name]:.4f}"
        for name, label in fields
        if name in values
    )


def log_step_metrics(
    step: int,
    values: Mapping[str, float],
    *,
    output_dir: str | Path,
    block_size: int,
    diagnostics_every_steps: int,
) -> None:
    """Log one optimizer step on rank zero, using the trainer's logging cadence."""
    record: dict[str, int | float] = {"step": step}
    for name, _ in (*_PROGRESS_FIELDS, *_OBJECTIVE_FIELDS, *_NORM_FIELDS, *_LOGIT_FIELDS):
        if name in values:
            record[_PUBLIC_LOSS_NAMES.get(name, name)] = values[name]
    for index in range(block_size):
        name = f"{_RANK_PREFIX}{index}"
        if name in values:
            record[name] = values[name]

    target = Path(output_dir) / "metrics.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")

    progress = _format_fields(values, _PROGRESS_FIELDS)
    lines = [f"step={step}" + (f" | {progress}" if progress else "")]
    objective = _format_fields(values, _OBJECTIVE_FIELDS)
    if objective:
        lines.append(f"  {objective}")
    if step % diagnostics_every_steps == 0:
        lines.append(f"[diagnostics step={step}]")
        for fields in (_NORM_FIELDS, _LOGIT_FIELDS):
            formatted = _format_fields(values, fields)
            if formatted:
                lines.append(f"  {formatted}")
        for start in range(0, block_size, 16):
            end = min(start + 16, block_size)
            ranks = ", ".join(
                f"{values[f'{_RANK_PREFIX}{index}']:.4f}" if f"{_RANK_PREFIX}{index}" in values else "n/a"
                for index in range(start, end)
            )
            lines.append(f"  order_avg_rank[{start}:{end}] = [{ranks}]")
    print("\n".join(lines), flush=True)


__all__ = ["log_step_metrics"]
