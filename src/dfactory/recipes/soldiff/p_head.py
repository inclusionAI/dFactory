"""Public scorer primitives for the SolDiff recipe."""

from dfactory.models.qwen3.heads import (
    PWarmupHeadConfig,
    QwenStyleRMSNorm,
    TokenScoreMLP,
    attach_p_head,
    get_p_head,
)

__all__ = ["PWarmupHeadConfig", "QwenStyleRMSNorm", "TokenScoreMLP", "attach_p_head", "get_p_head"]
