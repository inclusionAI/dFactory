"""vLLM inference engine connector."""

from __future__ import annotations

from anexus.rl.engines.openai_compatible import (
    OpenAICompatibleEngine,
    base_url_from_env,
    session_id_from_env,
)


class VLLMEngine(OpenAICompatibleEngine):
    """Inference engine backed by vLLM's OpenAI-compatible server."""


class VLLMConnector:
    """Connect to a vLLM OpenAI-compatible server from runtime env."""

    def connect(self, env: dict[str, str]) -> VLLMEngine:
        base_url = base_url_from_env(
            env,
            url_keys=("VLLM_BASE_URL", "OPENAI_BASE_URL"),
            host_key="VLLM_HOST",
            port_key="VLLM_PORT",
            label="vllm",
        )
        session_id = session_id_from_env(
            env,
            keys=("VLLM_SESSION_ID", "ROLLOUT_SESSION_ID", "ANEXUS_SESSION_ID"),
        )
        return VLLMEngine(base_url, session_id=session_id)
