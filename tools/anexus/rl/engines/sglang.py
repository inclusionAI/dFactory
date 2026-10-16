"""SGLang inference engine connector."""

from __future__ import annotations

from anexus.rl.engines.openai_compatible import (
    OpenAICompatibleEngine,
    base_url_from_env,
    session_id_from_env,
)


class SGLangEngine(OpenAICompatibleEngine):
    """Inference engine backed by SGLang's OpenAI-compatible server."""


class SGLangConnector:
    """Connect to an SGLang OpenAI-compatible server from runtime env."""

    def connect(self, env: dict[str, str]) -> SGLangEngine:
        base_url = base_url_from_env(
            env,
            url_keys=("SGLANG_BASE_URL", "OPENAI_BASE_URL"),
            host_key="SGLANG_HOST",
            port_key="SGLANG_PORT",
            label="sglang",
        )
        session_id = session_id_from_env(
            env,
            keys=("SGLANG_SESSION_ID", "ROLLOUT_SESSION_ID", "ANEXUS_SESSION_ID"),
        )
        return SGLangEngine(base_url, session_id=session_id)
