"""verl rollout engine connector."""

from __future__ import annotations

from anexus.rl.engines.openai_compatible import (
    OpenAICompatibleEngine,
    base_url_from_env,
    session_id_from_env,
)


class VerlEngine(OpenAICompatibleEngine):
    """Inference engine backed by a verl rollout server endpoint."""


class VerlConnector:
    """Connect to a verl rollout server from runtime env."""

    def connect(self, env: dict[str, str]) -> VerlEngine:
        base_url = base_url_from_env(
            env,
            url_keys=("VERL_OPENAI_BASE_URL", "VERL_BASE_URL", "OPENAI_BASE_URL"),
            host_key="VERL_HOST",
            port_key="VERL_PORT",
            label="verl",
        )
        session_id = session_id_from_env(
            env,
            keys=("VERL_SESSION_ID", "ROLLOUT_SESSION_ID", "ANEXUS_SESSION_ID"),
        )
        return VerlEngine(base_url, session_id=session_id)
