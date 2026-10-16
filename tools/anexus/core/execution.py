"""Canonical execution contracts shared by run, RL, and swarm entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field

from anexus.core.context import TaskContext
from anexus.sandbox.provider import SandboxLaunchSpec
from anexus.tasks.graders import GraderConfig
from anexus.tasks.spec import TaskSpec


@dataclass(frozen=True)
class SandboxPlan:
    """Describe one fully resolved provider-neutral sandbox launch.

    Attributes:
        launch: Final provider create payload.
        role: Diagnostic role such as ``main`` or ``verify``.

    """

    launch: SandboxLaunchSpec
    role: str = "main"

    @property
    def name(self) -> str:
        """Return the logical sandbox name."""
        return self.launch.name


@dataclass(frozen=True)
class SidecarPlan:
    """Describe one attempt-scoped sidecar and its wiring contract.

    Attributes:
        launch: Fully resolved sidecar sandbox launch.
        env_url_key: Main-sandbox variable receiving the sidecar internal URL.
        health_path: Optional external readiness path.
        health_timeout_seconds: Maximum readiness wait in seconds.
        needs_provider_env: Whether to add the provider control-plane env.

    """

    launch: SandboxLaunchSpec
    env_url_key: str = ""
    health_path: str = ""
    health_timeout_seconds: float = 0.0
    needs_provider_env: bool = False

    def __post_init__(self) -> None:
        """Validate sidecar readiness settings."""
        if self.health_timeout_seconds < 0:
            raise ValueError("sidecar health_timeout_seconds must be >= 0")

    @property
    def name(self) -> str:
        """Return the logical sidecar name."""
        return self.launch.name


@dataclass(frozen=True)
class AgentExecutionPlan:
    """Describe sandbox-owned agent execution for one attempt.

    Attributes:
        payload_path: Sandbox path receiving the UTF-8 task JSON.
        commands: Final commands executed in order inside the main sandbox.
        timeout_seconds: Per-command cap bounded by the attempt deadline.

    """

    payload_path: str
    commands: tuple[str, ...]
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate payload and command fields."""
        if not self.payload_path.startswith("/"):
            raise ValueError("agent payload_path must be an absolute sandbox path")
        if not self.commands or any(not command.strip() for command in self.commands):
            raise ValueError("agent commands must contain only non-empty commands")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("agent timeout_seconds must be > 0")


@dataclass(frozen=True)
class EvaluationPlan:
    """Describe verification and result extraction for one attempt.

    Attributes:
        enabled: Whether an explicit verification phase is configured.
        command: Final verifier command, when configured directly.
        grader_config: Optional registered grader configuration.
        grader_eval_script: Statically generated verifier script text.
        sandbox: Optional isolated verifier sandbox plan.

    """

    enabled: bool = False
    command: str = ""
    grader_config: GraderConfig | None = None
    grader_eval_script: str = ""
    sandbox: SandboxPlan | None = None

    @property
    def isolated(self) -> bool:
        """Return whether evaluation uses a separate sandbox."""
        return self.sandbox is not None


@dataclass(frozen=True)
class GatewayPlan:
    """Describe one gateway session checkout.

    Attributes:
        route: Stable route selector such as ``kgb`` or ``none``.
        model: Resolved logical model name.
        agent_models: Agent model names keyed by runtime role.

    """

    route: str = "none"
    model: str = ""
    agent_models: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionRequest:
    """Represent the complete compiled input for one scheduled attempt."""

    context: TaskContext
    task: TaskSpec
    sandbox_plan: SandboxPlan
    sidecar_plans: tuple[SidecarPlan, ...]
    gateway_plan: GatewayPlan
    agent_plan: AgentExecutionPlan
    evaluation_plan: EvaluationPlan
    retry_number: int = 0

    def __post_init__(self) -> None:
        """Validate identity and attempt numbering."""
        if self.retry_number < 0:
            raise ValueError("retry_number must be >= 0")
        if (
            self.context.run_id,
            self.context.task_id,
            self.context.version,
            self.context.agent_name,
        ) != (
            self.task.run_id,
            self.task.task_id,
            self.task.version,
            self.task.agent_name,
        ):
            raise ValueError("execution context identity must match task identity")
        agent_spec_label = self.task.labels.get("agent_spec")
        if agent_spec_label and agent_spec_label != self.context.agent_spec_name:
            raise ValueError("execution context agent_spec_name must match task agent_spec label")
