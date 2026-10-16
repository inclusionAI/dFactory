"""Pure compilation of run plans into canonical execution requests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from anexus.config import RunPlan
from anexus.core.context import TaskContext
from anexus.core.execution import (
    AgentExecutionPlan,
    EvaluationPlan,
    ExecutionRequest,
    GatewayPlan,
    SandboxPlan,
    SidecarPlan,
)
from anexus.core.ids import new_id
from anexus.runtime.store import LocalRunStore, StoredAttempt, StoredAttemptIntent
from anexus.sandbox.network import sandbox_network_policy_from_config
from anexus.sandbox.provider import SandboxLaunchSpec
from anexus.tasks import TaskAttemptRecord, TaskSpec
from anexus.tasks.graders import GraderConfig, get_grader

AttemptIdFactory = Callable[[str], str]
TASK_PAYLOAD_PATH = "/tmp/.task_payload.json"
ACCEPTED_RESUME_STATUSES = frozenset({"completed", "cleanup_pending"})


class ResumeStateError(ValueError):
    """Raised when persisted slot state is ambiguous or corrupt."""


@dataclass(frozen=True, order=True)
class AttemptSlot:
    """Stable logical identity for one requested sample."""

    agent_name: str
    task_id: str
    payload_sha256: str
    sample_index: int


@dataclass(frozen=True)
class ExecutionSchedule:
    """Resume-aware requests and invocation counters for one run."""

    requests: tuple[ExecutionRequest, ...]
    resumed_attempts: tuple[StoredAttempt, ...]
    total_slot_count: int
    physical_attempt_count: int

    @property
    def resumed_slot_count(self) -> int:
        return len(self.resumed_attempts)

    @property
    def scheduled_slot_count(self) -> int:
        return len(self.requests)


class ExecutionRequestCompiler:
    """Compile one resolved run plan into complete attempt requests.

    Compilation expands attempts, applies resume selection, resolves task-level
    image and sidecar values, and lowers every runtime decision into one
    request. It never allocates a gateway or sandbox resource.

    Args:
        run_store: Optional durable store used for resume filtering.
        resume_completed: Whether completed attempts count toward the requested
            number of samples per task.
        id_factory: Function used to create attempt identifiers.

    """

    def __init__(
        self,
        *,
        run_store: LocalRunStore | None = None,
        resume_completed: bool = False,
        id_factory: AttemptIdFactory = new_id,
    ) -> None:
        """Initialize the compiler without creating runtime resources."""
        self._run_store = run_store
        self._resume_completed = resume_completed
        self._id_factory = id_factory

    def compile(self, plan: RunPlan) -> tuple[ExecutionRequest, ...]:
        """Compile all scheduled attempts for a run.

        Args:
            plan: Fully resolved, provider-free run plan.

        Returns:
            Requests in stable attempt-major, task-minor order.

        """
        return self.compile_schedule(plan).requests

    def compile_schedule(self, plan: RunPlan) -> ExecutionSchedule:
        """Compile stable sample slots using final records and recordless intents."""
        _require_unique_task_slots(plan.tasks)
        attempts = self._run_store.list_attempts(plan.run_id) if self._run_store else []
        intents = self._run_store.list_attempt_intents(plan.run_id) if self._run_store else []
        attempts_by_slot = _attempts_by_slot(attempts, plan)
        intents_by_slot = _intents_by_slot(intents, plan)
        resumed: list[StoredAttempt] = []
        requests: list[ExecutionRequest] = []
        for sample_index in range(1, plan.n_attempts + 1):
            for task in plan.tasks:
                slot = _slot(task, sample_index)
                slot_attempts = attempts_by_slot.get(slot, [])
                accepted = [attempt for attempt in slot_attempts if attempt.status in ACCEPTED_RESUME_STATUSES]
                if len(accepted) > 1:
                    raise ResumeStateError(
                        f"multiple accepted records exist for agent {task.agent_name!r} "
                        f"task {task.task_id!r} "
                        f"sample_index={sample_index}"
                    )
                if self._resume_completed and accepted:
                    resumed.append(accepted[0])
                    continue
                execution_index = 1
                if self._resume_completed:
                    previous_indexes = [attempt.execution_index for attempt in slot_attempts] + [
                        intent.execution_index for intent in intents_by_slot.get(slot, [])
                    ]
                    execution_index = max(previous_indexes, default=0) + 1
                requests.append(
                    self.compile_task(
                        plan,
                        task,
                        sample_index=sample_index,
                        execution_index=execution_index,
                    )
                )
        return ExecutionSchedule(
            requests=tuple(requests),
            resumed_attempts=tuple(resumed),
            total_slot_count=len(plan.tasks) * plan.n_attempts,
            physical_attempt_count=len(attempts),
        )

    def compile_task(
        self,
        plan: RunPlan,
        task: TaskSpec,
        *,
        sample_index: int = 1,
        execution_index: int = 1,
    ) -> ExecutionRequest:
        """Compile one task and scheduling slot into an execution request.

        Args:
            plan: Run-level configuration and execution template source.
            task: Task payload and identity.

        Returns:
            A request containing every statically known attempt decision.

        """
        if task.run_id != plan.run_id:
            raise ValueError("task run_id must match the run plan")
        agent = plan.agent(task.agent_name)
        if sample_index < 1:
            raise ValueError("sample_index must be >= 1")
        if execution_index < 1:
            raise ValueError("execution_index must be >= 1")
        context_scope = {
            key: task.labels[key]
            for key in ("swarm_id", "swarm_node_id", "gateway_session_group_id")
            if task.labels.get(key)
        }
        context_scope.update(
            sample_index=sample_index,
            execution_index=execution_index,
        )
        context = TaskContext(
            run_id=task.run_id,
            task_id=task.task_id,
            version=task.version,
            attempt_id=self._id_factory("att"),
            agent_name=task.agent_name,
            agent_spec_name=agent.agent_spec.name,
            scope=context_scope,
        )
        main_sandbox = compile_main_sandbox_plan(plan, task)
        gateway_route = agent.gateway.driver
        return ExecutionRequest(
            context=context,
            task=task,
            sandbox_plan=main_sandbox,
            sidecar_plans=compile_sidecar_plans(plan, task),
            gateway_plan=GatewayPlan(
                route=gateway_route,
                model=agent.model_id,
                agent_models=dict(agent.agent_models),
            ),
            agent_plan=compile_agent_plan(plan, task),
            evaluation_plan=compile_evaluation_plan(plan, task, main_sandbox),
        )


def select_logical_records(
    plan: RunPlan,
    attempts: Iterable[StoredAttempt],
    *,
    executed: Iterable[TaskAttemptRecord] = (),
) -> tuple[list[TaskAttemptRecord], int]:
    """Select one current record per logical slot and count physical records."""
    raw_by_attempt: dict[tuple[str, str, str], dict[str, object]] = {
        (attempt.agent_name, attempt.task_id, attempt.attempt_id): dict(attempt.record) for attempt in attempts
    }
    for record in executed:
        raw_by_attempt[(record.agent_name, record.task_id, record.attempt_id)] = record.to_dict()

    expected = {_slot(task, sample_index) for sample_index in range(1, plan.n_attempts + 1) for task in plan.tasks}
    by_slot: dict[AttemptSlot, list[dict[str, object]]] = {}
    for raw in raw_by_attempt.values():
        slot = _raw_record_slot(raw)
        if slot is not None and slot in expected:
            by_slot.setdefault(slot, []).append(raw)

    selected: list[TaskAttemptRecord] = []
    for sample_index in range(1, plan.n_attempts + 1):
        for task in plan.tasks:
            slot = _slot(task, sample_index)
            records = by_slot.get(slot, [])
            accepted = [record for record in records if str(record.get("status", "")) in ACCEPTED_RESUME_STATUSES]
            if len(accepted) > 1:
                raise ResumeStateError(
                    f"multiple accepted records exist for agent {task.agent_name!r} "
                    f"task {task.task_id!r} "
                    f"sample_index={sample_index}"
                )
            chosen = accepted[0] if accepted else _latest_physical_record(records)
            if chosen is not None:
                selected.append(TaskAttemptRecord.from_dict(chosen))
    return selected, len(raw_by_attempt)


def _require_unique_task_slots(tasks: Iterable[TaskSpec]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for task in tasks:
        key = (task.agent_name, task.task_id, task.payload_sha256)
        if key in seen:
            raise ValueError(
                f"duplicate task identity in run plan: agent_name={task.agent_name!r} "
                f"task_id={task.task_id!r} "
                f"payload_sha256={task.payload_sha256}"
            )
        seen.add(key)


def _slot(task: TaskSpec, sample_index: int) -> AttemptSlot:
    return AttemptSlot(task.agent_name, task.task_id, task.payload_sha256, sample_index)


def _attempts_by_slot(
    attempts: Iterable[StoredAttempt],
    plan: RunPlan,
) -> dict[AttemptSlot, list[StoredAttempt]]:
    expected_tasks = {(task.agent_name, task.task_id, task.payload_sha256) for task in plan.tasks}
    grouped: dict[AttemptSlot, list[StoredAttempt]] = {}
    for attempt in attempts:
        if (
            (attempt.agent_name, attempt.task_id, attempt.payload_sha256) not in expected_tasks
            or not 1 <= attempt.sample_index <= plan.n_attempts
            or attempt.execution_index < 1
        ):
            continue
        slot = AttemptSlot(
            attempt.agent_name,
            attempt.task_id,
            attempt.payload_sha256,
            attempt.sample_index,
        )
        grouped.setdefault(slot, []).append(attempt)
    return grouped


def _intents_by_slot(
    intents: Iterable[StoredAttemptIntent],
    plan: RunPlan,
) -> dict[AttemptSlot, list[StoredAttemptIntent]]:
    expected_tasks = {(task.agent_name, task.task_id, task.payload_sha256) for task in plan.tasks}
    grouped: dict[AttemptSlot, list[StoredAttemptIntent]] = {}
    for intent in intents:
        if (
            (intent.agent_name, intent.task_id, intent.payload_sha256) not in expected_tasks
            or not 1 <= intent.sample_index <= plan.n_attempts
            or intent.execution_index < 1
        ):
            continue
        slot = AttemptSlot(
            intent.agent_name,
            intent.task_id,
            intent.payload_sha256,
            intent.sample_index,
        )
        grouped.setdefault(slot, []).append(intent)
    return grouped


def _raw_record_slot(record: dict[str, object]) -> AttemptSlot | None:
    try:
        sample_index = int(record.get("sample_index", 0))
    except (TypeError, ValueError):
        return None
    task_id = str(record.get("task_id", ""))
    agent_name = str(record.get("agent_name", ""))
    payload_sha256 = str(record.get("payload_sha256", ""))
    if not agent_name or not task_id or not payload_sha256 or sample_index < 1:
        return None
    return AttemptSlot(agent_name, task_id, payload_sha256, sample_index)


def _latest_physical_record(
    records: list[dict[str, object]],
) -> dict[str, object] | None:
    if not records:
        return None

    def order(record: dict[str, object]) -> tuple[int, str]:
        try:
            execution_index = int(record.get("execution_index", 0))
        except (TypeError, ValueError):
            execution_index = 0
        return execution_index, str(record.get("attempt_id", ""))

    return max(records, key=order)


def compile_main_sandbox_plan(plan: RunPlan, task: TaskSpec) -> SandboxPlan:
    """Compile the final main sandbox launch for one task.

    Args:
        plan: Resolved run plan.
        task: Task that may select an image through its payload.

    Returns:
        A provider-neutral main sandbox plan.

    """
    agent = plan.agent(task.agent_name)
    sandbox = agent.sandbox
    env = dict(agent.agent_spec.env)
    env.update(sandbox.env)
    launch = SandboxLaunchSpec(
        image=_task_sandbox_image(plan, task),
        provider=sandbox.provider,
        port=sandbox.port,
        cpu_request=sandbox.cpu_request,
        cpu_limit=sandbox.cpu_limit,
        memory_request_mb=sandbox.memory_request_mb,
        memory_limit_mb=sandbox.memory_limit_mb,
        command=sandbox.command,
        env=env,
        file_overrides=[dict(item) for item in sandbox.file_overrides],
        name="main",
        network_policy=sandbox.network_policy,
        extra_config=dict(sandbox.extra_config) if sandbox.extra_config else None,
    )
    return SandboxPlan(launch=launch, role="main")


def compile_sidecar_plans(plan: RunPlan, task: TaskSpec) -> tuple[SidecarPlan, ...]:
    """Compile all task-specific sidecar launches.

    Args:
        plan: Resolved run plan containing sidecar templates.
        task: Task supplying allowlisted payload-derived environment values.

    Returns:
        Final sidecar plans in catalog order.

    """
    compiled: list[SidecarPlan] = []
    agent = plan.agent(task.agent_name)
    for sidecar in agent.sidecars:
        env = dict(sidecar.env)
        for env_key, payload_path in sidecar.env_from_payload.items():
            value = _payload_string(task.payload, payload_path)
            if value:
                env[env_key] = value
        launch = SandboxLaunchSpec(
            image=sidecar.image,
            provider=sidecar.provider,
            port=sidecar.port,
            ports=list(sidecar.ports),
            cpu_request=sidecar.cpu_request,
            cpu_limit=sidecar.cpu_limit,
            memory_request_mb=sidecar.memory_request_mb,
            memory_limit_mb=sidecar.memory_limit_mb,
            command=sidecar.command,
            name=sidecar.name,
            env=env,
            network_policy=sidecar.network_policy,
            extra_config=dict(sidecar.extra_config) if sidecar.extra_config else None,
        )
        compiled.append(
            SidecarPlan(
                launch=launch,
                env_url_key=sidecar.env_url_key,
                health_path=sidecar.health_path,
                health_timeout_seconds=sidecar.health_timeout_seconds,
                needs_provider_env=sidecar.needs_provider,
            )
        )
    return tuple(compiled)


def compile_agent_plan(plan: RunPlan, task: TaskSpec) -> AgentExecutionPlan:
    """Compile final sandbox commands for one agent attempt.

    Args:
        plan: Resolved run plan containing the task runner.
        task: Task supplying the attempt timeout.

    Returns:
        A final agent execution plan with payload-file injection.

    """
    agent = plan.agent(task.agent_name)
    task_runner = agent.sandbox.task_runner
    if isinstance(task_runner, list):
        commands = [str(command).strip() for command in task_runner if str(command).strip()]
    else:
        command = str(task_runner).strip()
        commands = [command] if command else []
    if not commands:
        raise ValueError(f"agent {task.agent_name!r} does not define sandbox.task_runner")
    commands[-1] = f"TASK_PAYLOAD_FILE={TASK_PAYLOAD_PATH} {commands[-1]}"
    return AgentExecutionPlan(
        payload_path=TASK_PAYLOAD_PATH,
        commands=tuple(commands),
        timeout_seconds=task.attempt_policy.timeout_seconds,
    )


def compile_evaluation_plan(
    plan: RunPlan,
    task: TaskSpec,
    main_sandbox: SandboxPlan,
) -> EvaluationPlan:
    """Compile verification config into a task-specific evaluation plan.

    Args:
        plan: Resolved run plan containing the verifier block.
        task: Task payload used by registered graders.
        main_sandbox: Resolved main sandbox used as verifier defaults.

    Returns:
        A disabled plan when no verifier exists, otherwise a fully resolved
        direct or isolated verification plan.

    Raises:
        ValueError: If verification is enabled but incomplete or references an
            unknown grader.

    """
    raw_verify = plan.agent(task.agent_name).sandbox.verify
    if not raw_verify:
        return EvaluationPlan()
    raw = dict(raw_verify)
    if bool(raw.get("enabled", True)) is False:
        return EvaluationPlan()

    grader_config = _grader_config_from_verify(raw)
    command = str(raw.get("command") or raw.get("eval_command") or "").strip()
    grader_eval_script = ""
    if grader_config is not None:
        try:
            grader = get_grader(grader_config.grader)
        except ValueError as exc:
            raise ValueError(f"unknown sandbox verifier grader: {grader_config.grader}") from exc
        grader_eval_script = grader.generate_eval_script(task.payload, grader_config)
        command = command or grader.build_command(task.payload, grader_config)
    if not command and grader_config is None:
        raise ValueError("sandbox verify config requires command or grader when enabled")

    verifier_config = dict(raw)
    if (
        grader_config is not None
        and "isolated" not in verifier_config
        and not _has_verifier_sandbox_config(verifier_config)
    ):
        verifier_config["isolated"] = True
    verifier = _compile_verifier_sandbox(main_sandbox, verifier_config)
    return EvaluationPlan(
        enabled=True,
        command=command,
        grader_config=grader_config,
        grader_eval_script=grader_eval_script,
        sandbox=verifier,
    )


def _compile_verifier_sandbox(
    main_sandbox: SandboxPlan,
    verify: dict[str, object],
) -> SandboxPlan | None:
    """Compile an optional isolated verifier sandbox."""
    raw_sandbox = verify.get("sandbox")
    sandbox = raw_sandbox if isinstance(raw_sandbox, dict) else {}
    isolated = bool(verify.get("isolated", False)) or bool(sandbox)
    if not isolated:
        return None
    base = main_sandbox.launch
    if base is None:
        raise ValueError("isolated verification requires a main sandbox launch")
    launch = SandboxLaunchSpec(
        image=str(sandbox.get("image") or base.image),
        provider=str(sandbox.get("provider") or base.provider),
        port=int(sandbox.get("port", base.port)),
        ports=[int(port) for port in sandbox.get("ports", base.ports)],
        cpu_request=int(sandbox.get("cpu_request", sandbox.get("cpu", base.cpu_request))),
        cpu_limit=int(sandbox.get("cpu_limit", sandbox.get("cpu", base.cpu_limit))),
        memory_request_mb=int(
            sandbox.get(
                "memory_request_mb",
                sandbox.get("memory_request", sandbox.get("memory", base.memory_request_mb)),
            )
        ),
        memory_limit_mb=int(
            sandbox.get(
                "memory_limit_mb",
                sandbox.get("memory_limit", sandbox.get("memory", base.memory_limit_mb)),
            )
        ),
        command=str(sandbox["command"]) if "command" in sandbox else None,
        name=str(sandbox.get("name", "verify")),
        env={str(key): str(value) for key, value in sandbox.get("env", {}).items()}
        if isinstance(sandbox.get("env"), dict)
        else dict(base.env),
        file_overrides=[
            {str(key): str(value) for key, value in item.items()}
            for item in sandbox.get("file_overrides", [])
            if isinstance(item, dict)
        ],
        network_policy=sandbox_network_policy_from_config(
            sandbox,
            default=base.network_policy,
        ),
        extra_config=dict(sandbox["extra_config"]) if isinstance(sandbox.get("extra_config"), dict) else None,
    )
    return SandboxPlan(launch=launch, role="verify")


_VERIFY_GRADER_CONTROL_KEYS = {
    "command",
    "enabled",
    "eval_command",
    "eval_timeout",
    "eval_timeout_seconds",
    "extra",
    "grader",
    "isolated",
    "sandbox",
}


def _grader_config_from_verify(raw: dict[str, object]) -> GraderConfig | None:
    """Parse a registered grader block from verifier config."""
    if "grader" not in raw:
        return None
    grader = str(raw.get("grader") or "").strip()
    if not grader:
        raise ValueError("sandbox verify grader must be a non-empty string")
    extra = {str(key): value for key, value in raw.items() if key not in _VERIFY_GRADER_CONTROL_KEYS}
    raw_extra = raw.get("extra")
    if isinstance(raw_extra, dict):
        extra.update({str(key): value for key, value in raw_extra.items()})
    return GraderConfig(
        enabled=True,
        grader=grader,
        eval_timeout_seconds=_positive_int(
            raw.get("eval_timeout_seconds", raw.get("eval_timeout", 300)),
            default=300,
        ),
        extra=extra,
    )


def _positive_int(value: object, *, default: int) -> int:
    """Return a positive integer or a configured default."""
    if isinstance(value, int):
        return value if value > 0 else default
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else default
    return default


def _has_verifier_sandbox_config(verify: dict[str, object]) -> bool:
    """Return whether verifier config declares a dedicated sandbox."""
    sandbox = verify.get("sandbox")
    return isinstance(sandbox, dict) and bool(sandbox)


def _task_sandbox_image(plan: RunPlan, task: TaskSpec) -> str:
    """Resolve a task-specific main sandbox image.

    Args:
        plan: Run plan containing the default image and payload selector.
        task: Task payload that may override the image.

    Returns:
        The task-selected image when present, otherwise the catalog image.

    """
    sandbox = plan.agent(task.agent_name).sandbox
    path = sandbox.image_from_payload.strip()
    value = _payload_value(task.payload, path) if path else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return sandbox.image


def _payload_string(payload: dict[str, object], path: str) -> str | None:
    """Return a trimmed string selected from a dot-separated payload path."""
    value = _payload_value(payload, path)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _payload_value(payload: dict[str, object], path: str) -> object:
    """Select a value from a nested payload without mutating it."""
    value: object = payload
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value
