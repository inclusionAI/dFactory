"""Attempt state and final record construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from anexus.core.context import TaskContext
from anexus.core.errors import ErrorEnvelope, ErrorKind
from anexus.core.execution import ExecutionRequest, SandboxPlan, SidecarPlan
from anexus.gateway.session import GatewaySession
from anexus.sandbox.provider import SandboxLease
from anexus.tasks.spec import EvaluationResult, TaskAttemptRecord, TaskLineage, TokenUsage


@dataclass
class SandboxResource:
    """Created sandbox tracked until cleanup finishes."""

    plan: SandboxPlan | SidecarPlan
    lease: SandboxLease
    released: bool = False
    cleanup_error: ErrorEnvelope | None = None


@dataclass
class AttemptResources:
    """Resources and data accumulated during one infrastructure round."""

    context: TaskContext
    sandbox: SandboxResource | None = None
    sidecars: list[SandboxResource] = field(default_factory=list)
    verifiers: list[SandboxResource] = field(default_factory=list)
    sandbox_stack: list[SandboxResource] = field(default_factory=list)
    gateway_session: GatewaySession | None = None
    gateway_released: bool = False
    gateway_cleanup_error: ErrorEnvelope | None = None
    gateway_uploaded_traces: list[str] = field(default_factory=list)
    phase: str = "attempt.start"
    fixture_results: list[dict[str, object]] = field(default_factory=list)
    artifact_results: list[dict[str, object]] = field(default_factory=list)
    diagnostics: list[ErrorEnvelope] = field(default_factory=list)
    evaluation_result: EvaluationResult | None = None

    def register_sandbox(self, resource: SandboxResource) -> SandboxResource:
        self.sandbox_stack.append(resource)
        return resource


@dataclass(frozen=True)
class AttemptRoundOutcome:
    """Business and cleanup result of one infrastructure retry round."""

    context: TaskContext
    resources: AttemptResources
    result: EvaluationResult | None
    primary_error: ErrorEnvelope | None
    diagnostic_errors: tuple[ErrorEnvelope, ...]
    cleanup_errors: tuple[ErrorEnvelope, ...]
    llm_traces: tuple[dict[str, Any], ...]

    @property
    def errors(self) -> tuple[ErrorEnvelope, ...]:
        primary = (self.primary_error,) if self.primary_error is not None else ()
        return (*primary, *self.diagnostic_errors, *self.cleanup_errors)


def build_final_record(
    request: ExecutionRequest,
    outcomes: list[AttemptRoundOutcome],
    errors: list[ErrorEnvelope],
    started: float,
) -> TaskAttemptRecord:
    """Reduce retry rounds into one canonical attempt record."""

    task = request.task
    final = outcomes[-1]
    primary_error = final.primary_error
    cleanup_pending = any(outcome.cleanup_errors for outcome in outcomes)
    if primary_error is None:
        status = "cleanup_pending" if cleanup_pending else "completed"
    elif primary_error.kind == ErrorKind.CANCELLED:
        status = "cancelled"
    elif primary_error.kind == ErrorKind.TIMEOUT:
        status = "timeout"
    else:
        status = "failed"

    result = final.result
    if result is None:
        result = EvaluationResult(
            passed=False,
            score=0.0,
            summary=primary_error.message if primary_error is not None else "not evaluated",
            details={"error": primary_error.to_dict() if primary_error is not None else {}},
        )

    resources = [resource for outcome in outcomes for resource in outcome.resources.sandbox_stack]
    main_resource = next(
        (outcome.resources.sandbox for outcome in reversed(outcomes) if outcome.resources.sandbox is not None),
        None,
    )
    gateway_session_id = next(
        (
            getattr(outcome.resources.gateway_session, "session_id", "")
            for outcome in reversed(outcomes)
            if outcome.resources.gateway_session is not None
        ),
        "",
    )
    traces = unique_trace_refs([trace for outcome in outcomes for trace in outcome.llm_traces])
    quality_tags = unique_strings(
        [error.kind.value for error in errors] + (["cleanup_pending"] if cleanup_pending else [])
    )
    return TaskAttemptRecord(
        run_id=task.run_id,
        task_id=task.task_id,
        version=task.version,
        attempt_id=request.context.attempt_id,
        agent_name=task.agent_name,
        agent_spec_name=request.context.agent_spec_name,
        payload_sha256=task.payload_sha256,
        status=status,
        score=result.score if final.result is not None else None,
        result=result,
        llm_traces=traces,
        fixtures=list(final.resources.fixture_results),
        artifacts={
            "manifest_uri": "",
            "objects": list(final.resources.artifact_results),
        },
        usage=TokenUsage(),
        timings={"attempt_seconds": perf_counter() - started},
        errors=list(errors),
        quality_tags=quality_tags,
        lineage=TaskLineage(
            source_uri=task.source_uri,
            payload_sha256=task.payload_sha256,
            agent_spec_name=request.context.agent_spec_name,
            model_ids=unique_strings(
                [
                    request.gateway_plan.model,
                    *request.gateway_plan.agent_models.values(),
                ]
            ),
        ),
        payload=task.payload,
        sandbox_id=main_resource.lease.sandbox_id if main_resource is not None else "",
        sandbox_provider=main_resource.lease.provider if main_resource is not None else "",
        gateway_session_id=gateway_session_id,
        sandbox_resources=[sandbox_resource_summary(resource) for resource in resources],
        sample_index=int(request.context.scope.get("sample_index", 0)),
        execution_index=int(request.context.scope.get("execution_index", 0)),
    )


def gateway_trace_refs(
    gateway_session: GatewaySession | None,
    *,
    uploaded_trace_files: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return session traces that were confirmed uploaded during release."""

    if gateway_session is None:
        return []
    refs = gateway_session.trace_refs()
    if not isinstance(refs, list):
        return []
    trace_refs = [dict(ref) for ref in refs if isinstance(ref, dict)]
    if uploaded_trace_files is None:
        return trace_refs
    uploaded = {str(item) for item in uploaded_trace_files}
    if not uploaded:
        return []
    return [ref for ref in trace_refs if str(ref.get("trace_file") or "") in uploaded]


def sandbox_resource_fields(resource: SandboxResource) -> dict[str, object]:
    plan = resource.plan
    return {
        "sandbox_id": resource.lease.sandbox_id,
        "provider": resource.lease.provider,
        "resource_name": plan.name,
        "resource_role": "sidecar" if isinstance(plan, SidecarPlan) else plan.role,
    }


def sandbox_resource_summary(resource: SandboxResource) -> dict[str, object]:
    return {
        **sandbox_resource_fields(resource),
        "cleanup_status": "cleaned" if resource.released else "pending",
        "cleanup_error": (resource.cleanup_error.to_dict() if resource.cleanup_error is not None else None),
    }


def terminal_event_name(status: str) -> str:
    if status == "cancelled":
        return "attempt.cancelled"
    if status in {"failed", "timeout"}:
        return "attempt.failed"
    return "attempt.completed"


def unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def unique_trace_refs(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        key = repr(sorted(value.items()))
        if key not in seen:
            seen.add(key)
            refs.append(dict(value))
    return refs
