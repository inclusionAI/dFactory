"""Execute self-contained agent attempts inside provider sandboxes."""

from __future__ import annotations

import asyncio
import shlex
from contextlib import suppress
from dataclasses import replace
from pathlib import PurePosixPath
from time import perf_counter

from anexus.core.context import TaskContext
from anexus.core.errors import AnexusError, ErrorEnvelope, ErrorKind, as_error_envelope
from anexus.core.execution import (
    AgentExecutionPlan,
    ExecutionRequest,
    SidecarPlan,
)
from anexus.events import EventBus, EventEnvelope
from anexus.gateway.session import (
    GatewaySession,
    GatewaySessionProvider,
    NoopGatewaySessionProvider,
)
from anexus.outputs.publisher import LocalOutputPublisher
from anexus.runtime.attempt import (
    AttemptResources,
    AttemptRoundOutcome,
    SandboxResource,
    build_final_record,
    gateway_trace_refs,
    sandbox_resource_fields,
    terminal_event_name,
    unique_strings,
)
from anexus.runtime.cleanup import CleanupBacklog, CleanupEntry
from anexus.runtime.grading import ProviderArtifactSandbox, get_grader_or_raise
from anexus.runtime.retry import RetryPolicy
from anexus.runtime.sidecar import wait_for_sidecar_health
from anexus.runtime.store import LocalRunStore, record_event_safely
from anexus.sandbox.provider import (
    ArtifactMapping,
    DownloadResult,
    ExecResult,
    FixtureMapping,
    SandboxLaunchSpec,
    SandboxLease,
    SandboxProvider,
    UploadResult,
)
from anexus.tasks.evaluation import EvaluationInput, StructuredOutputEvaluator, TaskEvaluator
from anexus.tasks.spec import EvaluationResult, TaskAttemptRecord

# Attempt execution


class AsyncTaskExecutor:
    """Async task executor implementing the Self-contained Agent model.

    Key design points:
    - All I/O is async (no threading layer)
    - Only Self-contained Agent mode (no host-directed LLM calls)
    - Fixtures download directly in sandbox (no host transit)
    - Artifacts upload directly from sandbox (no host transit)
    - Cancellation via asyncio.CancelledError (no threading.Event)
    """

    def __init__(
        self,
        *,
        sandbox_provider: SandboxProvider,
        bus: EventBus,
        output_publisher: LocalOutputPublisher,
        evaluator: TaskEvaluator | None = None,
        run_store: LocalRunStore | None = None,
        cleanup_backlog: CleanupBacklog | None = None,
        gateway_sessions: GatewaySessionProvider | None = None,
        cleanup_timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize an executor with run-scoped services only."""
        if cleanup_timeout_seconds <= 0:
            raise ValueError("cleanup_timeout_seconds must be > 0")

        self.sandbox_provider = sandbox_provider
        self.bus = bus
        self.output_publisher = output_publisher
        self.evaluator = evaluator or StructuredOutputEvaluator()
        self.run_store = run_store
        self.cleanup_backlog = cleanup_backlog
        self.gateway_sessions = gateway_sessions or NoopGatewaySessionProvider()
        self.cleanup_timeout_seconds = cleanup_timeout_seconds

    def log(
        self,
        name: str,
        context: TaskContext,
        payload: dict[str, object] | None = None,
        **fields: object,
    ) -> None:
        """Record one task event without interrupting the business flow."""
        event = EventEnvelope.for_task(name, context, {**(payload or {}), **fields})
        self.bus.publish(event)
        if self.run_store is not None:
            record_event_safely(self.run_store, event, self.bus.publish)

    async def run(self, request: ExecutionRequest) -> TaskAttemptRecord:
        """Execute one canonical request with its configured retry policy."""
        task = request.task
        context = request.context.with_worker_pid()
        started = perf_counter()
        self.log(
            "attempt.started",
            context,
            payload_sha256=task.payload_sha256,
            agent_spec_name=context.agent_spec_name,
        )

        retry_policy = RetryPolicy(task.attempt_policy)
        max_rounds = task.attempt_policy.max_retries + 1
        deadline = started + task.attempt_policy.timeout_seconds
        outcomes: list[AttemptRoundOutcome] = []
        errors: list[ErrorEnvelope] = []

        for run_round in range(1, max_rounds + 1):
            round_context = context.with_scope(
                retry_number=request.retry_number + run_round - 1,
                run_round=run_round,
            )
            remaining = deadline - perf_counter()
            outcome = await self._execute_round(
                request,
                round_context,
                timeout_seconds=max(remaining, 0.001),
            )
            outcomes.append(outcome)
            errors.extend(outcome.errors)
            if outcome.primary_error is None:
                break

            primary_error = outcome.primary_error
            decision = retry_policy.decide(primary_error, failures_seen=run_round)
            remaining = deadline - perf_counter()
            if (
                not decision.should_retry
                or run_round >= max_rounds
                or remaining <= 0
                or primary_error.kind in {ErrorKind.CANCELLED, ErrorKind.TIMEOUT}
            ):
                break
            self.log(
                "attempt.retrying",
                outcome.context,
                {
                    **primary_error.to_dict(),
                    "run_round": run_round,
                    "retry_reason": decision.reason,
                },
            )
            wait_seconds = min(
                retry_policy.wait_seconds(failures_seen=run_round),
                max(deadline - perf_counter(), 0.0),
            )
            try:
                await asyncio.sleep(wait_seconds)
            except asyncio.CancelledError:
                cancelled_resources = AttemptResources(context=round_context)
                cancelled_error = _attempt_cancelled_error(round_context, phase="attempt.retry_wait")
                outcomes.append(
                    AttemptRoundOutcome(
                        context=round_context,
                        resources=cancelled_resources,
                        result=None,
                        primary_error=cancelled_error,
                        diagnostic_errors=(),
                        cleanup_errors=(),
                        llm_traces=(),
                    )
                )
                errors.append(cancelled_error)
                break

        return self._finalize_attempt(request, outcomes, errors, started)

    async def _execute_round(
        self,
        request: ExecutionRequest,
        context: TaskContext,
        *,
        timeout_seconds: float,
    ) -> AttemptRoundOutcome:
        """Execute one retry round and always resolve its cleanup outcome."""
        resources = AttemptResources(context=context)
        result: EvaluationResult | None = None
        primary_error: ErrorEnvelope | None = None
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await self._execute_business_phases(request, resources)
        except TimeoutError:
            primary_error = _attempt_timeout_error(
                resources.context,
                phase=resources.phase,
                timeout_seconds=timeout_seconds,
            )
        except asyncio.CancelledError:
            primary_error = _attempt_cancelled_error(
                resources.context,
                phase=resources.phase,
            )
        except AnexusError as exc:
            primary_error = _enrich_error(
                as_error_envelope(exc, component="runtime.executor"),
                resources.context,
                resources.phase,
            )
        except Exception as exc:
            primary_error = _enrich_error(
                as_error_envelope(exc, component="runtime.executor"),
                resources.context,
                resources.phase,
            )

        if result is None:
            result = resources.evaluation_result
        cleanup_errors = await self._cleanup_with_deadline(
            resources,
            graceful=primary_error is None or primary_error.kind != ErrorKind.CANCELLED,
        )
        llm_traces = tuple(
            gateway_trace_refs(
                resources.gateway_session,
                uploaded_trace_files=resources.gateway_uploaded_traces,
            )
        )
        return AttemptRoundOutcome(
            context=resources.context,
            resources=resources,
            result=result,
            primary_error=primary_error,
            diagnostic_errors=tuple(resources.diagnostics),
            cleanup_errors=tuple(cleanup_errors),
            llm_traces=llm_traces,
        )

    async def _execute_business_phases(
        self,
        request: ExecutionRequest,
        resources: AttemptResources,
    ) -> EvaluationResult:
        """Execute business phases without publishing a terminal result."""
        task = request.task
        resources.phase = "gateway.checkout"
        resources.gateway_session = await self._phase_checkout_gateway(
            request,
            resources.context,
        )
        session_id = getattr(resources.gateway_session, "session_id", "")
        if session_id:
            resources.context = resources.context.with_gateway(session_id)
        self.log("gateway.ready", resources.context)

        resources.phase = "sidecars.create"
        await self._phase_create_sidecars(request, resources)
        sidecar_env = await self._collect_sidecar_env(resources.sidecars)

        resources.phase = "sandbox.create"
        resources.sandbox = await self._phase_create_sandbox(
            request,
            resources,
            sidecar_env,
            gateway_session=resources.gateway_session,
        )
        lease = resources.sandbox.lease
        resources.context = resources.context.with_sandbox(lease.sandbox_id)
        self.log(
            "sandbox.created",
            resources.context,
            sandbox_id=lease.sandbox_id,
            provider=lease.provider,
        )

        resources.phase = "fixtures.download"
        await self._phase_download_fixtures(
            request,
            lease,
            resources.context,
            resources,
        )
        self.log(
            "fixtures.ready",
            resources.context,
            _transfer_counts(resources.fixture_results),
        )

        resources.phase = "sidecars.ready"
        await self._phase_wait_ready(resources, resources.context)
        resources.phase = "payload.inject"
        await self._phase_inject_payload(request, lease)

        resources.phase = "agent.execute"
        self.log("agent.started", resources.context, agent_name=task.agent_name)
        exec_result = await self._phase_execute_agent(request.agent_plan, lease)
        self.log("agent.completed", resources.context, agent_name=task.agent_name)

        resources.phase = "evaluation.execute"
        result = await self._phase_verify_and_evaluate(
            request,
            resources.context,
            resources,
            exec_result,
        )
        resources.evaluation_result = result
        self.log(
            "attempt.evaluated",
            resources.context,
            passed=result.passed,
            score=result.score,
        )

        resources.phase = "artifacts.upload"
        await self._phase_upload_artifacts(
            request,
            lease,
            resources.context,
            resources,
        )
        self.log(
            "artifacts.committed",
            resources.context,
            _transfer_counts(resources.artifact_results),
        )
        resources.phase = "business.completed"
        return result

    async def _phase_checkout_gateway(
        self,
        request: ExecutionRequest,
        context: TaskContext,
    ) -> GatewaySession:
        """Checkout the gateway session declared by the request."""
        return await self.gateway_sessions.checkout(context, request.gateway_plan)

    async def _phase_create_sidecars(
        self,
        request: ExecutionRequest,
        resources: AttemptResources,
    ) -> None:
        """Create all statically compiled attempt sidecars."""
        for plan in request.sidecar_plans:
            launch = plan.launch
            if plan.needs_provider_env:
                provider_env = self.sandbox_provider.control_plane_env(launch.provider)
                merged_env = dict(launch.env)
                for key, value in provider_env.items():
                    merged_env.setdefault(key, value)
                launch = replace(launch, env=merged_env)
            lease = await self.sandbox_provider.create(resources.context, launch)
            resource = resources.register_sandbox(SandboxResource(plan=plan, lease=lease))
            resources.sidecars.append(resource)

    async def _phase_create_sandbox(
        self,
        request: ExecutionRequest,
        resources: AttemptResources,
        sidecar_env: dict[str, str],
        gateway_session: GatewaySession | None = None,
    ) -> SandboxResource:
        """Create the request's main sandbox with dynamic service env."""
        launch = _launch_with_env(request.sandbox_plan.launch, sidecar_env)
        if gateway_session is not None:
            launch = _launch_with_env(launch, gateway_session.agent_env())
        lease = await self.sandbox_provider.create(resources.context, launch)
        return resources.register_sandbox(SandboxResource(plan=request.sandbox_plan, lease=lease))

    async def _phase_download_fixtures(
        self,
        request: ExecutionRequest,
        lease: SandboxLease,
        context: TaskContext,
        resources: AttemptResources,
    ) -> None:
        """Download fixtures and preserve required/optional semantics."""
        for fixture in request.task.fixtures.fixtures:
            source_uri = (fixture.source_uri or fixture.source_path).strip()
            target_path = (fixture.sandbox_path or fixture.target_path).strip()
            mapping = FixtureMapping(
                source_uri=source_uri,
                target_path=target_path,
                kind=fixture.kind,
            )
            try:
                if not source_uri or not target_path:
                    raise ValueError("fixture requires source and sandbox target")
                results = await self.sandbox_provider.download_fixtures(
                    lease,
                    [mapping],
                )
                result = _require_single_transfer_result(
                    results,
                    operation="fixture download",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                base = as_error_envelope(exc, component="sandbox.download_fixtures")
                error = replace(
                    base,
                    kind=(ErrorKind.SANDBOX_ERROR if base.kind == ErrorKind.UNKNOWN else base.kind),
                    details={
                        **context.to_fields(),
                        **base.details,
                        "source_uri": source_uri,
                        "target_path": target_path,
                        "kind": fixture.kind,
                        "required": fixture.required,
                        "phase": "fixtures.download",
                        "operation": "download",
                    },
                )
                resources.fixture_results.append(
                    {
                        "source_uri": source_uri,
                        "target_path": target_path,
                        "kind": fixture.kind,
                        "required": fixture.required,
                        "size_bytes": 0,
                        "file_count": 0,
                        "status": "failed",
                        "error": error.to_dict(),
                    }
                )
                self.log("fixture.download_failed", context, error.to_dict())
                if fixture.required:
                    raise AnexusError(
                        error.message,
                        kind=error.kind,
                        component=error.component,
                        retryable=error.retryable,
                        details=error.details,
                    ) from exc
                resources.diagnostics.append(error)
            else:
                resources.fixture_results.append(
                    {
                        "source_uri": source_uri,
                        "target_path": target_path,
                        "kind": fixture.kind,
                        "required": fixture.required,
                        "size_bytes": result.size_bytes,
                        "file_count": result.file_count,
                        "status": "downloaded",
                        "error": None,
                    }
                )

    async def _phase_wait_ready(
        self,
        resources: AttemptResources,
        context: TaskContext,
    ) -> None:
        """Wait for sidecars to become healthy (async)."""
        for sidecar in resources.sidecars:
            plan = sidecar.plan
            if not isinstance(plan, SidecarPlan) or not plan.health_timeout_seconds:
                continue
            health_url = await self.sandbox_provider.port_url(sidecar.lease, plan.launch.port, internal=False)
            await wait_for_sidecar_health(plan, health_url)
            self.log(
                "sidecar.ready",
                context,
                name=plan.name,
                sandbox_id=sidecar.lease.sandbox_id,
            )

    async def _phase_inject_payload(
        self,
        request: ExecutionRequest,
        lease: SandboxLease,
    ) -> None:
        """Write the task payload to the statically compiled sandbox path."""
        import json as _json

        payload_json = _json.dumps(request.task.payload, ensure_ascii=False)
        await self._put_text_file(lease, payload_json, request.agent_plan.payload_path)

    async def _put_text_file(self, lease: SandboxLease, text: str, target_path: str) -> None:
        command = _write_text_file_command(target_path)
        result = await self.sandbox_provider.exec(lease, command, stdin=text, timeout=30)
        if result.exit_code != 0:
            raise AnexusError(
                f"failed to write sandbox text file: {target_path}",
                kind=ErrorKind.SANDBOX_ERROR,
                component="sandbox.write_text_file",
                retryable=True,
                details={
                    "command": command,
                    "target_path": target_path,
                    "exit_code": result.exit_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            )

    async def _phase_execute_agent(
        self,
        plan: AgentExecutionPlan,
        lease: SandboxLease,
    ) -> ExecResult:
        """Execute agent command in sandbox. Agent makes its own LLM calls."""
        result = ExecResult(exit_code=0, stdout="", stderr="")
        for command in plan.commands:
            result = await self.sandbox_provider.exec(
                lease,
                command,
                timeout=plan.timeout_seconds,
            )
            if result.exit_code != 0:
                raise AnexusError(
                    f"agent command failed with exit code {result.exit_code}",
                    kind=ErrorKind.AGENT_ERROR,
                    component="agent",
                    retryable=True,
                    details={
                        "command": command,
                        "exit_code": result.exit_code,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                )
        return result

    async def _phase_verify_and_evaluate(
        self,
        request: ExecutionRequest,
        context: TaskContext,
        resources: AttemptResources,
        exec_result: ExecResult,
    ) -> EvaluationResult:
        """Run verification and evaluation."""
        task = request.task
        verification = request.evaluation_plan

        if verification.enabled and resources.sandbox is not None:
            lease = resources.sandbox.lease
            grader_config = verification.grader_config
            grader = None
            verification_command = verification.command
            grader_artifact = ""
            grader_eval_script = ""

            if grader_config is not None:
                grader = get_grader_or_raise(grader_config)
                artifact_sandbox = ProviderArtifactSandbox(
                    sandbox_provider=self.sandbox_provider,
                    lease=lease,
                )
                grader_artifact = await grader.extract_artifact(artifact_sandbox, task.payload, grader_config)
                if not grader_artifact.strip():
                    return EvaluationResult(passed=False, score=0.0, summary="verify_artifact_empty", details={})
                grader_eval_script = verification.grader_eval_script
                if not grader_eval_script.strip():
                    return EvaluationResult(passed=False, score=0.0, summary="no_eval_script", details={})
                verification_command = verification_command or grader.build_command(task.payload, grader_config)

            verifier_lease = lease
            if verification.isolated and verification.sandbox is not None:
                verifier_launch = verification.sandbox.launch
                if verifier_launch is None:
                    raise ValueError("isolated verifier sandbox launch is required")
                verifier_resource = resources.register_sandbox(
                    SandboxResource(
                        plan=verification.sandbox,
                        lease=await self.sandbox_provider.create(
                            context,
                            verifier_launch,
                        ),
                    )
                )
                resources.verifiers.append(verifier_resource)
                verifier_lease = verifier_resource.lease

            if grader is not None and grader_config is not None:
                await self._put_text_file(verifier_lease, grader_artifact, "/tmp/agent_patch.diff")
                await self._put_text_file(verifier_lease, grader_eval_script, "/tmp/eval_script.sh")

            if verification_command:
                verify_result = await self.sandbox_provider.exec(
                    verifier_lease,
                    verification_command,
                    timeout=grader_config.eval_timeout_seconds if grader_config else None,
                )
                if grader is not None and grader_config is not None:
                    grader_result = grader.grade_output(
                        verify_result.stdout,
                        verify_result.exit_code,
                        task.payload,
                        grader_config,
                    )
                    return grader_result.to_evaluation_result()
                return self.evaluator.evaluate(
                    EvaluationInput(
                        task=task,
                        agent_text="",
                        sandbox_stdout=verify_result.stdout,
                        sandbox_stderr=verify_result.stderr,
                    )
                )

        return self.evaluator.evaluate(
            EvaluationInput(
                task=task,
                agent_text="",
                sandbox_stdout=exec_result.stdout,
                sandbox_stderr=exec_result.stderr,
            )
        )

    async def _phase_upload_artifacts(
        self,
        request: ExecutionRequest,
        lease: SandboxLease,
        context: TaskContext,
        resources: AttemptResources,
    ) -> None:
        """Upload artifacts and preserve required/optional semantics."""
        for artifact in request.task.artifacts:
            source_path = artifact.source_path.strip()
            target_uri = artifact.uri.strip()
            kind = artifact.storage_kind or "file"
            mapping = ArtifactMapping(
                source_path=source_path,
                target_uri=target_uri,
                kind=kind,
            )
            try:
                if not source_path or not target_uri:
                    raise ValueError("artifact requires sandbox source_path and target uri")
                results = await self.sandbox_provider.upload_artifacts(
                    lease,
                    [mapping],
                )
                result = _require_single_transfer_result(
                    results,
                    operation="artifact upload",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                base = as_error_envelope(exc, component="sandbox.upload_artifacts")
                error = replace(
                    base,
                    kind=(ErrorKind.ARTIFACT_ERROR if base.kind == ErrorKind.UNKNOWN else base.kind),
                    details={
                        **context.to_fields(),
                        **base.details,
                        "object_id": artifact.object_id,
                        "semantic_kind": artifact.kind,
                        "source_path": source_path,
                        "target_uri": target_uri,
                        "kind": kind,
                        "required": artifact.required,
                        "phase": "artifacts.upload",
                        "operation": "upload",
                    },
                )
                resources.artifact_results.append(
                    {
                        "object_id": artifact.object_id,
                        "semantic_kind": artifact.kind,
                        "source_path": source_path,
                        "target_uri": target_uri,
                        "kind": kind,
                        "required": artifact.required,
                        "size_bytes": 0,
                        "file_count": 0,
                        "status": "failed",
                        "error": error.to_dict(),
                    }
                )
                self.log("artifact.upload_failed", context, error.to_dict())
                if artifact.required:
                    raise AnexusError(
                        error.message,
                        kind=error.kind,
                        component=error.component,
                        retryable=error.retryable,
                        details=error.details,
                    ) from exc
                resources.diagnostics.append(error)
            else:
                resources.artifact_results.append(
                    {
                        "object_id": artifact.object_id,
                        "semantic_kind": artifact.kind,
                        "source_path": source_path,
                        "target_uri": target_uri,
                        "kind": kind,
                        "required": artifact.required,
                        "size_bytes": result.size_bytes,
                        "file_count": result.file_count,
                        "status": "uploaded",
                        "error": None,
                    }
                )

    async def _collect_sidecar_env(self, sidecars: list[SandboxResource]) -> dict[str, str]:
        """Collect provider-issued internal sidecar endpoints."""
        env: dict[str, str] = {}
        for sidecar in sidecars:
            plan = sidecar.plan
            if isinstance(plan, SidecarPlan) and plan.env_url_key:
                url = await self.sandbox_provider.port_url(
                    sidecar.lease,
                    plan.launch.port,
                    internal=True,
                )
                env[plan.env_url_key] = url
        return env

    async def _cleanup_with_deadline(
        self,
        resources: AttemptResources,
        *,
        graceful: bool,
    ) -> list[ErrorEnvelope]:
        """Run reverse-order cleanup under an independent bounded deadline."""
        self.log("cleanup.started", resources.context)
        cleanup_task = asyncio.create_task(self._cleanup_resources(resources, graceful=graceful))
        try:
            errors = await asyncio.wait_for(
                asyncio.shield(cleanup_task),
                timeout=self.cleanup_timeout_seconds,
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task
            errors = self._record_unreleased_resources(resources, exc)
        event_name = "cleanup.failed" if errors else "cleanup.completed"
        self.log(
            event_name,
            resources.context,
            errors=[error.to_dict() for error in errors],
        )
        return errors

    async def _cleanup_resources(
        self,
        resources: AttemptResources,
        *,
        graceful: bool,
    ) -> list[ErrorEnvelope]:
        """Release registered resources in strict reverse acquisition order."""
        errors: list[ErrorEnvelope] = []
        for resource in reversed(resources.sandbox_stack):
            try:
                await self.sandbox_provider.delete(resource.lease)
            except Exception as exc:
                error = self._sandbox_cleanup_error(resources.context, resource, exc)
                resource.cleanup_error = error
                errors.append(error)
                self._record_sandbox_cleanup_pending(resources.context, resource, error)
                self.log("sandbox.cleanup_failed", resources.context, error.to_dict())
            else:
                resource.released = True
                self.log(
                    "sandbox.deleted",
                    resources.context,
                    sandbox_resource_fields(resource),
                )

        release = getattr(resources.gateway_session, "release", None)
        if callable(release):
            if graceful:
                await self._record_final_gateway_activity(resources)
            try:
                uploaded = await release(graceful=graceful)
            except Exception as exc:
                error = self._gateway_cleanup_error(resources.context, exc)
                resources.gateway_cleanup_error = error
                errors.append(error)
                self._publish_gateway_release_failure(resources.context, error)
            else:
                if isinstance(uploaded, list):
                    resources.gateway_uploaded_traces.extend(str(item) for item in uploaded if item)
                resources.gateway_released = True
        return errors

    async def _record_final_gateway_activity(self, resources: AttemptResources) -> None:
        snapshot_fn = getattr(resources.gateway_session, "activity_snapshot", None)
        if not callable(snapshot_fn):
            return
        try:
            snapshot = await snapshot_fn()
        except Exception as exc:
            self.log(
                "gateway.activity",
                resources.context,
                sample="final",
                available=False,
                error_type=exc.__class__.__name__,
                error=str(exc) or exc.__class__.__name__,
            )
            return
        if snapshot is None:
            return
        self.log(
            "gateway.activity",
            resources.context,
            snapshot.event_payload(sample="final"),
        )

    def _record_unreleased_resources(
        self,
        resources: AttemptResources,
        exc: BaseException,
    ) -> list[ErrorEnvelope]:
        """Persist resources left uncertain after cleanup timeout/cancellation."""
        errors = [resource.cleanup_error for resource in resources.sandbox_stack if resource.cleanup_error is not None]
        for resource in reversed(resources.sandbox_stack):
            if resource.released or resource.cleanup_error is not None:
                continue
            error = self._sandbox_cleanup_error(resources.context, resource, exc)
            resource.cleanup_error = error
            errors.append(error)
            self._record_sandbox_cleanup_pending(resources.context, resource, error)
        if resources.gateway_session is not None and not resources.gateway_released:
            if resources.gateway_cleanup_error is not None:
                errors.append(resources.gateway_cleanup_error)
            else:
                error = self._gateway_cleanup_error(resources.context, exc)
                resources.gateway_cleanup_error = error
                errors.append(error)
        return errors

    def _sandbox_cleanup_error(
        self,
        context: TaskContext,
        resource: SandboxResource,
        exc: BaseException,
    ) -> ErrorEnvelope:
        """Build a resource-specific sandbox cleanup error."""
        base = as_error_envelope(exc, component="runtime.cleanup")
        fields = sandbox_resource_fields(resource)
        return replace(
            base,
            kind=(
                ErrorKind.SANDBOX_ERROR
                if base.kind in {ErrorKind.UNKNOWN, ErrorKind.CANCELLED, ErrorKind.TIMEOUT}
                else base.kind
            ),
            message=f"sandbox cleanup failed for {resource.lease.sandbox_id}: {base.message}",
            component="runtime.cleanup",
            retryable=True,
            details={
                **base.details,
                **context.to_fields(),
                **fields,
                "phase": "cleanup.sandbox.delete",
                "operation": "delete",
            },
        )

    def _gateway_cleanup_error(
        self,
        context: TaskContext,
        exc: BaseException,
    ) -> ErrorEnvelope:
        """Normalize a gateway release failure without hiding provider detail."""
        base = as_error_envelope(exc, component="gateway.cleanup")
        return replace(
            base,
            details={
                **context.to_fields(),
                **base.details,
                "phase": base.details.get("phase", "cleanup.gateway.release"),
                "operation": "release",
            },
        )

    def _record_sandbox_cleanup_pending(
        self,
        context: TaskContext,
        resource: SandboxResource,
        error: ErrorEnvelope,
    ) -> None:
        """Append one retryable sandbox leak to the cleanup backlog."""
        if self.cleanup_backlog is None:
            return
        fields = sandbox_resource_fields(resource)
        self.cleanup_backlog.add(
            context.run_id,
            CleanupEntry.sandbox_pending(
                context=context,
                lease=resource.lease,
                error=error,
                resource_name=str(fields["resource_name"]),
                resource_role=str(fields["resource_role"]),
            ),
        )

    def _finalize_attempt(
        self,
        request: ExecutionRequest,
        outcomes: list[AttemptRoundOutcome],
        errors: list[ErrorEnvelope],
        started: float,
    ) -> TaskAttemptRecord:
        """Publish one final record after every acquired resource is resolved."""
        if not outcomes:
            raise RuntimeError("attempt produced no execution outcome")
        record = build_final_record(request, outcomes, errors, started)
        context = outcomes[-1].context
        try:
            published_ref = self.output_publisher.publish(record)
        except Exception as exc:
            output_error = _enrich_error(
                as_error_envelope(exc, component="outputs.publisher"),
                context,
                "outputs.records.publish",
            )
            record = replace(
                record,
                status="failed",
                errors=[*record.errors, output_error],
                quality_tags=unique_strings([*record.quality_tags, output_error.kind.value]),
            )
            self.log("attempt.output_failed", context, output_error.to_dict())
            if self.run_store is None:
                raise
            self.run_store.write_attempt_record(record.to_dict())
            published_ref = "run_store_fallback"

        self.log(
            "attempt.output_published",
            context,
            status=record.status,
            reference=str(published_ref),
        )
        self.log(
            terminal_event_name(record.status),
            context,
            status=record.status,
            score=record.score,
            error_count=len(record.errors),
        )
        return record

    def _publish_gateway_release_failure(
        self,
        context: TaskContext,
        error: ErrorEnvelope,
    ) -> None:
        phase = str(error.details.get("phase", ""))
        event_name = "gateway.trace_export.failed" if phase == "gateway.kgb.trace_export" else "gateway.release.failed"
        self.log(event_name, context, error.to_dict())


# Error and command helpers


def _attempt_timeout_error(
    context: TaskContext,
    *,
    phase: str,
    timeout_seconds: float,
) -> ErrorEnvelope:
    """Build a structured attempt-deadline error."""
    return ErrorEnvelope(
        kind=ErrorKind.TIMEOUT,
        message=f"attempt exceeded remaining timeout {timeout_seconds:.3f}s",
        component="runtime.executor",
        details={
            **context.to_fields(),
            "phase": phase,
            "operation": "execute",
            "timeout_seconds": timeout_seconds,
        },
    )


def _attempt_cancelled_error(context: TaskContext, *, phase: str) -> ErrorEnvelope:
    """Build a structured attempt cancellation error."""
    return ErrorEnvelope(
        kind=ErrorKind.CANCELLED,
        message="attempt cancelled",
        component="runtime.executor",
        details={
            **context.to_fields(),
            "phase": phase,
            "operation": "execute",
        },
    )


def _enrich_error(
    error: ErrorEnvelope,
    context: TaskContext,
    phase: str,
) -> ErrorEnvelope:
    """Add attempt identity and phase without replacing source ownership."""
    return replace(
        error,
        details={
            **context.to_fields(),
            **error.details,
            "phase": error.details.get("phase", phase),
        },
    )


def _transfer_counts(
    outcomes: list[dict[str, object]],
) -> dict[str, int]:
    """Count successful and failed data-plane transfers."""
    failed = sum(outcome["status"] == "failed" for outcome in outcomes)
    return {
        "transfer_count": len(outcomes),
        "succeeded_count": len(outcomes) - failed,
        "failed_count": failed,
    }


def _require_single_transfer_result(
    results: list[DownloadResult] | list[UploadResult],
    *,
    operation: str,
) -> DownloadResult | UploadResult:
    """Require a provider to return one result for one transfer mapping.

    Args:
        results: Provider results for a single mapping.
        operation: Human-readable operation used in contract errors.

    Returns:
        The only provider transfer result.

    Raises:
        RuntimeError: If the provider violates the one-mapping contract.

    """
    if len(results) != 1:
        raise RuntimeError(f"provider returned {len(results)} results for one {operation} mapping")
    return results[0]


def _write_text_file_command(target_path: str) -> str:
    target = str(PurePosixPath(target_path))
    parent = str(PurePosixPath(target).parent)
    if not target or target == ".":
        raise ValueError("target_path must be a non-empty sandbox path")
    return f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(target)}"


def _launch_with_env(
    launch: SandboxLaunchSpec | None,
    env: dict[str, str],
) -> SandboxLaunchSpec | None:
    """Merge dynamic service environment into a compiled launch spec.

    Args:
        launch: Statically compiled provider launch payload.
        env: Dynamic sidecar or gateway environment.

    Returns:
        A copied launch with dynamic values taking precedence.

    """
    if launch is None or not env:
        return launch
    return replace(launch, env={**launch.env, **env})
