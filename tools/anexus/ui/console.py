"""Human-readable run progress reporter for the CLI."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from anexus.events import EventEnvelope
from anexus.gateway.session import GatewayActivitySnapshot

KGBLogCountGetter = Callable[[Iterable[str]], Mapping[str, int]]
KGBTotalLogGetter = Callable[[], int]
ActivitySnapshotGetter = Callable[[], Iterable[GatewayActivitySnapshot]]
RunProgressMode = str
ProgressKey = tuple[str, str, str, str]

_PHASES = ("gateway", "sandbox", "fixtures", "agent", "evaluate", "artifact", "cleanup")


@dataclass
class _ActivityView:
    available: bool = True
    request_count: int = 0
    last_request_time: str = ""
    response_status: int | None = None
    total_time_ms: int | None = None
    observed_at: str = ""


@dataclass
class _TaskProgress:
    task_id: str
    agent_name: str = ""
    attempt_id: str = ""
    gateway_session_id: str = ""
    status: str = "started"
    score: float | None = None
    kgb_logs: int | None = None
    activity: _ActivityView | None = None
    activity_delta: int = 0
    activity_delta_at: float = 0.0


class RunConsoleReporter:
    """Render event-driven run progress with Rich or stable plain text."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        mode: RunProgressMode = "compact",
        status_interval: float = 10.0,
        kgb_log_counts: KGBLogCountGetter | None = None,
        kgb_total_logs: KGBTotalLogGetter | None = None,
        activity_snapshots: ActivitySnapshotGetter | None = None,
        output_dir: str | Path | None = None,
        native_stderr_path: str | Path | None = None,
        application_log_path: str | Path | None = None,
        max_active_tasks: int = 10,
        max_recent_tasks: int = 3,
        force_rich: bool | None = None,
    ) -> None:
        if mode not in {"compact", "verbose"}:
            raise ValueError("mode must be compact or verbose")
        if status_interval <= 0:
            raise ValueError("status_interval must be > 0")
        if max_active_tasks < 1:
            raise ValueError("max_active_tasks must be >= 1")
        if max_recent_tasks < 1:
            raise ValueError("max_recent_tasks must be >= 1")
        self.stream = stream or sys.stderr
        self.mode = mode
        self.status_interval = status_interval
        self.kgb_log_counts = kgb_log_counts
        self.kgb_total_logs = kgb_total_logs
        self.activity_snapshots = activity_snapshots
        self.output_dir = str(output_dir) if output_dir is not None else ""
        self.native_stderr_path = str(native_stderr_path) if native_stderr_path else ""
        self.application_log_path = str(application_log_path) if application_log_path else ""
        self.max_active_tasks = max_active_tasks
        self._recent: deque[tuple[str, str, float | None, int | None, str]] = deque(maxlen=max_recent_tasks)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None
        self._run_id = ""
        self._agent_name = ""
        self._trace_viewer_url = ""
        self._total_tasks = 0
        self._concurrency = 0
        self._started_at = 0.0
        self._queued = 0
        self._admitted = 0
        self._completed = 0
        self._failed = 0
        self._active: dict[ProgressKey, _TaskProgress] = {}
        self._scores: dict[ProgressKey, float] = {}
        self._finished_kgb_logs = 0
        self._kgb_total = 0
        self._last_poll_error = ""
        self._console: Console | None = None
        self._live: Live | None = None
        terminal = _is_terminal(self.stream)
        self._rich_enabled = terminal if force_rich is None else force_rich

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()
        if self._rich_enabled:
            self._console = Console(
                file=self.stream,
                force_terminal=True,
                no_color=os.environ.get("NO_COLOR") is not None,
                highlight=False,
            )
            self._live = Live(
                self._render_rich(),
                console=self._console,
                auto_refresh=False,
                transient=False,
            )
            self._live.start(refresh=True)
        self._thread = threading.Thread(
            target=self._status_loop,
            name="anexus-run-console",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._live is not None:
            self._refresh_activity()
            self._live.update(self._render_rich(), refresh=True)
            self._live.stop()
            self._live = None

    def emit(self, event: EventEnvelope) -> None:
        line = self._handle_event(event)
        if self._rich_enabled:
            self._refresh_live()
        elif line:
            self._print(line)

    def print_status(self) -> None:
        """Refresh gateway activity and render one terminal update."""
        self._refresh_activity()
        if self._rich_enabled:
            self._refresh_live()
            return
        with self._lock:
            line = self._status_line()
        self._print(line)

    def _status_loop(self) -> None:
        interval = 1.0 if self._rich_enabled else self.status_interval
        while not self._stop_event.wait(interval):
            with self._lock:
                if not self._running:
                    return
            self.print_status()

    def _handle_event(self, event: EventEnvelope) -> str:
        name = event.name
        context = event.context
        payload = event.payload
        now = time.time()
        with self._lock:
            if name == "run.trace_viewer":
                self._run_id = str(context.get("run_id") or self._run_id)
                self._agent_name = str(payload.get("agent_name") or self._agent_name)
                self._trace_viewer_url = str(payload.get("trace_viewer_url") or "")
                if self._trace_viewer_url:
                    return f"== Trace Viewer: {self._trace_viewer_url}"
                return ""
            if name == "run.started":
                self._run_id = str(context.get("run_id") or "")
                raw_agent_names = payload.get("agent_names")
                self._agent_name = (
                    ",".join(str(item) for item in raw_agent_names) if isinstance(raw_agent_names, list) else ""
                )
                self._total_tasks = _int(payload.get("request_count") or payload.get("task_count"))
                self._concurrency = _int(payload.get("n_concurrent"))
                self._started_at = now
                output = f" output={self.output_dir}" if self.output_dir else ""
                return (
                    f"== run started agents={_cell(self._agent_name)} "
                    f"tasks={self._total_tasks} concurrency={self._concurrency} "
                    f"run={_cell(self._run_id)}{output}"
                )
            if name == "attempt.queued":
                self._queued += 1
                return ""
            if name == "attempt.admitted":
                self._admitted += 1
                self._set_active(context, "gateway")
                return ""
            if name == "attempt.started":
                progress = self._set_active(context, "gateway")
                return f"> start {_task_label(progress)}" if self.mode == "verbose" else ""
            if name == "gateway.ready":
                self._set_active(context, "sandbox")
                return ""
            if name == "sandbox.created":
                self._set_active(context, "fixtures")
                return ""
            if name in {"fixtures.ready", "agent.started"}:
                self._set_active(context, "agent")
                return ""
            if name == "agent.completed":
                self._set_active(context, "evaluate")
                return ""
            if name == "attempt.evaluated":
                progress = self._set_active(context, "artifact")
                score = _optional_float(payload.get("score"))
                if score is not None:
                    progress.score = score
                    self._scores[_progress_key(context)] = score
                return ""
            if name == "artifacts.committed":
                self._set_active(context, "cleanup")
                return ""
            if name == "gateway.activity":
                progress = self._set_active(context, self._current_phase(context))
                self._apply_activity(progress, _activity_from_payload(payload), now=now)
                self._recalculate_kgb_total()
                return ""
            if name == "attempt.retrying":
                task_id = str(context.get("task_id") or "")
                agent_name = str(context.get("agent_name") or "")
                progress = self._active.get(_progress_key(context))
                self._recent.append(
                    (
                        _identity_label(agent_name, task_id),
                        "retrying",
                        progress.score if progress is not None else None,
                        progress.kgb_logs if progress is not None else None,
                        _message(payload.get("message")),
                    )
                )
                return self._retry_line(context, payload)
            if name == "attempt.completed":
                status = str(payload.get("status") or "completed")
                return self._complete_task(context, payload, status=status)
            if name in {"attempt.failed", "attempt.cancelled"}:
                return self._complete_task(context, payload, status="failed")
            if name in {"run.completed", "run.failed", "run.cancelled"}:
                status = str(payload.get("status") or name.removeprefix("run."))
                counts = payload.get("status_counts")
                if isinstance(counts, Mapping):
                    self._completed = _int(counts.get("completed"))
                    self._failed = sum(_int(value) for key, value in counts.items() if str(key) != "completed")
                return self._final_line(status)
        return ""

    def _current_phase(self, context: Mapping[str, object]) -> str:
        progress = self._active.get(_progress_key(context))
        return progress.status if progress is not None else "gateway"

    def _set_active(
        self,
        context: Mapping[str, object],
        status: str,
    ) -> _TaskProgress:
        key = _progress_key(context)
        task_id = str(context.get("task_id") or "")
        agent_name = str(context.get("agent_name") or "")
        attempt_id = str(context.get("attempt_id") or "")
        gateway_session_id = str(context.get("gateway_session_id") or "")
        progress = self._active.get(key)
        if progress is None:
            progress = _TaskProgress(
                task_id=task_id,
                agent_name=agent_name,
                attempt_id=attempt_id,
                gateway_session_id=gateway_session_id,
                status=status,
            )
            self._active[key] = progress
        else:
            progress.status = status
            progress.gateway_session_id = gateway_session_id or progress.gateway_session_id
        return progress

    def _complete_task(
        self,
        context: Mapping[str, object],
        payload: Mapping[str, object],
        *,
        status: str,
    ) -> str:
        key = _progress_key(context)
        task_id = str(context.get("task_id") or "")
        agent_name = str(context.get("agent_name") or "")
        progress = self._active.pop(
            key,
            _TaskProgress(task_id=task_id, agent_name=agent_name),
        )
        score = _optional_float(payload.get("score"))
        if score is None:
            score = progress.score
        if score is not None:
            self._scores[key] = score
        logs = progress.kgb_logs
        if logs is not None:
            self._finished_kgb_logs += logs
        if status == "completed":
            self._completed += 1
        else:
            self._failed += 1
        message = _message(payload.get("message")) if status != "completed" else ""
        task_label = _task_label(progress)
        self._recent.append((task_label, status, score, logs, message))
        self._recalculate_kgb_total()
        if status != "completed":
            return self._failure_line(task_label, payload)
        if self.mode != "verbose":
            return ""
        return (
            f"< done {task_label} status={status} score={_score(score)} "
            f"kgb_logs={_logs(logs)} progress={self._completed + self._failed}/{self._total_tasks}"
        )

    def _refresh_activity(self) -> None:
        if self.activity_snapshots is not None:
            try:
                snapshots = list(self.activity_snapshots())
            except Exception as exc:
                with self._lock:
                    self._last_poll_error = exc.__class__.__name__
                return
            now = time.time()
            with self._lock:
                for snapshot in snapshots:
                    key = _progress_key(snapshot.context.to_fields())
                    progress = self._active.get(key)
                    if progress is None:
                        progress = next(
                            (
                                item
                                for item in self._active.values()
                                if item.gateway_session_id == snapshot.gateway_session_id
                            ),
                            None,
                        )
                    if progress is None:
                        continue
                    self._apply_activity(progress, _activity_from_snapshot(snapshot), now=now)
                self._last_poll_error = ""
                self._recalculate_kgb_total()
            return
        self._poll_legacy_kgb_counts()

    def _poll_legacy_kgb_counts(self) -> None:
        with self._lock:
            active_task_ids = [progress.task_id for progress in self._active.values()]
        try:
            counts = dict(self.kgb_log_counts(active_task_ids)) if self.kgb_log_counts and active_task_ids else {}
            total = self.kgb_total_logs() if self.kgb_total_logs else None
        except Exception as exc:
            with self._lock:
                self._last_poll_error = exc.__class__.__name__
            return
        with self._lock:
            for progress in self._active.values():
                if progress.task_id in counts:
                    progress.kgb_logs = _int(counts[progress.task_id])
            if total is not None:
                self._kgb_total = _int(total)
            else:
                self._recalculate_kgb_total()
            self._last_poll_error = ""

    def _apply_activity(self, progress: _TaskProgress, activity: _ActivityView, *, now: float) -> None:
        previous = progress.kgb_logs or 0
        if activity.request_count > previous:
            progress.activity_delta = activity.request_count - previous
            progress.activity_delta_at = now
        progress.activity = activity
        progress.kgb_logs = activity.request_count

    def _recalculate_kgb_total(self) -> None:
        active_total = sum(progress.kgb_logs or 0 for progress in self._active.values())
        self._kgb_total = max(self._kgb_total, self._finished_kgb_logs + active_total)

    def _status_line(self) -> str:
        if self.mode == "verbose":
            return self._verbose_status_line()
        return self._compact_status_line()

    def _compact_status_line(self) -> str:
        done = self._completed + self._failed
        phases = self._phase_counts_text()
        score_text = self._score_summary_text()
        elapsed = _elapsed(time.time() - self._started_at) if self._started_at else "0s"
        queue_depth = max(self._queued - self._admitted, 0)
        poll_error = f" kgb_poll_error={self._last_poll_error}" if self._last_poll_error else ""
        return (
            f"-- KGB logs={self._kgb_total} slots active={len(self._active)}/{self._concurrency} "
            f"phases {phases} done={done}/{self._total_tasks} failed={self._failed}"
            f"{score_text} queue={queue_depth} elapsed={elapsed}{poll_error}"
        )

    def _verbose_status_line(self) -> str:
        done = self._completed + self._failed
        pct = 100.0 * done / self._total_tasks if self._total_tasks else 0.0
        active = self._active_text()
        recent = self._recent_text()
        score_text = self._score_summary_text()
        elapsed = _elapsed(time.time() - self._started_at) if self._started_at else "0s"
        failed = f" failed={self._failed}" if self._failed else ""
        poll_error = f" kgb_poll_error={self._last_poll_error}" if self._last_poll_error else ""
        return (
            f"-- running={len(self._active)} [{active}] "
            f"done={done}/{self._total_tasks} ({pct:.0f}%){failed} "
            f"kgb_logs={self._kgb_total}{score_text} recent=[{recent}] "
            f"elapsed={elapsed}{poll_error}"
        )

    def _final_line(self, status: str) -> str:
        done = self._completed + self._failed
        score = self._score_summary_text()
        app_log = f" log={self.application_log_path}" if self.application_log_path else ""
        native = f" native_stderr={self.native_stderr_path}" if self.native_stderr_path else ""
        return (
            f"== run {status} done={done}/{self._total_tasks} "
            f"completed={self._completed} failed={self._failed} "
            f"KGB logs={self._kgb_total}{score}{app_log}{native}"
        )

    def _retry_line(self, context: Mapping[str, object], payload: Mapping[str, object]) -> str:
        task_id = str(context.get("task_id") or "")
        agent_name = str(context.get("agent_name") or "")
        return (
            f"! retry {_identity_label(agent_name, task_id)} "
            f"round={_int(payload.get('run_round'))} "
            f"reason={str(payload.get('retry_reason') or '-')} "
            f"kind={str(payload.get('kind') or '-')} "
            f"component={str(payload.get('component') or '-')} "
            f"message={_message(payload.get('message'))}"
        )

    def _failure_line(self, task_id: str, payload: Mapping[str, object]) -> str:
        kind = str(payload.get("kind") or "-")
        component = str(payload.get("component") or "-")
        retryable = payload.get("retryable")
        retryable_text = f" retryable={str(bool(retryable)).lower()}" if retryable is not None else ""
        return (
            f"! failed {task_id} status=failed kind={kind} component={component}"
            f"{retryable_text} message={_message(payload.get('message'))} "
            f"progress={self._completed + self._failed}/{self._total_tasks}"
        )

    def _active_text(self) -> str:
        rows = [
            f"{_task_label(progress)}:{progress.status}:logs={_logs(progress.kgb_logs)}"
            for progress in list(self._active.values())[: self.max_active_tasks]
        ]
        remaining = len(self._active) - len(rows)
        if remaining > 0:
            rows.append(f"+{remaining}")
        return ", ".join(rows) if rows else "-"

    def _recent_text(self) -> str:
        rows = [
            f"{task_id}:{status}:score={_score(score)}:logs={_logs(logs)}"
            for task_id, status, score, logs, _message_text in self._recent
        ]
        return ", ".join(rows) if rows else "-"

    def _score_summary_text(self) -> str:
        if not self._scores:
            return " score=-"
        values = list(self._scores.values())
        return f" score={sum(values) / len(values):.4f}({len(values)})"

    def _phase_counts_text(self) -> str:
        counts = {phase: 0 for phase in _PHASES}
        for progress in self._active.values():
            if progress.status in counts:
                counts[progress.status] += 1
        return " ".join(f"{phase}={counts[phase]}" for phase in _PHASES)

    def _refresh_live(self) -> None:
        live = self._live
        if live is not None:
            live.update(self._render_rich(), refresh=True)

    def _render_rich(self) -> RenderableType:
        with self._lock:
            done = self._completed + self._failed
            total = self._total_tasks
            header = Text()
            header.append("ANexus", style="bold cyan")
            header.append(f"  {_cell(self._run_id)}", style="bold")
            header.append(f"  {self._agent_name or '-'}", style="dim")
            header.append(f"  concurrency {self._concurrency}", style="dim")

            progress = Progress(
                TextColumn("Tasks"),
                BarColumn(bar_width=None),
                TextColumn("{task.completed:.0f}/{task.total:.0f}"),
                TimeElapsedColumn(),
                expand=True,
            )
            progress.add_task("run", total=max(total, 1), completed=min(done, max(total, 1)))
            stats = Text(
                f"active {len(self._active)}/{self._concurrency}  failed {self._failed}  "
                f"requests {self._kgb_total}{self._score_summary_text()}  "
                f"queue {max(self._queued - self._admitted, 0)}",
                style="dim",
            )
            phase_counts = [
                f"{phase} {count}"
                for phase in _PHASES
                if (count := sum(item.status == phase for item in self._active.values()))
            ]
            renderables: list[RenderableType] = [header]
            if self._trace_viewer_url:
                viewer = Text()
                viewer.append(
                    "Open Trace Viewer",
                    style=f"link {self._trace_viewer_url} underline cyan",
                )
                renderables.append(viewer)
            renderables.extend((progress, stats))
            if phase_counts:
                renderables.append(Text("phases  " + "  ".join(phase_counts), style="dim"))
            if self._active:
                renderables.append(self._activity_table())
            failures = [row for row in self._recent if row[1] != "completed"]
            if failures:
                renderables.append(self._failure_table(failures))
            paths = []
            if self.application_log_path:
                paths.append(f"log {self.application_log_path}")
            if self.native_stderr_path:
                paths.append(f"native {self.native_stderr_path}")
            if paths:
                renderables.append(Text("  ".join(paths), style="dim"))
            return Group(*renderables)

    def _activity_table(self) -> Table:
        width = self._console.size.width if self._console is not None else 120
        table = Table(box=None, padding=(0, 1), expand=True)
        table.add_column("Task", overflow="ellipsis", no_wrap=True, ratio=3)
        table.add_column("Phase", width=9)
        table.add_column("LLM", width=7)
        table.add_column("Requests", justify="right", width=8)
        show_delta = width >= 90
        show_latest = width >= 110
        if show_delta:
            table.add_column("Delta", justify="right", width=5)
        table.add_column("Last", justify="right", width=7)
        if show_latest:
            table.add_column("Latest", justify="right", width=12)
        now = datetime.now(UTC)
        rows = list(self._active.values())[: self.max_active_tasks]
        for progress in rows:
            activity = progress.activity
            cells: list[RenderableType] = [_task_label(progress), progress.status]
            cells.append(_activity_status(activity, now=now))
            cells.append(str(progress.kgb_logs or 0))
            if show_delta:
                delta = (
                    f"+{progress.activity_delta}"
                    if progress.activity_delta and time.time() - progress.activity_delta_at <= 10
                    else "-"
                )
                cells.append(Text(delta, style="green" if delta != "-" else "dim"))
            cells.append(_last_request_text(activity, now=now))
            if show_latest:
                cells.append(_latest_text(activity))
            table.add_row(*cells)
        remaining = len(self._active) - len(rows)
        if remaining > 0:
            table.add_row(f"+ {remaining} more", "", "", "", *("" for _ in range(len(table.columns) - 4)))
        return table

    def _failure_table(
        self,
        failures: list[tuple[str, str, float | None, int | None, str]],
    ) -> Table:
        table = Table(title="Recent failures", title_style="bold red", box=None, expand=True)
        table.add_column("Task", overflow="ellipsis", no_wrap=True, ratio=1)
        table.add_column("Message", overflow="ellipsis", ratio=3)
        for task_id, _status, _score_value, _logs_value, message in failures[-3:]:
            table.add_row(task_id, message)
        return table

    def _print(self, line: str) -> None:
        print(line, file=self.stream, flush=True)


def _progress_key(context: Mapping[str, object]) -> ProgressKey:
    return (
        str(context.get("run_id") or ""),
        str(context.get("agent_name") or ""),
        str(context.get("task_id") or ""),
        str(context.get("attempt_id") or ""),
    )


def _task_label(progress: _TaskProgress) -> str:
    return _identity_label(progress.agent_name, progress.task_id)


def _identity_label(agent_name: str, task_id: str) -> str:
    return f"{agent_name}/{task_id}" if agent_name else task_id


def _activity_from_snapshot(snapshot: GatewayActivitySnapshot) -> _ActivityView:
    return _ActivityView(
        available=snapshot.available,
        request_count=snapshot.request_count,
        last_request_time=snapshot.last_request_time,
        response_status=snapshot.response_status,
        total_time_ms=snapshot.total_time_ms,
        observed_at=snapshot.observed_at,
    )


def _activity_from_payload(payload: Mapping[str, object]) -> _ActivityView:
    return _ActivityView(
        available=bool(payload.get("available", True)),
        request_count=_int(payload.get("request_count")),
        last_request_time=str(payload.get("last_request_time") or ""),
        response_status=_optional_int(payload.get("response_status")),
        total_time_ms=_optional_int(payload.get("total_time_ms")),
        observed_at=str(payload.get("observed_at") or ""),
    )


def _activity_status(activity: _ActivityView | None, *, now: datetime) -> Text:
    if activity is None or not activity.available:
        return Text("unknown", style="yellow")
    if activity.request_count == 0:
        return Text("none", style="dim")
    age = _request_age(activity, now=now)
    if age is not None and age <= 10:
        return Text("recent", style="green")
    return Text("quiet", style="cyan")


def _last_request_text(activity: _ActivityView | None, *, now: datetime) -> str:
    age = _request_age(activity, now=now)
    return _elapsed(age) if age is not None else "-"


def _request_age(activity: _ActivityView | None, *, now: datetime) -> float | None:
    if activity is None or not activity.last_request_time:
        return None
    try:
        requested_at = datetime.fromisoformat(activity.last_request_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    if requested_at.tzinfo is None:
        requested_at = requested_at.replace(tzinfo=UTC)
    return max((now - requested_at.astimezone(UTC)).total_seconds(), 0.0)


def _latest_text(activity: _ActivityView | None) -> Text:
    if activity is None or not activity.available:
        return Text("-", style="dim")
    status = str(activity.response_status) if activity.response_status is not None else "-"
    latency = f"{activity.total_time_ms / 1000:.1f}s" if activity.total_time_ms is not None else "-"
    style = "red" if activity.response_status is not None and activity.response_status >= 400 else ""
    return Text(f"{status} / {latency}", style=style)


def _is_terminal(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def _int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _logs(value: int | None) -> str:
    return "?" if value is None else str(value)


def _cell(value: str) -> str:
    return value if value else "-"


def _message(value: object, *, max_len: int = 160) -> str:
    text = " ".join(str(value or "-").split())
    return text if len(text) <= max_len else f"{text[: max_len - 1]}..."


def _elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
