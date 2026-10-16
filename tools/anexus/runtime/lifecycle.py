"""Parent-child process lifecycle pipes."""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from multiprocessing.connection import Connection


def create_lifecycle_pipe(
    context: mp.context.BaseContext | None = None,
) -> tuple[Connection, Connection]:
    """Create a parent-child lifecycle pipe.

    The parent keeps the first connection and passes the second connection to
    the child. Parent messages and parent-side EOF both tell the child to abort.
    """

    selected_context = context or mp.get_context()
    return selected_context.Pipe()


def start_lifecycle_monitor(
    lifecycle_conn: Connection,
    abort_event: threading.Event,
    *,
    stop_event: threading.Event | None = None,
    peer_conn: Connection | None = None,
    hard_exit_seconds: float = 5.0,
    poll_seconds: float = 0.2,
    exit_process: bool = True,
) -> threading.Thread:
    """Start a daemon monitor that sets ``abort_event`` on pipe message or EOF."""

    thread = threading.Thread(
        target=_lifecycle_monitor_loop,
        args=(
            lifecycle_conn,
            abort_event,
            stop_event,
            peer_conn,
            max(hard_exit_seconds, 0.0),
            max(poll_seconds, 0.01),
            exit_process,
        ),
        name="anexus-lifecycle-monitor",
        daemon=True,
    )
    thread.start()
    return thread


def _lifecycle_monitor_loop(
    lifecycle_conn: Connection,
    abort_event: threading.Event,
    stop_event: threading.Event | None,
    peer_conn: Connection | None,
    hard_exit_seconds: float,
    poll_seconds: float,
    exit_process: bool,
) -> None:
    conns = [lifecycle_conn]
    if peer_conn is not None:
        conns.append(peer_conn)

    while not abort_event.is_set():
        if stop_event is not None and stop_event.is_set():
            return
        for conn in conns:
            try:
                if not conn.poll(poll_seconds):
                    continue
                conn.recv()
            except (EOFError, OSError, ValueError):
                abort_event.set()
                break
            else:
                abort_event.set()
                break

    if not exit_process:
        return
    time.sleep(hard_exit_seconds)
    os._exit(1)
