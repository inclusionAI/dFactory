"""Retry policy for task-attempt operations."""

from __future__ import annotations

from dataclasses import dataclass

from anexus.core.errors import ErrorEnvelope, ErrorKind
from anexus.tasks.spec import AttemptPolicy


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    reason: str


class RetryPolicy:
    """Decide retries from structured errors, not exception strings."""

    def __init__(self, attempt_policy: AttemptPolicy):
        self.attempt_policy = attempt_policy

    def decide(self, error: ErrorEnvelope, *, failures_seen: int) -> RetryDecision:
        if failures_seen > self.attempt_policy.max_retries:
            return RetryDecision(False, "max_retries_exhausted")
        if error.kind == ErrorKind.CANCELLED:
            return RetryDecision(False, "cancelled")
        if not error.retryable:
            return RetryDecision(False, "error_not_retryable")
        return RetryDecision(True, "retryable_error")

    def wait_seconds(self, *, failures_seen: int) -> float:
        """Return the configured backoff delay before the next retry."""

        if self.attempt_policy.min_retry_wait_seconds <= 0:
            return 0.0
        retry_index = max(failures_seen - 1, 0)
        wait_seconds = self.attempt_policy.min_retry_wait_seconds * (
            self.attempt_policy.retry_wait_multiplier**retry_index
        )
        return min(wait_seconds, self.attempt_policy.max_retry_wait_seconds)
