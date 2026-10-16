"""AKernel exception normalization."""

from __future__ import annotations

from anexus.core.errors import AnexusError, ErrorKind

_TRANSIENT_CODES = {4005, 3002, 3006}
_FATAL_CODES = {1003, 1007, 9000}
_RESOURCE_CODES = {1002, 4306}
_NETWORK_TRANSIENT_MARKERS = (
    "Connection refused",
    "Connection reset by peer",
    "Connection timed out",
    "DeadlineExceeded",
    "ResourceExhausted",
    "Unavailable",
    "failed to handshake",
    "handshake error",
    "transport error",
)
_REMOTE_GONE_MARKERS = (
    "instance not exist",
    "instance is not found",
    "instance not found",
    "instance may have been killed",
    "not created or killed",
)


def classify_akernel_error(exc: BaseException, *, operation: str = "") -> AnexusError:
    """Map an AKernel or transport exception to a stable operational error."""

    code = _yr_error_code(exc)
    message = str(exc)
    remote_gone = any(marker in message.lower() for marker in _REMOTE_GONE_MARKERS)
    transient = any(marker in message for marker in _NETWORK_TRANSIENT_MARKERS)
    details: dict[str, object] = {
        "operation": operation,
        "exception_type": exc.__class__.__name__,
    }
    if code is not None:
        details["yr_error_code"] = code

    operation_name = operation or "operation"
    if remote_gone:
        details["reason"] = "remote_gone"
        if operation == "delete":
            kind = ErrorKind.SANDBOX_ERROR
            error_message = "akernel sandbox is already gone during delete"
            retryable = False
        else:
            kind = ErrorKind.INFRASTRUCTURE_ERROR
            error_message = f"akernel sandbox disappeared during {operation_name}"
            retryable = True
    elif code in _RESOURCE_CODES:
        kind = ErrorKind.RESOURCE_EXHAUSTED
        error_message = f"akernel resource unavailable during {operation_name}"
        retryable = True
    elif code in _TRANSIENT_CODES or transient:
        kind = ErrorKind.INFRASTRUCTURE_ERROR
        error_message = f"akernel transient failure during {operation_name}"
        retryable = True
    elif code in _FATAL_CODES:
        kind = ErrorKind.SANDBOX_ERROR
        error_message = f"akernel sandbox is no longer available during {operation_name}"
        retryable = False
    else:
        kind = ErrorKind.SANDBOX_ERROR
        error_message = f"akernel operation failed during {operation_name}"
        retryable = False
        details["message"] = message

    return AnexusError(
        error_message,
        kind=kind,
        component="sandbox.akernel",
        retryable=retryable,
        details=details,
    )


def delete_failure_means_remote_gone(exc: AnexusError) -> bool:
    """Return whether a failed delete confirms the remote sandbox is gone."""

    if (
        exc.component != "sandbox.akernel"
        or exc.details.get("operation") != "delete"
        or exc.kind != ErrorKind.SANDBOX_ERROR
    ):
        return False
    code = exc.details.get("yr_error_code")
    return exc.details.get("reason") == "remote_gone" or (isinstance(code, int) and code in _FATAL_CODES)


def _yr_error_code(exc: BaseException) -> int | None:
    """Read a structured YR error code through SDK exception wrappers."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for value in (
            getattr(current, "code", None),
            getattr(current, "error_code", None),
        ):
            if isinstance(value, int):
                return value
            enum_value = getattr(value, "value", None)
            if isinstance(enum_value, int):
                return enum_value
        explicit_cause = getattr(current, "cause", None)
        current = (
            explicit_cause if isinstance(explicit_cause, BaseException) else current.__cause__ or current.__context__
        )
    return None
