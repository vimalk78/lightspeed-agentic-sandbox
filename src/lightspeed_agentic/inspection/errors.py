"""Safe inspection error types."""

from __future__ import annotations

import re
from typing import Any

CLASSIFIER_FAILURE_MESSAGE = (
    "Lightspeed stopped the operation because a tool result failed the safety inspection."
)


_SAFE_ERROR_TYPE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def provider_error_metadata(
    exc: BaseException,
) -> tuple[int | None, str | None, str | None]:
    """Extract bounded HTTP metadata without retaining provider response content."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int) or not 100 <= status_code <= 599:
        status_code = None

    body: Any = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    error_type = error.get("type") if isinstance(error, dict) else None
    if not isinstance(error_type, str) or _SAFE_ERROR_TYPE.fullmatch(error_type) is None:
        error_type = None

    message = error.get("message") if isinstance(error, dict) else None
    reason = None
    if isinstance(message, str):
        lowered = message.lower()
        reason = (
            "temperature_invalid"
            if "temperature" in lowered
            else "max_tokens_invalid"
            if "max_tokens" in lowered or "max tokens" in lowered
            else "tool_request_invalid"
            if "tool" in lowered
            else "model_invalid"
            if "model" in lowered
            else "message_invalid"
            if "message" in lowered or "content" in lowered
            else "invalid_request"
        )
    return status_code, error_type, reason


class InspectionError(RuntimeError):
    """A classifier failure that carries no untrusted content."""

    def __init__(
        self,
        message: str = CLASSIFIER_FAILURE_MESSAGE,
        *,
        failure_type: str = "provider_error",
        attempt_count: int = 0,
        provider_status_code: int | None = None,
        provider_error_type: str | None = None,
        provider_error_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.attempt_count = attempt_count
        self.provider_status_code = provider_status_code
        self.provider_error_type = provider_error_type
        self.provider_error_reason = provider_error_reason
