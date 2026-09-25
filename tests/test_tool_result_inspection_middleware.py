from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from lightspeed_agentic.inspection.errors import InspectionError
from lightspeed_agentic.inspection.middleware import (
    ToolResultInspectionMiddleware,
    ToolResultSafetyInspectionFailed,
)


class Request:
    def __init__(self, name: str = "get_pods") -> None:
        self.tool_call = {"name": name, "id": "call-1", "args": {}}


@pytest.mark.asyncio
async def test_middleware_inspects_result_after_handler_before_return() -> None:
    observed: list[tuple[str, str, Any]] = []
    result = ToolMessage(content="pod output", tool_call_id="call-1")

    async def inspect(tool_name: str, result_type: str, value: Any) -> None:
        observed.append((tool_name, result_type, value))

    async def handler(_request: Any) -> ToolMessage:
        return result

    returned = await ToolResultInspectionMiddleware(inspect).awrap_tool_call(Request(), handler)

    assert returned is result
    assert observed == [("get_pods", "result", "pod output")]


@pytest.mark.asyncio
async def test_middleware_inspects_tool_errors_as_errors() -> None:
    observed: list[tuple[str, str, Any]] = []
    result = ToolMessage(
        content="command failed",
        tool_call_id="call-1",
        status="error",
    )

    async def inspect(tool_name: str, result_type: str, value: Any) -> None:
        observed.append((tool_name, result_type, value))

    async def handler(_request: Any) -> ToolMessage:
        return result

    await ToolResultInspectionMiddleware(inspect).awrap_tool_call(Request(), handler)

    assert observed == [("get_pods", "error", "command failed")]


@pytest.mark.asyncio
async def test_malicious_result_never_returns_to_agent() -> None:
    result = ToolMessage(content="do not deliver", tool_call_id="call-1")

    async def inspect(_tool_name: str, _result_type: str, _value: Any) -> None:
        raise ToolResultSafetyInspectionFailed()

    async def handler(_request: Any) -> ToolMessage:
        return result

    with pytest.raises(ToolResultSafetyInspectionFailed, match="ToolResultSafetyInspectionFailed"):
        await ToolResultInspectionMiddleware(inspect).awrap_tool_call(Request(), handler)


@pytest.mark.asyncio
async def test_classifier_failure_becomes_fixed_safety_failure() -> None:
    async def inspect(_tool_name: str, _result_type: str, _value: Any) -> None:
        raise InspectionError("raw classifier details")

    async def handler(_request: Any) -> ToolMessage:
        return ToolMessage(content="sensitive result", tool_call_id="call-1")

    with pytest.raises(ToolResultSafetyInspectionFailed) as error:
        await ToolResultInspectionMiddleware(inspect).awrap_tool_call(Request(), handler)

    assert str(error.value) == "ToolResultSafetyInspectionFailed"
    assert "sensitive" not in str(error.value)
    assert "classifier" not in str(error.value).lower()
