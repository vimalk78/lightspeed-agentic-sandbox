"""DeepAgents middleware that gates model-visible tool results."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from langchain.agents.middleware import AgentMiddleware


class ToolResultInspector(Protocol):
    def __call__(
        self,
        tool_name: str,
        result_type: str,
        value: Any,
    ) -> Awaitable[Any]: ...


class ToolResultSafetyInspectionFailed(RuntimeError):  # noqa: N818
    """Stable termination marker for a rejected or uninspectable tool result."""

    def __init__(self) -> None:
        super().__init__("ToolResultSafetyInspectionFailed")


class ToolResultInspectionMiddleware(AgentMiddleware[Any, Any, Any]):
    """Inspect tool output after execution and before DeepAgents delivery."""

    def __init__(self, inspector: ToolResultInspector) -> None:
        self._inspector = inspector

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        tool_name = request.tool_call.get("name", "")
        try:
            result = await handler(request)
        except Exception as exc:
            await self._inspect_or_fail(tool_name, "error", str(exc))
            raise

        content = getattr(result, "content", None)
        if content is None:
            return result
        result_type = "error" if getattr(result, "status", "success") == "error" else "result"
        await self._inspect_or_fail(tool_name, result_type, content)
        return result

    async def _inspect_or_fail(self, tool_name: str, result_type: str, value: Any) -> None:
        try:
            outcome = await self._inspector(tool_name, result_type, value)
        except ToolResultSafetyInspectionFailed:
            raise
        except Exception as exc:
            raise ToolResultSafetyInspectionFailed() from exc

        if getattr(outcome, "passed", True) is False:
            raise ToolResultSafetyInspectionFailed()
