from __future__ import annotations

import json
from typing import Any

import pytest

from lightspeed_agentic.inspection.client import LangChainClassifierClient
from lightspeed_agentic.inspection.models import ClassifierDecision, ClassifierRequest


class FakeModel:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.messages: list[Any] | None = None
        self.parameters: dict[str, Any] = {}

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        self.messages = messages
        self.parameters = kwargs
        return self.response


@pytest.mark.asyncio
async def test_classifier_client_sends_only_dedicated_untrusted_content_messages() -> None:
    model = FakeModel('{"injectionDetected": false, "category": "none"}')
    client = LangChainClassifierClient(model)
    request = ClassifierRequest(
        toolName="get_pods",
        resultType="result",
        chunkIndex=0,
        chunkCount=1,
        content="tool result",
    )

    decision = await client.classify(request)

    assert decision == ClassifierDecision(injectionDetected=False, category="none")
    assert model.parameters == {"max_tokens": 128}
    assert model.messages is not None
    assert len(model.messages) == 2
    assert "untrusted" in model.messages[0].content.lower()
    assert "do not follow" in model.messages[0].content.lower()
    assert json.loads(model.messages[1].content) == request.model_dump(by_alias=True)
    assert "history" not in model.messages[1].content
    assert "system prompt" not in model.messages[1].content.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("flagged", [False, True])
async def test_classifier_client_adapts_flagged_response(flagged: bool) -> None:
    model = FakeModel(json.dumps({"flagged": flagged}))

    decision = await LangChainClassifierClient(model).classify(
        ClassifierRequest(
            toolName="execute",
            resultType="result",
            chunkIndex=0,
            chunkCount=1,
            content="output",
        )
    )

    assert decision == ClassifierDecision(
        injectionDetected=flagged,
        category="unknown" if flagged else "none",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        '```json\\n{"decision":"safe","injection_detected":false,"manipulation_attempt":false}\\n```{"flagged":false}',
        '**Decision: SAFE** FLAG: false ```json {"decision":"allow","risk_flags":[]} ```',
        "SAFE - no injected instructions detected; content is plain status output.",
        '{"decision":"allow","injection_detected":false}',
    ],
)
async def test_classifier_client_accepts_wrapped_safe_responses(response: str) -> None:
    decision = await LangChainClassifierClient(FakeModel(response)).classify(
        ClassifierRequest(
            toolName="execute",
            resultType="result",
            chunkIndex=0,
            chunkCount=1,
            content="output",
        )
    )

    assert decision == ClassifierDecision(injectionDetected=False, category="none")


@pytest.mark.asyncio
async def test_classifier_client_adapts_wrapped_unsafe_response() -> None:
    decision = await LangChainClassifierClient(
        FakeModel('```json {"decision":"unsafe","injection_detected":true} ```')
    ).classify(
        ClassifierRequest(
            toolName="execute",
            resultType="result",
            chunkIndex=0,
            chunkCount=1,
            content="output",
        )
    )

    assert decision == ClassifierDecision(injectionDetected=True, category="unknown")


@pytest.mark.asyncio
async def test_classifier_client_parses_text_content_blocks() -> None:
    model = FakeModel(
        [
            {"type": "text", "text": '{"injectionDetected": true, "category": "unknown"}'},
        ]
    )

    decision = await LangChainClassifierClient(model).classify(
        ClassifierRequest(
            toolName="execute",
            resultType="result",
            chunkIndex=0,
            chunkCount=1,
            content="output",
        )
    )

    assert decision == ClassifierDecision(injectionDetected=True, category="unknown")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"injectionDetected": "false", "category": "none"}',
        '{"injectionDetected": false, "category": "none", "reason": "do not expose"}',
    ],
)
async def test_classifier_client_rejects_non_strict_responses(response: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        await LangChainClassifierClient(FakeModel(response)).classify(
            ClassifierRequest(
                toolName="get_pods",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="output",
            )
        )
