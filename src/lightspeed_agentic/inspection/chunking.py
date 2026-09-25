"""Serialization-aware chunking for complete tool-result inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

CHUNK_OVERLAP_TOKENS = 256


class Utf8ByteCodec:
    """Conservative fallback codec used when the provider exposes no tokenizer."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")


class TokenCodec(Protocol):
    """Token encoder/decoder for the active classifier model."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


@dataclass(frozen=True)
class ToolResultChunk:
    content: str
    chunk_index: int
    chunk_count: int


def serialize_tool_result(value: Any) -> str:
    """Serialize the effective tool result before it is split for inspection."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("tool result bytes must be valid UTF-8") from exc
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("tool result is not JSON serializable") from exc


def chunk_tool_result(
    value: Any,
    *,
    codec: TokenCodec,
    context_window_tokens: int,
    instruction_tokens: int,
    output_tokens: int,
) -> list[ToolResultChunk]:
    """Serialize and split a complete result with the required overlap."""
    serialized = serialize_tool_result(value)
    capacity = context_window_tokens - instruction_tokens - output_tokens
    if capacity <= CHUNK_OVERLAP_TOKENS:
        raise ValueError("classifier context budget must exceed the 256-token overlap")

    tokens = codec.encode(serialized)
    if not tokens:
        return [ToolResultChunk(content="", chunk_index=0, chunk_count=1)]

    raw_chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + capacity, len(tokens))
        raw_chunks.append(codec.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - CHUNK_OVERLAP_TOKENS

    count = len(raw_chunks)
    return [
        ToolResultChunk(content=content, chunk_index=index, chunk_count=count)
        for index, content in enumerate(raw_chunks)
    ]
