"""LLM gateway request and response format conversions."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from anexus.core.ids import new_id

__all__ = [
    "anthropic_to_openai_chat",
    "openai_chat_to_anthropic_message",
    "wrap_anthropic_message_as_sse",
    "wrap_openai_chat_as_sse",
]


def anthropic_to_openai_chat(request: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic Messages request into OpenAI Chat Completions."""

    messages: list[dict[str, Any]] = []
    system_text = _anthropic_system_text(request.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for message in request.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        if role == "assistant":
            messages.append(_anthropic_assistant_to_openai(message))
        else:
            messages.extend(_anthropic_user_to_openai(message))

    converted: dict[str, Any] = {
        "model": request.get("model", "auto"),
        "messages": messages,
        "max_completion_tokens": request.get("max_tokens", 4096),
        "temperature": request.get("temperature", 1.0),
        "top_p": request.get("top_p", 1.0),
    }
    if request.get("stop_sequences"):
        converted["stop"] = request["stop_sequences"]
    tools = _anthropic_tools_to_openai(request.get("tools"))
    if tools:
        converted["tools"] = tools
    tool_choice = _anthropic_tool_choice_to_openai(request.get("tool_choice"))
    if tool_choice is not None:
        converted["tool_choice"] = tool_choice
    return converted


def openai_chat_to_anthropic_message(
    response: dict[str, Any],
    *,
    original_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an OpenAI Chat Completion response into Anthropic Messages format."""

    if hasattr(response, "model_dump"):
        response = response.model_dump()
    original_request = original_request or {}
    choices = response.get("choices", [])
    if not choices:
        return {
            "type": "error",
            "error": {"type": "api_error", "message": "OpenAI response has no choices"},
        }

    choice = choices[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    content: list[dict[str, Any]] = []

    reasoning = message.get("reasoning_content")
    if reasoning:
        content.append({"type": "thinking", "thinking": str(reasoning)})

    text = message.get("content")
    if text:
        content.append({"type": "text", "text": str(text)})

    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        content.append(
            {
                "type": "tool_use",
                "id": str(tool_call.get("id") or new_id("tool")),
                "name": str(function.get("name", "")),
                "input": arguments,
            }
        )

    if not content:
        content.append({"type": "text", "text": ""})

    usage = response.get("usage") or {}
    return {
        "id": str(response.get("id") or new_id("msg")),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": str(response.get("model") or original_request.get("model") or "unknown"),
        "stop_reason": _openai_finish_reason_to_anthropic(
            str(choice.get("finish_reason") or "stop"),
            has_tool_calls=bool(tool_calls),
        ),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        },
    }


def wrap_openai_chat_as_sse(response: dict[str, Any]) -> Iterator[str]:
    """Yield OpenAI-compatible SSE chunks for a full chat completion response."""

    choices = response.get("choices", [])
    if not choices:
        yield "data: [DONE]\n\n"
        return

    choice = choices[0]
    message = choice.get("message") or {}
    chunk_id = str(response.get("id") or new_id("chatcmpl"))
    model = str(response.get("model", "unknown"))
    created = int(response.get("created", int(time.time())))

    yield _openai_sse_chunk(
        chunk_id=chunk_id,
        created=created,
        model=model,
        delta={"role": "assistant"},
    )
    if message.get("reasoning_content"):
        yield _openai_sse_chunk(
            chunk_id=chunk_id,
            created=created,
            model=model,
            delta={"reasoning_content": message["reasoning_content"]},
        )
    if message.get("content"):
        yield _openai_sse_chunk(
            chunk_id=chunk_id,
            created=created,
            model=model,
            delta={"content": message["content"]},
        )
    for index, tool_call in enumerate(message.get("tool_calls") or []):
        function = tool_call.get("function") or {}
        yield _openai_sse_chunk(
            chunk_id=chunk_id,
            created=created,
            model=model,
            delta={
                "tool_calls": [
                    {
                        "index": index,
                        "id": str(tool_call.get("id") or new_id("call")),
                        "type": "function",
                        "function": {
                            "name": str(function.get("name", "")),
                            "arguments": function.get("arguments", "{}"),
                        },
                    }
                ]
            },
        )
    yield _openai_sse_chunk(
        chunk_id=chunk_id,
        created=created,
        model=model,
        delta={},
        finish_reason=str(choice.get("finish_reason") or "stop"),
    )
    yield "data: [DONE]\n\n"


def wrap_anthropic_message_as_sse(message: dict[str, Any]) -> Iterator[str]:
    """Yield Anthropic-compatible SSE events for a full Messages response."""

    message_id = str(message.get("id") or new_id("msg"))
    model = str(message.get("model", "unknown"))
    usage = message.get("usage") or {}
    content = message.get("content") or []
    yield _anthropic_sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": int(usage.get("input_tokens", 0)), "output_tokens": 0},
            },
        },
    )
    yield _anthropic_sse("ping", {"type": "ping"})
    for index, block in enumerate(content):
        block_type = str(block.get("type", "text"))
        if block_type == "thinking":
            yield from _anthropic_text_block(
                index=index,
                block_type="thinking",
                delta_type="thinking_delta",
                field="thinking",
                value=str(block.get("thinking", "")),
            )
        elif block_type == "tool_use":
            yield from _anthropic_tool_block(index=index, block=block)
        else:
            yield from _anthropic_text_block(
                index=index,
                block_type="text",
                delta_type="text_delta",
                field="text",
                value=str(block.get("text", "")),
            )
    yield _anthropic_sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": message.get("stop_reason", "end_turn"),
                "stop_sequence": message.get("stop_sequence"),
            },
            "usage": usage,
        },
    )
    yield _anthropic_sse("message_stop", {"type": "message_stop"})


def _anthropic_system_text(system: object) -> str:
    if isinstance(system, str):
        return system.strip()
    if isinstance(system, list):
        parts = [
            str(block.get("text", "")) for block in system if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n\n".join(part for part in parts if part).strip()
    return ""


def _anthropic_user_to_openai(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if content is None:
        return [{"role": "user", "content": ""}]
    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    user_blocks: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            user_blocks.append({"type": "text", "text": str(block.get("text", ""))})
        elif block_type == "image":
            image_url = _anthropic_image_url(block)
            if image_url:
                user_blocks.append({"type": "image_url", "image_url": {"url": image_url}})
        elif block_type == "tool_result":
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id", "")),
                    "content": _tool_result_content(block.get("content", "")),
                }
            )

    messages: list[dict[str, Any]] = []
    if len(user_blocks) == 1 and user_blocks[0]["type"] == "text":
        messages.append({"role": "user", "content": user_blocks[0]["text"]})
    elif user_blocks:
        messages.append({"role": "user", "content": user_blocks})
    messages.extend(tool_messages)
    return messages or [{"role": "user", "content": ""}]


def _anthropic_assistant_to_openai(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if content is None or isinstance(content, str):
        return {"role": "assistant", "content": content}

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "thinking":
            thinking_parts.append(str(block.get("thinking", "")))
        elif block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or new_id("call")),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                }
            )

    converted: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if thinking_parts:
        converted["reasoning_content"] = "\n".join(thinking_parts)
    if tool_calls:
        converted["tool_calls"] = tool_calls
    return converted


def _anthropic_tools_to_openai(tools: object) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or not str(tool.get("name", "")).strip():
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool.get("name", "")),
                    "description": str(tool.get("description", "")),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return converted


def _anthropic_tool_choice_to_openai(choice: object) -> object | None:
    if not isinstance(choice, dict):
        return None
    choice_type = choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": str(choice["name"])}}
    return None


def _anthropic_image_url(block: dict[str, Any]) -> str:
    source = block.get("source") or {}
    if not isinstance(source, dict) or source.get("type") != "base64":
        return ""
    media_type = str(source.get("media_type") or "image/png")
    data = str(source.get("data") or "")
    return f"data:{media_type};base64,{data}" if data else ""


def _tool_result_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    return str(content)


def _openai_finish_reason_to_anthropic(reason: str, *, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_use"
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }.get(reason, "end_turn")


def _openai_sse_chunk(
    *,
    chunk_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    return (
        "data: "
        + json.dumps(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            },
            ensure_ascii=False,
        )
        + "\n\n"
    )


def _anthropic_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _anthropic_text_block(
    *,
    index: int,
    block_type: str,
    delta_type: str,
    field: str,
    value: str,
) -> Iterator[str]:
    yield _anthropic_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": block_type, field: ""},
        },
    )
    yield _anthropic_sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": delta_type, field: value},
        },
    )
    yield _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": index})


def _anthropic_tool_block(index: int, block: dict[str, Any]) -> Iterator[str]:
    yield _anthropic_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": str(block.get("id", "")),
                "name": str(block.get("name", "")),
                "input": {},
            },
        },
    )
    yield _anthropic_sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(block.get("input", {}), ensure_ascii=False),
            },
        },
    )
    yield _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": index})
