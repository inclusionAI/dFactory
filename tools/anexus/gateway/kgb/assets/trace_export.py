#!/usr/bin/env python3
"""Export and merge conversation traces from KGB llm_gateway.db.

Builds a prefix tree from API call histories, then outputs one JSONL line
per root-to-leaf path — each line is a complete conversation session.
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from api_protocol_converter import convert_request


def parse_openai_response(raw: str, is_stream: bool = True) -> dict:
    """Parse an OpenAI response into content, reasoning, tools, and usage."""

    if not raw:
        return {"content": ""}
    if not is_stream:
        return _parse_openai_json_response(raw)
    return _parse_openai_stream_response(raw)


def _parse_openai_json_response(raw: str) -> dict:
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return {"content": ""}

    message = body.get("choices", [{}])[0].get("message", {})
    result = {"content": message.get("content", "") or ""}
    reasoning = _reasoning_value(message)
    if reasoning:
        result["reasoning_content"] = reasoning
    if message.get("tool_calls"):
        result["tool_calls"] = [
            {
                "id": tool_call.get("id", ""),
                "type": "function",
                "function": {
                    "name": tool_call.get("function", {}).get("name", ""),
                    "arguments": _tool_arguments_text(tool_call.get("function", {}).get("arguments", "")),
                },
            }
            for tool_call in message["tool_calls"]
        ]
    if body.get("usage"):
        result["usage"] = body["usage"]
    return result


def _parse_openai_stream_response(raw: str) -> dict:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    usage = {}

    for chunk in _sse_json_events(raw):
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        if delta.get("content"):
            content_parts.append(delta["content"])
        reasoning = _reasoning_value(delta)
        if reasoning:
            reasoning_parts.append(reasoning)
        for tool_call in delta.get("tool_calls") or []:
            _append_openai_tool_delta(tool_calls, tool_call)

    result = {"content": "".join(content_parts)}
    if reasoning_parts:
        result["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        result["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    if usage:
        result["usage"] = usage
    return result


def _append_openai_tool_delta(tool_calls: dict[int, dict], delta: dict) -> None:
    index = delta.get("index", 0)
    tool_call = tool_calls.setdefault(
        index,
        {
            "id": delta.get("id", ""),
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if delta.get("id"):
        tool_call["id"] = delta["id"]
    function = delta.get("function", {})
    if function.get("name"):
        tool_call["function"]["name"] = function["name"]
    if "arguments" in function and function.get("arguments") is not None:
        tool_call["function"]["arguments"] += _tool_arguments_text(function.get("arguments", ""))


def parse_anthropic_response(raw: str, is_stream: bool = True) -> dict:
    """Parse an Anthropic response into content, reasoning, tools, and usage."""

    if not raw:
        return {"content": ""}
    if not is_stream:
        return _parse_anthropic_json_response(raw)
    return _parse_anthropic_stream_response(raw)


def _parse_anthropic_json_response(raw: str) -> dict:
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return {"content": ""}

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in body.get("content", []):
        block_type = block.get("type", "")
        if block_type == "text":
            content_parts.append(block.get("text", ""))
        elif block_type == "thinking":
            reasoning_parts.append(block.get("thinking", ""))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": _stable_json(block.get("input", {})),
                    },
                }
            )

    result = {"content": "".join(content_parts)}
    if reasoning_parts:
        result["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        result["tool_calls"] = tool_calls
    usage = body.get("usage", {})
    if usage:
        result["usage"] = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        }
    return result


def _parse_anthropic_stream_response(raw: str) -> dict:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    usage = {}

    for event in _sse_json_events(raw):
        event_type = event.get("type", "")
        if event_type == "message_start":
            event_usage = event.get("message", {}).get("usage", {})
            if event_usage:
                usage["prompt_tokens"] = event_usage.get("input_tokens", 0)
        elif event_type == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type", "text") == "tool_use":
                index = event.get("index", 0)
                tool_calls[index] = {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": ""},
                }
        elif event_type == "content_block_delta":
            _append_anthropic_content_delta(
                event,
                content_parts=content_parts,
                reasoning_parts=reasoning_parts,
                tool_calls=tool_calls,
            )
        elif event_type == "message_delta":
            event_usage = event.get("usage", {})
            if event_usage:
                usage["completion_tokens"] = event_usage.get("output_tokens", 0)

    result = {"content": "".join(content_parts)}
    if reasoning_parts:
        result["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        result["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    if usage:
        result["usage"] = usage
    return result


def _append_anthropic_content_delta(
    event: dict,
    *,
    content_parts: list[str],
    reasoning_parts: list[str],
    tool_calls: dict[int, dict],
) -> None:
    index = event.get("index", 0)
    delta = event.get("delta", {})
    delta_type = delta.get("type", "")
    if delta_type == "text_delta":
        content_parts.append(delta.get("text", ""))
    elif delta_type == "thinking_delta":
        reasoning_parts.append(delta.get("thinking", ""))
    elif delta_type == "input_json_delta" and index in tool_calls:
        tool_calls[index]["function"]["arguments"] += delta.get("partial_json", "")


def _sse_json_events(raw: str):
    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].lstrip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def normalize_request(rb: dict, protocol: str) -> dict:
    """Normalize any protocol's request body to OpenAI Chat format using the SDK."""
    if not protocol or protocol in ("openai_chat", "openai_classic", "openai"):
        return rb
    if protocol in ("anthropic", "anthropic_messages"):
        return _anthropic_request_to_openai_chat(rb)
    # Map KGB protocol names to converter protocol names
    protocol_map = {
        "anthropic": "anthropic_messages",
        "anthropic_messages": "anthropic_messages",
    }
    converter_protocol = protocol_map.get(protocol, protocol)
    try:
        return convert_request(converter_protocol, "openai_chat", rb)
    except Exception as exc:
        print(
            f"warning: request conversion from {converter_protocol!r} failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return rb


def _reasoning_value(message: dict) -> str:
    for key in ("reasoning_content", "reasoning", "reasoning_text", "thinking"):
        value = message.get(key)
        if value:
            return str(value)
    return ""


def _tool_arguments_text(arguments) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    return _stable_json(arguments)


def _anthropic_request_to_openai_chat(request: dict) -> dict:
    """Convert Anthropic Messages request history without dropping trace fields."""
    messages = []
    system_text = _anthropic_system_text(request.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for raw in request.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role", "user"))
        if role == "assistant":
            messages.append(_anthropic_assistant_to_openai(raw))
        else:
            messages.extend(_anthropic_user_to_openai(raw))

    converted = {
        "model": request.get("model", "auto"),
        "messages": messages,
    }
    if request.get("stream") is not None:
        converted["stream"] = bool(request.get("stream"))
    if request.get("max_tokens") is not None:
        converted["max_completion_tokens"] = request.get("max_tokens")
    if request.get("temperature") is not None:
        converted["temperature"] = request.get("temperature")
    if request.get("top_p") is not None:
        converted["top_p"] = request.get("top_p")
    if request.get("stop_sequences"):
        converted["stop"] = request["stop_sequences"]
    tools = _anthropic_tools_to_openai(request.get("tools"))
    if tools:
        converted["tools"] = tools
    tool_choice = _anthropic_tool_choice_to_openai(request.get("tool_choice"))
    if tool_choice is not None:
        converted["tool_choice"] = tool_choice
    return converted


def _anthropic_system_text(system) -> str:
    if isinstance(system, str):
        return system.strip()
    if isinstance(system, list):
        parts = [
            str(block.get("text", "")) for block in system if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n\n".join(part for part in parts if part).strip()
    return ""


def _anthropic_user_to_openai(message: dict) -> list[dict]:
    content = message.get("content")
    if content is None:
        return [{"role": "user", "content": ""}]
    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    user_blocks = []
    tool_messages = []
    for block in content or []:
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

    messages = []
    if len(user_blocks) == 1 and user_blocks[0].get("type") == "text":
        messages.append({"role": "user", "content": user_blocks[0].get("text", "")})
    elif user_blocks:
        messages.append({"role": "user", "content": user_blocks})
    messages.extend(tool_messages)
    return messages or [{"role": "user", "content": ""}]


def _anthropic_assistant_to_openai(message: dict) -> dict:
    content = message.get("content")
    if content is None or isinstance(content, str):
        return {"role": "assistant", "content": content}

    text_parts = []
    thinking_parts = []
    tool_calls = []
    for block in content or []:
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
                    "id": str(block.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": _stable_json(block.get("input", {})),
                    },
                }
            )

    # Adjacent Anthropic content blocks do not imply a text separator.
    converted = {"role": "assistant", "content": "".join(text_parts) or None}
    if thinking_parts:
        converted["reasoning_content"] = "\n".join(thinking_parts)
    if tool_calls:
        converted["tool_calls"] = tool_calls
    return converted


def _anthropic_tools_to_openai(tools) -> list[dict]:
    if not isinstance(tools, list):
        return []
    converted = []
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


def _anthropic_tool_choice_to_openai(choice):
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


def _anthropic_image_url(block: dict) -> str:
    source = block.get("source") or {}
    if not isinstance(source, dict) or source.get("type") != "base64":
        return ""
    media_type = str(source.get("media_type") or "image/png")
    data = str(source.get("data") or "")
    return f"data:{media_type};base64,{data}" if data else ""


def _tool_result_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    return str(content)


def _normalize_id(id_str: str) -> str:
    """Normalize tool_call IDs by stripping dashes for consistent matching.

    Some models (Kimi) return UUIDs with dashes (fc-xxxx-xxxx) in SSE responses,
    but clients like OpenClaw strip dashes when storing history. Normalizing
    ensures prefix matching works regardless of format.
    """
    return id_str.replace("-", "") if id_str else ""


def _strip_billing_header(content: str) -> str:
    """Strip Claude Code billing header (cch= changes every request)."""
    return re.sub(r"x-anthropic-billing-header:[^\n]*\n*", "", content, count=1)


def _stable_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _hash_value(value) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _argument_fingerprint(arguments) -> str:
    """Fingerprint tool arguments with JSON normalization.

    When arguments are valid JSON, we parse and re-serialize with
    _stable_json (sorted keys, compact separators) so that whitespace
    or key-ordering differences don't cause false incompatibility.
    This is critical because:
    - KGB records arguments verbatim from streaming API deltas
    - nanobot's to_openai_tool_call() may json.dumps(dict) with
      different whitespace than the original streaming string
    - nanobot's tool_arguments_json_for_replay() may re-format
    Without normalization, semantically identical arguments with
    different formatting cause _messages_compatible to fail, which
    forks the prefix-tree trace.
    For malformed/partial JSON, fall back to byte-for-byte comparison.
    """
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, (dict, list)):
                return _stable_json(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
        return arguments
    return _stable_json(arguments)


def _tool_call_fingerprint(tool_call: dict) -> tuple:
    function = tool_call.get("function") or {}
    return (
        _normalize_id(str(tool_call.get("id", ""))),
        tool_call.get("type", "function"),
        function.get("name", ""),
        _argument_fingerprint(function.get("arguments", "")),
    )


def _function_call_fingerprint(function_call: dict | None) -> tuple | None:
    if not function_call:
        return None
    return (
        function_call.get("name", ""),
        _argument_fingerprint(function_call.get("arguments", "")),
    )


def _content_hash(msg: dict) -> str:
    content = msg.get("content", "")
    if msg.get("role", "") == "system" and isinstance(content, str):
        content = _strip_billing_header(content)
    return _hash_value(content or "")


def _msg_fingerprint(msg: dict, *, include_reasoning: bool = True) -> tuple:
    """Stable fingerprint for matching trace messages."""
    role = msg.get("role", "")
    reasoning_content = msg.get("reasoning_content", "") if include_reasoning else ""
    return (
        role,
        _content_hash(msg),
        _hash_value(reasoning_content or ""),
        _normalize_id(str(msg.get("tool_call_id", ""))),
        msg.get("name", ""),
        tuple(_tool_call_fingerprint(tc) for tc in msg.get("tool_calls") or []),
        _function_call_fingerprint(msg.get("function_call")),
    )


def _assistant_response_msg(response: dict) -> dict:
    msg: dict = {"role": "assistant", "content": response.get("content", "") or ""}
    if response.get("reasoning_content"):
        msg["reasoning_content"] = response["reasoning_content"]
    if response.get("tool_calls"):
        msg["tool_calls"] = response["tool_calls"]
    if response.get("function_call"):
        msg["function_call"] = response["function_call"]
    return msg


def _node_turn_fingerprint(node: dict) -> tuple:
    request_fps = tuple(_msg_fingerprint(m) for m in node["messages"])
    return request_fps + (_msg_fingerprint(_assistant_response_msg(node["response"])),)


def _values_compatible(left, right) -> bool:
    return not left or not right or left == right


def _tool_calls_compatible(left_calls: list[dict], right_calls: list[dict]) -> bool:
    if len(left_calls) != len(right_calls):
        return False

    for left, right in zip(left_calls, right_calls):
        left_id = _normalize_id(str(left.get("id", "")))
        right_id = _normalize_id(str(right.get("id", "")))

        if left_id and right_id:
            if left_id != right_id:
                return False
        elif _tool_call_fingerprint(left) != _tool_call_fingerprint(right):
            return False

        left_fn = left.get("function") or {}
        right_fn = right.get("function") or {}
        if not _values_compatible(left.get("type"), right.get("type")):
            return False
        if not _values_compatible(left_fn.get("name"), right_fn.get("name")):
            return False

    return True


def _messages_compatible(left: dict, right: dict, *, _diag: bool = False) -> bool | str:
    """Check compatibility. If _diag=True, returns a reason string instead of bool."""
    reason = _message_incompatibility(left, right)
    return reason if _diag else not reason


def _message_incompatibility(left: dict, right: dict) -> str:
    if left.get("role", "") != right.get("role", ""):
        return "role mismatch"
    if _content_hash(left) != _content_hash(right):
        left_c = left.get("content", "")
        right_c = right.get("content", "")
        return (
            f"content hash mismatch: left_type={type(left_c).__name__} "
            f"right_type={type(right_c).__name__} left_repr={repr(left_c)[:120]} "
            f"right_repr={repr(right_c)[:120]}"
        )
    if _normalize_id(str(left.get("tool_call_id", ""))) != _normalize_id(str(right.get("tool_call_id", ""))):
        return f"tool_call_id mismatch: left={left.get('tool_call_id')!r} right={right.get('tool_call_id')!r}"
    if left.get("name", "") != right.get("name", ""):
        return f"name mismatch: left={left.get('name')!r} right={right.get('name')!r}"
    left_tool_calls = left.get("tool_calls") or []
    right_tool_calls = right.get("tool_calls") or []
    if not _tool_calls_compatible(left_tool_calls, right_tool_calls):
        return _tool_call_incompatibility(left_tool_calls, right_tool_calls)
    if _function_call_fingerprint(left.get("function_call")) != _function_call_fingerprint(
        right.get("function_call"),
    ):
        return f"function_call mismatch: left={left.get('function_call')!r} right={right.get('function_call')!r}"

    left_reasoning = left.get("reasoning_content")
    right_reasoning = right.get("reasoning_content")
    if left_reasoning and right_reasoning and left_reasoning != right_reasoning:
        return f"reasoning_content mismatch: left={repr(left_reasoning)[:60]} right={repr(right_reasoning)[:60]}"
    return ""


def _tool_call_incompatibility(left_calls: list[dict], right_calls: list[dict]) -> str:
    for index, (left, right) in enumerate(zip(left_calls, right_calls)):
        left_id = _normalize_id(str(left.get("id", "")))
        right_id = _normalize_id(str(right.get("id", "")))
        if left_id and right_id and left_id != right_id:
            return f"tool_calls[{index}] id mismatch: left={left.get('id')!r} right={right.get('id')!r}"
        left_function = left.get("function") or {}
        right_function = right.get("function") or {}
        if left_function.get("name") != right_function.get("name"):
            return (
                f"tool_calls[{index}] fn.name mismatch: "
                f"left={left_function.get('name')!r} right={right_function.get('name')!r}"
            )
        left_arguments = _argument_fingerprint(left_function.get("arguments", ""))
        right_arguments = _argument_fingerprint(right_function.get("arguments", ""))
        if left_arguments != right_arguments:
            return (
                f"tool_calls[{index}] args fingerprint mismatch: "
                f"left_fn={left_function.get('name')!r} left_fp={left_arguments[:80]} "
                f"right_fp={right_arguments[:80]}"
            )
    return f"tool_calls count mismatch: left={len(left_calls)} right={len(right_calls)}"


def _is_parent_node(parent: dict, child: dict, *, _diag: bool = False) -> bool | str:
    parent_messages = parent["messages"]
    child_messages = child["messages"]
    parent_turn_len = len(parent_messages) + 1

    if parent_turn_len >= len(child_messages):
        return "parent_turn_len >= child_messages" if _diag else False

    for idx, parent_msg in enumerate(parent_messages):
        compat = _messages_compatible(parent_msg, child_messages[idx], _diag=_diag)
        if _diag:
            if compat == "":
                continue
            return f"messages[{idx}] incompatible: {compat}"
        elif not compat:
            return False

    parent_response_msg = _assistant_response_msg(parent["response"])
    compat = _messages_compatible(parent_response_msg, child_messages[len(parent_messages)], _diag=_diag)
    if _diag:
        if compat == "":
            return ""
        return f"response vs child_messages[{len(parent_messages)}] incompatible: {compat}"
    return compat


def _parse_log_node(log: dict) -> dict:
    """Parse a single DB log row into a node for tree building."""
    rb = log.get("request_body")
    if isinstance(rb, str):
        try:
            rb = json.loads(rb)
        except json.JSONDecodeError:
            rb = {}

    # Normalize request body to OpenAI Chat format
    request_protocol = log.get("request_protocol") or "openai_chat"
    if rb:
        rb = normalize_request(rb, request_protocol)

    messages = rb.get("messages", []) if rb else []
    tools = rb.get("tools") if rb else None

    # Parse response: prefer upstream_response_body (preserves reasoning_content
    # that KGB's protocol conversion may drop), fallback to response_body.
    is_stream = bool(log.get("is_stream"))
    supplier_protocol = log.get("supplier_protocol") or "openai_chat"

    upstream_response = log.get("upstream_response_body", "") or ""
    downstream_response = log.get("response_body", "") or ""

    if upstream_response:
        if supplier_protocol in ("anthropic_messages", "anthropic"):
            response = parse_anthropic_response(upstream_response, is_stream)
        else:
            response = parse_openai_response(upstream_response, is_stream)
    elif downstream_response:
        if request_protocol in ("anthropic_messages", "anthropic"):
            response = parse_anthropic_response(downstream_response, is_stream)
        else:
            response = parse_openai_response(downstream_response, is_stream)
    else:
        response = {"content": ""}

    meta = {
        "request_id": log.get("id"),
        "model": log.get("requested_model"),
        "target_model": log.get("target_model"),
        "provider": log.get("provider_name"),
        "response_status": log.get("response_status"),
        "tokens": {
            "input": log.get("input_tokens"),
            "output": log.get("output_tokens"),
        },
        "time_ms": log.get("total_time_ms"),
        "request_time": log.get("request_time"),
    }
    if response.get("usage"):
        meta["usage"] = response["usage"]

    return {
        "id": log.get("id"),
        "messages": messages,
        "tools": tools,
        "response": response,
        "meta": meta,
        "is_stream": is_stream,
        "parent_idx": None,
        "children": [],
    }


def merge_conversations(logs: list[dict]) -> list[dict]:
    """Return one merged conversation for each root-to-leaf request path."""

    nodes = _deduplicate_nodes([_parse_log_node(log) for log in logs])
    fork_log = _build_parent_links(nodes)
    if fork_log:
        print("  [FORK-DIAG] " + "\n  [FORK-DIAG] ".join(fork_log))

    leaves = (index for index, node in enumerate(nodes) if not node["children"])
    return [_conversation_record(nodes, _conversation_path(nodes, leaf)) for leaf in leaves]


def _deduplicate_nodes(raw_nodes: list[dict]) -> list[dict]:
    """Collapse duplicate turns, preferring complete non-stream responses."""

    seen: dict[tuple, int] = {}
    nodes: list[dict] = []
    for node in raw_nodes:
        fingerprint = _node_turn_fingerprint(node)
        previous = seen.get(fingerprint)
        if previous is None:
            seen[fingerprint] = len(nodes)
            nodes.append(node)
        elif not node["is_stream"] and nodes[previous]["is_stream"]:
            nodes[previous] = node

    for node in nodes:
        node["parent_idx"] = None
        node["children"] = []
    return nodes


def _build_parent_links(nodes: list[dict]) -> list[str]:
    """Link each node to the longest compatible earlier request prefix."""

    fork_log: list[str] = []
    for index, node in enumerate(nodes):
        parent_index, closest_failure = _best_parent(nodes, index)
        if parent_index is not None:
            node["parent_idx"] = parent_index
            nodes[parent_index]["children"].append(index)
        else:
            fork_log.extend(_fork_diagnostics(nodes, index, closest_failure))
    return fork_log


def _best_parent(nodes: list[dict], child_index: int) -> tuple[int | None, str]:
    best_parent = None
    best_prefix_len = -1
    closest_failure = ""
    for candidate_index in range(child_index - 1, -1, -1):
        candidate = nodes[candidate_index]
        candidate_len = len(candidate["messages"]) + 1
        if candidate_len <= best_prefix_len:
            continue
        result = _is_parent_node(candidate, nodes[child_index], _diag=True)
        if result == "":
            best_parent = candidate_index
            best_prefix_len = candidate_len
        elif not closest_failure:
            closest_failure = f"  closest candidate j={candidate_index} (cand_len={candidate_len}): {result}"
    return best_parent, closest_failure


def _fork_diagnostics(nodes: list[dict], index: int, closest_failure: str) -> list[str]:
    node = nodes[index]
    message_count = len(node["messages"])
    diagnostics = [(f"FORK: node[{index}] id={node.get('id')} msgs={message_count} has NO parent\n{closest_failure}")]
    if index == 0:
        return diagnostics

    root = nodes[0]
    child_message_index = len(root["messages"])
    if child_message_index >= message_count:
        return diagnostics

    parent_response = _assistant_response_msg(root["response"])
    child_message = node["messages"][child_message_index]
    mismatch = _messages_compatible(parent_response, child_message, _diag=True)
    diagnostics.append(
        f"  vs root node[0] (msgs={len(root['messages'])}): response vs child[{child_message_index}]: {mismatch}"
    )
    diagnostics.extend(_message_diagnostics("parent_response", parent_response))
    diagnostics.extend(_message_diagnostics(f"child[{child_message_index}]", child_message))
    return diagnostics


def _message_diagnostics(label: str, message: dict) -> list[str]:
    tool_calls = message.get("tool_calls") or []
    lines = [
        (
            f"    {label}: role={message.get('role')} "
            f"content_repr={repr(message.get('content', ''))[:100]} "
            f"tc_count={len(tool_calls)}"
        )
    ]
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") or {}
        lines.append(
            f"      tc[{index}] id={tool_call.get('id')!r} "
            f"fn={function.get('name')!r} "
            f"args_fp={_argument_fingerprint(function.get('arguments', ''))[:80]}"
        )
    return lines


def _conversation_path(nodes: list[dict], leaf_index: int) -> list[int]:
    path: list[int] = []
    index = leaf_index
    while index is not None:
        path.append(index)
        index = nodes[index]["parent_idx"]
    path.reverse()
    return path


def _conversation_record(nodes: list[dict], path: list[int]) -> dict:
    messages: list[dict] = []
    tools = None
    previous_message_count = 0
    tool_call_ids: dict[str, str] = {}

    for node_index in path:
        node = nodes[node_index]
        if node["tools"]:
            tools = node["tools"]

        for message in node["messages"][previous_message_count:]:
            messages.append(_history_message(message, tool_call_ids))

        response = node["response"]
        assistant_message: dict = {"role": "assistant", "meta": node["meta"]}
        if response.get("content"):
            assistant_message["content"] = response["content"]
        if response.get("reasoning_content"):
            assistant_message["reasoning_content"] = response["reasoning_content"]
        if response.get("tool_calls"):
            assistant_message["tool_calls"] = response["tool_calls"]
            _remember_tool_call_ids(response["tool_calls"], tool_call_ids)
        messages.append(assistant_message)
        previous_message_count = len(node["messages"]) + 1

    record: dict = {"messages": messages}
    if tools:
        record["tools"] = tools
    return record


def _history_message(message: dict, tool_call_ids: dict[str, str]) -> dict:
    entry = {"role": message["role"]}
    if message.get("content") is not None:
        entry["content"] = message["content"]
    if message.get("reasoning_content"):
        entry["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        entry["tool_calls"] = message["tool_calls"]
        _remember_tool_call_ids(message["tool_calls"], tool_call_ids)
    if message.get("function_call"):
        entry["function_call"] = message["function_call"]
    if message.get("tool_call_id"):
        raw_id = message["tool_call_id"]
        entry["tool_call_id"] = tool_call_ids.get(_normalize_id(raw_id), raw_id)
    if message.get("name"):
        entry["name"] = message["name"]
    return entry


def _remember_tool_call_ids(tool_calls: list[dict], known_ids: dict[str, str]) -> None:
    for tool_call in tool_calls:
        original_id = tool_call.get("id", "")
        if original_id:
            known_ids[_normalize_id(original_id)] = original_id


def _connect_db(db_path: str):
    """Connect to SQLite file or PostgreSQL via connection string.

    For PostgreSQL, pass a connection string starting with "postgresql" or "postgres".
    Requires psycopg2 to be installed when using PostgreSQL.

    Returns a DB-API 2.0 connection with dict-like row access.
    """
    if db_path.startswith(("postgresql", "postgres")):
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(db_path)
        # Use RealDictCursor so rows behave like dicts (row[key] and row.keys())
        conn.cursor_factory = psycopg2.extras.RealDictCursor
        return conn

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def export_db(db_path: str, output_dir: str) -> dict[str, str]:
    """Export traces from llm_gateway DB, grouped by api_key_name.

    Args:
        db_path: SQLite file path or PostgreSQL connection string.

    Returns dict mapping api_key_name to output file path.
    """
    conn = _connect_db(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT l.id, l.request_time, l.api_key_name, l.requested_model,
               l.target_model, l.provider_name, l.input_tokens, l.output_tokens,
               l.total_time_ms, l.response_status, l.is_stream,
               l.request_protocol, l.supplier_protocol,
               d.request_body, d.response_body, d.upstream_response_body
        FROM request_logs l
        JOIN request_log_details d ON l.id = d.log_id
        WHERE l.response_status BETWEEN 200 AND 399
        ORDER BY l.id
    """)
    rows = cur.fetchall()

    # Count filtered-out requests for diagnostics
    cur.execute("SELECT COUNT(*) AS cnt FROM request_logs")
    row = cur.fetchone()
    total_count = row["cnt"] if isinstance(row, dict) else row[0]
    cur.execute("""
        SELECT response_status, COUNT(*) as cnt
        FROM request_logs
        WHERE response_status NOT BETWEEN 200 AND 399
        GROUP BY response_status
    """)
    failed_rows = cur.fetchall()
    conn.close()

    if failed_rows:
        failed_summary = ", ".join(f"{r['response_status']}:{r['cnt']}" for r in failed_rows)
        print(f"  Filtered out {total_count - len(rows)}/{total_count} failed requests ({failed_summary})")

    # Group by api_key_name
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        record = {key: row[key] for key in row.keys()}
        key_name = record.get("api_key_name") or "unknown"
        groups[key_name].append(record)

    # Merge and export each group
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}

    for key_name, logs in groups.items():
        conversations = merge_conversations(logs)
        safe_name = key_name.replace("/", "_")
        out_path = out_dir / f"{safe_name}.jsonl"

        with open(out_path, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False, default=str) + "\n")

        total_msgs = sum(len(c["messages"]) for c in conversations)
        output_files[key_name] = str(out_path)
        print(
            f"  {key_name}: {len(logs)} requests -> {len(conversations)} path(s), {total_msgs} messages -> {out_path}"
        )

    return output_files


def main():
    if len(sys.argv) < 2:
        print("Usage: trace_export.py <db_path_or_url> [--output-dir <dir>]")
        sys.exit(1)

    db_input = sys.argv[1]
    output_dir = "./traces"

    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]

    # Download if URL
    if db_input.startswith("http://") or db_input.startswith("https://"):
        print("Downloading from URL...")
        tmp = tempfile.mktemp(suffix=".db")
        urllib.request.urlretrieve(db_input, tmp)
        db_path = tmp
        print(f"  -> {tmp} ({os.path.getsize(tmp)} bytes)")
    else:
        db_path = db_input

    print(f"Exporting traces from {db_path} -> {output_dir}/")
    files = export_db(db_path, output_dir)
    print(f"\nDone. Exported {len(files)} conversation(s).")


if __name__ == "__main__":
    main()
