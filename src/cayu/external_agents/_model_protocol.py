"""The gateway's stateless, client-tool-only model protocol boundary."""

from __future__ import annotations

import json
from typing import Any


def _object(value: Any, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) - fields:
        raise ValueError("Unsupported model request object.")
    return value


def _array(value: Any) -> list[Any]:
    if type(value) is not list:
        raise ValueError("Expected an inline model request array.")
    return value


def _schema(value: Any) -> None:
    # A schema is data, but external references must not become a fetch surface.
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"$ref", "$dynamicRef"} and (
                not isinstance(item, str) or not item.startswith("#")
            ):
                raise ValueError("External schema references are unavailable.")
            _schema(item)
    elif isinstance(value, list):
        for item in value:
            _schema(item)


def _cache(value: Any) -> None:
    value = _object(value, {"type", "ttl"})
    if value.get("type") != "ephemeral" or value.get("ttl", "5m") != "5m":
        raise ValueError("Unsupported prompt cache control.")


def _inline_citation(value: Any, documents: list[dict[str, Any]]) -> None:
    ranges = {
        "char_location": ("start_char_index", "end_char_index", 0),
        "page_location": ("start_page_number", "end_page_number", 1),
    }
    if type(value) is not dict or value.get("type") not in ranges:
        raise ValueError("Only inline document citations are supported.")
    start, end, minimum = ranges[value["type"]]
    citation = _object(
        value, {"type", "cited_text", "document_index", "document_title", "file_id", start, end}
    )
    index = citation.get("document_index")
    if (
        type(index) is not int
        or not 0 <= index < len(documents)
        or not isinstance(citation.get("cited_text"), str)
        or not isinstance(citation.get("document_title"), str | type(None))
        or citation.get("file_id") is not None
        or type(citation.get(start)) is not int
        or type(citation.get(end)) is not int
        or not minimum <= citation[start] < citation[end]
    ):
        raise ValueError("Invalid inline document citation.")
    source = documents[index]["source"]
    if citation["type"] == "char_location":
        if source["type"] != "text" or citation[end] > len(source["data"]):
            raise ValueError("Citation does not refer to the inline text document.")
    elif source["type"] != "base64" or source.get("media_type") != "application/pdf":
        raise ValueError("Citation does not refer to an inline PDF document.")


def _anthropic_content(value: Any, documents: list[dict[str, Any]], citations: list[Any]) -> None:
    if isinstance(value, str):
        return
    fields = {
        "text": {"text", "citations"},
        "thinking": {"thinking", "signature"},
        "redacted_thinking": {"data"},
        "tool_use": {"id", "name", "input"},
        "tool_result": {"tool_use_id", "content", "is_error"},
        "tool_reference": {"tool_name"},
        "image": {"source"},
        "document": {"source", "title", "context", "citations"},
    }
    for block in _array(value):
        if type(block) is not dict or block.get("type") not in fields:
            raise ValueError("Only inline client content is supported.")
        kind = block["type"]
        _object(block, {"type", "cache_control"} | fields[kind])
        if "cache_control" in block:
            _cache(block["cache_control"])
        if kind == "tool_result":
            _anthropic_content(block.get("content", ""), documents, citations)
        elif kind == "text" and block.get("citations") is not None:
            citations.extend(_array(block["citations"]))
        elif kind in {"image", "document"}:
            source = _object(block.get("source"), {"type", "media_type", "data"})
            if source.get("type") not in {"base64", "text"} or not isinstance(
                source.get("data"), str
            ):
                raise ValueError("Model attachments must be inline data.")
            if kind == "document":
                documents.append(block)
                if block.get("citations") is not None:
                    config = _object(block["citations"], {"enabled"})
                    if "enabled" in config and type(config["enabled"]) is not bool:
                        raise ValueError("Invalid inline document citation configuration.")


def _anthropic(body: dict[str, Any]) -> None:
    _object(
        body,
        {
            "model",
            "messages",
            "max_tokens",
            "metadata",
            "stop_sequences",
            "stream",
            "system",
            "temperature",
            "tool_choice",
            "tools",
            "top_k",
            "top_p",
            "thinking",
            "output_config",
            "output_format",
            "cache_control",
            "service_tier",
            "context_management",
        },
    )
    documents: list[dict[str, Any]] = []
    citations: list[Any] = []
    for message in _array(body.get("messages", [])):
        message = _object(message, {"role", "content"})
        if message.get("role") not in {"user", "assistant"}:
            raise ValueError("Unsupported message role.")
        _anthropic_content(message.get("content"), documents, citations)
    if "system" in body:
        _anthropic_content(body["system"], documents, citations)
    for citation in citations:
        _inline_citation(citation, documents)
    if "cache_control" in body:
        _cache(body["cache_control"])
    for tool in _array(body.get("tools", [])):
        tool = _object(
            tool,
            {
                "type",
                "name",
                "description",
                "input_schema",
                "cache_control",
                "strict",
                "defer_loading",
                "eager_input_streaming",
                "allowed_callers",
                "input_examples",
            },
        )
        if tool.get("type", "custom") != "custom" or tool.get("allowed_callers", ["direct"]) != [
            "direct"
        ]:
            raise ValueError("Only client-executed custom tools are supported.")
        _schema(tool.get("input_schema"))
        if "cache_control" in tool:
            _cache(tool["cache_control"])
    if "tool_choice" in body:
        choice = _object(body["tool_choice"], {"type", "name", "disable_parallel_tool_use"})
        if choice.get("type") not in {"auto", "any", "none", "tool"}:
            raise ValueError("Unsupported tool choice.")
    if "thinking" in body:
        _object(body["thinking"], {"type", "budget_tokens", "display"})
    if "context_management" in body:
        management = _object(body["context_management"], {"edits"})
        for edit in _array(management.get("edits", [])):
            edit = _object(edit, {"type", "keep"})
            if edit.get("type") != "clear_thinking_20251015":
                raise ValueError("Unsupported context management operation.")
    if "output_config" in body:
        output = _object(body["output_config"], {"effort", "format"})
        if "format" in output:
            _schema(_object(output["format"], {"type", "schema"}).get("schema"))
    if "output_format" in body:
        _schema(_object(body["output_format"], {"type", "schema"}).get("schema"))


def _responses_content(value: Any) -> None:
    if isinstance(value, str):
        return
    for block in _array(value):
        if type(block) is not dict:
            raise ValueError("Expected inline message content.")
        kind = block.get("type")
        if kind in {"input_text", "output_text"}:
            _object(block, {"type", "text", "annotations", "logprobs"})
            if block.get("annotations"):
                raise ValueError("Referenced output annotations are unavailable.")
        elif kind == "input_image":
            _object(block, {"type", "image_url", "detail"})
            url = block.get("image_url")
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise ValueError("Images must be inline data URLs.")
        elif kind == "input_file":
            _object(block, {"type", "filename", "file_data"})
            if not isinstance(block.get("file_data"), str):
                raise ValueError("Files must be inline data.")
        elif kind == "refusal":
            _object(block, {"type", "refusal"})
        else:
            raise ValueError("Unsupported model content type.")


def _response_tools(value: Any) -> None:
    for tool in _array(value):
        if type(tool) is not dict:
            raise ValueError("Expected a client tool definition.")
        kind = tool.get("type")
        if kind == "function":
            _object(tool, {"type", "name", "description", "parameters", "strict"})
            _schema(tool.get("parameters"))
        elif kind == "custom":
            _object(tool, {"type", "name", "description", "format"})
            if "format" in tool:
                _object(tool["format"], {"type", "syntax", "definition"})
        elif kind == "namespace":
            _object(tool, {"type", "name", "description", "tools"})
            _response_tools(tool.get("tools"))
        else:
            raise ValueError("Only client-executed function and custom tools are supported.")


def _responses(body: dict[str, Any]) -> None:
    _object(
        body,
        {
            "model",
            "input",
            "instructions",
            "max_output_tokens",
            "stream",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "text",
            "temperature",
            "top_p",
            "store",
            "include",
            "prompt_cache_key",
            "prompt_cache_retention",
            "service_tier",
            "metadata",
            "truncation",
            "stream_options",
            "client_metadata",
        },
    )
    if body.get("store", False) is not False:
        raise ValueError("Stored responses are unavailable.")
    body["store"] = False
    if any(item != "reasoning.encrypted_content" for item in _array(body.get("include", []))):
        raise ValueError("Only stateless reasoning inclusion is supported.")
    _response_tools(body.get("tools", []))
    choice = body.get("tool_choice", "auto")
    if isinstance(choice, dict):
        _object(choice, {"type", "name"})
        if choice.get("type") not in {"function", "custom"}:
            raise ValueError("Unsupported tool choice.")
    elif choice not in ("auto", "none", "required"):
        raise ValueError("Unsupported tool choice.")
    value = body.get("input", [])
    if not isinstance(value, str):
        for item in _array(value):
            if type(item) is not dict:
                raise ValueError("Expected inline response input.")
            kind = item.get("type", "message")
            if kind == "message":
                _object(item, {"type", "id", "role", "content", "status", "phase"})
                if item.get("role") not in {"user", "assistant", "developer", "system"}:
                    raise ValueError("Unsupported message role.")
                _responses_content(item.get("content"))
            elif kind in {"function_call", "custom_tool_call"}:
                _object(
                    item,
                    {"type", "id", "call_id", "name", "namespace", "arguments", "input", "status"},
                )
                if "arguments" not in item and "input" not in item:
                    raise ValueError("Tool calls must be supplied inline.")
            elif kind in {"function_call_output", "custom_tool_call_output"}:
                _object(item, {"type", "id", "call_id", "output", "status"})
                _responses_content(item.get("output"))
            elif kind == "reasoning":
                _object(item, {"type", "id", "summary", "content", "encrypted_content", "status"})
                if "summary" not in item and "encrypted_content" not in item:
                    raise ValueError("Reasoning must be supplied inline.")
            else:
                raise ValueError("Stored references and server-tool inputs are unavailable.")
    if "reasoning" in body:
        _object(body["reasoning"], {"effort", "summary", "generate_summary"})
    if "text" in body:
        config = _object(body["text"], {"format", "verbosity"})
        if "format" in config:
            output = _object(config["format"], {"type", "name", "description", "schema", "strict"})
            _schema(output.get("schema"))
    if "stream_options" in body:
        _object(body["stream_options"], {"include_usage", "include_obfuscation"})


def validate_model_request(body: dict[str, Any], protocol: str) -> None:
    """Reject provider-side execution, network fetches and stored-state references."""
    if "stream" in body and type(body["stream"]) is not bool:
        raise ValueError("Streaming must be a boolean.")
    (_anthropic if protocol == "messages" else _responses)(body)


def response_usage(
    raw: bytes, protocol: str, *, streaming: bool
) -> tuple[dict[str, Any], str | None]:
    """Read complete upstream receipts; absent or incomplete usage stays unknown."""
    usage: dict[str, Any] = {}
    model = None
    complete = False
    final_message_usage = False
    try:
        if not streaming:
            response = json.loads(raw)
            usage, model = response.get("usage") or {}, response.get("model")
            complete = True
        else:
            for frame in raw.replace(b"\r\n", b"\n").split(b"\n\n"):
                data = b"\n".join(
                    line[5:].lstrip() for line in frame.splitlines() if line.startswith(b"data:")
                )
                if not data or data == b"[DONE]":
                    continue
                event = json.loads(data)
                kind = event.get("type")
                if protocol == "messages":
                    if kind == "message_start":
                        usage = dict(event["message"].get("usage") or {})
                        model = event["message"].get("model")
                    elif kind == "message_delta":
                        usage.update(event.get("usage") or {})
                        final_message_usage = "output_tokens" in (event.get("usage") or {})
                    elif kind == "message_stop":
                        complete = final_message_usage
                elif kind in {"response.completed", "response.incomplete", "response.failed"}:
                    response = event["response"]
                    usage, model = response.get("usage") or {}, response.get("model")
                    complete = True
        if (
            not complete
            or type(usage) is not dict
            or any(
                type(usage.get(key)) is not int or usage[key] < 0
                for key in ("input_tokens", "output_tokens")
            )
        ):
            return {}, None
        result = {key: usage[key] for key in ("input_tokens", "output_tokens")}
        if protocol == "messages":
            for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
                value = usage.get(key, 0)
                if type(value) is not int or value < 0:
                    return {}, None
                result[key] = value
        else:
            details = usage.get("input_tokens_details") or {}
            result["input_tokens_details"] = {}
            for field in ("cached_tokens", "cache_write_tokens"):
                count = details.get(field, 0)
                if type(count) is not int or not 0 <= count <= result["input_tokens"]:
                    return {}, None
                result["input_tokens_details"][field] = count
            reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
            if type(reasoning) is not int or not 0 <= reasoning <= result["output_tokens"]:
                return {}, None
            result["output_tokens_details"] = {"reasoning_tokens": reasoning}
        return result, model if isinstance(model, str) else None
    except (ValueError, TypeError, KeyError, AttributeError):
        return {}, None
