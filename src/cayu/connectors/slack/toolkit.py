"""Slack's Web API as tools, through a connection slot."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from cayu.connections import ConnectionCallError, ConnectionClient, ConnectionResponse
from cayu.connectors import _common
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy

DEFAULT_SLOT = "slack"
#: Messages, replies or channels one read answers unless the call asks for another number.
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
#: The longest message text Slack accepts.
MAX_MESSAGE_CHARACTERS = 40_000
#: How long a read message's text may be, tried in turn until the answer fits.
_READ_TEXT_LIMITS = (4_000, 1_000, 300, 100)
#: How long a channel's topic and purpose may be, tried in turn until the page fits.
_TOPIC_LIMITS = (250, 80, 0)
_JSON = {"content-type": "application/json; charset=utf-8"}

POST_MESSAGE = "slack_post_message"
UPDATE_MESSAGE = "slack_update_message"
LIST_CHANNELS = "slack_list_channels"
READ_CHANNEL = "slack_read_channel"
READ_THREAD = "slack_read_thread"
LOOKUP_USER = "slack_lookup_user"

_CHANNEL_ID = {"type": "string", "description": "The channel's ID (C...), not its name."}
_POST_CHANNEL = {
    "type": "string",
    "description": "The channel's ID (C...), or its name (#general).",
}
_PAGE = {
    "limit": {
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_PAGE_SIZE,
        "description": f"How many to return (default {DEFAULT_PAGE_SIZE}).",
    },
    "cursor": {"type": "string", "description": "The next_cursor of the previous page."},
}


def tools(
    client: ConnectionClient | None = None,
    slot: str = DEFAULT_SLOT,
    *,
    include_writes: bool = False,
) -> list[Tool]:
    """The Slack read tools on ``slot``, and with ``include_writes=True`` the post and edit tools
    too (pair them with ``approval_policy()``), calling through ``client`` (default: Cayu
    Cloud's broker)."""

    resolved = _common.resolved_client(client)
    kinds: tuple[type[_SlackTool], ...] = (_ListChannels, _ReadChannel, _ReadThread, _LookupUser)
    if include_writes is True:
        kinds = (_PostMessage, _UpdateMessage, *kinds)
    return [kind(resolved, slot) for kind in kinds]


def approval_policy(inner: ToolPolicy | None = None) -> ToolPolicy:
    """Ask for approval before posting or editing a message, naming the channel; every other
    decision is ``inner``'s (default: allow)."""

    return _common.approval_policy(inner, _REASONS)


def review_fields() -> dict[str, tuple[str, ...]]:
    """The arguments an approver sees, for ``ToolArgumentReview``."""

    return {
        POST_MESSAGE: ("channel", "thread_ts", "text"),
        UPDATE_MESSAGE: ("channel", "ts", "text"),
    }


def _post_reason(arguments: Mapping[str, Any]) -> str:
    channel = _common.reason_argument(arguments.get("channel"))
    if arguments.get("thread_ts"):
        return f"Reply in a Slack thread in {channel}."
    return f"Post a Slack message to {channel}."


def _update_reason(arguments: Mapping[str, Any]) -> str:
    return f"Edit a Slack message in {_common.reason_argument(arguments.get('channel'))}."


_REASONS: dict[str, Callable[[Mapping[str, Any]], str]] = {
    POST_MESSAGE: _post_reason,
    UPDATE_MESSAGE: _update_reason,
}


class _SlackError(Exception):
    def __init__(self, result: ToolResult) -> None:
        super().__init__(result.content)
        self.result = result


class _InvalidArguments(ValueError):
    pass


class _SlackTool(Tool):
    def __init__(self, client: ConnectionClient | None, slot: str) -> None:
        super().__init__()
        self._client = client
        self._slot = slot

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        try:
            return await self._run(args)
        except _SlackError as error:
            return error.result
        except _InvalidArguments as error:
            return _common.invalid_arguments(str(error))

    async def _run(self, args: dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    async def _call(
        self,
        method: str,
        *,
        query: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Slack's answer to ``method``: its JSON body when ``ok``; otherwise a tool error."""

        if self._client is None:
            raise _SlackError(_common.no_client_result())
        try:
            if body is None:
                response = await self._client.request(
                    self._slot, "GET", f"/api/{method}", query=query
                )
            else:
                response = await self._client.request(
                    self._slot,
                    "POST",
                    f"/api/{method}",
                    headers=_JSON,
                    body=json.dumps(body, ensure_ascii=False).encode(),
                )
        except ConnectionCallError as error:
            raise _SlackError(
                _common.call_error_result(
                    error,
                    self._slot,
                    write=body is not None,
                    check="read the channel's latest messages",
                )
            ) from None
        return _answer(method, response)


def _answer(method: str, response: ConnectionResponse) -> dict[str, Any]:
    if response.status == 429:
        raise _SlackError(_common.connector_rate_limited(response, "Slack"))
    answer = _common.json_body(response)
    if response.status != 200 or answer is None:
        raise _SlackError(
            _common.error_result(
                "upstream_status",
                f"Slack answered {method} with HTTP {response.status}.",
                _common.TEMPORARY if response.status >= 500 or response.status == 200 else None,
                status=response.status,
            )
        )
    if answer.get("ok") is not True:
        error = answer.get("error")
        error = error if isinstance(error, str) and error else "unknown_error"
        raise _SlackError(
            _common.error_result(
                "slack_error", f"Slack refused {method}: {error}.", slack_error=error
            )
        )
    return answer


def _required(args: Mapping[str, Any], name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise _InvalidArguments(f"{name} must be a nonblank string.")
    return value


def _optional(args: Mapping[str, Any], name: str) -> str | None:
    return None if args.get(name) is None else _required(args, name)


def _page(args: Mapping[str, Any]) -> dict[str, str]:
    limit = args.get("limit", DEFAULT_PAGE_SIZE)
    if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
        raise _InvalidArguments(f"limit must be an integer from 1 to {MAX_PAGE_SIZE}.")
    query = {"limit": str(limit)}
    cursor = _optional(args, "cursor")
    if cursor is not None:
        query["cursor"] = cursor
    return query


def _items(answer: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    return [item for item in answer.get(name) or [] if isinstance(item, dict)]


def _next_cursor(answer: Mapping[str, Any]) -> str | None:
    metadata = answer.get("response_metadata")
    cursor = metadata.get("next_cursor") if isinstance(metadata, dict) else None
    return cursor if isinstance(cursor, str) and cursor else None


def _message(item: Mapping[str, Any], limit: int) -> dict[str, Any]:
    text, cut = _common.shortened(str(item.get("text") or ""), limit)
    message: dict[str, Any] = {
        name: item[name]
        for name in ("ts", "user", "bot_id", "subtype", "thread_ts", "reply_count")
        if item.get(name) is not None
    }
    message["text"] = text
    if cut:
        message["text_shortened"] = True
    return message


def _messages_result(channel: str, answer: Mapping[str, Any]) -> ToolResult:
    """The page of messages, each text shortened as far as the answer needs to fit."""

    items = _items(answer, "messages")
    for limit in _READ_TEXT_LIMITS:
        result = _common.answer_result(
            {
                "channel": channel,
                "messages": [_message(item, limit) for item in items],
                "has_more": answer.get("has_more") is True,
                "next_cursor": _next_cursor(answer),
            }
        )
        if _common.answer_fits(result):
            return result
    return _common.error_result(
        "answer_too_large",
        "The page is too large to answer.",
        "Ask for fewer messages with a smaller limit.",
    )


def _spec(
    name: str, description: str, properties: dict[str, Any], required: list[str], **kwargs: Any
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        max_terminal_payload_bytes=_common.MAX_TERMINAL_PAYLOAD_BYTES,
        **kwargs,
    )


class _PostMessage(_SlackTool):
    spec = _spec(
        POST_MESSAGE,
        "Post a message to a public Slack channel, or reply in a thread with thread_ts. It may "
        "need a person's approval first.",
        {
            "channel": _POST_CHANNEL,
            "text": {"type": "string", "maxLength": MAX_MESSAGE_CHARACTERS},
            "thread_ts": {"type": "string", "description": "The thread's parent message ts."},
        },
        ["channel", "text"],
        effect=ToolEffect.EXTERNAL,
        parallel_safe=False,
    )

    async def _run(self, args: dict[str, Any]) -> ToolResult:
        body: dict[str, Any] = {"channel": _required(args, "channel"), "text": _message_text(args)}
        thread_ts = _optional(args, "thread_ts")
        if thread_ts is not None:
            body["thread_ts"] = thread_ts
        answer = await self._call("chat.postMessage", body=body)
        return _common.answer_result(
            {"channel": answer.get("channel"), "ts": answer.get("ts"), "thread_ts": thread_ts}
        )


class _UpdateMessage(_SlackTool):
    spec = _spec(
        UPDATE_MESSAGE,
        "Replace the text of a message this app posted, in the channel given by its ID. It may "
        "need a person's approval first.",
        {
            "channel": _CHANNEL_ID,
            "ts": {"type": "string", "description": "The message's ts."},
            "text": {"type": "string", "maxLength": MAX_MESSAGE_CHARACTERS},
        },
        ["channel", "ts", "text"],
        effect=ToolEffect.IDEMPOTENT,
        parallel_safe=False,
    )

    async def _run(self, args: dict[str, Any]) -> ToolResult:
        answer = await self._call(
            "chat.update",
            body={
                "channel": _required(args, "channel"),
                "ts": _required(args, "ts"),
                "text": _message_text(args),
            },
        )
        return _common.answer_result({"channel": answer.get("channel"), "ts": answer.get("ts")})


def _message_text(args: Mapping[str, Any]) -> str:
    text = _required(args, "text")
    if len(text) > MAX_MESSAGE_CHARACTERS:
        raise _InvalidArguments(f"text is at most {MAX_MESSAGE_CHARACTERS} characters.")
    return text


class _ListChannels(_SlackTool):
    spec = _spec(
        LIST_CHANNELS,
        "List the workspace's public channels, a page at a time.",
        dict(_PAGE),
        [],
        effect=ToolEffect.NONE,
    )

    async def _run(self, args: dict[str, Any]) -> ToolResult:
        query = {"types": "public_channel", "exclude_archived": "true", **_page(args)}
        answer = await self._call("conversations.list", query=query)
        items = _items(answer, "channels")
        for limit in _TOPIC_LIMITS:
            result = _channels_result(items, limit, answer)
            if _common.answer_fits(result):
                return result
        # Even without topics the page doesn't fit: ask again for as many as do, from the same
        # cursor, so the next page's cursor still follows the last channel answered.
        fitting = max(len(items) - 1, 1)
        while fitting > 1 and not _common.answer_fits(_channels_result(items[:fitting], 0, {})):
            fitting = fitting * 3 // 4
        answer = await self._call("conversations.list", query={**query, "limit": str(fitting)})
        return _channels_result(_items(answer, "channels")[:fitting], 0, answer)


def _channels_result(
    items: list[dict[str, Any]], topic_limit: int, answer: Mapping[str, Any]
) -> ToolResult:
    return _common.answer_result(
        {
            "channels": [_channel(item, topic_limit) for item in items],
            "next_cursor": _next_cursor(answer),
        }
    )


def _channel(item: Mapping[str, Any], topic_limit: int) -> dict[str, Any]:
    channel = {
        name: item[name]
        for name in ("id", "name", "is_member", "num_members")
        if item.get(name) is not None
    }
    for name in ("topic", "purpose") if topic_limit else ():
        detail = item.get(name)
        value = detail.get("value") if isinstance(detail, dict) else None
        if isinstance(value, str) and value:
            channel[name] = _common.shortened(value, topic_limit)[0]
    return channel


class _ReadChannel(_SlackTool):
    spec = _spec(
        READ_CHANNEL,
        "Read a public channel's messages, newest first, a page at a time. oldest and latest "
        "bound them by ts.",
        {
            "channel": _CHANNEL_ID,
            **_PAGE,
            "oldest": {"type": "string", "description": "Only messages after this ts."},
            "latest": {"type": "string", "description": "Only messages before this ts."},
        },
        ["channel"],
        effect=ToolEffect.NONE,
    )

    async def _run(self, args: dict[str, Any]) -> ToolResult:
        channel = _required(args, "channel")
        query = {"channel": channel, **_page(args)}
        for bound in ("oldest", "latest"):
            value = _optional(args, bound)
            if value is not None:
                query[bound] = value
        return _messages_result(channel, await self._call("conversations.history", query=query))


class _ReadThread(_SlackTool):
    spec = _spec(
        READ_THREAD,
        "Read a thread: its parent message, then its replies, a page at a time.",
        {
            "channel": _CHANNEL_ID,
            "thread_ts": {"type": "string", "description": "The thread's parent message ts."},
            **_PAGE,
        },
        ["channel", "thread_ts"],
        effect=ToolEffect.NONE,
    )

    async def _run(self, args: dict[str, Any]) -> ToolResult:
        channel = _required(args, "channel")
        query = {"channel": channel, "ts": _required(args, "thread_ts"), **_page(args)}
        return _messages_result(channel, await self._call("conversations.replies", query=query))


class _LookupUser(_SlackTool):
    spec = _spec(
        LOOKUP_USER,
        "Look up a Slack user by user_id (U...) or by email; give exactly one.",
        {"user_id": {"type": "string"}, "email": {"type": "string"}},
        [],
        effect=ToolEffect.NONE,
    )

    async def _run(self, args: dict[str, Any]) -> ToolResult:
        user_id = _optional(args, "user_id")
        email = _optional(args, "email")
        if user_id is not None and email is None:
            answer = await self._call("users.info", query={"user": user_id})
        elif email is not None and user_id is None:
            answer = await self._call("users.lookupByEmail", query={"email": email})
        else:
            raise _InvalidArguments("Give exactly one of user_id and email.")
        user = answer.get("user")
        return _common.answer_result({"user": _user(user if isinstance(user, dict) else {})})


def _user(item: Mapping[str, Any]) -> dict[str, Any]:
    profile = item.get("profile")
    profile = profile if isinstance(profile, dict) else {}
    user = {
        name: item[name]
        for name in ("id", "name", "real_name", "tz", "is_bot", "deleted")
        if item.get(name) is not None
    }
    for name in ("display_name", "email", "title"):
        if profile.get(name):
            user[name] = profile[name]
    return user
