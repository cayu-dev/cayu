"""The Slack toolkit: each tool's request through a slot, its answer, and its approval."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from tests.connector_broker_support import (
    Answer,
    FakeBroker,
    Refusal,
    Sent,
    assert_within_budgets,
    context,
    policy_request,
)

from cayu.approvals import ToolArgumentReview
from cayu.connections import LocalConnections, LocalSlot
from cayu.connectors import slack
from cayu.connectors._common import MAX_TERMINAL_PAYLOAD_BYTES
from cayu.tools import (
    AlwaysRequireApprovalToolPolicy,
    StaticToolPolicy,
    Tool,
    ToolEffect,
    ToolPolicyDecision,
    ToolResult,
)


def _tools(broker: FakeBroker, slot: str = "team-slack") -> dict[str, Tool]:
    return {tool.name: tool for tool in slack.tools(broker.client, slot=slot, include_writes=True)}


def _run(tool: Tool, **args: Any) -> ToolResult:
    return asyncio.run(tool.run(context(), args))


def _ok(**body: Any) -> Answer:
    return Answer(body={"ok": True, **body})


def test_the_toolkit_has_read_tools_and_write_tools_only_when_asked() -> None:
    reads = slack.tools(LocalConnections({}))
    assert [tool.name for tool in reads] == [
        "slack_list_channels",
        "slack_read_channel",
        "slack_read_thread",
        "slack_lookup_user",
    ]
    tools = slack.tools(LocalConnections({}), include_writes=True)
    assert {tool.name: tool.spec.effect for tool in tools} == {
        "slack_post_message": ToolEffect.EXTERNAL,
        "slack_update_message": ToolEffect.IDEMPOTENT,
        "slack_list_channels": ToolEffect.NONE,
        "slack_read_channel": ToolEffect.NONE,
        "slack_read_thread": ToolEffect.NONE,
        "slack_lookup_user": ToolEffect.NONE,
    }
    assert all(tool.spec.max_terminal_payload_bytes == MAX_TERMINAL_PAYLOAD_BYTES for tool in tools)
    schemas = {
        tool.name: tool.schema["properties"]["channel"]["description"]
        for tool in tools
        if "channel" in tool.schema["properties"]
    }
    assert "#general" in schemas["slack_post_message"]
    for name in ("slack_update_message", "slack_read_channel", "slack_read_thread"):
        assert "not its name" in schemas[name]
    assert all("may need a person's approval" in tool.description for tool in tools[:2])


def test_post_message_sends_json_to_chat_post_message() -> None:
    broker = FakeBroker(lambda _: _ok(channel="C1", ts="1700.1"))
    result = _run(_tools(broker)["slack_post_message"], channel="C1", text="Héllo", thread_ts="9.9")

    [sent] = broker.sent
    assert (sent.slot, sent.method, sent.path) == ("team-slack", "POST", "/api/chat.postMessage")
    assert sent.headers == {"content-type": "application/json; charset=utf-8"}
    assert sent.json() == {"channel": "C1", "text": "Héllo", "thread_ts": "9.9"}
    assert not result.is_error
    assert result.structured == {"channel": "C1", "ts": "1700.1", "thread_ts": "9.9"}
    assert json.loads(result.content) == result.structured


def test_update_message_sends_json_to_chat_update() -> None:
    broker = FakeBroker(lambda _: _ok(channel="C1", ts="1700.1"))
    result = _run(_tools(broker)["slack_update_message"], channel="C1", ts="1700.1", text="Edited")

    [sent] = broker.sent
    assert (sent.method, sent.path) == ("POST", "/api/chat.update")
    assert sent.json() == {"channel": "C1", "ts": "1700.1", "text": "Edited"}
    assert result.structured == {"channel": "C1", "ts": "1700.1"}


def test_slack_ok_false_is_a_tool_error_with_slacks_error() -> None:
    broker = FakeBroker(lambda _: Answer(body={"ok": False, "error": "channel_not_found"}))
    result = _run(_tools(broker)["slack_post_message"], channel="C404", text="hi")

    assert result.is_error
    assert result.structured is not None
    assert result.structured["error"] == "slack_error"
    assert result.structured["slack_error"] == "channel_not_found"
    assert "channel_not_found" in result.content


def test_list_channels_reads_public_channels_with_a_cursor() -> None:
    def connector(sent: Sent) -> Answer:
        return _ok(
            channels=[
                {
                    "id": "C1",
                    "name": "general",
                    "is_member": True,
                    "num_members": 12,
                    "topic": {"value": "t" * 400},
                    "purpose": {"value": ""},
                    "created": 1,
                }
            ],
            response_metadata={"next_cursor": "dGVhbTpDMQ=="},
        )

    broker = FakeBroker(connector)
    tool = _tools(broker)["slack_list_channels"]
    first = _run(tool)
    _run(tool, limit=10, cursor="dGVhbTpDMQ==")

    assert [(sent.method, sent.path) for sent in broker.sent] == [
        ("GET", "/api/conversations.list")
    ] * 2
    assert broker.sent[0].query == {
        "types": "public_channel",
        "exclude_archived": "true",
        "limit": "50",
    }
    assert broker.sent[1].query["limit"] == "10"
    assert broker.sent[1].query["cursor"] == "dGVhbTpDMQ=="
    assert first.structured is not None
    [channel] = first.structured["channels"]
    assert channel["id"] == "C1" and "created" not in channel and "purpose" not in channel
    assert len(channel["topic"]) == 250
    assert first.structured["next_cursor"] == "dGVhbTpDMQ=="


def test_a_last_page_has_no_cursor_and_limit_is_bounded() -> None:
    broker = FakeBroker(lambda _: _ok(channels=[], response_metadata={"next_cursor": ""}))
    tool = _tools(broker)["slack_list_channels"]
    assert _run(tool).structured == {"channels": [], "next_cursor": None}
    for limit in (0, slack.MAX_PAGE_SIZE + 1, "5", True):
        refused = _run(tool, limit=limit)
        assert refused.is_error
        assert refused.structured is not None
        assert refused.structured["error"] == "invalid_arguments"
    assert len(broker.sent) == 1


def _long_channels(name_length: int) -> FakeBroker:
    def channel(index: int) -> dict[str, Any]:
        return {
            "id": f"C{index:010d}",
            "name": f"{index:03d}-" + "n" * (name_length - 4),
            "is_member": True,
            "num_members": 1000,
            "topic": {"value": "t" * 250},
            "purpose": {"value": "p" * 250},
        }

    def connector(sent: Sent) -> Answer:
        limit = int(sent.query["limit"])
        return _ok(
            channels=[channel(index) for index in range(limit)],
            response_metadata={"next_cursor": f"after-{limit}"},
        )

    return FakeBroker(connector)


def test_a_page_of_200_long_channels_fits_without_losing_a_channel() -> None:
    broker = _long_channels(80)
    result = _run(_tools(broker)["slack_list_channels"], limit=200)

    assert result.structured is not None
    channels = result.structured["channels"]
    assert all("topic" not in item and "purpose" not in item for item in channels)
    assert_within_budgets(result)
    # The page answered ends where the next page's cursor starts.
    assert int(broker.sent[-1].query["limit"]) == len(channels) > 100
    assert result.structured["next_cursor"] == f"after-{len(channels)}"


def test_a_page_that_never_fits_is_asked_for_again_smaller_from_the_same_cursor() -> None:
    broker = _long_channels(400)
    result = _run(_tools(broker)["slack_list_channels"], limit=200, cursor="c1")

    assert not result.is_error
    assert result.structured is not None
    channels = result.structured["channels"]
    [_, again] = broker.sent
    assert again.query["cursor"] == "c1"
    assert int(again.query["limit"]) == len(channels) < 200
    assert result.structured["next_cursor"] == f"after-{len(channels)}"
    assert_within_budgets(result)


def test_a_page_whose_topics_dont_fit_keeps_them_shorter() -> None:
    channels = [
        {
            "id": f"C{index}",
            "name": f"c{index}",
            "topic": {"value": "t" * 250},
            "purpose": {"value": "p" * 250},
        }
        for index in range(120)
    ]
    broker = FakeBroker(lambda _: _ok(channels=channels))
    result = _run(_tools(broker)["slack_list_channels"], limit=120)
    assert result.structured is not None
    assert len(result.structured["channels"]) == 120
    assert {len(item["topic"]) for item in result.structured["channels"]} == {80}
    assert {len(item["purpose"]) for item in result.structured["channels"]} == {80}
    assert len(broker.sent) == 1


def test_read_channel_pages_history_and_shortens_long_messages() -> None:
    def connector(sent: Sent) -> Answer:
        return _ok(
            messages=[
                {"ts": "2.0", "user": "U1", "text": "x" * 5_000, "blocks": [{"big": True}]},
                {"ts": "1.0", "user": "U2", "text": "short", "thread_ts": "1.0", "reply_count": 3},
            ],
            has_more=True,
            response_metadata={"next_cursor": "bmV4dA=="},
        )

    broker = FakeBroker(connector)
    result = _run(
        _tools(broker)["slack_read_channel"], channel="C1", limit=2, oldest="0.5", latest="3.0"
    )

    [sent] = broker.sent
    assert (sent.method, sent.path) == ("GET", "/api/conversations.history")
    assert sent.query == {"channel": "C1", "limit": "2", "oldest": "0.5", "latest": "3.0"}
    assert result.structured is not None
    long, short = result.structured["messages"]
    assert len(long["text"]) == 4_000 and long["text_shortened"] is True
    assert "blocks" not in long
    assert short == {
        "ts": "1.0",
        "user": "U2",
        "thread_ts": "1.0",
        "reply_count": 3,
        "text": "short",
    }
    assert result.structured["has_more"] is True
    assert result.structured["next_cursor"] == "bmV4dA=="


def test_a_page_too_large_for_the_answer_is_shortened_further_to_fit() -> None:
    messages = [{"ts": f"{index}.0", "text": "y" * 4_000} for index in range(50)]
    broker = FakeBroker(lambda _: _ok(messages=messages))
    result = _run(_tools(broker)["slack_read_channel"], channel="C1")

    assert not result.is_error
    assert result.structured is not None
    assert len(result.structured["messages"]) == 50
    assert all(len(message["text"]) < 4_000 for message in result.structured["messages"])
    assert_within_budgets(result)


def test_read_thread_reads_replies() -> None:
    broker = FakeBroker(lambda _: _ok(messages=[{"ts": "1.0", "text": "parent"}]))
    result = _run(_tools(broker)["slack_read_thread"], channel="C1", thread_ts="1.0")

    [sent] = broker.sent
    assert (sent.method, sent.path) == ("GET", "/api/conversations.replies")
    assert sent.query == {"channel": "C1", "ts": "1.0", "limit": "50"}
    assert result.structured is not None
    assert result.structured["messages"] == [{"ts": "1.0", "text": "parent"}]
    assert result.structured["next_cursor"] is None


def test_lookup_user_by_id_or_email_and_never_both() -> None:
    user = {
        "id": "U1",
        "name": "ada",
        "real_name": "Ada L",
        "is_bot": False,
        "profile": {"email": "ada@example.test", "display_name": "ada", "image_512": "x"},
    }
    broker = FakeBroker(lambda _: _ok(user=user))
    tool = _tools(broker)["slack_lookup_user"]
    by_id = _run(tool, user_id="U1")
    _run(tool, email="ada@example.test")

    assert [(sent.path, sent.query) for sent in broker.sent] == [
        ("/api/users.info", {"user": "U1"}),
        ("/api/users.lookupByEmail", {"email": "ada@example.test"}),
    ]
    assert by_id.structured == {
        "user": {
            "id": "U1",
            "name": "ada",
            "real_name": "Ada L",
            "is_bot": False,
            "display_name": "ada",
            "email": "ada@example.test",
        }
    }
    for args in ({}, {"user_id": "U1", "email": "ada@example.test"}):
        refused = _run(tool, **args)
        assert refused.is_error
        assert refused.structured is not None
        assert refused.structured["error"] == "invalid_arguments"
    assert len(broker.sent) == 2


@pytest.mark.parametrize(
    ("refusal", "code", "advice"),
    [
        (Refusal(404, "connection_unbound"), "connection_unbound", "An admin must bind"),
        (Refusal(409, "connection_not_ready"), "connection_not_ready", "An admin must bind"),
        (Refusal(422, "scope_not_allowed"), "scope_not_allowed", "don't retry"),
        (Refusal(422, "path_not_allowed"), "path_not_allowed", "don't retry"),
        (Refusal(502, "upstream_failed"), "upstream_failed", "retry later"),
        (
            Refusal(429, "too_many_calls", headers={"Retry-After": "2"}),
            "too_many_calls",
            "retry after 2 seconds",
        ),
    ],
)
def test_a_broker_refusal_is_a_tool_error_naming_its_code_and_what_to_do(
    refusal: Refusal, code: str, advice: str
) -> None:
    broker = FakeBroker(lambda _: refusal)
    result = _run(_tools(broker)["slack_list_channels"])

    assert result.is_error
    assert result.structured is not None
    assert result.structured["error"] == code
    assert advice in result.structured["advice"]
    assert code in result.content
    if code == "connection_unbound":
        assert "team-slack" in result.structured["advice"]
    if code == "too_many_calls":
        assert result.structured["retry_after_seconds"] == 2


def test_slacks_own_rate_limit_carries_its_delay() -> None:
    broker = FakeBroker(lambda _: Answer(status=429, body=b"", headers={"Retry-After": "30"}))
    result = _run(_tools(broker)["slack_read_channel"], channel="C1")

    assert result.structured is not None
    assert result.structured["error"] == "too_many_calls"
    assert result.structured["retry_after_seconds"] == 30
    assert "30 seconds" in result.content


def test_a_failed_slack_answer_is_a_temporary_tool_error() -> None:
    broker = FakeBroker(lambda _: Answer(status=503, body=b"<html>"))
    result = _run(_tools(broker)["slack_read_channel"], channel="C1")
    assert result.structured is not None
    assert (result.structured["error"], result.structured["status"]) == ("upstream_status", 503)


def test_local_connections_send_the_same_requests_with_the_slots_token() -> None:
    sent: list[httpx.Request] = []

    def slack_api(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"ok": True, "channel": "C1", "ts": "1.1"})

    local = LocalConnections(
        {"slack": LocalSlot("https://slack.com", {"authorization": "Bearer xoxb-dev"})},
        transport=httpx.MockTransport(slack_api),
    )
    [post, *_] = slack.tools(local, include_writes=True)
    result = _run(post, channel="C1", text="hi")

    [request] = sent
    assert str(request.url) == "https://slack.com/api/chat.postMessage"
    assert request.headers["authorization"] == "Bearer xoxb-dev"
    assert request.headers["content-type"] == "application/json; charset=utf-8"
    assert result.structured == {"channel": "C1", "ts": "1.1", "thread_ts": None}


def test_without_a_client_outside_cloud_the_tools_answer_an_error(monkeypatch) -> None:
    monkeypatch.delenv("CAYU_CONNECTIONS_URL", raising=False)
    [post, *_] = slack.tools(include_writes=True)
    result = _run(post, channel="C1", text="hi")
    assert result.is_error
    assert result.structured is not None
    assert result.structured["error"] == "no_connection_client"


def test_approval_policy_asks_before_writes_naming_the_channel() -> None:
    policy = slack.approval_policy()

    def decide(tool_name: str, **arguments: Any):
        return asyncio.run(policy.authorize(policy_request(tool_name, **arguments)))

    post = decide("slack_post_message", channel="C123", text="hi")
    reply = decide("slack_post_message", channel="#ops", text="hi", thread_ts="1.0")
    edit = decide("slack_update_message", channel="C123", ts="1.0", text="hi")
    read = decide("slack_read_channel", channel="C123")

    assert post.decision == ToolPolicyDecision.REQUIRE_APPROVAL
    assert post.reason == "Post a Slack message to C123."
    assert reply.reason == "Reply in a Slack thread in #ops."
    assert edit.decision == ToolPolicyDecision.REQUIRE_APPROVAL
    assert edit.reason == "Edit a Slack message in C123."
    assert read.decision == ToolPolicyDecision.ALLOW
    long = decide("slack_post_message", channel="C1\n" + "C" * 500, text="hi")
    assert long.reason is not None and len(long.reason) < 120 and "\n" not in long.reason


def test_approval_policy_leaves_every_other_decision_to_inner() -> None:
    denying = slack.approval_policy(
        StaticToolPolicy(deny=["slack_post_message", "slack_read_thread"])
    )
    asking = slack.approval_policy(
        AlwaysRequireApprovalToolPolicy(tools=["slack_lookup_user"], approval_gate="people")
    )

    def decide(policy, tool_name: str):
        return asyncio.run(policy.authorize(policy_request(tool_name, channel="C1")))

    assert decide(denying, "slack_post_message").decision == ToolPolicyDecision.DENY
    assert decide(denying, "slack_read_thread").decision == ToolPolicyDecision.DENY
    assert decide(denying, "slack_update_message").decision == ToolPolicyDecision.REQUIRE_APPROVAL
    gated = decide(asking, "slack_lookup_user")
    assert gated.decision == ToolPolicyDecision.REQUIRE_APPROVAL
    assert gated.metadata == {"cayu:approval_gate": "people"}


def test_review_fields_show_the_channel_and_the_text() -> None:
    fields = slack.review_fields()
    assert fields == {
        "slack_post_message": ("channel", "thread_ts", "text"),
        "slack_update_message": ("channel", "ts", "text"),
    }
    ToolArgumentReview(fields, binding_key=b"k" * 32)


@pytest.mark.parametrize(
    "refusal",
    [Refusal(502, "upstream_failed"), Refusal(502, "upstream_response_too_large")],
)
def test_a_post_that_may_have_run_is_ambiguous_and_never_advised_to_retry(
    refusal: Refusal,
) -> None:
    broker = FakeBroker(lambda _: refusal)
    tools = _tools(broker)
    result = _run(tools["slack_post_message"], channel="C1", text="Deployed.")

    assert result.is_error and result.structured is not None
    assert result.structured["error"] == refusal.code
    assert result.structured["ambiguous"] is True
    assert "may have taken effect" in result.structured["advice"]
    assert "read the channel's latest messages" in result.structured["advice"]
    assert "retry later" not in result.content

    read = _run(tools["slack_list_channels"])
    assert read.structured is not None and "ambiguous" not in read.structured

    refused = FakeBroker(lambda _: Refusal(422, "scope_not_allowed"))
    unsent = _run(_tools(refused)["slack_post_message"], channel="C1", text="Deployed.")
    assert unsent.structured is not None and "ambiguous" not in unsent.structured
