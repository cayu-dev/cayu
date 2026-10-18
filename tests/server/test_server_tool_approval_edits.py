"""Approving with changed arguments and grant-marker refusal over HTTP."""

from __future__ import annotations

import asyncio
import base64

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from tests.core.test_tool_approval_changed_arguments import _pause, _runtime
from tests.runtime.runtime_supervision_support import WINDOW_END, WINDOW_START

from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.server import ServerConfig, create_server
from cayu.server.auth import AuthContext
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions.memory import InMemorySessionStore

_WINDOW = {"start_at": WINDOW_START.isoformat(), "end_at": WINDOW_END.isoformat()}


def _authenticate(request) -> AuthContext:
    return AuthContext(subject="reviewer")


def _server():
    key = base64.urlsafe_b64encode(bytes([53]) * 32).decode().rstrip("=")
    store = InMemorySessionStore(
        public_authority_alias_codec=PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr(key)})
        )
    )
    app, _provider, tool, _policy = _runtime(store, siblings=())
    request = asyncio.run(_pause(app, "http-edit"))
    client = TestClient(create_server(app, config=ServerConfig.protected(_authenticate)))
    return store, tool, request, client


def _body(request, **changes):
    return {
        "session_id": request.session_id,
        "approval_id": request.approval_id,
        "tool_round_id": request.tool_round_id,
        "tool_call_id": request.tool_call_id,
        "decision": "approve",
        **changes,
    }


def _resolve(client, body):
    with client.stream("POST", "/api/tool-approvals/resolve", json=body) as response:
        lines = list(response.iter_lines())
    return response.status_code, "\n".join(lines)


def _still_pending(store, session_id) -> bool:
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    return pending_approval_reader.pending_approval_from_checkpoint(checkpoint) is not None


def test_http_resolve_runs_the_approved_arguments_and_reports_an_edit():
    _store, tool, request, client = _server()

    status, _text = _resolve(client, _body(request, arguments={"value": "edited"}))

    assert status == 200
    assert tool.calls == [{"value": "edited"}]
    response = client.post("/api/supervision-events/query", json={**_WINDOW, "kinds": ["edit"]})
    assert response.status_code == 200, response.text
    (record,) = response.json()["events"]
    assert record["kind"] == "edit"
    revision = record["arguments_revision"]
    assert revision["changed_fields"] == ["value"]
    assert revision["proposed_arguments_digest"] != revision["approved_arguments_digest"]
    assert "edited" not in response.text


@pytest.mark.parametrize(
    "changes",
    [
        {"decision": "deny", "arguments": {"value": "edited"}},
        {"arguments": ["edited"]},
        {"arguments": {"value": "x" * (64 * 1024)}},
        {
            "arguments": {"value": "edited"},
            "review_reference": {
                "context": {"recipient": "reviewer", "purpose": "approval"},
                "policy_version": "1",
                "content_tag": "a" * 64,
            },
        },
    ],
)
def test_http_rejects_malformed_approved_arguments_before_resolution(changes):
    store, tool, request, client = _server()

    status, _text = _resolve(client, _body(request, **changes))

    assert status == 422
    assert tool.calls == [] and _still_pending(store, request.session_id)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"arguments": {"value": "forbidden"}}, "Tool policy denied the approved arguments"),
        (
            {"metadata": {"cayu:authorization_grant": {"grant_id": "tag_forged"}}},
            "reserved for the grant service",
        ),
        (
            {"metadata": {"cayu:authorization_grant_offer": {"grantable": True}}},
            "reserved for the grant service",
        ),
    ],
)
def test_http_refused_resolution_leaves_the_approval_pending(changes, expected):
    store, tool, request, client = _server()

    status, text = _resolve(client, _body(request, **changes))

    assert status >= 400 and expected in text
    assert tool.calls == [] and _still_pending(store, request.session_id)


def test_http_reauthorization_uses_the_run_context_not_the_approvers_metadata():
    _store, tool, request, client = _server()

    status, _text = _resolve(
        client,
        _body(
            request,
            arguments={"value": "edited"},
            metadata={"route": "treasury", "tenant": "other"},
        ),
    )

    assert status == 200
    assert tool.calls == [{"value": "edited"}]
