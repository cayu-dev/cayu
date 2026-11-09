"""A fake connection broker for connector toolkit tests: it answers each request the way Cayu
Cloud's broker does, from a fake connector, and records what it was sent."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from cayu import AgentSpec
from cayu.connections import CloudConnectionBroker
from cayu.sessions.records import Session, SessionStatus
from cayu.tools import ToolContext, ToolPolicyRequest
from tests._session_provenance import fixture_session_invocation

BROKER = "https://api.example.test/v1/agent-connections"


@dataclass(frozen=True)
class Sent:
    slot: str
    method: str
    path: str
    query: dict[str, Any]
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


@dataclass(frozen=True)
class Answer:
    status: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Refusal:
    """The broker's own refusal, never reaching the connector."""

    status: int
    code: str
    message: str = "Refused."
    headers: dict[str, str] = field(default_factory=dict)


Connector = Callable[[Sent], Answer | Refusal]


class FakeBroker:
    def __init__(self, connector: Connector) -> None:
        self.connector = connector
        self.sent: list[Sent] = []
        self.client = CloudConnectionBroker(
            url=BROKER, token="cayu_conn_test", transport=httpx.MockTransport(self._handle)
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        slot = request.url.path.removeprefix("/v1/agent-connections/").removesuffix("/request")
        document = json.loads(request.content)
        sent = Sent(
            slot=slot,
            method=document["method"],
            path=document["path"],
            query=document["query"],
            headers=document["headers"],
            body=base64.b64decode(document["body"]),
        )
        self.sent.append(sent)
        answer = self.connector(sent)
        if isinstance(answer, Refusal):
            return httpx.Response(
                answer.status,
                headers=answer.headers,
                json={"detail": {"code": answer.code, "message": answer.message}},
            )
        body = answer.body if isinstance(answer.body, bytes) else json.dumps(answer.body).encode()
        return httpx.Response(
            200,
            json={
                "status": answer.status,
                "headers": [[name.lower(), value] for name, value in answer.headers.items()],
                "body": base64.b64encode(body).decode(),
            },
        )


def context(**kwargs: Any) -> ToolContext:
    return ToolContext(session_id="session-1", **kwargs)


def policy_request(tool_name: str, **arguments: Any) -> ToolPolicyRequest:
    """A tool policy's request for one call to ``tool_name`` with ``arguments``."""

    return ToolPolicyRequest(
        session=Session(
            id="s",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            invocation=fixture_session_invocation("s"),
            status=SessionStatus.RUNNING,
        ),
        agent=AgentSpec(name="assistant", model="fake-model"),
        tool_name=tool_name,
        tool_call_id="call_1",
        arguments=arguments,
    )
