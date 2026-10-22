"""Service-owned model credential proxy and per-run native-tool MCP endpoint.

This optional HTTP surface requires ``cayu[server]``. Mount its router only on
the deployment's internal sandbox network. Every route also requires a fresh,
revocable run capability; model credentials never enter the guest environment.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from cayu.deadlines import execution_deadline_scope
from cayu.external_agents._model_protocol import response_usage, validate_model_request
from cayu.external_agents.base import (
    ExternalAgentBudgetExceeded,
    ExternalAgentContext,
    ExternalAgentError,
    ExternalAgentEvent,
)
from cayu.external_agents.environment import join_cleanup
from cayu.external_agents.tools import CallableToolEndpoint
from cayu.runtime.mcp_manifest_policy import McpManifestPolicyError
from cayu.sessions.mcp_manifest_history import McpManifestHistoryConflict
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    import httpx


@dataclass(frozen=True)
class ExternalModelRoute:
    """Trusted upstream configuration. Headers may contain service credentials."""

    base_url: str
    headers: Mapping[str, str]
    paths: tuple[str, ...]


@dataclass(frozen=True)
class ExternalAgentCapability:
    token: str
    model_base_url: str
    mcp_url: str


@dataclass
class _Grant:
    context: ExternalAgentContext
    route: str
    models: frozenset[str]
    tools: CallableToolEndpoint | None
    capability_redactor: SecretRedactor
    tasks: set[asyncio.Task] = field(default_factory=set)
    responses: set[Any] = field(default_factory=set)


def _require_gateway_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Gateway URLs require a fixed HTTP(S) origin and path without credentials."
        )
    return value


class ExternalAgentGateway:
    """Application-owned HTTP gateway. The caller owns the HTTP client lifetime.

    Each capability can use exactly one upstream route and the explicitly
    named models until its execution deadline or scope exit. Configure route
    paths explicitly, e.g. ``('/v1/messages',)`` and ``('/responses',)``.
    Redirects are never followed and caller headers cannot replace credentials.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        public_base_url: str,
        routes: Mapping[str, ExternalModelRoute],
    ) -> None:
        self.client = client
        self.public_base_url = public_base_url.rstrip("/")
        self.routes = dict(routes)
        self.grants: dict[str, _Grant] = {}
        for name, route in self.routes.items():
            if not name.isidentifier() or not route.base_url.startswith(("http://", "https://")):
                raise ValueError("External model routes need fixed names and HTTP(S) upstreams.")
            if not route.paths or any(
                path not in {"/messages", "/v1/messages", "/responses", "/v1/responses"}
                for path in route.paths
            ):
                raise ValueError("External model routes support only Messages and Responses paths.")
        for value in (public_base_url, *(route.base_url for route in self.routes.values())):
            _require_gateway_url(value)
        self.router = APIRouter()
        self.router.add_api_route(
            "/models/{route}/{path:path}", self.model_request, methods=["POST"]
        )
        self.router.add_api_route("/mcp", self.mcp_request, methods=["POST"])

    @asynccontextmanager
    async def grant(
        self,
        context: ExternalAgentContext,
        *,
        route: str,
        models: frozenset[str],
        tools: CallableToolEndpoint | None = None,
        public_base_url: str | None = None,
    ) -> AsyncIterator[ExternalAgentCapability]:
        """Issue one run's bearer capability, revoked when the block exits.

        ``public_base_url`` replaces the gateway's base URL in this capability's
        URLs. Use it when the guest reaches the gateway at a different address,
        such as ``runner_port_forward(...).guest_url``.
        """
        if route not in self.routes or not models:
            raise ValueError("A grant requires an existing route and explicit models.")
        base_url = (
            self.public_base_url
            if public_base_url is None
            else _require_gateway_url(public_base_url).rstrip("/")
        )
        context.execution_deadline.require_admission("external gateway grant")
        if context.check_budget is not None and context.record_model_usage is None:
            raise ValueError("Budgeted gateway grants require authoritative usage recording.")
        if context.claim_model_usage is not None:
            context.claim_model_usage()
        token = secrets.token_urlsafe(32)
        if context.protect_secret is not None:
            context.protect_secret(token)
        grant = _Grant(
            context, route, frozenset(models), tools, SecretRedactor().with_secret(token)
        )
        self.grants[token] = grant
        try:
            yield ExternalAgentCapability(token, f"{base_url}/models/{route}", f"{base_url}/mcp")
        finally:
            self.grants.pop(token, None)

            async def revoke() -> None:
                if tools is not None:
                    await tools.close()
                owned = tuple(grant.tasks)
                for task in owned:
                    task.cancel()
                await asyncio.gather(*owned, return_exceptions=True)
                await asyncio.gather(*(response.aclose() for response in tuple(grant.responses)))
                grant.responses.clear()

            await join_cleanup(revoke())

    @asynccontextmanager
    async def _owned(self, request: Request, grant: _Grant):
        if self._grant(request) is not grant:
            raise HTTPException(401, "External run capability changed.")
        task = asyncio.current_task()
        assert task is not None
        grant.tasks.add(task)
        try:
            async with execution_deadline_scope(grant.context.execution_deadline):
                yield
        finally:
            grant.tasks.discard(task)

    def _grant(self, request: Request, *, require_accounted: bool = False) -> _Grant:
        authorization = request.headers.get("authorization", "")
        token = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
        grant = self.grants.get(token)
        if grant is None:
            raise HTTPException(401, "External run capability is unavailable.")
        if grant.context.execution_deadline.expired:
            raise HTTPException(403, "External run capability has expired.")
        if require_accounted and grant.context.pending_model_usage:
            raise HTTPException(403, "External run accounting requires reconciliation.")
        return grant

    async def _body(self, request: Request) -> dict[str, Any]:
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 8 * 1024 * 1024:
                raise HTTPException(413, "External request is too large.")
        try:
            value = json.loads(body)
        except ValueError:
            raise HTTPException(400, "External request must be JSON.") from None
        if type(value) is not dict:
            raise HTTPException(400, "External request must be an object.")
        return value

    async def model_request(self, route: str, path: str, request: Request) -> Response:
        grant = self._grant(request)
        configured = self.routes.get(route)
        if configured is None or grant.route != route or f"/{path}" not in configured.paths:
            raise HTTPException(403, "External model route is not authorized.")
        async with self._owned(request, grant):
            body = await self._body(request)
            if type(body.get("model")) is not str or body["model"] not in grant.models:
                raise HTTPException(403, "External model is not authorized.")
            protocol = path.rsplit("/", 1)[-1]
            try:
                validate_model_request(body, protocol)
            except (ValueError, TypeError, RecursionError):
                raise HTTPException(400, "Unsupported external model request.") from None
            # The lock belongs to the run context, including when the service
            # grants several capabilities. Usage commits before another dispatch
            # or before the guest receives a completed response.
            async with grant.context.model_request_lock:
                self._grant(request, require_accounted=True)
                if grant.context.check_budget is not None:
                    try:
                        await grant.context.check_budget()
                    except ExternalAgentBudgetExceeded:
                        raise HTTPException(403, "External run budget exhausted.") from None
                headers = {
                    name: value
                    for name, value in request.headers.items()
                    if name.lower() in {"anthropic-version", "accept"}
                }
                # Beta features are selected only by trusted service configuration.
                headers.update(configured.headers)
                headers["content-type"] = "application/json"
                upstream = self.client.build_request(
                    "POST",
                    f"{configured.base_url.rstrip('/')}/{path}",
                    headers=headers,
                    json=body,
                    timeout=grant.context.execution_deadline.remaining_seconds(),
                )
                response = None
                usage: dict[str, Any] = {}
                model = body["model"]
                observation_id = f"gateway:{uuid4().hex}"
                try:
                    response = await self.client.send(upstream, stream=True, follow_redirects=False)
                    grant.responses.add(response)
                    if response.is_redirect:
                        raise HTTPException(502, "External upstream redirect was refused.")
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        self._grant(request)
                        content.extend(chunk)
                        if len(content) > 64 * 1024 * 1024:
                            raise HTTPException(
                                502, "External model response exceeds its size limit."
                            )
                    received = bytes(content)
                    del content
                    if response.is_success:
                        usage, reported_model = response_usage(
                            received, protocol, streaming=body.get("stream", False)
                        )
                        model = reported_model or model
                    else:
                        # A fully received rejection did not admit inference.
                        # Interrupted reads and incomplete success receipts keep
                        # unknown usage, so they cannot authorize another call.
                        usage = {"input_tokens": 0, "output_tokens": 0}
                finally:

                    async def finish() -> None:
                        try:
                            if response is not None:
                                await response.aclose()
                                grant.responses.discard(response)
                        finally:
                            if grant.context.record_model_usage is not None:
                                observation = ExternalAgentEvent(
                                    "usage",
                                    usage,
                                    observation_id,
                                    model,
                                    usage_dialect="anthropic"
                                    if protocol == "messages"
                                    else "openai",
                                )
                                grant.context.pending_model_usage[observation_id] = observation
                                await grant.context.record_model_usage(observation)
                                grant.context.pending_model_usage.pop(observation_id)

                            if grant.context.check_budget is not None:
                                # Losing execution authority during cancellation
                                # must not replace the original cancellation after
                                # the interrupted call's receipt is committed.
                                with suppress(ExternalAgentError):
                                    await grant.context.check_budget()

                    await join_cleanup(finish())
                secrets_to_redact = tuple(configured.headers.values()) + tuple(
                    value.removeprefix("Bearer ")
                    for value in configured.headers.values()
                    if value.startswith("Bearer ")
                )
                redactor = SecretRedactor(secrets_to_redact).stream_bytes(
                    max_retained_bytes=1024 * 1024
                )
                safe = redactor.feed(received) + redactor.finish_complete()
                return Response(
                    safe,
                    status_code=response.status_code,
                    media_type=response.headers.get("content-type", "application/json"),
                )

    async def mcp_request(self, request: Request) -> Response:
        grant = self._grant(request)
        if grant.tools is None:
            raise HTTPException(403, "No native tools were granted.")
        async with self._owned(request, grant):
            body = await self._body(request)
            if grant.capability_redactor.redact_json(body) != body or (
                grant.context.contains_secret is not None and grant.context.contains_secret(body)
            ):
                raise HTTPException(400, "MCP arguments must not contain protected credentials.")
            method, request_id = body.get("method"), body.get("id")
            if body.get("jsonrpc") != "2.0" or not isinstance(method, str):
                raise HTTPException(400, "Malformed MCP request.")
            if request_id is None:
                return Response(status_code=202)
            params = body.get("params", {})
            if type(params) is not dict:
                raise HTTPException(400, "Malformed MCP parameters.")
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "cayu-native-tools", "version": "1"},
                }
            elif method == "ping":
                result = {}
            elif method in {"tools/list", "tools/call"}:
                try:
                    async with grant.context.model_request_lock:
                        self._grant(request, require_accounted=True)
                        if grant.context.check_budget is not None:
                            try:
                                await grant.context.check_budget()
                            except ExternalAgentBudgetExceeded:
                                raise HTTPException(403, "External run budget exhausted.") from None
                        result = await grant.tools.dispatch(method, params)
                except (McpManifestPolicyError, McpManifestHistoryConflict):
                    raise HTTPException(403, "Native tool manifest exposure was refused.") from None
                except (ValueError, KeyError, TypeError):
                    return JSONResponse(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32602, "message": "Invalid parameters."},
                        }
                    )
            else:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "Method not found."},
                    }
                )
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})
