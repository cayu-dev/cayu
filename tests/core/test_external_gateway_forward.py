"""A CLI in a runner reaches the model only through the gateway, over a guest port forward."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from tests.core.test_external_runner_launch import _run_shim
from tests.runners.lambda_microvm_harness import ConformanceLambdaClient
from tests.runners.test_aws_lambda_microvm_interactive import InteractiveTransport
from tests.runners.test_aws_lambda_microvm_port_forward import ForwardingTransport

from cayu.deadlines import ExecutionDeadline
from cayu.external_agents import (
    codex_proxy_overrides,
    runner_agent_launch,
    runner_port_forward,
)
from cayu.external_agents.base import ExternalAgentBudgetExceeded, ExternalAgentContext
from cayu.external_agents.server import ExternalAgentGateway, ExternalModelRoute
from cayu.runners import LambdaMicroVMRunner

ANTHROPIC_KEY = "sk-ant-service-key-never-in-guest"
OPENAI_KEY = "sk-openai-service-key-never-in-guest"
CLAUDE_MODEL = "claude-test"
CODEX_MODEL = "codex-test"

# Stands in for the vendor CLI: one model call with the guest's configuration,
# reported with the guest environment so the test can check what it held.
_FAKE_CLI = """
import json, os, sys, urllib.error, urllib.request
base, token, path, body = sys.argv[1:5]
request = urllib.request.Request(
    os.environ[base] + path,
    data=body.encode(),
    headers={
        "authorization": "Bearer " + os.environ[token],
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": "guest-supplied",
    },
)
try:
    with urllib.request.urlopen(request, timeout=20) as response:
        status, text = response.status, response.read().decode()
except urllib.error.HTTPError as error:
    status, text = error.code, error.read().decode()
print(json.dumps({"status": status, "body": text, "environment": dict(os.environ)}))
"""


def _runner(root: Path, transport: Any) -> LambdaMicroVMRunner:
    return LambdaMicroVMRunner(
        ConformanceLambdaClient(),
        microvm_id="mvm-gateway",
        endpoint="gateway.lambda-microvm.invalid",
        region_name="us-east-1",
        default_cwd=str(root),
        endpoint_transport=transport,
        poll_interval_s=0,
    )


@asynccontextmanager
async def _gateway(
    upstream: Callable[[httpx.Request], httpx.Response],
) -> AsyncIterator[tuple[ExternalAgentGateway, int]]:
    """The gateway on this host's loopback only, with a fake upstream model API."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        gateway = ExternalAgentGateway(
            client=client,
            public_base_url="http://127.0.0.1",
            routes={
                "anthropic": ExternalModelRoute(
                    "https://api.anthropic.test", {"x-api-key": ANTHROPIC_KEY}, ("/v1/messages",)
                ),
                "openai": ExternalModelRoute(
                    "https://api.openai.test/v1",
                    {"authorization": f"Bearer {OPENAI_KEY}"},
                    ("/responses",),
                ),
            },
        )
        surface = FastAPI()
        surface.include_router(gateway.router)
        server = uvicorn.Server(
            uvicorn.Config(
                surface,
                host="127.0.0.1",
                port=0,
                log_level="warning",
                lifespan="off",
                ws="none",
            )
        )
        serving = asyncio.create_task(server.serve())
        try:
            while not server.started:
                assert not serving.done()
                await asyncio.sleep(0.01)
            yield gateway, server.servers[0].sockets[0].getsockname()[1]
        finally:
            server.should_exit = True
            await serving


def _context(*, check_budget=None, usage: list | None = None, secrets: list | None = None):
    async def record(event) -> None:
        assert usage is not None
        usage.append(event)

    return ExternalAgentContext(
        "run",
        "prompt",
        ExecutionDeadline.after(120),
        check_budget=check_budget,
        record_model_usage=record if usage is not None else None,
        protect_secret=None if secrets is None else secrets.append,
    )


def _messages_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": CLAUDE_MODEL,
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
    )


async def _call(launch, base: str, token: str, path: str, body: dict) -> dict:
    code, stdout = await _run_shim(
        launch, "-c", _FAKE_CLI, base, token, path, json.dumps(body), stdin=b""
    )
    assert code == 0, stdout
    return json.loads(stdout)


def test_claude_configuration_reaches_the_model_only_through_the_gateway(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    usage: list = []
    secrets: list[str] = []
    exhausted = False

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _messages_response(request)

    async def check_budget() -> None:
        if exhausted:
            raise ExternalAgentBudgetExceeded("External run budget exhausted.")

    async def scenario() -> None:
        nonlocal exhausted
        transport = ForwardingTransport(tmp_path)
        runner = _runner(tmp_path, transport)
        request = {
            "model": CLAUDE_MODEL,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        }
        async with _gateway(upstream) as (gateway, gateway_port):
            context = _context(check_budget=check_budget, usage=usage, secrets=secrets)
            async with runner_port_forward(
                runner, host="127.0.0.1", port=gateway_port, timeout_s=300
            ) as forward:
                async with gateway.grant(
                    context,
                    route="anthropic",
                    models=frozenset({CLAUDE_MODEL}),
                    public_base_url=forward.guest_url,
                ) as capability:
                    assert capability.model_base_url == f"{forward.guest_url}/models/anthropic"
                    async with runner_agent_launch(
                        runner=runner,
                        cli=sys.executable,
                        environment={
                            "ANTHROPIC_BASE_URL": capability.model_base_url,
                            "ANTHROPIC_AUTH_TOKEN": capability.token,
                        },
                        timeout_s=300,
                    ) as launch:
                        answered = await _call(
                            launch,
                            "ANTHROPIC_BASE_URL",
                            "ANTHROPIC_AUTH_TOKEN",
                            "/v1/messages",
                            request,
                        )
                        assert answered["status"] == 200
                        assert json.loads(answered["body"])["content"][0]["text"] == "hello"
                        [sent] = seen
                        assert sent.url == "https://api.anthropic.test/v1/messages"
                        assert sent.headers["x-api-key"] == ANTHROPIC_KEY
                        assert "authorization" not in sent.headers
                        [receipt] = usage
                        assert receipt.payload["input_tokens"] == 3
                        assert receipt.payload["output_tokens"] == 2
                        assert ANTHROPIC_KEY not in json.dumps(answered["environment"])

                        # The budget is checked before the call leaves the gateway.
                        exhausted = True
                        refused = await _call(
                            launch,
                            "ANTHROPIC_BASE_URL",
                            "ANTHROPIC_AUTH_TOKEN",
                            "/v1/messages",
                            request,
                        )
                        assert refused["status"] == 403
                        assert len(seen) == 1
                token = capability.token
                assert secrets == [token]
            # After the run the token is revoked and the guest listener is gone.
            async with httpx.AsyncClient() as client:
                late = await client.post(
                    f"http://127.0.0.1:{gateway_port}/models/anthropic/v1/messages",
                    headers={"authorization": f"Bearer {token}"},
                    json=request,
                )
            assert late.status_code == 401
            with pytest.raises(OSError):
                await asyncio.to_thread(
                    socket.create_connection, (forward.guest_host, forward.guest_port), 2
                )
            assert transport.forwards._forwards == {}
        assert len(seen) == 1

    asyncio.run(scenario())


def test_codex_configuration_reaches_the_model_only_through_the_gateway(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "model": CODEX_MODEL,
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    async def scenario() -> None:
        transport = ForwardingTransport(tmp_path)
        runner = _runner(tmp_path, transport)
        async with (
            _gateway(upstream) as (gateway, gateway_port),
            runner_port_forward(runner, host="127.0.0.1", port=gateway_port) as forward,
            gateway.grant(
                _context(),
                route="openai",
                models=frozenset({CODEX_MODEL}),
                public_base_url=forward.guest_url,
            ) as capability,
            runner_agent_launch(
                runner=runner,
                cli=sys.executable,
                environment={
                    "CAYU_MODEL_PROXY_TOKEN": capability.token,
                    "CAYU_TEST_BASE_URL": capability.model_base_url,
                },
                codex_config_overrides=codex_proxy_overrides(capability.model_base_url),
            ) as launch,
        ):
            assert (
                f"model_providers.cayu.base_url={json.dumps(capability.model_base_url)}"
                in launch.codex_config_overrides
            )
            answered = await _call(
                launch,
                "CAYU_TEST_BASE_URL",
                "CAYU_MODEL_PROXY_TOKEN",
                "/responses",
                {"model": CODEX_MODEL, "input": "hi"},
            )
            assert answered["status"] == 200
            assert OPENAI_KEY not in json.dumps(answered["environment"])
        [sent] = seen
        assert sent.url == "https://api.openai.test/v1/responses"
        assert sent.headers["authorization"] == f"Bearer {OPENAI_KEY}"

    asyncio.run(scenario())


def test_forward_requires_a_runner_that_supports_it(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = _runner(tmp_path, InteractiveTransport(tmp_path))
        with pytest.raises(ValueError, match="forward guest ports"):
            async with runner_port_forward(runner, host="127.0.0.1", port=1):
                pass

    asyncio.run(scenario())


def test_grant_refuses_a_guest_url_with_credentials() -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as client:
            gateway = ExternalAgentGateway(
                client=client,
                public_base_url="http://127.0.0.1:1",
                routes={"anthropic": ExternalModelRoute("https://a.test", {}, ("/v1/messages",))},
            )
            with pytest.raises(ValueError, match="without credentials"):
                async with gateway.grant(
                    _context(),
                    route="anthropic",
                    models=frozenset({CLAUDE_MODEL}),
                    public_base_url="http://user:pw@192.0.2.1:18500",
                ):
                    pass

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_open", [False, True])
@pytest.mark.parametrize("close_failures", [3, 6])
def test_forward_context_retains_failed_open_cleanup(tmp_path, cancel_open, close_failures):
    from cayu.external_agents._cleanup import ExternalCleanupIncomplete, settle_external_cleanup
    from cayu.runners.aws_lambda_microvm import LambdaMicroVMEndpointTransientError

    class UnavailableTransport(ForwardingTransport):
        failures = close_failures

        async def open_forward(self, **kwargs):
            response = await super().open_forward(**kwargs)
            if cancel_open:
                raise asyncio.CancelledError()
            return response

        async def close_forward(self, **kwargs):
            if self.failures:
                self.failures -= 1
                raise LambdaMicroVMEndpointTransientError("offline")
            return await super().close_forward(**kwargs)

    async def scenario():
        transport = UnavailableTransport(tmp_path)
        if not cancel_open:
            transport.fail_next_open = 3
        runner = _runner(tmp_path, transport)
        original = asyncio.CancelledError if cancel_open else LambdaMicroVMEndpointTransientError
        expected = ExternalCleanupIncomplete if close_failures == 6 else original
        with pytest.raises(expected) as caught:
            async with runner_port_forward(runner, host="127.0.0.1", port=1):
                pytest.fail("A failed open must never enter the context body")
        if isinstance(caught.value, ExternalCleanupIncomplete):
            assert len(transport.forwards._forwards) == 1
            await settle_external_cleanup(caught.value, asyncio.Event())
        assert transport.forwards._forwards == {}

    asyncio.run(scenario())
