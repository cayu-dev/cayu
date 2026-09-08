"""Host bootstrap delivery uses only private runner stdin and strict receipts."""

import asyncio
import json

import pytest
from tests.core.test_browser_session import _WireRunner

from cayu.core.tools import ToolContext
from cayu.runners import ExecResult
from cayu.tools.browser_session import BrowserSessionTool, _RunnerBrowserSessionBackend


@pytest.mark.parametrize("valid", [False, True])
def test_backend_bootstrap_uses_private_transport_without_retry(valid):
    calls = []

    class Runner(_WireRunner):
        async def _exec_private_browser_control(self, command, **kwargs):
            calls.append(json.loads(kwargs["stdin"]))
            assert kwargs["output_limit_bytes"] == 1024
            return ExecResult(
                stdout=json.dumps(
                    {
                        "schema_version": 1 if valid else True,
                        "bootstrap_accepted": True,
                    }
                )
            )

        async def exec(self, command, **kwargs):
            pytest.fail("Bootstrap must not use ordinary runner output.")

    async def scenario():
        context = ToolContext(session_id="parent", runner=Runner())
        backend = BrowserSessionTool()._backend
        assert isinstance(backend, _RunnerBrowserSessionBackend)
        operation = backend.bootstrap_control(
            context,
            browser_session_id="bs_delivery",
            endpoint="wss://control.example/guest",
            credential="a" * 64,
            scope_sha256="b" * 64,
        )
        if valid:
            await operation
        else:
            with pytest.raises(RuntimeError, match="acknowledgement is invalid"):
                await operation
        assert len(calls) == 1
        assert calls[0] == {
            "protocol_version": "cayu.browser-control-bootstrap.v1",
            "session_id": "bs_delivery",
            "endpoint": "wss://control.example/guest",
            "credential": "a" * 64,
            "scope_sha256": "b" * 64,
        }

    asyncio.run(scenario())
