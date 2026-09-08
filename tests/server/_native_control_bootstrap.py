"""Local native delivery for the production service/private-bootstrap path."""

import asyncio
import json

import pytest
from tests.core.test_browser_control_service import bound_browser_context
from tests.core.test_browser_session import _WireRunner

from cayu.runners import ExecResult
from cayu.tools import _browser_guest
from cayu.tools._browser_control_transport import open_guest_control_channel
from cayu.tools.browser_session import BrowserSessionTool, _RunnerBrowserSessionBackend


async def bootstrap_native_control(service, daemon, identity, tls):
    connected = asyncio.get_running_loop().create_future()
    calls = []

    async def connect(*, endpoint, credential):
        connection = await open_guest_control_channel(
            endpoint=endpoint, credential=credential, tls=tls
        )
        connected.set_result(connection)
        return connection

    class NativeRunner(_WireRunner):
        async def _exec_private_browser_control(self, command, **kwargs):
            calls.append(1)
            raw = json.loads(kwargs["stdin"])
            assert raw.pop("protocol_version") == "cayu.browser-control-bootstrap.v1"
            assert raw.pop("session_id") == daemon.session_id
            accepted = await daemon.bootstrap_operator_channel(raw)
            return ExecResult(stdout=json.dumps(accepted))

        async def exec(self, command, **kwargs):
            raise AssertionError("Bootstrap entered ordinary runner execution")

    context = bound_browser_context(identity, runner=NativeRunner())
    backend = BrowserSessionTool()._backend
    assert isinstance(backend, _RunnerBrowserSessionBackend)
    # Only the local test certificate trust differs from production connection.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_browser_guest, "open_guest_control_channel", connect)
        bound = await service.bootstrap(
            context,
            backend=backend,
            browser_session_id=daemon.session_id,
            arguments={"operation": "navigate"},
        )
        assert (
            await service.bootstrap(
                context,
                backend=backend,
                browser_session_id=daemon.session_id,
                arguments={"operation": "navigate"},
            )
        ) == bound
    assert calls == [1]
    assert daemon._operator_bootstrap_task is not None
    return await connected, daemon._operator_bootstrap_task
