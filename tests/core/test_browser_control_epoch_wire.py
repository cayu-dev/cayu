"""Private epoch survives tool preparation and the real guest request parser."""

import asyncio
import json
from typing import Any

import pytest
from tests.core.test_browser_session import _durable_context, _WireRunner

from cayu.tools._browser_control_guest import GuestControlFailure, GuestControlFence
from cayu.tools._browser_guest import _interactive_request_from_json
from cayu.tools.browser_session import BrowserSessionTool


@pytest.mark.parametrize("caller_supplied", [False, True])
def test_runtime_epoch_wire_is_not_a_public_tool_argument(tmp_path, caller_supplied):
    async def scenario():
        captured = []

        class Runner(_WireRunner):
            async def exec(self, command, **kwargs):
                raw = json.loads(kwargs["stdin"])
                parsed = _interactive_request_from_json(raw)
                assert parsed.invocation_control_epoch == 1
                fence = GuestControlFence(worker_instance="vw_" + "a" * 32)
                fence.bind("a" * 64)
                fence.check_model(parsed.invocation_control_epoch, parsed.operation)
                fence.epoch = 2
                with pytest.raises(GuestControlFailure):
                    fence.check_model(parsed.invocation_control_epoch, parsed.operation)
                captured.append(raw)
                return await super().exec(command, **kwargs)

        preparations = []

        async def epoch(browser_session_id, operation):
            preparations.append((browser_session_id, operation))
            return 1

        args: dict[str, Any] = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "private-epoch-wire",
        }
        if caller_supplied:
            args["invocation_control_epoch"] = 1
        records = {}
        tool = BrowserSessionTool(expected_runner_candidate="wire-browser")
        result = await tool.run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                runner=Runner(),
                browser_control_epoch=epoch,
            ),
            args,
        )
        if caller_supplied:
            assert result.is_error
            assert result.structured is not None
            assert result.structured["error"] == "invalid_arguments"
            assert not preparations and not captured and not records
        else:
            assert not result.is_error
            assert len(captured) == len(preparations) == 1
            operation = next(
                r for r in records.values() if r.get("record_type") == "cayu.browser-operation"
            )
            assert operation["state"] == "terminal"
            assert operation["invocation_control_epoch"] == captured[0]["invocation_control_epoch"]

    asyncio.run(scenario())
