"""Tool-to-wire accounting composition and durable parent reconstruction."""

import asyncio
import json
from dataclasses import replace

import pytest
from tests.core.test_browser_session import (
    _durable_context,
    _FakeBrowserBackend,
    _ProfileWireRunner,
    _tool,
)

from cayu.core.tools import _RuntimeBrowserControlAdmission
from cayu.tools import browser_session as browser_module
from cayu.tools.browser_session import BrowserSessionTool


@pytest.mark.parametrize("admitted", [0, 2, 3, 4])
def test_tool_requires_exact_operator_delta_and_reconstructs_consumed_counts(tmp_path, admitted):
    async def scenario():
        class Runner(_ProfileWireRunner):
            async def exec(self, command, **kwargs):
                assert "_operator_page_operations" not in json.loads(kwargs["stdin"])
                return await super().exec(command, **kwargs)

        runner = Runner()
        records = {}
        page_id = None

        async def admission(browser_session_id, operation):
            if page_id is None:
                return None
            return _RuntimeBrowserControlAdmission(3, ((page_id, admitted),) if admitted else ())

        async def run(tool, arguments):
            return await tool.run(
                _durable_context(
                    tmp_path,
                    args=arguments,
                    records=records,
                    runner=runner,
                    tool_call_id=arguments["operation_id"],
                    browser_control_epoch=admission,
                ),
                arguments,
            )

        tool = BrowserSessionTool(expected_runner_candidate="wire-browser")
        opened = await run(
            tool, {"operation": "navigate", "operation_id": "open", "url": "https://example.test/"}
        )
        assert not opened.is_error
        page_id = opened.structured["page_id"]
        browser_id = opened.structured["session_id"]
        runner.operation_count += 3
        runner.control_epoch += 1
        observed = await run(
            tool,
            {
                "operation": "observe",
                "operation_id": "handback",
                "session_id": browser_id,
                "page_id": page_id,
            },
        )
        assert observed.is_error is (admitted != 3)
        if admitted != 3:
            assert observed.structured["error"] == "browser_crash"
            return
        sessions = [
            value
            for value in records.values()
            if value.get("record_type") == "cayu.browser-session"
        ]
        assert len(sessions) == 1
        assert sessions[0]["operator_page_operations"] == [[page_id, 3]]
        # A reconstructed tool consumes the stored baseline, not the same three
        # operations again. Its next model-only observation adds exactly one.
        restarted = BrowserSessionTool(expected_runner_candidate="wire-browser")
        again = await run(
            restarted,
            {
                "operation": "observe",
                "operation_id": "after-restart",
                "session_id": browser_id,
                "page_id": page_id,
            },
        )
        assert not again.is_error, again
        assert again.structured["page_set"]["total_operations"] == 6

    asyncio.run(scenario())


def test_cancellation_during_terminal_readback_remains_cancellation(tmp_path, monkeypatch):
    async def scenario():
        records = {}
        backend = _FakeBrowserBackend()
        tool = _tool(backend)
        entered = asyncio.Event()
        blocking = True
        original = browser_module._runtime_tool_invocation_authority

        def authority_for(ctx):
            authority = original(ctx)
            if authority is None:
                return None

            async def load(key):
                if blocking and any(
                    value.get("record_type") == "cayu.browser-operation"
                    and value.get("state") == "terminal"
                    for value in records.values()
                ):
                    entered.set()
                    await asyncio.Event().wait()
                return await authority.load_durable_operation(key)

            return replace(authority, load_durable_operation=load)

        monkeypatch.setattr(browser_module, "_runtime_tool_invocation_authority", authority_for)
        args = {
            "operation": "navigate",
            "operation_id": "cancel-readback",
            "url": "https://example.test/",
        }
        task = asyncio.create_task(
            tool.run(
                _durable_context(tmp_path, args=args, records=records, fail_after_state="terminal"),
                args,
            )
        )
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert task.cancelling() == 1
        assert len(backend.calls) == 1
        blocking = False
        recovered = await tool.run(_durable_context(tmp_path, args=args, records=records), args)
        assert not recovered.is_error, recovered
        assert len(backend.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("target", ["parent", "operation", "session"])
@pytest.mark.parametrize("fault", ["missing", "conflicting", "unreadable"])
def test_terminal_readback_requires_all_publication_evidence(tmp_path, target, fault):
    async def scenario():
        class Records(dict):
            faults = 0
            enabled = True

            def get(self, key, default=None):
                value = super().get(key, default)
                committed = any(
                    record.get("record_type") == "cayu.browser-operation"
                    and record.get("state") == "terminal"
                    for record in self.values()
                )
                if (
                    self.enabled
                    and committed
                    and isinstance(value, dict)
                    and value.get("record_type") == f"cayu.browser-{target}"
                ):
                    self.faults += 1
                    if fault == "missing":
                        return None
                    if fault == "unreadable":
                        raise ConnectionError("readback unavailable")
                    return {**value, "conflicting_evidence": True}
                return value

        records = Records()
        backend = _FakeBrowserBackend()
        tool = _tool(backend)
        args = {
            "operation": "navigate",
            "operation_id": "readback",
            "url": "https://example.test/",
        }
        result = await tool.run(
            _durable_context(tmp_path, args=args, records=records, fail_after_state="terminal"),
            args,
        )
        assert records.faults > 0
        assert result.is_error
        assert result.structured is not None
        assert result.structured["error"] == "outcome_ambiguous"
        assert len(backend.calls) == 1
        records.enabled = False
        recovered = await tool.run(_durable_context(tmp_path, args=args, records=records), args)
        assert not recovered.is_error, recovered
        assert len(backend.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("publication_failure", ["before", "after"])
def test_operator_counts_survive_terminal_publication_loss(tmp_path, restart, publication_failure):
    async def scenario():
        class Runner(_ProfileWireRunner):
            def __init__(self):
                super().__init__()
                self.receipts = {}

            async def exec(self, command, **kwargs):
                request = json.loads(kwargs["stdin"])
                key = request["operation_id"]
                if request.get("reconcile_only"):
                    return self.receipts[key]
                assert key not in self.receipts
                result = await super().exec(command, **kwargs)
                self.receipts[key] = result
                return result

        runner = Runner()
        records = {}
        page_id = None
        replay = False

        async def admission(browser_session_id, operation):
            assert not replay, "Receipt recovery must not acquire new authority."
            return None if page_id is None else _RuntimeBrowserControlAdmission(3, ((page_id, 3),))

        async def run(tool, args, **failure):
            return await tool.run(
                _durable_context(
                    tmp_path,
                    args=args,
                    records=records,
                    runner=runner,
                    tool_call_id=args["operation_id"],
                    browser_control_epoch=admission,
                    **failure,
                ),
                args,
            )

        tool = BrowserSessionTool(expected_runner_candidate="wire-browser")
        opened = await run(
            tool, {"operation": "navigate", "operation_id": "open", "url": "https://example.test/"}
        )
        assert not opened.is_error
        page_id = opened.structured["page_id"]
        args = {
            "operation": "observe",
            "operation_id": "handback",
            "session_id": opened.structured["session_id"],
            "page_id": page_id,
        }
        runner.operation_count += 3
        runner.control_epoch += 1
        first = await run(tool, args, **{f"fail_{publication_failure}_state": "terminal"})
        if publication_failure == "before":
            assert first.is_error
        replay = True
        if publication_failure == "before" and not restart:
            key = next(
                key
                for key, value in records.items()
                if value.get("record_type") == "cayu.browser-operation"
                and value.get("state") == "dispatched"
            )
            evidence = records.pop(key)
            missing = await run(tool, args)
            assert missing.is_error
            assert runner.operations == ["navigate", "observe"]
            records[key] = evidence
        if restart:
            tool = BrowserSessionTool(expected_runner_candidate="wire-browser")
        recovered = await run(tool, args)
        assert not recovered.is_error, recovered
        assert recovered.structured["page_set"]["total_operations"] == 5
        assert runner.operations == ["navigate", "observe"]
        session = next(
            value
            for value in records.values()
            if value.get("record_type") == "cayu.browser-session"
        )
        assert session["operator_page_operations"] == [[page_id, 3]]
        replay = False
        following = await run(tool, {**args, "operation_id": "following"})
        assert not following.is_error, following
        assert following.structured["page_set"]["total_operations"] == 6
        assert runner.operations == ["navigate", "observe", "observe"]

    asyncio.run(scenario())
