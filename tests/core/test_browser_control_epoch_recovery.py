"""Tool re-entry reconstructs the original guest epoch without new admission."""

import asyncio

import pytest
from tests.core.test_browser_session import _durable_context, _FakeBrowserBackend, _tool


@pytest.mark.parametrize("corruption", [None, True, 0, "3", 2**53])
def test_dispatched_reconciliation_preserves_original_epoch(tmp_path, corruption):
    async def scenario():
        class Backend(_FakeBrowserBackend):
            async def execute(self, ctx, request):
                self.response = await super().execute(ctx, request)
                return self.response

            async def reconcile(self, ctx, request):
                reconciled.append(dict(request))
                assert request["invocation_control_epoch"] == 3
                return self.response

        backend = Backend()
        records = {}
        reconciled = []
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "epoch-recovery",
        }

        async def original_epoch(browser_session_id, operation):
            assert operation == "navigate"
            return 3

        async def forbid_new_epoch(*unused):
            raise AssertionError("Reconciliation must not acquire new dispatch authority.")

        first = await _tool(backend).run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                fail_before_state="terminal",
                browser_control_epoch=original_epoch,
            ),
            args,
        )
        assert first.is_error
        assert backend.calls[0]["invocation_control_epoch"] == 3
        operation = next(
            r for r in records.values() if r.get("record_type") == "cayu.browser-operation"
        )
        assert operation["state"] == "dispatched"
        assert operation["invocation_control_epoch"] == 3
        if corruption is not None:
            operation["invocation_control_epoch"] = corruption
        result = await _tool(backend).run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                browser_control_epoch=forbid_new_epoch,
            ),
            args,
        )
        assert len(backend.calls) == 1
        if corruption is None:
            assert not result.is_error
            assert len(reconciled) == 1
            settled = next(
                r for r in records.values() if r.get("record_type") == "cayu.browser-operation"
            )
            assert settled["state"] == "terminal"
            assert settled["invocation_control_epoch"] == 3
        else:
            assert result.is_error
            assert not reconciled

    asyncio.run(scenario())
