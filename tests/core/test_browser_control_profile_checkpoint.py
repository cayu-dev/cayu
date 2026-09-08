"""Post-handback checkpoint admission through the tool and private profile wire.

The control snapshot represents a settled handback. Full HTTP handback and native
profile export remain separate acceptance coverage, not claims of this fixture.
"""

import asyncio
import json

import pytest
from tests.core.test_browser_control import identity, request
from tests.core.test_browser_session import (
    _browser_profile_binding,
    _durable_context,
    _ProfileWireRunner,
)

from cayu.browser_profiles import (
    BrowserProfileCheckpointPolicy,
    InMemoryBrowserProfileStore,
    SQLiteBrowserProfileStore,
)
from cayu.runtime._browser_control_model import browser_model_control_epoch
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlCheckpoint,
    BrowserControlRecord,
)
from cayu.runtime.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY
from cayu.tools._browser_control_guest import GuestControlFence
from cayu.tools._browser_guest import _interactive_request_from_json
from cayu.tools.browser_session import BrowserSessionTool


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("consent", ["allow", "deny", "undecided"])
@pytest.mark.parametrize("fresh_required", [False, True])
@pytest.mark.parametrize("policy", list(BrowserProfileCheckpointPolicy))
def test_profile_checkpoint_consent_survives_tool_and_wire(
    tmp_path, persistent, consent, fresh_required, policy
):
    async def scenario():
        store = (
            SQLiteBrowserProfileStore(tmp_path / "profile.db", store_id="profiles")
            if persistent
            else InMemoryBrowserProfileStore(store_id="profiles")
        )
        binding = _browser_profile_binding(store, checkpoint_policy=policy)
        await binding.initialize()
        snapshot = None
        exported_epochs = []

        class Runner(_ProfileWireRunner):
            async def exec(self, command, **kwargs):
                raw = json.loads(kwargs["stdin"])
                if snapshot is not None and raw["operation"] == "profile_checkpoint":
                    parsed = _interactive_request_from_json(raw)
                    fence = GuestControlFence(worker_instance="vw_" + "a" * 32)
                    fence.bind("a" * 64)
                    fence.epoch = 3
                    fence.check_model(parsed.invocation_control_epoch, parsed.operation)
                    exported_epochs.append(parsed.invocation_control_epoch)
                return await super().exec(command, **kwargs)

        async def epoch(browser_session_id, operation):
            if snapshot is None:
                return None
            allocation = BrowserControlAllocation.model_validate(
                snapshot.records[0].identity.model_dump(exclude={"worker_instance_id"})
            )
            assert allocation.browser_session_id == browser_session_id
            return browser_model_control_epoch(
                {BROWSER_CONTROLS_CHECKPOINT_KEY: snapshot.model_dump(mode="json")},
                allocation=allocation,
                operation_name=operation,
            )

        runner = Runner()
        tool = BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_wait_ms=1000,
            idle_timeout_seconds=60,
            max_sessions=1,
        )
        records = {}

        async def run(args):
            return await tool.run(
                _durable_context(
                    tmp_path,
                    args=args,
                    records=records,
                    runner=runner,
                    tool_call_id=args["operation_id"],
                    browser_control_epoch=epoch,
                ),
                args,
            )

        opened = await run(
            {"operation": "navigate", "url": "https://example.test/login", "operation_id": "open"}
        )
        assert not opened.is_error
        browser_id = opened.structured["session_id"]
        exact = identity().model_copy(update={"browser_session_id": browser_id})
        takeover = request().model_copy(update={"identity": exact, "checkpoint_consent": consent})
        snapshot = BrowserControlCheckpoint(
            records=(
                BrowserControlRecord(
                    identity=exact,
                    revision=6,
                    control_epoch=3,
                    request=takeover,
                    checkpoint_consent=consent,
                    fresh_observation_required=fresh_required,
                ),
            )
        )
        before = runner.operations.count("profile_checkpoint")
        operation = "close" if policy is BrowserProfileCheckpointPolicy.ON_CLOSE else "observe"
        args = {"operation": operation, "session_id": browser_id, "operation_id": "after-handback"}
        if operation == "observe":
            args["page_id"] = opened.structured["page_id"]
        result = await run(args)
        assert not result.is_error
        expected = int(
            consent == "allow"
            and not fresh_required
            and policy is not BrowserProfileCheckpointPolicy.DISABLED
        )
        assert runner.operations.count("profile_checkpoint") == before + expected
        assert exported_epochs == ([3] if expected else [])
        inspection = await store.inspect_profile(binding.access)
        assert inspection.generation == before + expected

    asyncio.run(scenario())
