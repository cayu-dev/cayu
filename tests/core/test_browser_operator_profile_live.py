"""Native tool/profile-wire handoff; HTTP control admission is covered separately.

The test owns a local fulfilled HTTPS page and supplies settled durable handback
evidence. Browser creation, guest restore/export, input, observation and encrypted
profile publication are real. This is not a remote-runner or full UI proof.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from tests.core.test_browser_control import identity, request
from tests.core.test_browser_control_guest import takeover_material
from tests.core.test_browser_session import (
    _browser_profile_binding,
    _durable_context,
    _ProfileWireRunner,
)

from cayu.browser_profiles import (
    AESGCMBrowserProfileKeyAuthority,
    BrowserProfileBinding,
    SQLiteBrowserProfileStore,
)
from cayu.runners import ExecResult
from cayu.runtime._browser_control_model import browser_model_control_admission
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlCheckpoint,
    BrowserControlRecord,
    BrowserOperatorPageOperations,
)
from cayu.runtime.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY
from cayu.tools._browser_guest import _interactive_request_from_json, _InteractiveDaemon
from cayu.tools.browser_session import BrowserSessionTool


@pytest.mark.skipif(
    os.environ.get("CAYU_BROWSER_CONTROL_LIVE") != "1",
    reason="Opt-in isolated local browser profile acceptance.",
)
@pytest.mark.parametrize("consent", ["allow", "deny"])
def test_native_manual_login_checkpoint_and_reopened_profile(tmp_path, monkeypatch, consent):
    from playwright.async_api import async_playwright

    from cayu.tools import browser_session as browser_session_module

    original_transition = browser_session_module._browser_page_set_transition_is_valid
    operation_counts = []

    def trace_transition(previous, current, delta, **kwargs):
        if previous is not None and kwargs["request"]["operation"] == "observe":
            operation_counts.append((previous.total_operations, current.total_operations))
        return original_transition(previous, current, delta, **kwargs)

    monkeypatch.setattr(
        browser_session_module, "_browser_page_set_transition_is_valid", trace_transition
    )

    async def scenario():
        path = tmp_path / "profiles.sqlite"
        store = SQLiteBrowserProfileStore(path, store_id="profiles")
        snapshot = None
        records = {}
        daemons = {}
        async with async_playwright() as playwright:

            class NativeRunner(_ProfileWireRunner):
                async def exec(self, command, **kwargs):
                    raw = json.loads(kwargs["stdin"])
                    parsed = _interactive_request_from_json(raw)
                    self.operations.append(parsed.operation)
                    daemon = daemons.get(parsed.session_id)
                    if daemon is None:
                        daemon = _InteractiveDaemon(parsed.session_id)
                        daemon.browser = await playwright.chromium.launch(
                            executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"),
                            headless=True,
                        )
                        daemons[parsed.session_id] = daemon
                    result = await daemon.execute(parsed)
                    if parsed.operation == "profile_restore":
                        await daemon.context.route(
                            "https://example.test/**",
                            lambda route: route.fulfill(
                                body="""<form onsubmit="event.preventDefault();
                                  localStorage.setItem('session',document.getElementById('code').value);
                                  document.title=localStorage.getItem('session');
                                  document.body.dataset.authenticated='yes';">
                                  <input id='code' type='password'><button>Login</button></form>""",
                                content_type="text/html",
                            ),
                        )
                    return ExecResult(stdout=json.dumps(result))

            runner = NativeRunner()

            async def epoch(browser_session_id, operation):
                if snapshot is None:
                    return None
                allocation = BrowserControlAllocation.model_validate(
                    snapshot.records[0].identity.model_dump(exclude={"worker_instance_id"})
                )
                assert allocation.browser_session_id == browser_session_id
                return browser_model_control_admission(
                    {BROWSER_CONTROLS_CHECKPOINT_KEY: snapshot.model_dump(mode="json")},
                    allocation=allocation,
                    operation_name=operation,
                )

            async def run(tool, args):
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

            try:
                binding = _browser_profile_binding(store)
                await binding.initialize()
                tool = BrowserSessionTool(
                    expected_runner_candidate="wire-browser",
                    browser_profile=binding,
                    max_wait_ms=1000,
                    idle_timeout_seconds=60,
                    max_sessions=1,
                )
                opened = await run(
                    tool,
                    {
                        "operation": "navigate",
                        "url": "https://example.test/login",
                        "operation_id": "open",
                    },
                )
                assert not opened.is_error, opened
                browser_id, page_id = opened.structured["session_id"], opened.structured["page_id"]
                daemon = daemons[browser_id]
                page = daemon.pages[page_id]
                daemon.claim_operator_channel("profile-test-channel")
                await daemon.bind_operator_control("a" * 64)
                material = takeover_material(daemon)
                await daemon.acquire_operator_control(**material)
                await daemon.enter_operator_sensitive_entry(
                    request_id=material["request_id"], epoch=2
                )
                for sequence, kind, value in (
                    (1, "key", "tab"),
                    (2, "text", "native-profile-private-canary"),
                    (3, "key", "enter"),
                ):
                    arguments = dict(
                        request_id=material["request_id"],
                        epoch=2,
                        sequence=sequence,
                        page_id=page_id,
                        page_epoch=page.control_epoch,
                    )
                    if kind == "key":
                        await daemon.operator_key_input(**arguments, key=value)
                    else:
                        await daemon.operator_text_input(**arguments, text=value)
                assert await page.page.locator("body").get_attribute("data-authenticated") == "yes"
                await daemon.handback_operator_control(request_id=material["request_id"], epoch=2)
                exact = identity().model_copy(update={"browser_session_id": browser_id})
                takeover = request().model_copy(
                    update={"identity": exact, "checkpoint_consent": consent}
                )
                snapshot = BrowserControlCheckpoint(
                    records=(
                        BrowserControlRecord(
                            identity=exact,
                            revision=6,
                            control_epoch=3,
                            request=takeover,
                            checkpoint_consent=consent,
                            fresh_observation_required=True,
                            capture_restricted=True,
                            settled_input_sequence=3,
                            operator_page_operations=(
                                BrowserOperatorPageOperations(page_id=page_id, operations=3),
                            ),
                        ),
                    )
                )
                observed = await run(
                    tool,
                    {
                        "operation": "observe",
                        "session_id": browser_id,
                        "page_id": page_id,
                        "operation_id": "handback",
                    },
                )
                assert not observed.is_error, (observed, operation_counts)
                assert not daemon.control.fresh_observation_required
                assert "native-profile-private-canary" not in repr(observed)
                snapshot = BrowserControlCheckpoint(
                    records=(
                        snapshot.records[0].model_copy(
                            update={"revision": 7, "fresh_observation_required": False}
                        ),
                    )
                )
                closed = await run(
                    tool, {"operation": "close", "session_id": browser_id, "operation_id": "close"}
                )
                assert not closed.is_error, closed
                assert closed.structured["allocation_disposition"] == "retired"
                assert runner.operations.count("profile_checkpoint") == int(consent == "allow")
                assert (await store.inspect_profile(binding.access)).generation == int(
                    consent == "allow"
                )
                assert not (await store.inspect_profile(binding.access)).active_writer
                assert "native-profile-private-canary" not in repr(records)
                await store.close()
                assert b"native-profile-private-canary" not in path.read_bytes()
                store = SQLiteBrowserProfileStore(path, store_id="profiles")
                # Reconstruct a fresh allocation, not a handle to the daemon
                # whose physical close has already positively settled.
                daemons.clear()
                runner = NativeRunner()
                restored_binding = BrowserProfileBinding(
                    authority=binding.authority,
                    store=store,
                    key_authority=AESGCMBrowserProfileKeyAuthority(
                        authority_id="browser-profile-test-key", key=b"p" * 32
                    ),
                    lease_seconds=120,
                )
                restored_tool = BrowserSessionTool(
                    expected_runner_candidate="wire-browser",
                    browser_profile=restored_binding,
                    max_wait_ms=1000,
                    idle_timeout_seconds=60,
                    max_sessions=1,
                )
                snapshot, records = None, {}
                reopened = await run(
                    restored_tool,
                    {
                        "operation": "navigate",
                        "url": "https://example.test/login",
                        "operation_id": "reopen",
                    },
                )
                assert not reopened.is_error, reopened
                restored = daemons[reopened.structured["session_id"]]
                restored_page = restored.pages[reopened.structured["page_id"]].page
                assert await restored_page.evaluate("localStorage.getItem('session')") == (
                    "native-profile-private-canary" if consent == "allow" else None
                )
                assert "native-profile-private-canary" not in repr(reopened)
            finally:
                for daemon in daemons.values():
                    await daemon.close()
                await store.close()

    asyncio.run(scenario())
