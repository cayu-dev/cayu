"""An interrupt from another worker during targeted grant binding ends the session interrupted.

Targeted ``call_tool`` resolution binds the grant use under a running-session
fence. When a second ``CayuApp`` on the same store moves the session to
``interrupting`` while that bind is in flight, the fence raises
``SessionStatusConflict``. The round must treat that as the interruption it is,
close the round and end ``interrupted``, without running the tool. A status
conflict that no interruption explains must still fail the session, and a local
interrupt that cancels the run just after resolution commits its checkpoint must
still end ``interrupted``.
"""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_targeted_tool_grants import (
    _GatewayProvider,
    _GatewayRememberTool,
    targeted_store,  # noqa: F401 - imported pytest fixture
)

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.runtime._tool_invocation.admission import ToolInvocationAdmission
from cayu.sessions.records import SessionStatus, SessionStatusConflict
from cayu.sessions.requests import InterruptSessionRequest
from cayu.tools.exposure import StaticToolExposurePolicy
from cayu.tools.grants import TargetedToolGrant

SESSION_ID = "targeted-remote-binding-race"


def _app(store, tool: _GatewayRememberTool) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    provider = _GatewayProvider()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model", provider_name=provider.name),
        targeted_tool_mode="call_tool",
        tools=[tool],
        tool_exposure_policy=StaticToolExposurePolicy(profile_id="targeted-only", tools=()),
    )
    return app


async def _collect(stream):
    return [event async for event in stream]


async def _wait_for(predicate, what: str) -> None:
    for _ in range(2000):
        if await predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.mark.parametrize(
    "case",
    ["normal-binding", "remote-interrupt", "unrelated-conflict", "local-interrupt-after-commit"],
)
def test_interrupt_during_targeted_binding_ends_interrupted(case, targeted_store, monkeypatch):  # noqa: F811
    asyncio.run(_scenario(case=case, store=targeted_store, monkeypatch=monkeypatch))


async def _scenario(*, case: str, store, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _GatewayRememberTool()
    worker = _app(store, tool)
    controller = _app(store, _GatewayRememberTool())

    entered = asyncio.Event()
    release = asyncio.Event()
    original_bind = store.bind_targeted_tool_grant_use
    original_resolve = ToolInvocationAdmission.resolve_targeted_calls

    async def delayed_bind(request, *, observed_at):
        entered.set()
        await release.wait()
        if case == "unrelated-conflict":
            raise SessionStatusConflict("Unrelated status conflict.")
        return await original_bind(request, observed_at=observed_at)

    async def held_resolve(self, **kwargs):
        # Resolution has committed the resolved round to the checkpoint; the
        # run is cancelled before its caller sees the resolved calls, as when
        # a store delivers a cancellation only after its write commits.
        resolved = await original_resolve(self, **kwargs)
        entered.set()
        await asyncio.Event().wait()
        return resolved

    if case == "local-interrupt-after-commit":
        monkeypatch.setattr(ToolInvocationAdmission, "resolve_targeted_calls", held_resolve)
    else:
        store.bind_targeted_tool_grant_use = delayed_bind

    run_task = asyncio.create_task(
        _collect(
            worker.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=SESSION_ID,
                    messages=[Message.text("user", "Remember a reviewed fact.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="remember", tool_id="cayu:remember", max_calls=1
                        ),
                    ),
                )
            )
        )
    )
    interrupt_task = None
    try:

        async def binding_reached() -> bool:
            return entered.is_set() or run_task.done()

        await _wait_for(binding_reached, "targeted binding")
        assert entered.is_set(), run_task.result() if run_task.done() else None
        if case == "remote-interrupt":
            interrupt_task = asyncio.create_task(
                _collect(
                    controller.interrupt_session(
                        InterruptSessionRequest(
                            session_id=SESSION_ID, reason="stop before targeted dispatch"
                        )
                    )
                )
            )

            async def interrupting() -> bool:
                return (await store.load(SESSION_ID)).status is SessionStatus.INTERRUPTING

            await _wait_for(interrupting, "the session to be interrupting")
        elif case == "local-interrupt-after-commit":
            interrupt_task = asyncio.create_task(
                _collect(
                    worker.interrupt_session(
                        InterruptSessionRequest(
                            session_id=SESSION_ID, reason="stop after targeted resolution"
                        )
                    )
                )
            )
        release.set()
        events = await asyncio.wait_for(run_task, 10)
        failures = [event.payload for event in events if event.type is EventType.SESSION_FAILED]
        status = (await store.load(SESSION_ID)).status
        if case in {"remote-interrupt", "local-interrupt-after-commit"}:
            assert not failures, failures
            assert interrupt_task is not None
            receipt = await asyncio.wait_for(interrupt_task, 10)
            assert any(event.type is EventType.SESSION_INTERRUPTED for event in events + receipt)
            assert status is SessionStatus.INTERRUPTED
            assert not tool.calls
        elif case == "unrelated-conflict":
            [failure] = failures
            assert failure["error_type"] == "SessionStatusConflict"
            assert failure["error"] == "Unrelated status conflict."
            assert status is SessionStatus.FAILED
            assert not tool.calls
        else:
            assert not failures, failures
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(tool.calls) == 1
    finally:
        release.set()
        tasks = [run_task] + ([] if interrupt_task is None else [interrupt_task])
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
