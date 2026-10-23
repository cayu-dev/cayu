"""A historical deferred pointer must not strand a later model call after upgrade."""

from __future__ import annotations

import asyncio
import os
import sys
from uuid import uuid4

import pytest
from tests.core.test_abandoned_model_call_resume import _build, _consume
from tests.core.test_execution_profiles import RecordingTool
from tests.runtime.runtime_provider_support import VersionedFakeProvider

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.build_provenance import RUNTIME_BUILD_PROVENANCE_ENV, current_runtime_build_provenance
from cayu.providers import ModelStreamEvent
from cayu.runtime._model_completion_contracts import ModelCompletionManualRecoveryRequired
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.migrations import SchemaMode


class _CrashAfterToolProvider(VersionedFakeProvider):
    async def stream(self, request):
        if os.environ.get("CAYU_TEST_CRASH_AFTER_TOOL") == "1" and self.requests:
            os._exit(137)
        async for event in super().stream(request):
            yield event


def _store(backend, location):
    if backend == "sqlite":
        return SQLiteSessionStore(location)
    return PostgresSessionStore(location, schema_mode=SchemaMode.MIGRATE)


def _app(store, provider):
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"), tools=[RecordingTool("record")]
    )
    return app


async def _child():
    store = _store(os.environ["CAYU_TEST_BACKEND"], os.environ["CAYU_TEST_STORE"])
    provider = _CrashAfterToolProvider(
        [
            ModelStreamEvent.tool_call(id="call-one", name="record", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    await _consume(
        _app(store, provider).run(
            RunRequest(
                agent_name="assistant",
                session_id=os.environ["CAYU_TEST_SESSION"],
                messages=[Message.text("user", "Record this.")],
            )
        )
    )
    raise AssertionError("The second model call did not crash.")


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("receipt_fault", [None, "missing", "wrong_step"])
def test_upgrade_closes_later_model_only_with_completed_tool_receipt(
    backend, receipt_fault, request, tmp_path, monkeypatch
):
    async def scenario():
        sid = "upgrade-after-tool-" + uuid4().hex
        location = (
            str(tmp_path / "store.db")
            if backend == "sqlite"
            else request.getfixturevalue("postgres_dsn")
        )
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import asyncio; from tests.core.test_upgrade_after_completed_tool import _child; "
            "asyncio.run(_child())",
            env={
                **os.environ,
                "CAYU_TEST_CRASH_AFTER_TOOL": "1",
                "CAYU_TEST_BACKEND": backend,
                "CAYU_TEST_STORE": location,
                "CAYU_TEST_SESSION": sid,
                RUNTIME_BUILD_PROVENANCE_ENV: _build("a"),
            },
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(child.communicate(), 30)
            assert child.returncode == 137, (out, err)
        finally:
            if child.returncode is None:
                child.kill()
                await child.wait()
        monkeypatch.setenv(RUNTIME_BUILD_PROVENANCE_ENV, _build("b"))
        current_runtime_build_provenance.cache_clear()
        store = _store(backend, location)
        try:
            checkpoint = await store.load_checkpoint(sid)
            assert checkpoint["last_model_step_publication"]["assistant_message_deferred"]
            assert "pending_tool_round" not in checkpoint
            assert checkpoint["invocation_configuration"]["session_id"] == sid
            active = await store.load_active_model_completion_stage(sid)
            assert active is not None
            provider = _CrashAfterToolProvider(
                [ModelStreamEvent.text_delta("Done."), ModelStreamEvent.completed()]
            )
            app = _app(store, provider)
            resume = ResumeRequest(session_id=sid, messages=[Message.text("user", "Continue.")])
            if receipt_fault is not None:
                original = store.load_runtime_publication_receipt

                async def faulty_receipt(session_id, publication_id):
                    receipt = await original(session_id, publication_id)
                    if publication_id.startswith("tool-round:"):
                        if receipt_fault == "missing":
                            return None
                        return receipt.model_copy(
                            update={"intent": {**receipt.intent, "model_step_id": "wrong-step"}}
                        )
                    return receipt

                with monkeypatch.context() as patch:
                    patch.setattr(store, "load_runtime_publication_receipt", faulty_receipt)
                    with pytest.raises(ModelCompletionManualRecoveryRequired):
                        await _consume(app.resume(resume))
                assert not provider.requests
                assert await store.load_active_model_completion_stage(sid) == active
                assert (
                    await store.load_model_completion_stage_settlement(sid, active.stage.stage_id)
                    is None
                )
            await _consume(app.resume(resume))
            assert (await store.load(sid)).status is SessionStatus.COMPLETED
            assert len(provider.requests) == 1
            assert await store.load_active_model_completion_stage(sid) is None
            settlement = await store.load_model_completion_stage_settlement(
                sid, active.stage.stage_id
            )
            assert settlement.disposition.value == "provider_effect_outcome_unknown"
            events = await store.load_events(sid)
            interruptions = [e for e in events if e.type is EventType.SESSION_INTERRUPTED]
            assert len(interruptions) == 1
            assert interruptions[0].payload["terminalization_only"] is True
        finally:
            await store.close()

    try:
        asyncio.run(scenario())
    finally:
        current_runtime_build_provenance.cache_clear()
