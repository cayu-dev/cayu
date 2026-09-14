"""Public metadata admission after native child identity enrichment."""

from __future__ import annotations

import json

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.exposure import StaticToolExposurePolicy
from cayu.workflows.base import WorkflowSpec
from cayu.workflows.models import StepError
from cayu.workflows.workflow import StepRunOptions, WorkflowBase, step


class MetadataWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="metadata-control")

    async def run(self, session_id):
        yield await self.context(session_id).start()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "native_child,payload_bytes,admitted",
    [
        (False, 3000, True),
        (False, 3700, True),
        (False, 4200, False),
        (True, 3000, True),
        (True, 3500, True),
        (True, 3700, False),
        (True, 4200, False),
    ],
)
async def test_metadata_enrichment_has_predictable_admission_and_reopened_evidence(
    tmp_path, native_child, payload_bytes, admitted
):
    path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(path)
    provider = ScriptedModelProvider(
        [[ModelStreamEvent.text_delta("accepted"), ModelStreamEvent.completed()]]
    )
    app = CayuApp(enable_logging=False, session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="probe", model="synthetic"),
        tool_exposure_policy=StaticToolExposurePolicy(profile_id="empty", tools=()),
    )
    metadata = {"application_value": "x" * payload_bytes}
    original = json.dumps(metadata, sort_keys=True)
    session_id = "metadata-direct"
    try:
        if native_child:
            ctx = MetadataWorkflow(app).context("metadata-parent")
            await ctx.start()
            try:
                child = await step(
                    ctx,
                    agent="probe",
                    step_id="metadata-child",
                    prompt="synthetic",
                    run_options=StepRunOptions(metadata=metadata),
                )
                session_id = child.session_id
            except StepError as error:
                assert not admitted
                assert "metadata cannot exceed 4096 canonical JSON bytes" in str(error)
                session_id = error.session_id
        else:
            async for _ in app.run(
                RunRequest(
                    agent_name="probe",
                    session_id=session_id,
                    messages=[Message.text("user", "synthetic")],
                    metadata=metadata,
                )
            ):
                pass
        assert session_id is not None
        events = await store.load_events(session_id)
        session = await store.load(session_id)
        assert len(provider.requests) == int(admitted)
        assert json.dumps(metadata, sort_keys=True) == original
        terminals = [e for e in events if str(e.type) in {"session.completed", "session.failed"}]
        assert len(terminals) == 1
        assert str(terminals[0].type) == ("session.completed" if admitted else "session.failed")
        if not admitted:
            assert "metadata cannot exceed 4096 canonical JSON bytes" in json.dumps(
                terminals[0].payload
            )
        await store.close()
        store = SQLiteSessionStore(path)
        assert await store.load_events(session_id) == events
        assert await store.load(session_id) == session
        assert len(provider.requests) == int(admitted)
    finally:
        await store.close()
