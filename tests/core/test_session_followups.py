"""Ordinary follow-up admission preserves verified invocation semantics."""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest
from tests.runtime.runtime_provider_support import FakeProvider

from cayu import AgentSpec, CayuApp, Message, ResumeRequest, RunRequest
from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME, StructuredOutputSpec
from cayu.messages import TextPart
from cayu.providers import ModelProviderError, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.overview import SessionOverviewReader
from cayu.storage import SQLiteSessionStore


class FollowupProvider(FakeProvider):
    supports_native_structured_output = True

    def __init__(self, store, session_id, strategy, first_state):
        self.store, self.session_id = store, session_id
        self.first_state = first_state
        self.observations = []
        output = {"answer": "first"}

        def batch(value):
            answer = (
                ModelStreamEvent.tool_call(
                    id="answer-call", name=STRUCTURED_OUTPUT_TOOL_NAME, arguments={"output": value}
                )
                if strategy == "tool"
                else ModelStreamEvent.text_delta(
                    json.dumps(value) if strategy == "native" else "answer"
                )
            )
            return [answer, ModelStreamEvent.completed()]

        super().__init__([batch(output), batch({"answer": "second"})])

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:followup-provider", behavior_version="1", implementation_version="1"
        )

    async def stream(self, request):
        self.observations.append(
            (await SessionOverviewReader(self.store).read((self.session_id,)))[0]
        )
        if self.first_state == "provider_failed" and not self.requests:
            self.requests.append(request)
            raise ModelProviderError(
                "planned rejection", provider=self.name, retryable=False, status_code=400
            )
        async for event in super().stream(request):
            yield event


def spec_for(strategy):
    if strategy == "plain":
        return None
    return StructuredOutputSpec(
        name="Answer",
        strategy=strategy,
        max_retries=0,
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )


def app_for(store, provider):
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("strategy", ["plain", "tool", "native"])
@pytest.mark.parametrize("first_state", ["completed", "provider_failed"])
async def test_finished_session_accepts_omitted_followup(
    tmp_path, request, backend, strategy, first_state
):
    if backend == "memory":
        store = InMemorySessionStore()
    elif backend == "sqlite":
        store = SQLiteSessionStore(tmp_path / "followup.db")
    else:
        from cayu.storage import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    session_id = "followup-" + uuid4().hex
    provider = FollowupProvider(store, session_id, strategy, first_state)
    app = app_for(store, provider)
    try:
        first = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "First")],
                    structured_output=spec_for(strategy),
                )
            )
        ]
        assert any(
            event.type.value
            == ("session.completed" if first_state == "completed" else "session.failed")
            for event in first
        )
        before = (await SessionOverviewReader(store).read((session_id,)))[0]
        assert before.ended_at is not None
        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(session_id=session_id, messages=[Message.text("user", "Again")])
            )
        ]
        assert any(event.type.value == "session.completed" for event in resumed)
        after = (await SessionOverviewReader(store).read((session_id,)))[0]
        assert after.session.id == before.session.id
        assert after.session.instance_id == before.session.instance_id
        assert after.session.run_epoch > before.session.run_epoch
        assert provider.observations[-1].status.value == "running"
        assert provider.observations[-1].ended_at is None
        assert after.ended_at >= before.ended_at
        assert len(provider.requests) == 2
        transcript = await store.load_transcript(session_id)
        assert (
            sum(
                any(isinstance(part, TextPart) and part.text == "Again" for part in message.content)
                for message in transcript
            )
            == 1
        )
    finally:
        if backend != "memory":
            await store.close()


@pytest.mark.anyio
@pytest.mark.parametrize("strategy", ["tool", "native"])
@pytest.mark.parametrize("first_state", ["completed", "provider_failed"])
async def test_followup_reopens_persistent_invocation_settings(tmp_path, strategy, first_state):
    path = tmp_path / "reopened.db"
    session_id = "reopened"
    first_store = SQLiteSessionStore(path)
    first_provider = FollowupProvider(first_store, session_id, strategy, first_state)
    first_app = app_for(first_store, first_provider)
    try:
        _ = [
            event
            async for event in first_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "First")],
                    structured_output=spec_for(strategy),
                )
            )
        ]
    finally:
        await first_store.close()
    reopened = SQLiteSessionStore(path)
    provider = FollowupProvider(reopened, session_id, strategy, "completed")
    app = app_for(reopened, provider)
    try:
        events = [
            event
            async for event in app.resume(
                ResumeRequest(session_id=session_id, messages=[Message.text("user", "Again")])
            )
        ]
        assert events[-1].type.value == "session.completed"
        assert len(provider.requests) == 1
        assert provider.observations[0].ended_at is None
    finally:
        await reopened.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "corruption",
    [
        "session_id",
        "session_instance_id",
        "execution_profile_fingerprint",
        "spec",
        "max_steps",
        "limits",
        "retry_policy",
        "missing",
        "malformed",
        "boolean_version",
    ],
)
async def test_failed_followup_rejects_unbound_configuration(corruption):
    from cayu.runtime.execution_profiles import ExecutionProfileMismatchError

    store = InMemorySessionStore()
    session_id = "unbound"
    provider = FollowupProvider(store, session_id, "native", "provider_failed")
    app = app_for(store, provider)
    _ = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "First")],
                structured_output=spec_for("native"),
            )
        )
    ]
    # Simulate corrupted durable state, bypassing the public writer protections.
    checkpoint = store._checkpoints[session_id]
    key = "invocation_configuration"
    assert key in checkpoint
    if corruption == "missing":
        del checkpoint[key]
    elif corruption == "malformed":
        checkpoint[key] = {"unexpected": True}
    elif corruption == "boolean_version":
        checkpoint[key]["schema_version"] = True
    elif corruption == "spec":
        checkpoint[key]["structured_output"]["max_retries"] = 1
    elif corruption == "max_steps":
        checkpoint[key]["max_steps"] = 18
    elif corruption == "limits":
        checkpoint[key]["limits"]["max_tool_calls"] = 7
    elif corruption == "retry_policy":
        checkpoint[key]["retry_policy"]["max_attempts"] = 2
    else:
        checkpoint[key][corruption] = "incorrect-binding"
    with pytest.raises((ExecutionProfileMismatchError, ValueError)):
        _ = [
            event
            async for event in app.resume(
                ResumeRequest(session_id=session_id, messages=[Message.text("user", "Again")])
            )
        ]
    assert len(provider.requests) == 1
    assert (await store.load(session_id)).status.value == "failed"
    transcript = await store.load_transcript(session_id)
    assert not any(
        isinstance(part, TextPart) and part.text == "Again"
        for message in transcript
        for part in message.content
    )


@pytest.mark.anyio
async def test_generic_checkpoint_writes_cannot_replace_structured_configuration():
    store = InMemorySessionStore()
    session_id = "protected"
    provider = FollowupProvider(store, session_id, "native", "provider_failed")
    app = app_for(store, provider)
    _ = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "First")],
                structured_output=spec_for("native"),
            )
        )
    ]
    key = "invocation_configuration"
    before = await store.load_checkpoint(session_id)
    assert key in before
    observed = []

    def forge(_session, checkpoint):
        observed.append(checkpoint)
        return {**(checkpoint or {}), key: {"structured_output": {"max_retries": 99}}}

    await store.transform_checkpoint(session_id, forge)
    after = await store.load_checkpoint(session_id)
    assert all(key not in checkpoint for checkpoint in observed)
    assert after[key] == before[key]
    events = [
        event
        async for event in app.resume(
            ResumeRequest(session_id=session_id, messages=[Message.text("user", "Again")])
        )
    ]
    assert events[-1].type.value == "session.completed"


class BlockingFollowupProvider(FollowupProvider):
    def __init__(self, store, session_id, strategy):
        super().__init__(store, session_id, strategy, "completed")
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request):
        if not self.requests:
            self.started.set()
            await self.release.wait()
        async for event in super().stream(request):
            yield event


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("strategy", ["plain", "tool", "native"])
async def test_running_followup_delivers_once_at_next_turn(tmp_path, backend, strategy):
    from cayu.sessions.messaging import EnqueueSessionMessageRequest, SessionMessageDeliveryMode

    store = (
        InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(tmp_path / "queue.db")
    )
    session_id = "queued-followup"
    provider = BlockingFollowupProvider(store, session_id, strategy)
    app = app_for(store, provider)

    async def execute():
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "First")],
                    structured_output=spec_for(strategy),
                )
            )
        ]

    task = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        request = EnqueueSessionMessageRequest(
            session_id=session_id,
            idempotency_key="next",
            content="Again",
            delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
        )
        accepted = await app.enqueue_session_message(request)
        duplicate = await app.enqueue_session_message(request)
        assert duplicate.replayed
        assert duplicate.message.queue_id == accepted.message.queue_id
        assert not provider.requests
        provider.release.set()
        events = await asyncio.wait_for(task, timeout=15)
        assert events[-1].type.value == "session.completed"
        assert len(provider.requests) == 2
        assert all(observation.ended_at is None for observation in provider.observations)
        transcript = await store.load_transcript(session_id)
        assert (
            sum(
                isinstance(part, TextPart) and part.text == "Again"
                for message in transcript
                for part in message.content
            )
            == 1
        )
    finally:
        provider.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if backend != "memory":
            await store.close()


@pytest.mark.anyio
@pytest.mark.parametrize("first_state", ["completed", "provider_failed"])
async def test_finished_followup_inherits_nondefault_loop_controls(first_state):
    from cayu.budgets.run_limits import RunLimits
    from cayu.providers.retry_policy import RetryPolicy

    store = InMemorySessionStore()
    session_id = "custom-controls"
    provider = FollowupProvider(store, session_id, "plain", first_state)
    app = app_for(store, provider)
    _ = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "First")],
                max_steps=18,
                limits=RunLimits(max_tool_calls=7),
                retry_policy=RetryPolicy(max_attempts=2),
            )
        )
    ]
    events = [
        event
        async for event in app.resume(
            ResumeRequest(session_id=session_id, messages=[Message.text("user", "Again")])
        )
    ]
    assert events[-1].type.value == "session.completed"


@pytest.mark.anyio
async def test_prompt_followup_does_not_bypass_pending_user_input():
    from tests.core.test_user_input import _build

    app, store = _build([("call_input", "ask_user", {"question": "Continue?"})])
    _ = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="pending",
                messages=[Message.text("user", "First")],
            )
        )
    ]
    before = await store.load_checkpoint("pending")
    assert before["pending_user_input"]
    with pytest.raises((RuntimeError, ValueError)):
        _ = [
            event
            async for event in app.resume(
                ResumeRequest(session_id="pending", messages=[Message.text("user", "Again")])
            )
        ]
    after = await store.load_checkpoint("pending")
    assert after["pending_user_input"] == before["pending_user_input"]
    transcript = await store.load_transcript("pending")
    assert not any(
        isinstance(part, TextPart) and part.text == "Again"
        for message in transcript
        for part in message.content
    )


class LostAdmissionAcknowledgementStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, phase):
        super().__init__()
        self.phase = phase
        self.injected = False

    async def create(self, *args, **kwargs):
        session = await super().create(*args, **kwargs)
        if self.phase == "create" and not self.injected:
            self.injected = True
            raise OSError("create acknowledgement lost")
        return session

    async def admit_session_invocation(self, *args, **kwargs):
        session = await super().admit_session_invocation(*args, **kwargs)
        if self.phase == "admit" and not self.injected:
            self.injected = True
            raise OSError("admission acknowledgement lost")
        return session


@pytest.mark.anyio
@pytest.mark.parametrize("phase", ["create", "admit"])
async def test_followup_configuration_survives_lost_admission_acknowledgement(phase):
    store = LostAdmissionAcknowledgementStore(phase)
    session_id = "lost-ack"
    provider = FollowupProvider(store, session_id, "native", "provider_failed")
    app = app_for(store, provider)
    _ = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "First")],
                structured_output=spec_for("native"),
            )
        )
    ]
    events = [
        event
        async for event in app.resume(
            ResumeRequest(session_id=session_id, messages=[Message.text("user", "Again")])
        )
    ]
    assert events[-1].type.value == "session.completed"
    assert store.injected
    assert len(provider.requests) == 2
    transcript = await store.load_transcript(session_id)
    assert (
        sum(
            isinstance(part, TextPart) and part.text == "Again"
            for message in transcript
            for part in message.content
        )
        == 1
    )
    configuration = (await store.load_checkpoint(session_id))["invocation_configuration"]
    assert configuration["structured_output"] == spec_for("native").model_dump(mode="json")
    assert configuration["session_instance_id"] == (await store.load(session_id)).instance_id


@pytest.mark.anyio
@pytest.mark.parametrize("first_state", ["completed", "provider_failed"])
async def test_followup_profile_comparison_uses_retained_configuration(first_state):
    from cayu.budgets.run_limits import RunLimits
    from cayu.execution_profiles import ExecutionProfileComponentClass
    from cayu.providers.retry_policy import RetryPolicy

    store = InMemorySessionStore()
    provider = FollowupProvider(store, "comparison", "native", first_state)
    app = app_for(store, provider)
    _ = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="comparison",
                messages=[Message.text("user", "First")],
                structured_output=spec_for("native"),
                max_steps=18,
                limits=RunLimits(max_tool_calls=7),
                retry_policy=RetryPolicy(max_attempts=2),
            )
        )
    ]
    comparison = await app._compare_session_execution_profile("comparison")
    unknown = {component for component, _ in comparison.not_compared}
    for component in (
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
        ExecutionProfileComponentClass.FINALIZATION,
    ):
        assert component not in unknown
        assert comparison.candidate_profile.component(
            component
        ) == comparison.stored_profile.component(component)


@pytest.mark.anyio
@pytest.mark.parametrize("override", ["omitted", "null", "same"])
async def test_ordinary_followup_keeps_tool_completion_explicit(override):
    from tests.core.test_tool_completion import FinalTool

    from cayu.runtime import ExecutionProfileMismatchError, ToolCompletionPolicy

    store = InMemorySessionStore()
    provider = FollowupProvider(store, "tool-policy", "plain", "completed")
    provider.event_batches = [
        [
            ModelStreamEvent.tool_call(id="final-call", name="ask_customer", arguments={}),
            ModelStreamEvent.completed(),
        ]
        for _ in range(2)
    ]
    tool = FinalTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
    policy = ToolCompletionPolicy(tool_names=("ask_customer",))
    _ = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="tool-policy",
                messages=[Message.text("user", "First")],
                tool_completion=policy,
            )
        )
    ]
    controls = (
        {} if override == "omitted" else {"tool_completion": policy if override == "same" else None}
    )
    request = ResumeRequest(
        session_id="tool-policy", messages=[Message.text("user", "Again")], **controls
    )
    if override == "same":
        events = [event async for event in app.resume(request)]
        assert events[-1].type.value == "session.completed"
        assert tool.calls == 2
    else:
        with pytest.raises(ExecutionProfileMismatchError):
            _ = [event async for event in app.resume(request)]
        assert tool.calls == 1
        assert len(provider.requests) == 1


@pytest.mark.anyio
async def test_prompt_followup_does_not_bypass_pending_approval():
    from tests.core.test_tool_completion import FinalTool

    from cayu.tools import AlwaysRequireApprovalToolPolicy

    store = InMemorySessionStore()
    provider = FollowupProvider(store, "approval", "plain", "completed")
    provider.event_batches = [
        [
            ModelStreamEvent.tool_call(id="call", name="ask_customer", arguments={}),
            ModelStreamEvent.completed(),
        ]
    ]
    tool = FinalTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )
    _ = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="approval",
                messages=[Message.text("user", "First")],
            )
        )
    ]
    before = await store.load_checkpoint("approval")
    assert before["pending_tool_approval"]
    with pytest.raises((RuntimeError, ValueError)):
        _ = [
            event
            async for event in app.resume(
                ResumeRequest(session_id="approval", messages=[Message.text("user", "Again")])
            )
        ]
    after = await store.load_checkpoint("approval")
    assert after["pending_tool_approval"] == before["pending_tool_approval"]
    assert tool.calls == 0
    assert len(provider.requests) == 1


@pytest.mark.anyio
async def test_followup_retains_deep_supported_structured_schema():
    schema = {}
    for _ in range(29):
        schema = {"$defs": {"nested": schema}}
    schema = {"if": schema}
    store = InMemorySessionStore()
    provider = FollowupProvider(store, "deep-schema", "native", "completed")
    app = app_for(store, provider)
    spec = StructuredOutputSpec(strategy="native", json_schema=schema)
    first = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="deep-schema",
                messages=[Message.text("user", "First")],
                structured_output=spec,
            )
        )
    ]
    assert first[-1].type.value == "session.completed"
    events = [
        event
        async for event in app.resume(
            ResumeRequest(session_id="deep-schema", messages=[Message.text("user", "Again")])
        )
    ]
    assert events[-1].type.value == "session.completed"


@pytest.mark.parametrize("secret", ["run", "interrupt", "session"])
def test_followup_configuration_preserves_protocol_and_runtime_identity(secret):
    from cayu.vaults import SecretRedactor

    async def scenario():
        session_id = "f123"
        store = InMemorySessionStore()
        provider = FollowupProvider(store, session_id, "plain", "completed")
        app = CayuApp(
            session_store=store, enable_logging=False, secret_redactor=SecretRedactor(secret)
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "begin")],
                )
            )
        ]
        assert not [event for event in events if event.type == "session.failed"], events
        assert (await store.load(session_id)).status == "completed"
        record = (await store.load_checkpoint(session_id))["invocation_configuration"]
        assert record["session_id"] == session_id
        assert record["limits"]["scope"] == "run"

    asyncio.run(scenario())


def test_admitted_configuration_secret_identity_requires_runtime_session():
    from cayu.sessions._checkpoint_secret_validation import require_secret_free_durable_object
    from cayu.vaults import SecretRedactor

    async def scenario():
        session_id = "configuration-private-session"
        store = InMemorySessionStore()
        provider = FollowupProvider(store, session_id, "plain", "completed")
        app = app_for(store, provider)
        [
            event
            async for event in app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "begin")],
                )
            )
        ]
        session = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        record = {"invocation_configuration": checkpoint["invocation_configuration"]}
        redactor = SecretRedactor(session_id)
        with pytest.raises(ValueError, match="contains a workload secret"):
            require_secret_free_durable_object(record, redactor=redactor, field_name="checkpoint")
        assert (
            require_secret_free_durable_object(
                record, redactor=redactor, field_name="checkpoint", runtime_session=session
            )
            == record
        )
        mismatched = session.model_copy(update={"instance_id": "another-instance"})
        with pytest.raises(ValueError, match="contains a workload secret"):
            require_secret_free_durable_object(
                record, redactor=redactor, field_name="checkpoint", runtime_session=mismatched
            )
        assert await store.load_checkpoint(session_id) == checkpoint

    asyncio.run(scenario())
