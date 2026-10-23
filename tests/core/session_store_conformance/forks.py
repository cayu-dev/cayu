"""Shared session-store forks contract cases."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
)
from tests.core.session_store_conformance.support import (
    _assert_exception_omits_private_values,
    _AuthorizeForkProfilePolicy,
    _close_store,
    _identity,
    _open_store,
    _RejectingPortableForkProvider,
    _reopen_store,
    _UnusedForkProvider,
    _UnusedNamedForkProvider,
)

import cayu.sessions.base as sessions_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    ResolutionActor,
    ResolutionActorSource,
)
from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    Message,
    ToolCallPart,
)
from cayu.observability.events import InMemoryEventSink
from cayu.observability.hooks import RuntimeHook, RuntimeHookContext
from cayu.runtime.authority import SessionRunFenced
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileDecisionKind,
)
from cayu.sessions._model_stage import (
    ModelCompletionStageRequest,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.forks import (
    ForkExecutionProfileSelection,
    ForkSystemPromptReplacement,
    fork_session_invocation,
)
from cayu.sessions.invocation import (
    InvocationOriginClaim,
    InvocationOriginTrust,
    SessionExecutionSource,
)
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import Session, SessionIdentity, SessionStatus
from cayu.sessions.requests import (
    ForkSessionRequest,
    ResumeRequest,
    RunRequest,
)
from cayu.sessions.transcript_queries import TranscriptQuery, TranscriptSnapshot
from cayu.tools.exposure import (
    ToolCapabilityCeiling,
)
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.postgres


def test_session_store_conformance_parent_deletion_preserves_root_invocation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            parent = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"invocation-parent-{session_store_case[0]}",
                    messages=[],
                ),
                identity=_identity(),
            )
            child = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"invocation-child-{session_store_case[0]}",
                    parent_session_id=parent.id,
                    messages=[],
                ),
                identity=_identity(),
            )

            await store.delete_session(parent.id)
            loaded = await store.load(child.id)

            assert loaded is not None
            assert loaded.parent_session_id is None
            assert loaded.invocation == child.invocation
            assert loaded.invocation.root_session_id == parent.id
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_parent_deletion_rejects_durable_subagent_child(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            parent = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"durable-delete-parent-{session_store_case[0]}",
                    messages=[],
                ),
                identity=_identity(),
            )
            child = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"durable-delete-child-{session_store_case[0]}",
                    parent_session_id=parent.id,
                    messages=[],
                    metadata={"subagent": {"mode": "durable"}},
                ),
                identity=_identity(),
            )

            with pytest.raises(
                ValueError,
                match="durable subagent child authority still exists",
            ):
                await store.delete_session(parent.id)

            retained_parent = await store.load(parent.id)
            retained_child = await store.load(child.id)
            assert retained_parent is not None
            assert retained_child is not None
            assert retained_child.parent_session_id == parent.id

            await store.delete_session(child.id)
            await store.delete_session(parent.id)
            assert await store.load(child.id) is None
            assert await store.load(parent.id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reused_root_session_id_cannot_rebind_invocation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            root_session_id = f"invocation-reused-root-{session_store_case[0]}"
            original_root = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=root_session_id,
                    messages=[],
                    invocation_origin=InvocationOriginClaim(subject="original-user"),
                ),
                identity=_identity(),
            )
            child = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"invocation-reused-child-{session_store_case[0]}",
                    parent_session_id=original_root.id,
                    messages=[],
                ),
                identity=_identity(),
            )

            await store.delete_session(original_root.id)
            replacement_root = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=root_session_id,
                    messages=[],
                    invocation_origin=InvocationOriginClaim(subject="replacement-user"),
                ),
                identity=_identity(),
            )
            store = await _reopen_store(session_store_case, store)
            loaded_child = await store.load(child.id)
            loaded_replacement = await store.load(replacement_root.id)

            assert loaded_child is not None
            assert loaded_replacement is not None
            assert loaded_child.invocation.root_session_id == root_session_id
            assert loaded_replacement.invocation.root_session_id == root_session_id
            assert (
                loaded_child.invocation.root_invocation_id
                == original_root.invocation.root_invocation_id
            )
            assert (
                loaded_replacement.invocation.root_invocation_id
                == replacement_root.invocation.root_invocation_id
            )
            assert (
                loaded_child.invocation.root_invocation_id
                != loaded_replacement.invocation.root_invocation_id
            )
            assert loaded_child.invocation.origin.subject == "original-user"
            assert loaded_replacement.invocation.origin.subject == "replacement-user"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reconstructs_exact_root_invocation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            root_request = RunRequest(
                agent_name="assistant",
                session_id=f"invocation-root-{session_store_case[0]}",
                messages=[],
                invocation_origin=InvocationOriginClaim(
                    subject="application-user",
                    tenant="customer-a",
                ),
            )
            root = await store.create(root_request, identity=_identity())
            child_request = RunRequest(
                agent_name="assistant",
                session_id=f"invocation-child-reload-{session_store_case[0]}",
                parent_session_id=root.id,
                messages=[],
            )
            child = await store.create(child_request, identity=_identity())

            with pytest.raises(ValueError, match="Session already exists"):
                await store.create(child_request, identity=_identity())

            store = await _reopen_store(session_store_case, store)
            loaded_root = await store.load(root.id)
            loaded_child = await store.load(child.id)

            assert loaded_root is not None
            assert loaded_child is not None
            assert loaded_root.invocation.origin.trust is InvocationOriginTrust.HOST_ASSERTED
            assert loaded_root.invocation.origin.subject == "application-user"
            assert loaded_root.invocation.origin.tenant == "customer-a"
            assert loaded_root.invocation.root_invocation_id == root.invocation.root_invocation_id
            assert loaded_root.invocation.root_session_id == root.id
            assert loaded_root.invocation.source is SessionExecutionSource.SDK_RUN
            assert loaded_child.invocation.origin == loaded_root.invocation.origin
            assert (
                loaded_child.invocation.root_invocation_id
                == loaded_root.invocation.root_invocation_id
            )
            assert loaded_child.invocation.root_session_id == loaded_root.id
            assert loaded_child.invocation.source is SessionExecutionSource.SDK_RUN
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_concurrent_child_creation_has_one_provenance(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            root = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"invocation-concurrent-root-{session_store_case[0]}",
                    messages=[],
                    invocation_origin=InvocationOriginClaim(subject="application-user"),
                ),
                identity=_identity(),
            )
            child_request = RunRequest(
                agent_name="assistant",
                session_id=f"invocation-concurrent-child-{session_store_case[0]}",
                parent_session_id=root.id,
                messages=[],
            )
            results = await asyncio.gather(
                store.create(child_request, identity=_identity()),
                store.create(child_request, identity=_identity()),
                return_exceptions=True,
            )

            created = [result for result in results if isinstance(result, Session)]
            rejected = [result for result in results if isinstance(result, ValueError)]
            assert len(created) == 1
            assert len(rejected) == 1
            assert "Session already exists" in str(rejected[0])
            loaded = await store.load(child_request.session_id or "")
            assert loaded is not None
            assert loaded.invocation == created[0].invocation
            assert loaded.invocation.origin == root.invocation.origin
            assert loaded.invocation.root_session_id == root.id
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_validates_fork_request_preamble(
    session_store_case,
) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            source = await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_fork_source",
                    messages=[Message.text("user", "fork")],
                ),
                identity=_identity(),
            )

            with pytest.raises(ValueError, match="Fork parent_session_id must match"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_parent",
                        agent_name="assistant",
                        provider_name="fake",
                        model="fake-model",
                        parent_session_id="sess_other",
                        causal_budget_id=source.causal_budget_id,
                        invocation=fork_session_invocation(source),
                        status=SessionStatus.PENDING,
                    ),
                    source_statuses={SessionStatus.PENDING},
                    expected_source_run_epoch=source.run_epoch,
                    transcript_cursor=None,
                    checkpoint_transform=None,
                )
            with pytest.raises(ValueError, match="transcript_cursor must be greater"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_cursor",
                        agent_name="assistant",
                        provider_name="fake",
                        model="fake-model",
                        parent_session_id=source.id,
                        causal_budget_id=source.causal_budget_id,
                        invocation=fork_session_invocation(source),
                        status=SessionStatus.PENDING,
                    ),
                    source_statuses={SessionStatus.PENDING},
                    expected_source_run_epoch=source.run_epoch,
                    transcript_cursor=-1,
                    checkpoint_transform=None,
                )
            with pytest.raises(ValueError, match="Source session status is not forkable"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_status_source",
                        agent_name="assistant",
                        provider_name="fake",
                        model="fake-model",
                        parent_session_id=source.id,
                        causal_budget_id=source.causal_budget_id,
                        invocation=fork_session_invocation(source),
                        status=SessionStatus.PENDING,
                    ),
                    source_statuses={SessionStatus.COMPLETED},
                    expected_source_run_epoch=source.run_epoch,
                    transcript_cursor=None,
                    checkpoint_transform=None,
                )
            with pytest.raises(ValueError, match="Fork status must match source session status"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_status_fork",
                        agent_name="assistant",
                        provider_name="fake",
                        model="fake-model",
                        parent_session_id=source.id,
                        causal_budget_id=source.causal_budget_id,
                        invocation=fork_session_invocation(source),
                        status=SessionStatus.COMPLETED,
                    ),
                    source_statuses={SessionStatus.PENDING},
                    expected_source_run_epoch=source.run_epoch,
                    transcript_cursor=None,
                    checkpoint_transform=None,
                )
            with pytest.raises(ValueError, match="Fork provider_name must match"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_provider",
                        agent_name="assistant",
                        provider_name="other",
                        model="fake-model",
                        parent_session_id=source.id,
                        causal_budget_id=source.causal_budget_id,
                        invocation=fork_session_invocation(source),
                        status=SessionStatus.PENDING,
                    ),
                    source_statuses={SessionStatus.PENDING},
                    expected_source_run_epoch=source.run_epoch,
                    transcript_cursor=None,
                    checkpoint_transform=None,
                )
        finally:
            await _close_store(session_store)

    asyncio.run(run())


def test_session_store_conformance_validates_exact_fork_transcript_atomically(
    session_store_case,
) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            source = await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_validated_fork_source",
                    messages=[Message.text("user", "fork")],
                ),
                identity=_identity(),
            )
            await session_store.append_transcript_messages(
                source.id,
                [Message.text("user", "copied-prefix")],
                interaction_id="source-interaction-copied",
            )
            await session_store.append_transcript_messages(
                source.id,
                [Message.text("user", "excluded-suffix")],
                interaction_id="source-interaction-excluded",
            )
            source = await session_store.update_status(
                source.id,
                SessionStatus.COMPLETED,
            )
            observed: list[tuple[tuple[Message, ...], TranscriptSnapshot]] = []

            fork = await session_store.create_fork_with_transcript_validation(
                source_session_id=source.id,
                fork=Session(
                    id="sess_validated_fork_child",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=source.id,
                    invocation=fork_session_invocation(source),
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=source.run_epoch,
                transcript_cursor=1,
                checkpoint_transform=None,
                transcript_validator=lambda messages, source_snapshot: (
                    not observed.append((messages, source_snapshot))
                ),
            )
            assert fork.id == "sess_validated_fork_child"
            assert len(observed) == 1
            observed_messages, observed_source_snapshot = observed[0]
            assert [message.content[0].text for message in observed_messages] == ["copied-prefix"]
            assert observed_source_snapshot.cursor == 2
            assert [record.index for record in observed_source_snapshot.records] == [0]
            assert [record.interaction_id for record in observed_source_snapshot.records] == [
                "source-interaction-copied"
            ]
            assert [
                message.content[0].text for message in await session_store.load_transcript(fork.id)
            ] == ["copied-prefix"]
            fork_transcript = await session_store.query_transcript(
                TranscriptQuery(session_id=fork.id, limit=10)
            )
            assert [record.interaction_id for record in fork_transcript.records] == [
                "source-interaction-copied"
            ]

            mutation_source = await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_validated_fork_mutation_source",
                    messages=[],
                ),
                identity=_identity(),
            )
            await session_store.append_transcript_messages(
                mutation_source.id,
                [
                    Message.tool_call(
                        tool_call_id="call_validator_mutation",
                        tool_name="validator_mutation",
                        arguments={"nested": {"value": "original"}},
                    )
                ],
            )
            mutation_source = await session_store.update_status(
                mutation_source.id,
                SessionStatus.COMPLETED,
            )

            def mutate_validation_projection(
                messages: tuple[Message, ...],
                source_snapshot: TranscriptSnapshot,
            ) -> bool:
                part = messages[0].content[0]
                assert isinstance(part, ToolCallPart)
                part.arguments["nested"]["value"] = "mutated"
                source_part = source_snapshot.records[0].message.content[0]
                assert isinstance(source_part, ToolCallPart)
                source_part.arguments["nested"]["value"] = "source-mutated"
                return True

            await session_store.create_fork_with_transcript_validation(
                source_session_id=mutation_source.id,
                fork=Session(
                    id="sess_validated_fork_mutation_child",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=mutation_source.id,
                    invocation=fork_session_invocation(mutation_source),
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=mutation_source.run_epoch,
                transcript_cursor=None,
                checkpoint_transform=None,
                transcript_validator=mutate_validation_projection,
            )
            for session_id in (
                mutation_source.id,
                "sess_validated_fork_mutation_child",
            ):
                transcript = await session_store.load_transcript(session_id)
                part = transcript[0].content[0]
                assert isinstance(part, ToolCallPart)
                assert part.arguments == {"nested": {"value": "original"}}

            for child_id, invalid_result in (
                ("sess_rejected_fork_false", False),
                ("sess_rejected_fork_ambiguous", 1),
            ):
                with pytest.raises(ValueError, match="atomic copy validation"):
                    await session_store.create_fork_with_transcript_validation(
                        source_session_id=source.id,
                        fork=Session(
                            id=child_id,
                            agent_name="assistant",
                            provider_name="fake",
                            model="fake-model",
                            parent_session_id=source.id,
                            invocation=fork_session_invocation(source),
                            status=SessionStatus.COMPLETED,
                        ),
                        source_statuses={SessionStatus.COMPLETED},
                        expected_source_run_epoch=source.run_epoch,
                        transcript_cursor=None,
                        checkpoint_transform=None,
                        transcript_validator=(
                            lambda _messages, _source_snapshot, result=invalid_result: result
                        ),
                    )
                assert await session_store.load(child_id) is None
        finally:
            await _close_store(session_store)

    asyncio.run(run())


def test_session_store_conformance_replaces_fork_system_prompt_atomically(
    session_store_case,
) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            source = await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_prompt_replacement_source",
                    messages=[],
                ),
                identity=_identity(),
            )
            await session_store.append_transcript_messages(
                source.id,
                [Message.text("system", "source prompt")],
            )
            await session_store.append_transcript_messages(
                source.id,
                [Message.text("user", "inherited question")],
                interaction_id="source-prompt-interaction",
            )
            source = await session_store.update_status(source.id, SessionStatus.COMPLETED)
            observed: list[tuple[Message, ...]] = []

            child = await session_store.create_fork_with_transcript_validation(
                source_session_id=source.id,
                fork=Session(
                    id="sess_prompt_replacement_child",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=source.id,
                    invocation=fork_session_invocation(source),
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=source.run_epoch,
                transcript_cursor=None,
                checkpoint_transform=None,
                system_prompt_replacement=ForkSystemPromptReplacement(
                    Message.text("system", "child prompt")
                ),
                transcript_validator=lambda messages, _source_snapshot: (
                    not observed.append(messages)
                ),
            )

            source_transcript = await session_store.load_transcript(source.id)
            child_transcript = await session_store.load_transcript(child.id)
            child_page = await session_store.query_transcript(
                TranscriptQuery(session_id=child.id, limit=10)
            )
            assert [message.content[0].text for message in source_transcript] == [
                "source prompt",
                "inherited question",
            ]
            assert [message.content[0].text for message in child_transcript] == [
                "child prompt",
                "inherited question",
            ]
            assert observed == [tuple(child_transcript)]
            assert [record.interaction_id for record in child_page.records] == [
                None,
                "source-prompt-interaction",
            ]
        finally:
            await _close_store(session_store)

    asyncio.run(run())


def test_session_store_conformance_rejects_unsafe_derived_fork_before_mutation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        source_id = f"derived_fork_source_{store_kind}"
        child_id = f"derived_fork_child_{store_kind}"
        secret = "derived-store-fork-secret"
        try:
            source = await store.create(
                RunRequest(
                    agent_name="source-agent",
                    session_id=source_id,
                    messages=[Message.text("user", "fork")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="source-model",
                ),
            )
            await store.append_transcript_messages(
                source.id,
                [Message.text("user", "copied transcript")],
            )
            await store.checkpoint(source.id, {"safe": "checkpoint"})
            await store.update_status(source.id, SessionStatus.COMPLETED)
            source_before = await store.load(source.id)
            transcript_before = await store.load_transcript(source.id)
            checkpoint_before = await store.load_checkpoint(source.id)
            events_before = await store.load_events(source.id)

            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="source-agent", model="source-model"))
            app.register_agent(AgentSpec(name="target-agent", model=secret))

            with pytest.raises(ValueError, match="model"):
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=source.id,
                            session_id=child_id,
                            agent_name="target-agent",
                            execution_profile_selection=(
                                ForkExecutionProfileSelection.CURRENT_CHILD
                            ),
                            profile_adoption=ExecutionProfileAdoptionIntent(
                                idempotency_key=f"derived-fork-{store_kind}",
                                reason="Exercise derived fork authority validation.",
                                requested_by=ResolutionActor(
                                    subject="test-caller",
                                    source=ResolutionActorSource.REQUEST,
                                ),
                            ),
                        )
                    )
                ]

            assert await store.load(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_transcript(child_id)
            assert await store.load_checkpoint(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_events(child_id)
            assert await store.load(source.id) == source_before
            assert await store.load_transcript(source.id) == transcript_before
            assert await store.load_checkpoint(source.id) == checkpoint_before
            assert await store.load_events(source.id) == events_before
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_same_model_cross_provider_preflight_is_atomic(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        source_id = f"portable_fork_source_{store_kind}_{uuid4().hex}"
        child_id = f"portable_fork_child_{store_kind}_{uuid4().hex}"
        try:
            source = await store.create(
                RunRequest(
                    agent_name="source-agent",
                    session_id=source_id,
                    messages=[Message.text("user", "fork source")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="source-provider",
                    model="shared-model",
                ),
            )
            await store.append_transcript_messages(
                source.id,
                [Message.text("user", "portable history")],
            )
            source = await store.update_status(source.id, SessionStatus.COMPLETED)
            source_before = await store.load(source.id)
            transcript_before = await store.load_transcript(source.id)
            events_before = await store.load_events(source.id)

            target_provider = _RejectingPortableForkProvider("target-provider")
            app = CayuApp(
                session_store=store,
                execution_profile_policy=_AuthorizeForkProfilePolicy(),
                enable_logging=False,
            )
            app.register_provider(_UnusedNamedForkProvider("source-provider"), default=True)
            app.register_provider(target_provider)
            app.register_agent(
                AgentSpec(
                    name="source-agent",
                    provider_name="source-provider",
                    model="shared-model",
                )
            )
            app.register_agent(
                AgentSpec(
                    name="target-agent",
                    provider_name="target-provider",
                    model="shared-model",
                )
            )

            with pytest.raises(ValueError, match="target provider rejected source history"):
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=source.id,
                            session_id=child_id,
                            agent_name="target-agent",
                            execution_profile_selection=(
                                ForkExecutionProfileSelection.CURRENT_CHILD
                            ),
                            profile_adoption=ExecutionProfileAdoptionIntent(
                                idempotency_key=f"portable-fork-{store_kind}",
                                reason="Authorize the target provider.",
                                requested_by=ResolutionActor(
                                    subject="test-caller",
                                    source=ResolutionActorSource.REQUEST,
                                ),
                            ),
                        )
                    )
                ]

            assert len(target_provider.preflight_calls) == 1
            model, messages, tools = target_provider.preflight_calls[0]
            assert model == "shared-model"
            assert list(messages) == transcript_before
            assert tools == ()
            assert await store.load(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_events(child_id)
            assert await store.load(source.id) == source_before
            assert await store.load_transcript(source.id) == transcript_before
            assert await store.load_events(source.id) == events_before
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_equal_current_child_authorization_is_replayable(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        source_id = f"equal_profile_fork_source_{store_kind}_{uuid4().hex}"
        child_id = f"equal_profile_fork_child_{store_kind}_{uuid4().hex}"
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "fork source")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                    provider=_UnusedForkProvider(),
                ),
            )
            source = await store.update_status(source.id, SessionStatus.COMPLETED)
            app = CayuApp(
                session_store=store,
                execution_profile_policy=_AuthorizeForkProfilePolicy(),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            request = ForkSessionRequest(
                source_session_id=source.id,
                session_id=child_id,
                execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                profile_adoption=ExecutionProfileAdoptionIntent(
                    idempotency_key=f"equal-profile-fork-{store_kind}",
                    reason="Authorize decision-bearing current registrations.",
                    requested_by=ResolutionActor(
                        subject="test-caller",
                        source=ResolutionActorSource.REQUEST,
                    ),
                ),
            )

            async def collect_fork() -> list[Event]:
                return [event async for event in app.fork_session(request)]

            first, replay = await asyncio.gather(collect_fork(), collect_fork())

            assert [event.type for event in first] == [EventType.SESSION_FORKED]
            assert [event.id for event in replay] == [event.id for event in first]
            child = await store.load(child_id)
            assert child is not None
            relationship = sessions_module.session_fork_profile_relationship(child)
            assert relationship is not None
            assert relationship.selected_profile == relationship.source_profile
            assert relationship.decision is not None
            assert relationship.decision.kind is ExecutionProfileDecisionKind.ADOPTED
            assert (
                relationship.decision.authority_decision
                is ExecutionProfileAuthorityDecision.AUTHORIZED
            )
            records = await store.query_events(EventQuery(session_id=child_id, limit=10))
            assert [record.event.type for record in records] == [
                EventType.SESSION_EXECUTION_PROFILE_DECIDED,
                EventType.SESSION_FORKED,
                EventType.TARGETED_TOOL_GRANT_FORK_RESET,
            ]
            assert records[-2].event.id == relationship.fork_event_id
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("profile_mode", ["inherit", "current-model", "current-provider"])
def test_session_store_conformance_profiled_fork_is_atomic_and_exactly_replayable(
    session_store_case,
    profile_mode: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        source_id = f"profiled_fork_source_{profile_mode}_{store_kind}_{uuid4().hex}"
        child_id = f"profiled_fork_child_{profile_mode}_{store_kind}_{uuid4().hex}"
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "fork source")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                ),
            )
            source = await store.update_status(source.id, SessionStatus.COMPLETED)
            app = CayuApp(
                session_store=store,
                execution_profile_policy=_AuthorizeForkProfilePolicy(),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            if profile_mode == "current-provider":
                app.register_provider(_UnusedNamedForkProvider("child-provider"))
                app.register_agent(
                    AgentSpec(
                        name="child-assistant",
                        provider_name="child-provider",
                        model="child-model",
                    )
                )
            request = (
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id=child_id,
                )
                if profile_mode == "inherit"
                else ForkSessionRequest(
                    source_session_id=source.id,
                    session_id=child_id,
                    agent_name=("child-assistant" if profile_mode == "current-provider" else None),
                    model=("child-model" if profile_mode == "current-model" else None),
                    execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key=f"profiled-fork-{profile_mode}-{store_kind}",
                        reason="Select the current child model profile.",
                        requested_by=ResolutionActor(
                            subject="test-caller",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )

            async def collect_fork() -> list[Event]:
                return [event async for event in app.fork_session(request)]

            first, replay = await asyncio.gather(
                collect_fork(),
                collect_fork(),
            )

            assert [event.type for event in first] == [EventType.SESSION_FORKED]
            assert [event.id for event in replay] == [event.id for event in first]
            child = await store.load(child_id)
            assert child is not None
            relationship = sessions_module.session_fork_profile_relationship(child)
            assert relationship is not None
            assert relationship.source_session_id == source.id
            if profile_mode == "inherit":
                assert relationship.selected_profile == relationship.source_profile
                expected_event_types = [
                    EventType.SESSION_FORKED,
                    EventType.TARGETED_TOOL_GRANT_FORK_RESET,
                ]
            else:
                assert relationship.selected_profile != relationship.source_profile
                assert relationship.decision is not None
                if profile_mode == "current-provider":
                    assert child.provider_name == "child-provider"
                expected_event_types = [
                    EventType.SESSION_EXECUTION_PROFILE_DECIDED,
                    EventType.SESSION_FORKED,
                    EventType.TARGETED_TOOL_GRANT_FORK_RESET,
                ]
            records = await store.query_events(EventQuery(session_id=child_id, limit=10))
            assert [record.event.type for record in records] == expected_event_types
            assert records[-2].event.id == relationship.fork_event_id
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_independent_forks_share_exact_source_authority(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        source_id = f"independent_fork_source_{store_kind}_{uuid4().hex}"
        child_ids = tuple(
            f"independent_fork_child_{suffix}_{store_kind}_{uuid4().hex}" for suffix in ("a", "b")
        )
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "checkpoint 14")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                    app=app,
                ),
            )
            await store.update_status(source.id, SessionStatus.COMPLETED)
            await store.checkpoint(
                source.id,
                {
                    "queued_dispatch_terminal_receipts": {
                        "version": 1,
                        "receipts": {
                            "independent-fork-source-operation": {
                                "queue_task_id": "independent-fork-source-task",
                                "terminal_event_id": "independent-fork-source-terminal",
                                "run_epoch": 1,
                            }
                        },
                    }
                },
            )
            exact_source = await app.snapshot_fork_source(source.id)

            async def fork_child(child_id: str) -> list[Event]:
                initial_dispatch_id = f"dispatch-{child_id}"
                return [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=source.id,
                            session_id=child_id,
                            expected_source=exact_source,
                            initial_invocation=ResumeRequest(
                                session_id=child_id,
                                messages=[Message.text("user", f"run {child_id}")],
                            ),
                            initial_dispatch_id=initial_dispatch_id,
                        )
                    )
                ]

            results = await asyncio.gather(*(fork_child(child_id) for child_id in child_ids))
            assert [[event.type for event in events] for events in results] == [
                [EventType.SESSION_FORKED],
                [EventType.SESSION_FORKED],
            ]
            for child_id in child_ids:
                child = await store.load(child_id)
                assert child is not None
                assert child.parent_session_id == source.id
                assert child.causal_budget_id == exact_source.causal_budget_id
                assert child.metadata["cayu:fork_source_snapshot"] == exact_source.model_dump(
                    mode="json"
                )
                relationship = sessions_module.session_fork_profile_relationship(child)
                assert relationship is not None
                assert relationship.initial_invocation_request_sha256 is not None
                assert relationship.initial_dispatch_id == f"dispatch-{child_id}"
                assert relationship.initial_invocation_profile is not None
                records = await store.query_events(
                    EventQuery(
                        session_id=child_id,
                        event_type=EventType.SESSION_FORKED,
                        limit=1,
                    )
                )
                assert records[0].event.payload["initial_dispatch_id"] == (
                    relationship.initial_dispatch_id
                )

            # Queue acknowledgement changes only source-owned terminal retention.
            # It must not invalidate authority already issued for child-relevant state.
            await store.checkpoint(source.id, {})
            post_ack_child_id = f"independent_fork_child_post_ack_{store_kind}_{uuid4().hex}"
            post_ack_events = await fork_child(post_ack_child_id)
            assert [event.type for event in post_ack_events] == [EventType.SESSION_FORKED]
            post_ack_child = await store.load(post_ack_child_id)
            assert post_ack_child is not None
            assert post_ack_child.metadata["cayu:fork_source_snapshot"] == (
                exact_source.model_dump(mode="json")
            )

            await store.publish_checkpoint_and_events(
                source.id,
                checkpoint_transform=lambda _session, _checkpoint: {"application_checkpoint": 15},
                events=[],
            )
            with pytest.raises(SessionRunFenced, match="exact snapshot"):
                await fork_child(f"independent_fork_child_stale_{store_kind}_{uuid4().hex}")
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_exact_fork_rejects_recreated_source_incarnation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        source_id = f"recreated_fork_source_{store_kind}_{uuid4().hex}"
        child_id = f"recreated_fork_child_{store_kind}_{uuid4().hex}"

        async def create_source() -> Session:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "same durable source")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                ),
            )
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            await store.append_event(
                created.id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=created.id,
                ),
            )
            return completed

        try:
            original = await create_source()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            stale_snapshot = await app.snapshot_fork_source(source_id)

            await store.delete_session(source_id)
            replacement = await create_source()
            replacement_snapshot = await app.snapshot_fork_source(source_id)

            assert replacement.instance_id != original.instance_id
            assert replacement_snapshot.source_instance_fingerprint != (
                stale_snapshot.source_instance_fingerprint
            )
            assert replacement_snapshot == stale_snapshot.model_copy(
                update={
                    "source_instance_fingerprint": (
                        replacement_snapshot.source_instance_fingerprint
                    )
                }
            )
            with pytest.raises(SessionRunFenced, match="identity or lifecycle"):
                async for _ in app.fork_session(
                    ForkSessionRequest(
                        source_session_id=source_id,
                        session_id=child_id,
                        expected_source=stale_snapshot,
                    )
                ):
                    pass

            assert await store.load(child_id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("child_identity", ("explicit", "generated"))
@pytest.mark.parametrize("causal_budget_identity", ("source", "independent"))
def test_session_store_conformance_exact_fork_aliases_survive_source_deletion(
    session_store_case,
    child_identity: str,
    causal_budget_identity: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        case_id = f"{child_identity}_{causal_budget_identity}_{store_kind}_{uuid4().hex}"
        private_source_id = f"private_fork_source_{case_id}"
        private_causal_budget_id = (
            private_source_id
            if causal_budget_identity == "source"
            else f"private_fork_budget_{case_id}"
        )
        requested_child_id = (
            f"public_fork_child_{case_id}" if child_identity == "explicit" else None
        )
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=private_source_id,
                    causal_budget_id=private_causal_budget_id,
                    messages=[Message.text("user", "checkpoint 14")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                ),
            )
            await store.update_status(source.id, SessionStatus.COMPLETED)
            await store.append_event(
                source.id,
                Event(type=EventType.SESSION_COMPLETED, session_id=source.id),
            )
            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor("private"),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            public_source_id = app.project_session_id_for_exposure(source.id)
            exact_source = await app.snapshot_fork_source(public_source_id)

            exact_source_json = exact_source.model_dump_json()
            assert private_source_id not in exact_source_json
            assert private_causal_budget_id not in exact_source_json
            assert exact_source.source_session_id == public_source_id
            alias_field_name = (
                "session_id" if causal_budget_identity == "source" else "causal_budget_id"
            )
            assert f".{alias_field_name}." in exact_source.causal_budget_id
            assert (
                await store.resolve_public_authority_alias(
                    exact_source.causal_budget_id,
                    field_name=alias_field_name,
                )
                == private_causal_budget_id
            )
            request = ForkSessionRequest(
                source_session_id=public_source_id,
                session_id=requested_child_id,
                expected_source=exact_source,
            )
            events = [event async for event in app.fork_session(request)]

            assert [event.type for event in events] == [EventType.SESSION_FORKED]
            assert all(private_source_id not in event.model_dump_json() for event in events)
            assert all(private_causal_budget_id not in event.model_dump_json() for event in events)
            child_id = events[0].session_id
            if requested_child_id is not None:
                assert child_id == requested_child_id
            child = await store.load(child_id)
            assert child is not None
            assert child.parent_session_id == private_source_id
            assert child.causal_budget_id == private_causal_budget_id
            private_snapshot = sessions_module.ForkSourceSnapshot.model_validate(
                child.metadata[sessions_module.FORK_SOURCE_SNAPSHOT_METADATA_KEY]
            )
            assert private_snapshot == exact_source.model_copy(
                update={
                    "source_session_id": private_source_id,
                    "causal_budget_id": private_causal_budget_id,
                },
            )
            child_events_before_delete = await store.load_events(child_id)

            await store.delete_session(source.id)

            detached_child = await store.load(child_id)
            assert detached_child is not None
            assert detached_child.parent_session_id is None
            assert (
                await store.resolve_public_authority_alias(
                    exact_source.causal_budget_id,
                    field_name=alias_field_name,
                )
                == private_causal_budget_id
            )
            replay = [event async for event in app.fork_session(request)]
            assert [event.id for event in replay] == [event.id for event in events]
            assert await store.load_events(child_id) == child_events_before_delete
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_generated_profiled_fork_is_idempotent(
    session_store_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        source_id = f"generated_profiled_fork_source_{store_kind}_{uuid4().hex}"
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "fork source")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                ),
            )
            source = await store.update_status(source.id, SessionStatus.COMPLETED)
            await store.append_event(
                source.id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=source.id,
                ),
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            request = ForkSessionRequest(source_session_id=source.id)

            async def collect_fork() -> list[Event]:
                return [event async for event in app.fork_session(request)]

            first, replay = await asyncio.gather(collect_fork(), collect_fork())

            assert [event.type for event in first] == [EventType.SESSION_FORKED]
            assert [event.id for event in replay] == [event.id for event in first]
            children = (
                await store.list_sessions(SessionQuery(parent_session_id=source.id, limit=10))
            ).sessions
            assert len(children) == 1
            child = children[0]
            relationship = sessions_module.session_fork_profile_relationship(child)
            assert relationship is not None
            assert relationship.source_session_id == source.id
            records = await store.query_events(EventQuery(session_id=child.id, limit=10))
            assert [record.event.type for record in records] == [
                EventType.SESSION_FORKED,
                EventType.TARGETED_TOOL_GRANT_FORK_RESET,
            ]

            await store.delete_session(source.id)
            detached_child = await store.load(child.id)
            assert detached_child is not None
            assert detached_child.parent_session_id is None
            assert sessions_module.session_fork_profile_relationship(detached_child) == relationship
            sessions_module.validate_profiled_fork_evidence(
                fork=detached_child,
                relationship=relationship,
                events=[record.event for record in records],
            )

            async def reject_source_admission(*_args, **_kwargs) -> None:
                raise AssertionError("exact child replay consulted source task authority")

            monkeypatch.setattr(
                "cayu.runtime._session_fork.require_ordinary_session_execution",
                reject_source_admission,
            )

            detached_replay = await collect_fork()
            assert [event.id for event in detached_replay] == [event.id for event in first]
            replayed_records = await store.query_events(EventQuery(session_id=child.id, limit=10))
            assert [record.event.id for record in replayed_records] == [
                record.event.id for record in records
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_hook_fork_preserves_generated_provenance(
    session_store_case,
) -> None:
    class ForkCompletedSessionHook(RuntimeHook):
        def __init__(self) -> None:
            self.calls = 0
            self.fork_events: list[Event] = []

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            self.calls += 1
            self.fork_events = await context.fork_session(
                ForkSessionRequest(source_session_id=context.session.id)
            )

    async def run() -> None:
        store = await _open_store(session_store_case)
        hook = ForkCompletedSessionHook()
        try:
            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor("-"),
                runtime_hooks=[hook],
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        messages=[Message.text("user", "start")],
                    )
                )
            ]

            assert hook.calls == 1
            assert [event.type for event in hook.fork_events] == [EventType.SESSION_FORKED]
            assert [event.type for event in events].count(EventType.SESSION_COMPLETED) == 1
            assert [event.type for event in events].count(EventType.HOOK_STARTED) == 1
            assert [event.type for event in events].count(EventType.HOOK_COMPLETED) == 1
            sessions = (await store.list_sessions(SessionQuery(limit=10))).sessions
            assert len(sessions) == 2
            source = next(session for session in sessions if session.parent_session_id is None)
            child = next(session for session in sessions if session.parent_session_id == source.id)
            assert "-" in source.id
            assert "-" in child.id
            assert "invocation_configuration" in (await store.load_checkpoint(source.id))
            assert "invocation_configuration" not in (await store.load_checkpoint(child.id) or {})
            source_events = await store.load_events(source.id)
            assert [event.type for event in source_events].count(EventType.SESSION_COMPLETED) == 1
            assert [event.type for event in source_events].count(EventType.HOOK_STARTED) == 1
            assert [event.type for event in source_events].count(EventType.HOOK_COMPLETED) == 1
            child_events = await store.load_events(child.id)
            assert [event.type for event in child_events] == [
                EventType.SESSION_FORKED,
                EventType.TARGETED_TOOL_GRANT_FORK_RESET,
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_fork_source_provenance_distinguishes_callers(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        secret = "private-fork-source"
        source_id = f"legacy-{secret}-session-{store_kind}"
        raw_child_id = f"raw-fork-child-{store_kind}"
        alias_child_id = f"alias-fork-child-{store_kind}"
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "source")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fakemodel",
                ),
            )
            await store.update_status(source.id, SessionStatus.COMPLETED)
            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

            with pytest.raises(ValueError, match="source_session_id"):
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=source.id,
                            session_id=raw_child_id,
                        )
                    )
                ]
            assert await store.load(raw_child_id) is None

            public_source_id = app.project_session_id_for_exposure(source.id)
            await store.register_public_authority_alias(
                public_source_id,
                field_name="session_id",
                private_value=source.id,
            )
            alias_events = [
                event
                async for event in app.fork_session(
                    ForkSessionRequest(
                        source_session_id=public_source_id,
                        session_id=alias_child_id,
                    )
                )
            ]

            assert [event.type for event in alias_events] == [EventType.SESSION_FORKED]
            child = await store.load(alias_child_id)
            assert child is not None
            assert child.parent_session_id == source.id
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("source_kind", ("generated", "legacy-secret"))
def test_session_store_conformance_stale_fork_alias_omits_private_authority(
    session_store_case,
    source_kind: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        secret = "stale-fork-private-secret" if source_kind == "legacy-secret" else "-"
        requested_source_id = (
            f"legacy-{secret}-session-{store_kind}" if source_kind == "legacy-secret" else None
        )
        child_id = f"stale_fork_child_{source_kind.replace('-', '_')}_{store_kind}"
        sink = InMemoryEventSink()
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=requested_source_id,
                    messages=[Message.text("user", "source")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fakemodel"),
            )
            await store.append_transcript_messages(
                source.id,
                [Message.text("user", "copied transcript")],
            )
            await store.checkpoint(source.id, {"safe": "checkpoint"})
            await store.update_status(source.id, SessionStatus.COMPLETED)
            await store.append_event(
                source.id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=source.id,
                ),
            )

            app = CayuApp(
                session_store=store,
                event_sinks=[sink],
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fakemodel"))
            public_source_id = app.project_session_id_for_exposure(source.id)
            await store.register_public_authority_alias(
                public_source_id,
                field_name="session_id",
                private_value=source.id,
            )

            await store.delete_session(source.id)
            assert await store.load(source.id) is None
            assert (
                await store.resolve_public_authority_alias(
                    public_source_id,
                    field_name="session_id",
                )
                == source.id
            )

            with pytest.raises(KeyError) as raised:
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=public_source_id,
                            session_id=child_id,
                        )
                    )
                ]

            assert raised.value.args == ("Fork source session was not found.",)
            private_values = (source.id, secret) if len(secret) > 1 else (source.id,)
            _assert_exception_omits_private_values(raised.value, *private_values)
            assert await store.load(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_transcript(child_id)
            assert await store.load_checkpoint(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_events(child_id)
            assert sink.events == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_fork_source_deleted_after_initial_load_omits_private_authority(
    session_store_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        secret = "fork-race-private-secret"
        source_id = f"legacy-{secret}-session-{store_kind}"
        child_id = f"fork_race_child_{store_kind}"
        sink = InMemoryEventSink()
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "source")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fakemodel",
                ),
            )
            await store.update_status(source.id, SessionStatus.COMPLETED)
            await store.append_event(
                source.id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=source.id,
                ),
            )

            app = CayuApp(
                session_store=store,
                event_sinks=[sink],
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fakemodel"))
            public_source_id = app.project_session_id_for_exposure(source.id)
            await store.register_public_authority_alias(
                public_source_id,
                field_name="session_id",
                private_value=source.id,
            )

            original_create_fork = store.create_profiled_fork
            deleted_after_initial_load = False

            async def delete_source_then_create_fork(**kwargs):
                nonlocal deleted_after_initial_load
                deleted_after_initial_load = True
                await store.delete_session(source.id)
                return await original_create_fork(**kwargs)

            monkeypatch.setattr(
                store,
                "create_profiled_fork",
                delete_source_then_create_fork,
            )

            with pytest.raises(KeyError) as raised:
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=public_source_id,
                            session_id=child_id,
                        )
                    )
                ]

            assert deleted_after_initial_load is True
            assert raised.value.args == ("Fork source session was not found.",)
            _assert_exception_omits_private_values(raised.value, source.id, secret)
            assert await store.load(source.id) is None
            assert await store.load(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_transcript(child_id)
            assert await store.load_checkpoint(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_events(child_id)
            assert sink.events == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_active_stage_fork_omits_private_source_authority(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        secret = "active-stage-fork-private-secret"
        source_id = f"legacy-{secret}-session-{store_kind}"
        child_id = f"active_stage_fork_child_{store_kind}"
        sink = InMemoryEventSink()
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "source")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fakemodel",
                ),
            )
            await store.append_transcript_messages(
                source.id,
                [Message.text("user", "source")],
            )
            running = await store.transition_status(
                source.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.prepare_model_completion_stage(
                source.id,
                request=ModelCompletionStageRequest(
                    stage_id="active-stage-fork-stage",
                    logical_step_id="active-stage-fork-step",
                    dispatch_ordinal=0,
                    intent={"request_fingerprint": "active-stage-fork"},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=1,
            )
            source = await store.update_status(source.id, SessionStatus.INTERRUPTED)
            await store.append_event(
                source.id,
                Event(type=EventType.SESSION_INTERRUPTED, session_id=source.id),
            )

            app = CayuApp(
                session_store=store,
                event_sinks=[sink],
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fakemodel"))
            public_source_id = app.project_session_id_for_exposure(source.id)
            await store.register_public_authority_alias(
                public_source_id,
                field_name="session_id",
                private_value=source.id,
            )

            with pytest.raises(ValueError) as raised:
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=public_source_id,
                            session_id=child_id,
                        )
                    )
                ]

            assert raised.value.args == (
                "Fork source session has an active model-completion stage.",
            )
            assert raised.value.__cause__ is None
            assert raised.value.__context__ is None
            _assert_exception_omits_private_values(raised.value, source.id, secret)
            assert await store.load(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_transcript(child_id)
            assert await store.load_checkpoint(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_events(child_id)
            assert sink.events == []
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("source_status", "copy_checkpoint"),
    (
        pytest.param(SessionStatus.INTERRUPTED, True, id="copy-checkpoint"),
        pytest.param(SessionStatus.FAILED, False, id="discard-checkpoint"),
    ),
)
def test_session_store_conformance_rejects_fork_with_pending_tool_round(
    session_store_case,
    source_status: SessionStatus,
    copy_checkpoint: bool,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        mode = "copy" if copy_checkpoint else "discard"
        source_id = f"sess_pending_round_fork_source_{store_kind}_{mode}"
        child_id = f"sess_pending_round_fork_child_{store_kind}_{mode}"
        identity = {
            "model_step_id": f"mstep_{'1' * 32}",
            "model_attempt_id": f"matt_{'2' * 32}",
            "tool_round_id": f"tround_{'3' * 32}",
        }
        checkpoint = {
            "pending_tool_round": {
                **identity,
                "agent_name": "assistant",
                "tool_calls": [
                    {
                        "tool_call_id": "call_external_effect",
                        "tool_name": "external_effect",
                        "arguments": {},
                    }
                ],
            }
        }
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "perform the effect")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                ),
            )
            await store.append_event(
                source.id,
                Event(
                    id=f"evt_pending_round_started_{store_kind}",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=source.id,
                    tool_name="external_effect",
                    payload={
                        **identity,
                        "tool_call_id": "call_external_effect",
                    },
                ),
            )
            await store.checkpoint(source.id, checkpoint)
            await store.update_status(source.id, source_status)
            source_events = await store.load_events(source.id)

            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))

            with pytest.raises(RuntimeError, match="pending tool round cannot be forked"):
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=source.id,
                            session_id=child_id,
                            copy_checkpoint=copy_checkpoint,
                        )
                    )
                ]

            assert await store.load(child_id) is None
            assert await store.load_checkpoint(source.id) == checkpoint
            assert await store.load_events(source.id) == source_events
        finally:
            await _close_store(store)

    asyncio.run(run())
