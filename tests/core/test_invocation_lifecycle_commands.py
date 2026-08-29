from __future__ import annotations

import asyncio
import copy
import pickle
from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import UTC, datetime, timedelta
from typing import Any, get_type_hints
from uuid import uuid4

import pytest

from cayu import AgentSpec, CayuApp, SQLiteSessionStore
from cayu._validation import canonical_durable_json_bytes
from cayu.core import Event, EventType, ExecutionProfileBehaviorIdentity, Message
from cayu.core.events import (
    event_envelope_authority_is_runtime_generated,
    event_with_runtime_envelope_authority,
)
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    AdmitInvocationCommand,
    CreateInvocationCommand,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InteractionTransitionSpec,
    InvocationCheckpointPatch,
    InvocationContext,
    InvocationLifecycleCommandConflict,
    InvocationMutationResult,
    InvocationReleaseResult,
    LoopPolicy,
    PreparedInvocationBinding,
    RejectInvocationCommand,
    ReleaseInvocationCommand,
    ResumeRequest,
    RunRequest,
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    SessionIdentity,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    SettleInvocationCommand,
    ToolCapabilityCeiling,
)
from cayu.runtime import _invocation_lifecycle as invocation_lifecycle_module
from cayu.runtime import sessions as sessions_module
from cayu.runtime._checkpoint_redaction import durable_value_contains_secret
from cayu.runtime._checkpoint_store import (
    load_runtime_session_checkpoint_snapshot,
    runtime_checkpoint_session_store,
)
from cayu.runtime._invocation_lifecycle import (
    INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS,
    _authenticated_invocation_context,
    _InvocationLifecycleCommandReceipt,
    _InvocationLifecycleReceiptLedger,
    _projected_invocation_release_receipt,
    _release_invocation_command_with_cleanup_authority,
    invocation_checkpoint_state_sha256,
    prepare_rebind_invocation_command,
)
from cayu.runtime.budgets import BudgetPolicy
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    CheckpointCompatibilityError,
)
from cayu.runtime.execution_profiles import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    EXECUTION_PROFILE_METADATA_KEY,
    ActiveInvocationExecutionProfile,
    ExecutionProfileIdentity,
    active_invocation_execution_profile_from_checkpoint,
    build_execution_profile_identity,
    changed_execution_profile_components,
    checkpoint_with_active_invocation_execution_profile,
)
from cayu.runtime.sessions import (
    _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
    ModelTarget,
    SessionInvocationAdmission,
    _current_session_run_epoch,
    run_request_with_runtime_session_instance_authority,
)
from cayu.runtime.tool_exposure import TOOL_CAPABILITY_CEILING_METADATA_KEY
from cayu.vaults import SecretRedactor


def test_session_store_lifecycle_command_runtime_annotations_are_resolvable() -> None:
    assert get_type_hints(SessionStore.apply_invocation_lifecycle_command) == {
        "command": object,
        "return": object,
    }


def _profile(*, tool_name: str = "original_tool") -> ExecutionProfileIdentity:
    return build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="test",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt="durable instructions",
        direct_tools=[
            {
                "tool_id": f"cayu:{tool_name}",
                "descriptor_version": f"sha256:{'d' * 64}",
            }
        ],
        tool_catalogue_revision=f"sha256:{'c' * 64}",
        tool_implementations=[{"implementation": f"test:{tool_name}:v1"}],
        tool_view_grants={
            "view_kind": "direct",
            "generation": 1,
            "grant_baseline": ["original_tool"],
        },
        effect_authority={"authority": "test-fixture"},
    )


class _NeverCalledProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        if False:  # pragma: no cover - keeps this an async generator.
            yield ModelStreamEvent.error("unreachable")


def _create_command(
    *,
    session_id: str,
    session_instance_id: str,
    interaction_id: str,
    profile: ExecutionProfileIdentity,
) -> CreateInvocationCommand:
    request = run_request_with_runtime_session_instance_authority(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[],
            tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
        ),
        session_instance_id=session_instance_id,
    )
    return CreateInvocationCommand(
        session_id=session_id,
        expected_session_instance_id=session_instance_id,
        request=request,
        identity=SessionIdentity(
            provider_name="fake",
            model="fake-model",
            runtime_name="cayu",
            runtime_version="test",
            execution_profile=profile,
        ),
        active_profile=ActiveInvocationExecutionProfile(
            session_id=session_id,
            interaction_id=interaction_id,
            run_epoch=1,
            profile=profile,
        ),
        interaction_started_event=Event(
            type=EventType.INTERACTION_STARTED,
            session_id=session_id,
            interaction_id=interaction_id,
            agent_name="assistant",
        ),
        interaction_source_messages=(),
        checkpoint_patch=InvocationCheckpointPatch(
            mutation=RuntimePublicationMutation(
                operations=(
                    RuntimePublicationCheckpointOperation(
                        key="invocation-test",
                        expected_value_digest=None,
                        action="set",
                        value={"prepared": True},
                    ),
                )
            )
        ),
    )


def _transition_event(
    *,
    session_id: str,
    interaction_id: str,
    status: SessionStatus,
) -> Event:
    event_type = {
        SessionStatus.COMPLETED: EventType.INTERACTION_COMPLETED,
        SessionStatus.FAILED: EventType.INTERACTION_FAILED,
        SessionStatus.INTERRUPTED: EventType.INTERACTION_INTERRUPTED,
    }[status]
    return event_with_runtime_envelope_authority(
        Event(
            type=event_type,
            session_id=session_id,
            interaction_id=interaction_id,
            agent_name="assistant",
        ),
        "session_id",
        "interaction_id",
    )


def _rejection_event(
    *,
    session_id: str,
    expected: ExecutionProfileIdentity,
    candidate: ExecutionProfileIdentity,
) -> Event:
    changed = changed_execution_profile_components(expected, candidate)
    return Event(
        type=EventType.SESSION_EXECUTION_PROFILE_REJECTED,
        session_id=session_id,
        agent_name="assistant",
        payload={
            "expected_profile_fingerprint": expected.fingerprint,
            "candidate_profile_fingerprint": candidate.fingerprint,
            "changed_component_classes": [item.value for item in changed],
        },
    )


async def _assert_invocation_command_conformance(store, suffix: str) -> None:
    session_id = f"invocation-command-{suffix}-{uuid4().hex}"
    session_instance_id = str(uuid4())
    interaction_id = f"interaction-{uuid4().hex}"
    profile = _profile()
    create_command = _create_command(
        session_id=session_id,
        session_instance_id=session_instance_id,
        interaction_id=interaction_id,
        profile=profile,
    )
    created = await store.apply_invocation_lifecycle_command(create_command)
    assert type(created) is InvocationMutationResult
    assert created.session.status is SessionStatus.RUNNING
    assert created.session.run_epoch == 1
    assert created.session.instance_id == session_instance_id
    assert created.active_profile.run_epoch == created.session.run_epoch
    checkpoint = await store.load_checkpoint(session_id)
    assert checkpoint is not None
    assert checkpoint["invocation-test"] == {"prepared": True}
    assert active_invocation_execution_profile_from_checkpoint(checkpoint) == (
        created.active_profile
    )
    for colliding_secret in ("command_identity", "create", "direct_tools"):
        assert not durable_value_contains_secret(
            checkpoint,
            redactor=SecretRedactor(colliding_secret),
        )
    replayed_create = await store.apply_invocation_lifecycle_command(create_command)
    assert replayed_create.replayed is True
    assert replayed_create.session == created.session
    conflicting_create = create_command.model_copy(
        update={
            "checkpoint_patch": InvocationCheckpointPatch(
                mutation=RuntimePublicationMutation(
                    operations=(
                        RuntimePublicationCheckpointOperation(
                            key="invocation-test",
                            expected_value_digest=None,
                            action="set",
                            value={"prepared": "conflicting"},
                        ),
                    )
                )
            )
        }
    )
    with pytest.raises(
        InvocationLifecycleCommandConflict,
        match="identity was reused with new authority",
    ):
        await store.apply_invocation_lifecycle_command(conflicting_create)

    def replace_private_lifecycle_authority(_session, current):
        assert current is not None
        assert ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY not in current
        assert INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY not in current
        replacement = dict(current)
        replacement[ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY] = {"caller": "forged"}
        replacement[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = {"caller": "forged"}
        return replacement

    await store.transform_checkpoint(session_id, replace_private_lifecycle_authority)
    protected_checkpoint = await store.load_checkpoint(session_id)
    assert protected_checkpoint is not None
    assert active_invocation_execution_profile_from_checkpoint(protected_checkpoint) == (
        created.active_profile
    )
    assert INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY in protected_checkpoint
    assert (await store.apply_invocation_lifecycle_command(create_command)).replayed is True
    assert await store.materialize_deferred_interaction_input(
        session_id,
        interaction_id=interaction_id,
    )
    assert (await store.apply_invocation_lifecycle_command(create_command)).replayed is True

    with pytest.raises(SessionRunFenced, match="cleanup has not proven quiescence"):
        await store.apply_invocation_lifecycle_command(
            ReleaseInvocationCommand(
                session_id=session_id,
                expected_session_instance_id=session_instance_id,
                expected_run_epoch=1,
                expected_active_profile=ActiveInvocationExecutionProfile(
                    session_id=session_id,
                    interaction_id=interaction_id,
                    run_epoch=1,
                    profile=_profile(tool_name="forged_tool"),
                ),
                settlement_transition=InteractionTransitionSpec(
                    event=_transition_event(
                        session_id=session_id,
                        interaction_id=interaction_id,
                        status=SessionStatus.INTERRUPTED,
                    ),
                    from_statuses=(SessionStatus.RUNNING,),
                    to_status=SessionStatus.INTERRUPTED,
                ),
            )
        )
    assert _current_session_run_epoch(session_id) == 1

    interrupted_event = _transition_event(
        session_id=session_id,
        interaction_id=interaction_id,
        status=SessionStatus.INTERRUPTED,
    )
    caller_reconstructed_event = Event.model_validate(interrupted_event.model_dump(mode="python"))
    with pytest.raises(
        ValueError,
        match="lacks runtime-owned session/interaction authority",
    ):
        SettleInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            expected_run_epoch=1,
            expected_active_profile=created.active_profile,
            transition=InteractionTransitionSpec(
                event=caller_reconstructed_event,
                from_statuses=(SessionStatus.RUNNING,),
                to_status=SessionStatus.INTERRUPTED,
            ),
        )
    premature_release = _release_invocation_command_with_cleanup_authority(
        ReleaseInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            expected_run_epoch=1,
            expected_active_profile=created.active_profile,
            settlement_transition=InteractionTransitionSpec(
                event=interrupted_event,
                from_statuses=(SessionStatus.RUNNING,),
                to_status=SessionStatus.INTERRUPTED,
            ),
        )
    )
    with pytest.raises(SessionRunFenced, match="exact durable terminal settlement"):
        await store.apply_invocation_lifecycle_command(premature_release)
    with pytest.raises(SessionRunFenced, match="authenticated cleanup/settlement"):
        await store.release_session_invocation(premature_release)
    before_settlement = await store.load(session_id)
    assert before_settlement is not None
    assert before_settlement.run_epoch == 1
    conflicting_event = _transition_event(
        session_id=session_id,
        interaction_id=interaction_id,
        status=SessionStatus.FAILED,
    )
    wrong_owner_events = (
        interrupted_event.model_copy(update={"agent_name": "another-agent"}),
        interrupted_event.model_copy(update={"environment_name": "another-environment"}),
    )
    for wrong_owner_event in wrong_owner_events:
        with pytest.raises(SessionRunFenced, match="event conflicts with session authority"):
            await store.apply_invocation_lifecycle_command(
                SettleInvocationCommand(
                    session_id=session_id,
                    expected_session_instance_id=session_instance_id,
                    expected_run_epoch=1,
                    expected_active_profile=created.active_profile,
                    transition=InteractionTransitionSpec(
                        event=wrong_owner_event,
                        from_statuses=(SessionStatus.RUNNING,),
                        to_status=SessionStatus.INTERRUPTED,
                    ),
                )
            )
    unchanged = await store.load(session_id)
    assert unchanged is not None
    assert unchanged.status is SessionStatus.RUNNING
    assert all(
        event.id not in {item.id for item in wrong_owner_events}
        for event in await store.load_events(session_id)
    )
    with pytest.raises(SessionRunFenced, match="session incarnation"):
        await store.apply_invocation_lifecycle_command(
            SettleInvocationCommand(
                session_id=session_id,
                expected_session_instance_id=str(uuid4()),
                expected_run_epoch=1,
                expected_active_profile=created.active_profile,
                transition=InteractionTransitionSpec(
                    event=conflicting_event,
                    from_statuses=(SessionStatus.RUNNING,),
                    to_status=SessionStatus.FAILED,
                ),
            )
        )
    unchanged = await store.load(session_id)
    assert unchanged is not None
    assert unchanged.status is SessionStatus.RUNNING
    assert all(event.id != conflicting_event.id for event in await store.load_events(session_id))

    settlement = SettleInvocationCommand(
        session_id=session_id,
        expected_session_instance_id=session_instance_id,
        expected_run_epoch=1,
        expected_active_profile=created.active_profile,
        transition=InteractionTransitionSpec(
            event=interrupted_event,
            from_statuses=(SessionStatus.RUNNING,),
            to_status=SessionStatus.INTERRUPTED,
        ),
    )
    settled = await store.apply_invocation_lifecycle_command(settlement)
    assert settled.session.status is SessionStatus.INTERRUPTED
    assert settled.replayed is False
    replayed_settlement = await store.apply_invocation_lifecycle_command(settlement)
    assert replayed_settlement.replayed is True
    assert replayed_settlement.session == settled.session
    delayed_create_replay = await store.apply_invocation_lifecycle_command(create_command)
    assert delayed_create_replay.replayed is True
    assert delayed_create_replay.session == created.session
    assert delayed_create_replay.active_profile == created.active_profile
    with pytest.raises(SessionRunFenced, match="another invocation authority"):
        await store.apply_invocation_lifecycle_command(
            settlement.model_copy(
                update={
                    "expected_active_profile": ActiveInvocationExecutionProfile(
                        session_id=session_id,
                        interaction_id=interaction_id,
                        run_epoch=1,
                        profile=_profile(tool_name="forged_tool"),
                    )
                }
            )
        )

    release = _release_invocation_command_with_cleanup_authority(
        ReleaseInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            expected_run_epoch=1,
            expected_active_profile=created.active_profile,
            settlement_transition=settlement.transition,
        )
    )
    released = await store.apply_invocation_lifecycle_command(release)
    assert type(released) is InvocationReleaseResult
    assert released.session.run_epoch == 2
    assert released.replayed is False
    replayed_release = await store.apply_invocation_lifecycle_command(release)
    assert replayed_release.session == released.session
    assert replayed_release.replayed is True

    rejected_profile = _profile(tool_name="replacement_tool")
    rejection = RejectInvocationCommand(
        session_id=session_id,
        expected_session_instance_id=session_instance_id,
        expected_statuses=(SessionStatus.INTERRUPTED,),
        expected_run_epoch=2,
        expected_profile=profile,
        candidate_profile=rejected_profile,
        event=_rejection_event(
            session_id=session_id,
            expected=profile,
            candidate=rejected_profile,
        ),
        expected_active_profile=created.active_profile,
    )
    with pytest.raises(SessionRunFenced, match="session incarnation"):
        await store.apply_invocation_lifecycle_command(
            rejection.model_copy(update={"expected_session_instance_id": str(uuid4())})
        )
    assert all(event.id != rejection.event.id for event in await store.load_events(session_id))

    rejected = await store.apply_invocation_lifecycle_command(rejection)
    assert rejected.replayed is False
    assert (await store.apply_invocation_lifecycle_command(rejection)).replayed is True

    rebound_profile = ActiveInvocationExecutionProfile(
        session_id=session_id,
        interaction_id=interaction_id,
        run_epoch=3,
        profile=profile,
    )
    rebind_source_session = await store.load(session_id)
    rebind_source_checkpoint = await store.load_checkpoint(session_id)
    assert rebind_source_session is not None
    assert rebind_source_checkpoint is not None

    def prepared_rebind(candidate: str):
        def transform(current_session, checkpoint):
            updated = dict(checkpoint or {})
            updated["rebind-winner"] = {"candidate": candidate}
            return checkpoint_with_active_invocation_execution_profile(
                updated,
                session_id=current_session.id,
                interaction_id=interaction_id,
                run_epoch=current_session.run_epoch + 1,
                profile=profile,
                expected=created.active_profile,
            )

        return prepare_rebind_invocation_command(
            rebind_source_session,
            rebind_source_checkpoint,
            expected_statuses={SessionStatus.INTERRUPTED},
            checkpoint_transform=transform,
        )

    stale_rebind = prepared_rebind("stale")

    def win_rebind_patch_race(_session, checkpoint):
        updated = dict(checkpoint or {})
        updated["rebind-winner"] = {"candidate": "concurrent"}
        return updated

    await store.transform_checkpoint(session_id, win_rebind_patch_race)
    with pytest.raises(
        SessionRunFenced,
        match="source state changed after command preparation",
    ):
        await store.apply_invocation_lifecycle_command(stale_rebind)
    rebind_source_session = await store.load(session_id)
    rebind_source_checkpoint = await store.load_checkpoint(session_id)
    assert rebind_source_session is not None
    assert rebind_source_checkpoint is not None

    rebinds = tuple(prepared_rebind(candidate) for candidate in ("first", "second"))
    rebind_outcomes = await asyncio.gather(
        *(store.apply_invocation_lifecycle_command(command) for command in rebinds),
        return_exceptions=True,
    )
    rebound_results = [
        outcome for outcome in rebind_outcomes if type(outcome) is InvocationMutationResult
    ]
    rebound_conflicts = [outcome for outcome in rebind_outcomes if isinstance(outcome, ValueError)]
    assert len(rebound_results) == 1
    assert len(rebound_conflicts) == 1
    rebound = rebound_results[0]
    assert type(rebound) is InvocationMutationResult
    assert rebound.session.run_epoch == 3
    assert rebound.active_profile == rebound_profile
    rebound_checkpoint = await store.load_checkpoint(session_id)
    assert rebound_checkpoint is not None
    assert rebound_checkpoint["rebind-winner"] in (
        {"candidate": "first"},
        {"candidate": "second"},
    )
    winning_rebind = rebinds[
        0 if rebound_checkpoint["rebind-winner"] == {"candidate": "first"} else 1
    ]
    assert (await store.apply_invocation_lifecycle_command(winning_rebind)).replayed is True

    delayed_release_replay = await store.apply_invocation_lifecycle_command(release)
    assert delayed_release_replay.replayed is True
    assert delayed_release_replay.session == released.session
    assert delayed_release_replay.active_profile == released.active_profile

    completed_event = _transition_event(
        session_id=session_id,
        interaction_id=interaction_id,
        status=SessionStatus.COMPLETED,
    )
    completed_settlement = SettleInvocationCommand(
        session_id=session_id,
        expected_session_instance_id=session_instance_id,
        expected_run_epoch=3,
        expected_active_profile=rebound_profile,
        transition=InteractionTransitionSpec(
            event=completed_event,
            from_statuses=(SessionStatus.INTERRUPTED,),
            to_status=SessionStatus.COMPLETED,
        ),
    )
    await store.apply_invocation_lifecycle_command(completed_settlement)
    rebound_release = _release_invocation_command_with_cleanup_authority(
        ReleaseInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            expected_run_epoch=3,
            expected_active_profile=rebound_profile,
            settlement_transition=completed_settlement.transition,
        )
    )
    concurrent_releases = await asyncio.gather(
        store.apply_invocation_lifecycle_command(rebound_release),
        store.apply_invocation_lifecycle_command(rebound_release),
    )
    assert sorted(result.replayed for result in concurrent_releases) == [False, True]
    assert {result.session.run_epoch for result in concurrent_releases} == {4}
    # Child tasks own their copied ContextVar state. The parent replays the exact
    # durable release to clear its own inherited run fence before later admission.
    owner_release_replay = await store.apply_invocation_lifecycle_command(rebound_release)
    assert owner_release_replay.replayed is True
    assert _current_session_run_epoch(session_id) is None
    delayed_rebind_replay = await store.apply_invocation_lifecycle_command(winning_rebind)
    assert delayed_rebind_replay.replayed is True
    assert delayed_rebind_replay.session == rebound.session
    assert delayed_rebind_replay.active_profile == rebound.active_profile

    new_interaction_id = f"interaction-{uuid4().hex}"
    admitted_profile = ActiveInvocationExecutionProfile(
        session_id=session_id,
        interaction_id=new_interaction_id,
        run_epoch=5,
        profile=profile,
    )
    admission_checkpoint = await store.load_checkpoint(session_id)
    admission = AdmitInvocationCommand(
        session_id=session_id,
        expected_session_instance_id=session_instance_id,
        expected_statuses=(SessionStatus.COMPLETED,),
        expected_run_epoch=4,
        expected_checkpoint_sha256=invocation_checkpoint_state_sha256(admission_checkpoint),
        target_active_profile=admitted_profile,
        interaction_started_event=Event(
            type=EventType.INTERACTION_STARTED,
            session_id=session_id,
            interaction_id=new_interaction_id,
            agent_name="assistant",
        ),
        interaction_source_messages=(),
        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
        expected_active_profile=rebound_profile,
    )
    assert admission.interaction_started_event is not None
    wrong_admission_events = (
        admission.interaction_started_event.model_copy(update={"agent_name": "another-agent"}),
        admission.interaction_started_event.model_copy(
            update={"environment_name": "another-environment"}
        ),
    )
    for wrong_owner_event in wrong_admission_events:
        with pytest.raises(SessionRunFenced, match="event conflicts with session authority"):
            await store.apply_invocation_lifecycle_command(
                admission.model_copy(update={"interaction_started_event": wrong_owner_event})
            )
    unchanged = await store.load(session_id)
    assert unchanged is not None
    assert unchanged.status is SessionStatus.COMPLETED
    assert unchanged.run_epoch == 4
    assert all(
        event.id not in {item.id for item in wrong_admission_events}
        for event in await store.load_events(session_id)
    )

    with pytest.raises(
        SessionRunFenced,
        match="lacks exact predecessor release authority",
    ):
        await store.apply_invocation_lifecycle_command(
            admission.model_copy(update={"expected_active_profile": None})
        )
    unchanged = await store.load(session_id)
    assert unchanged is not None
    assert unchanged.status is SessionStatus.COMPLETED
    assert unchanged.run_epoch == 4

    admitted = await store.apply_invocation_lifecycle_command(admission)
    assert type(admitted) is InvocationMutationResult
    assert admitted.session.status is SessionStatus.RUNNING
    assert admitted.session.run_epoch == 5
    assert (await store.apply_invocation_lifecycle_command(admission)).replayed is True
    final_settlement = SettleInvocationCommand(
        session_id=session_id,
        expected_session_instance_id=session_instance_id,
        expected_run_epoch=5,
        expected_active_profile=admitted_profile,
        transition=InteractionTransitionSpec(
            event=_transition_event(
                session_id=session_id,
                interaction_id=new_interaction_id,
                status=SessionStatus.COMPLETED,
            ),
            from_statuses=(SessionStatus.RUNNING,),
            to_status=SessionStatus.COMPLETED,
        ),
    )
    await store.apply_invocation_lifecycle_command(final_settlement)
    final_release = _release_invocation_command_with_cleanup_authority(
        ReleaseInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            expected_run_epoch=5,
            expected_active_profile=admitted_profile,
            settlement_transition=final_settlement.transition,
        )
    )
    await store.apply_invocation_lifecycle_command(final_release)
    delayed_admission_replay = await store.apply_invocation_lifecycle_command(admission)
    assert delayed_admission_replay.replayed is True
    assert delayed_admission_replay.session == admitted.session
    assert delayed_admission_replay.active_profile == admitted.active_profile

    recovery_session_id = f"recovery-release-{suffix}-{uuid4().hex}"
    recovery_instance_id = str(uuid4())
    recovery_interaction_id = f"interaction-{uuid4().hex}"
    recovery_create = _create_command(
        session_id=recovery_session_id,
        session_instance_id=recovery_instance_id,
        interaction_id=recovery_interaction_id,
        profile=profile,
    )
    recovery_created = await store.apply_invocation_lifecycle_command(recovery_create)
    assert type(recovery_created) is InvocationMutationResult
    await store.materialize_deferred_interaction_input(
        recovery_session_id,
        interaction_id=recovery_interaction_id,
    )
    await store.update_status(recovery_session_id, SessionStatus.INTERRUPTED)
    recovery_claim_id = str(uuid4())
    claimed_at = datetime.now(UTC)

    def add_recovery_claim(_session, current):
        assert current is not None
        updated = dict(current)
        updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
            "version": 1,
            "claim_id": recovery_claim_id,
            "claimed_at": claimed_at.isoformat(),
            "claim_expires_at": (claimed_at + timedelta(minutes=5)).isoformat(),
        }
        return updated

    await store.transform_checkpoint(recovery_session_id, add_recovery_claim)
    wrong_recovery_release = _release_invocation_command_with_cleanup_authority(
        ReleaseInvocationCommand(
            session_id=recovery_session_id,
            expected_session_instance_id=recovery_instance_id,
            expected_run_epoch=1,
            expected_active_profile=recovery_created.active_profile,
            recovery_claim_id=str(uuid4()),
        )
    )
    with pytest.raises(SessionRunFenced, match="exact terminal recovery claim"):
        await store.apply_invocation_lifecycle_command(wrong_recovery_release)
    unchanged_recovery = await store.load(recovery_session_id)
    assert unchanged_recovery is not None
    assert unchanged_recovery.run_epoch == 1

    recovery_release = _release_invocation_command_with_cleanup_authority(
        ReleaseInvocationCommand(
            session_id=recovery_session_id,
            expected_session_instance_id=recovery_instance_id,
            expected_run_epoch=1,
            expected_active_profile=recovery_created.active_profile,
            recovery_claim_id=recovery_claim_id,
        )
    )
    recovery_released = await store.apply_invocation_lifecycle_command(recovery_release)
    assert type(recovery_released) is InvocationReleaseResult
    assert recovery_released.session.run_epoch == 2
    assert recovery_released.replayed is False
    recovery_release_replay = await store.apply_invocation_lifecycle_command(recovery_release)
    assert recovery_release_replay.session == recovery_released.session
    assert recovery_release_replay.replayed is True

    terminal_session_id = f"terminal-event-release-{suffix}-{uuid4().hex}"
    terminal_instance_id = str(uuid4())
    terminal_interaction_id = f"interaction-{uuid4().hex}"
    terminal_created = await store.apply_invocation_lifecycle_command(
        _create_command(
            session_id=terminal_session_id,
            session_instance_id=terminal_instance_id,
            interaction_id=terminal_interaction_id,
            profile=profile,
        )
    )
    assert type(terminal_created) is InvocationMutationResult
    await store.materialize_deferred_interaction_input(
        terminal_session_id,
        interaction_id=terminal_interaction_id,
    )
    await store.update_status(terminal_session_id, SessionStatus.FAILED)
    caller_terminal_event = Event(
        type=EventType.SESSION_FAILED,
        session_id=terminal_session_id,
        agent_name="caller",
        payload={"error": "caller-shaped terminal evidence"},
    )
    await store.append_event(terminal_session_id, caller_terminal_event)
    forged_caller_release = _release_invocation_command_with_cleanup_authority(
        ReleaseInvocationCommand(
            session_id=terminal_session_id,
            expected_session_instance_id=terminal_instance_id,
            expected_run_epoch=1,
            expected_active_profile=terminal_created.active_profile,
            terminal_session_event=event_with_runtime_envelope_authority(
                caller_terminal_event,
                "session_id",
            ),
        )
    )
    with pytest.raises(SessionRunFenced, match="exact durable terminal session evidence"):
        await store.apply_invocation_lifecycle_command(forged_caller_release)
    terminal_event = event_with_runtime_envelope_authority(
        Event(
            type=EventType.SESSION_FAILED,
            session_id=terminal_session_id,
            agent_name="assistant",
            payload={"error": "setup failed before invocation dispatch"},
        ),
        "session_id",
    )
    await store.append_event(terminal_session_id, terminal_event)
    wrong_terminal_release = _release_invocation_command_with_cleanup_authority(
        ReleaseInvocationCommand(
            session_id=terminal_session_id,
            expected_session_instance_id=terminal_instance_id,
            expected_run_epoch=1,
            expected_active_profile=terminal_created.active_profile,
            terminal_session_event=terminal_event.model_copy(
                update={"payload": {"error": "conflicting evidence"}}
            ),
        )
    )
    with pytest.raises(SessionRunFenced, match="exact terminal session evidence"):
        await store.apply_invocation_lifecycle_command(wrong_terminal_release)
    reloaded_terminal_event = next(
        event
        for event in await store.load_events(terminal_session_id)
        if event.id == terminal_event.id
    )
    if suffix != "memory":
        assert not event_envelope_authority_is_runtime_generated(
            reloaded_terminal_event,
            field_name="session_id",
            value=terminal_session_id,
        )
    terminal_release = _release_invocation_command_with_cleanup_authority(
        ReleaseInvocationCommand(
            session_id=terminal_session_id,
            expected_session_instance_id=terminal_instance_id,
            expected_run_epoch=1,
            expected_active_profile=terminal_created.active_profile,
            terminal_session_event=reloaded_terminal_event,
        )
    )
    terminal_released = await store.apply_invocation_lifecycle_command(terminal_release)
    assert terminal_released.session.run_epoch == 2
    assert terminal_released.replayed is False

    released_repair = SettleInvocationCommand(
        session_id=terminal_session_id,
        expected_session_instance_id=terminal_instance_id,
        expected_run_epoch=2,
        expected_active_profile=terminal_created.active_profile,
        expected_authority_state="released",
        transition=InteractionTransitionSpec(
            event=_transition_event(
                session_id=terminal_session_id,
                interaction_id=terminal_interaction_id,
                status=SessionStatus.FAILED,
            ),
            from_statuses=(SessionStatus.FAILED,),
            to_status=SessionStatus.FAILED,
        ),
    )
    repaired = await store.apply_invocation_lifecycle_command(released_repair)
    assert repaired.session.run_epoch == 2
    assert repaired.replayed is False
    assert (await store.apply_invocation_lifecycle_command(released_repair)).replayed is True

    stale_repair = released_repair.model_copy(
        update={
            "transition": released_repair.transition.model_copy(
                update={
                    "event": _transition_event(
                        session_id=terminal_session_id,
                        interaction_id=terminal_interaction_id,
                        status=SessionStatus.FAILED,
                    )
                }
            )
        }
    )

    def rebind_terminal_session(current_session, checkpoint):
        return checkpoint_with_active_invocation_execution_profile(
            checkpoint,
            session_id=terminal_session_id,
            interaction_id=terminal_interaction_id,
            run_epoch=current_session.run_epoch + 1,
            profile=profile,
            expected=terminal_created.active_profile,
        )

    released_repair_source = await store.load(terminal_session_id)
    assert released_repair_source is not None
    successor = await store.apply_invocation_lifecycle_command(
        prepare_rebind_invocation_command(
            released_repair_source,
            await store.load_checkpoint(terminal_session_id),
            expected_statuses={SessionStatus.FAILED},
            checkpoint_transform=rebind_terminal_session,
            target_status=SessionStatus.RUNNING,
        )
    )
    assert type(successor) is InvocationMutationResult
    await store.update_status(terminal_session_id, SessionStatus.FAILED)
    stale_terminal_release = _release_invocation_command_with_cleanup_authority(
        ReleaseInvocationCommand(
            session_id=terminal_session_id,
            expected_session_instance_id=terminal_instance_id,
            expected_run_epoch=successor.active_profile.run_epoch,
            expected_active_profile=successor.active_profile,
            terminal_session_event=terminal_event,
        )
    )
    with pytest.raises(SessionRunFenced, match="another invocation authority"):
        await store.apply_invocation_lifecycle_command(stale_terminal_release)
    successor_terminal_event = event_with_runtime_envelope_authority(
        Event(
            type=EventType.SESSION_FAILED,
            session_id=terminal_session_id,
            agent_name="assistant",
            payload={"error": "successor failed"},
        ),
        "session_id",
    )
    await store.append_event(terminal_session_id, successor_terminal_event)
    await store.apply_invocation_lifecycle_command(
        _release_invocation_command_with_cleanup_authority(
            ReleaseInvocationCommand(
                session_id=terminal_session_id,
                expected_session_instance_id=terminal_instance_id,
                expected_run_epoch=successor.active_profile.run_epoch,
                expected_active_profile=successor.active_profile,
                terminal_session_event=successor_terminal_event,
            )
        )
    )
    with pytest.raises(SessionRunFenced, match="run epoch"):
        await store.apply_invocation_lifecycle_command(stale_repair)
    assert all(
        event.id != stale_repair.transition.event.id
        for event in await store.load_events(terminal_session_id)
    )

    multi_session_id = f"multi-interaction-settlement-{suffix}-{uuid4().hex}"
    multi_instance_id = str(uuid4())
    admission_interaction_id = f"interaction-{uuid4().hex}"
    terminal_interaction_id = f"interaction-{uuid4().hex}"
    multi_created = await store.apply_invocation_lifecycle_command(
        _create_command(
            session_id=multi_session_id,
            session_instance_id=multi_instance_id,
            interaction_id=admission_interaction_id,
            profile=profile,
        )
    )
    assert type(multi_created) is InvocationMutationResult
    await store.materialize_deferred_interaction_input(
        multi_session_id,
        interaction_id=admission_interaction_id,
    )
    await store.append_event(
        multi_session_id,
        Event(
            type=EventType.INTERACTION_STARTED,
            session_id=multi_session_id,
            interaction_id=terminal_interaction_id,
            agent_name="assistant",
        ),
    )
    multi_settlement = SettleInvocationCommand(
        session_id=multi_session_id,
        expected_session_instance_id=multi_instance_id,
        expected_run_epoch=1,
        expected_active_profile=multi_created.active_profile,
        transition=InteractionTransitionSpec(
            event=_transition_event(
                session_id=multi_session_id,
                interaction_id=terminal_interaction_id,
                status=SessionStatus.COMPLETED,
            ),
            from_statuses=(SessionStatus.RUNNING,),
            to_status=SessionStatus.COMPLETED,
        ),
    )
    await store.apply_invocation_lifecycle_command(multi_settlement)
    loaded_multi_settlement = await store.load_invocation_settlement_transition(
        multi_session_id,
        expected_session_instance_id=multi_instance_id,
        expected_active_invocation_profile=multi_created.active_profile,
    )
    assert loaded_multi_settlement == multi_settlement.transition
    multi_release = _release_invocation_command_with_cleanup_authority(
        ReleaseInvocationCommand(
            session_id=multi_session_id,
            expected_session_instance_id=multi_instance_id,
            expected_run_epoch=1,
            expected_active_profile=multi_created.active_profile,
            settlement_transition=multi_settlement.transition,
        )
    )
    released_multi = await store.apply_invocation_lifecycle_command(multi_release)
    assert type(released_multi) is InvocationReleaseResult
    assert released_multi.session.run_epoch == 2
    assert (await store.apply_invocation_lifecycle_command(multi_settlement)).replayed is True

    missing_profile_session_id = f"missing-profile-admission-{suffix}-{uuid4().hex}"
    missing_profile_instance_id = str(uuid4())
    missing_profile_created = await store.apply_invocation_lifecycle_command(
        _create_command(
            session_id=missing_profile_session_id,
            session_instance_id=missing_profile_instance_id,
            interaction_id=f"interaction-{uuid4().hex}",
            profile=profile,
        )
    )
    assert type(missing_profile_created) is InvocationMutationResult
    await store.update_status(missing_profile_session_id, SessionStatus.COMPLETED)

    def remove_active_profile(_session, current):
        assert current is not None
        updated = dict(current)
        updated.pop(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY)
        return updated

    with sessions_module._invocation_lifecycle_authority_mutation_scope():
        await store.transform_checkpoint(missing_profile_session_id, remove_active_profile)
    corrupted_checkpoint = await store.load_checkpoint(missing_profile_session_id)
    assert corrupted_checkpoint is not None
    assert INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY in corrupted_checkpoint
    assert ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY not in corrupted_checkpoint
    missing_profile_interaction_id = f"interaction-{uuid4().hex}"
    with pytest.raises(
        SessionRunFenced,
        match="lacks exact predecessor release authority",
    ):
        await store.apply_invocation_lifecycle_command(
            AdmitInvocationCommand(
                session_id=missing_profile_session_id,
                expected_session_instance_id=missing_profile_instance_id,
                expected_statuses=(SessionStatus.COMPLETED,),
                expected_run_epoch=missing_profile_created.session.run_epoch,
                expected_checkpoint_sha256=invocation_checkpoint_state_sha256(corrupted_checkpoint),
                target_active_profile=ActiveInvocationExecutionProfile(
                    session_id=missing_profile_session_id,
                    interaction_id=missing_profile_interaction_id,
                    run_epoch=missing_profile_created.session.run_epoch + 1,
                    profile=profile,
                ),
                interaction_started_event=Event(
                    type=EventType.INTERACTION_STARTED,
                    session_id=missing_profile_session_id,
                    interaction_id=missing_profile_interaction_id,
                    agent_name="assistant",
                ),
                interaction_source_messages=(),
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
            )
        )
    missing_profile_after = await store.load(missing_profile_session_id)
    assert missing_profile_after is not None
    assert missing_profile_after.status is SessionStatus.COMPLETED
    assert missing_profile_after.run_epoch == missing_profile_created.session.run_epoch

    unreceipted_release_session_id = f"unreceipted-release-{suffix}-{uuid4().hex}"
    unreceipted_release_instance_id = str(uuid4())
    unreceipted_release_created = await store.apply_invocation_lifecycle_command(
        _create_command(
            session_id=unreceipted_release_session_id,
            session_instance_id=unreceipted_release_instance_id,
            interaction_id=f"interaction-{uuid4().hex}",
            profile=profile,
        )
    )
    assert type(unreceipted_release_created) is InvocationMutationResult
    await store.update_status(unreceipted_release_session_id, SessionStatus.COMPLETED)
    await store.release_run_fence(unreceipted_release_session_id)
    unreceipted_release_session = await store.load(unreceipted_release_session_id)
    unreceipted_release_checkpoint = await store.load_checkpoint(unreceipted_release_session_id)
    assert unreceipted_release_session is not None
    assert unreceipted_release_session.run_epoch == 2
    unreceipted_interaction_id = f"interaction-{uuid4().hex}"
    with pytest.raises(
        SessionRunFenced,
        match="lacks exact durable released-profile authority",
    ):
        await store.apply_invocation_lifecycle_command(
            AdmitInvocationCommand(
                session_id=unreceipted_release_session_id,
                expected_session_instance_id=unreceipted_release_instance_id,
                expected_statuses=(SessionStatus.COMPLETED,),
                expected_run_epoch=unreceipted_release_session.run_epoch,
                expected_checkpoint_sha256=invocation_checkpoint_state_sha256(
                    unreceipted_release_checkpoint
                ),
                target_active_profile=ActiveInvocationExecutionProfile(
                    session_id=unreceipted_release_session_id,
                    interaction_id=unreceipted_interaction_id,
                    run_epoch=unreceipted_release_session.run_epoch + 1,
                    profile=profile,
                ),
                interaction_started_event=Event(
                    type=EventType.INTERACTION_STARTED,
                    session_id=unreceipted_release_session_id,
                    interaction_id=unreceipted_interaction_id,
                    agent_name="assistant",
                ),
                interaction_source_messages=(),
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
                expected_active_profile=unreceipted_release_created.active_profile,
            )
        )
    unreceipted_release_after = await store.load(unreceipted_release_session_id)
    assert unreceipted_release_after == unreceipted_release_session
    assert (
        await store.load_checkpoint(unreceipted_release_session_id)
        == unreceipted_release_checkpoint
    )

    direct_session_id = f"direct-admission-{suffix}-{uuid4().hex}"
    direct_instance_id = str(uuid4())
    direct_session = await store.create(
        run_request_with_runtime_session_instance_authority(
            RunRequest(
                agent_name="assistant",
                session_id=direct_session_id,
                messages=[],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
            ),
            session_instance_id=direct_instance_id,
        ),
        identity=SessionIdentity(
            provider_name="fake",
            model="fake-model",
            runtime_name="cayu",
            runtime_version="test",
            execution_profile=profile,
        ),
        result_checkpoint_transform=lambda _session, checkpoint: (
            checkpoint or {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
        ),
    )
    assert direct_session.status is SessionStatus.PENDING
    direct_checkpoint = await store.load_checkpoint(direct_session_id)
    direct_interaction_id = f"interaction-{uuid4().hex}"
    with pytest.raises(SessionRunFenced, match="typed lifecycle command boundary"):
        await store.admit_session_invocation(
            direct_session_id,
            admission=SessionInvocationAdmission(
                from_statuses=frozenset({SessionStatus.PENDING}),
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                execution_profile=profile,
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
                interaction_started_event=Event(
                    type=EventType.INTERACTION_STARTED,
                    session_id=direct_session_id,
                    interaction_id=direct_interaction_id,
                    agent_name="assistant",
                ),
                interaction_source_messages=(),
            ),
        )
    direct_after = await store.load(direct_session_id)
    assert direct_after == direct_session
    assert await store.load_checkpoint(direct_session_id) == direct_checkpoint
    assert await store.load_events(direct_session_id) == []


async def _close_store(store) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


class _UnknownInvocationCommandVersionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 2


class _OverriddenReleaseWithoutProofStore(InMemorySessionStore):
    async def release_session_invocation(self, command):
        raise AssertionError("The unauthenticated override must not receive a command.")


class _OverriddenAdmissionDependencyWithoutProofStore(InMemorySessionStore):
    async def transition_status_and_checkpoint(self, *args, **kwargs):
        raise AssertionError("The unauthenticated override must not receive a command.")


class _OverriddenRejectionWrapperWithoutProofStore(InMemorySessionStore):
    async def reject_active_invocation_execution_profile(self, *args, **kwargs):
        raise AssertionError("The unauthenticated override must not receive a command.")


class _OverriddenTransformWithoutProofStore(InMemorySessionStore):
    async def transform_checkpoint(self, *args, **kwargs):
        raise AssertionError("The unauthenticated override must not receive a callback.")


class _InheritedInvocationCommandStore(InMemorySessionStore):
    pass


class _ExplicitInheritedInvocationCommandStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1


class _LiteralCheckpointTransformStore(InMemorySessionStore):
    """Model a custom store that gives callbacks its literal durable object."""

    invocation_lifecycle_command_version = 1

    async def transform_checkpoint(self, session_id, checkpoint_transform) -> None:
        async with self._lock:
            session = self._sessions[session_id]
            current = copy.deepcopy(self._checkpoints.get(session_id))
            transformed = checkpoint_transform(session.model_copy(deep=True), current)
            if transformed is not None:
                self._checkpoints[session_id] = copy.deepcopy(transformed)


class _CommitThenRaiseCreateStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.raise_after_commit = True

    async def create(self, *args, **kwargs):
        session = await super().create(*args, **kwargs)
        if self.raise_after_commit:
            self.raise_after_commit = False
            raise RuntimeError("create acknowledgement lost")
        return session


class _CancelAfterCommitCreateStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.committed = asyncio.Event()
        self.release = asyncio.Event()

    async def create(self, *args, **kwargs):
        session = await super().create(*args, **kwargs)
        self.committed.set()
        await self.release.wait()
        return session


class _FailCreateResultPreparationMixin:
    invocation_lifecycle_command_version = 1
    fail_create_result_preparation = True

    async def create(self, *args, **kwargs):
        result_transform = kwargs.get("result_checkpoint_transform")
        assert result_transform is not None

        def fail_after_receipt_preparation(session, checkpoint):
            transformed = result_transform(session, checkpoint)
            if self.fail_create_result_preparation:
                self.fail_create_result_preparation = False
                raise OSError("create receipt preparation failed")
            return transformed

        kwargs["result_checkpoint_transform"] = fail_after_receipt_preparation
        return await super().create(*args, **kwargs)


class _FailCreateResultPreparationMemoryStore(
    _FailCreateResultPreparationMixin,
    InMemorySessionStore,
):
    invocation_lifecycle_command_version = 1


class _FailCreateResultPreparationSQLiteStore(
    _FailCreateResultPreparationMixin,
    SQLiteSessionStore,
):
    invocation_lifecycle_command_version = 1


class _MissingProfileAfterRecoveryClaimStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.corrupted_after_claim_renewal = False
        self.generic_run_fence_release_calls = 0

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform,
    ) -> None:
        await super().transform_checkpoint(session_id, checkpoint_transform)
        checkpoint = await InMemorySessionStore.load_checkpoint(self, session_id)
        claim = (
            None
            if checkpoint is None
            else checkpoint.get(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY)
        )
        if (
            self.corrupted_after_claim_renewal
            or type(claim) is not dict
            or "renewed_at" not in claim
        ):
            return

        def remove_active_profile(_session, current):
            assert current is not None
            updated = dict(current)
            updated.pop(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY)
            return updated

        self.corrupted_after_claim_renewal = True
        with sessions_module._invocation_lifecycle_authority_mutation_scope():
            await InMemorySessionStore.transform_checkpoint(
                self,
                session_id,
                remove_active_profile,
            )

    async def release_run_fence(self, session_id: str) -> None:
        self.generic_run_fence_release_calls += 1
        await super().release_run_fence(session_id)


async def _assert_create_result_preparation_failure_is_atomic(store, suffix: str) -> None:
    command = _create_command(
        session_id=f"create-preparation-failure-{suffix}-{uuid4().hex}",
        session_instance_id=str(uuid4()),
        interaction_id=f"interaction-{uuid4().hex}",
        profile=_profile(),
    )
    with pytest.raises(OSError, match="create receipt preparation failed"):
        await store.apply_invocation_lifecycle_command(command)
    assert await store.load(command.session_id) is None

    created = await store.apply_invocation_lifecycle_command(command)
    assert type(created) is InvocationMutationResult
    assert created.replayed is False
    replayed = await store.apply_invocation_lifecycle_command(command)
    assert replayed.replayed is True
    assert replayed.session == created.session


async def _assert_settlement_receipt_preparation_failure_is_atomic(
    store,
    suffix: str,
    *,
    monkeypatch,
    receipt_module,
) -> None:
    command = _create_command(
        session_id=f"settlement-preparation-failure-{suffix}-{uuid4().hex}",
        session_instance_id=str(uuid4()),
        interaction_id=f"interaction-{uuid4().hex}",
        profile=_profile(),
    )
    created = await store.apply_invocation_lifecycle_command(command)
    assert type(created) is InvocationMutationResult
    await store.materialize_deferred_interaction_input(
        command.session_id,
        interaction_id=command.active_profile.interaction_id,
    )
    settlement = SettleInvocationCommand(
        session_id=command.session_id,
        expected_session_instance_id=command.expected_session_instance_id,
        expected_run_epoch=1,
        expected_active_profile=created.active_profile,
        transition=InteractionTransitionSpec(
            event=_transition_event(
                session_id=command.session_id,
                interaction_id=command.active_profile.interaction_id,
                status=SessionStatus.COMPLETED,
            ),
            from_statuses=(SessionStatus.RUNNING,),
            to_status=SessionStatus.COMPLETED,
        ),
    )
    session_before = await store.load(command.session_id)
    checkpoint_before = await store.load_checkpoint(command.session_id)
    events_before = await store.load_events(command.session_id)

    def fail_receipt_preparation(*args, **kwargs):
        del args, kwargs
        raise OSError("settlement receipt preparation failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            receipt_module,
            "_interaction_transition_receipt_record",
            fail_receipt_preparation,
        )
        with pytest.raises(OSError, match="settlement receipt preparation failed"):
            await store.apply_invocation_lifecycle_command(settlement)

    assert await store.load(command.session_id) == session_before
    assert await store.load_checkpoint(command.session_id) == checkpoint_before
    assert await store.load_events(command.session_id) == events_before
    settled = await store.apply_invocation_lifecycle_command(settlement)
    assert settled.session.status is SessionStatus.COMPLETED
    assert settled.replayed is False


async def _assert_generic_checkpoint_replacements_preserve_authority(
    raw_store,
    suffix: str,
) -> None:
    session_id = f"generic-checkpoint-schema-{suffix}-{uuid4().hex}"
    created = await raw_store.apply_invocation_lifecycle_command(
        _create_command(
            session_id=session_id,
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
    )
    assert type(created) is InvocationMutationResult
    original = await raw_store.load_checkpoint(session_id)
    assert original is not None
    original_receipts = copy.deepcopy(original[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY])
    session_before_snapshot = await raw_store.load(session_id)
    assert session_before_snapshot is not None
    runtime_store = runtime_checkpoint_session_store(raw_store)
    snapshot_session, snapshot_checkpoint = await load_runtime_session_checkpoint_snapshot(
        runtime_store,
        session_id,
    )
    assert snapshot_session == session_before_snapshot
    assert snapshot_checkpoint == original
    assert await raw_store.load(session_id) == session_before_snapshot
    assert await raw_store.load_checkpoint(session_id) == original

    with pytest.raises(
        CheckpointCompatibilityError,
        match="uses a newer root checkpoint schema",
    ):
        await raw_store.checkpoint(
            session_id,
            {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
                "future": True,
            },
        )
    assert await raw_store.load_checkpoint(session_id) == original

    with pytest.raises(
        CheckpointCompatibilityError,
        match="uses a newer root checkpoint schema",
    ):
        await raw_store.transform_checkpoint(
            session_id,
            lambda _session, _checkpoint: {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
                "future-transform": True,
            },
        )
    assert await raw_store.load_checkpoint(session_id) == original

    await raw_store.checkpoint(
        session_id,
        {
            "ordinary-versionless": suffix,
            ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY: {"caller": "forged"},
            INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY: {"caller": "forged"},
        },
    )
    versionless = await raw_store.load_checkpoint(session_id)
    assert versionless is not None
    assert versionless[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    assert versionless["ordinary-versionless"] == suffix
    assert active_invocation_execution_profile_from_checkpoint(versionless) == (
        created.active_profile
    )
    assert versionless[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] == original_receipts

    await raw_store.checkpoint(
        session_id,
        {
            CHECKPOINT_SCHEMA_VERSION_KEY: 3,
            "ordinary-v3": suffix,
            "workspace_observations": {"caller": "forged"},
        },
    )
    migrated = await raw_store.load_checkpoint(session_id)
    assert migrated is not None
    assert migrated[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    assert migrated["ordinary-v3"] == suffix
    assert "workspace_observations" not in migrated
    assert active_invocation_execution_profile_from_checkpoint(migrated) == created.active_profile
    assert migrated[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] == original_receipts

    future_checkpoint = copy.deepcopy(migrated)
    future_checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
    future_checkpoint["future-private-root"] = {"backend": suffix}

    def install_future_checkpoint(_session, _current):
        return future_checkpoint

    with sessions_module._invocation_lifecycle_authority_mutation_scope():
        await raw_store.transform_checkpoint(session_id, install_future_checkpoint)

    for callback_result in (None, {"ordinary": "replacement"}):
        callback_calls = 0

        def inspect_future_checkpoint(
            _session,
            _checkpoint,
            result=callback_result,
        ):
            nonlocal callback_calls
            callback_calls += 1
            return result

        with pytest.raises(
            CheckpointCompatibilityError,
            match="uses a newer root checkpoint schema",
        ):
            await raw_store.transform_checkpoint(session_id, inspect_future_checkpoint)
        assert callback_calls == 0
        assert await raw_store.load_checkpoint(session_id) == future_checkpoint


def test_lifecycle_receipt_metadata_trust_fails_closed_per_receipt() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        command = _create_command(
            session_id=f"receipt-redaction-{uuid4().hex}",
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
        try:
            await store.apply_invocation_lifecycle_command(command)
            checkpoint = await store.load_checkpoint(command.session_id)
            assert checkpoint is not None
            raw_ledger = checkpoint[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
            assert type(raw_ledger) is dict
            raw_receipt = raw_ledger["receipts"][0]

            authenticated_identity = raw_receipt["command_identity"]
            assert not durable_value_contains_secret(
                checkpoint,
                redactor=SecretRedactor(authenticated_identity),
            )
            malformed_identity_ledger = copy.deepcopy(checkpoint)
            malformed_identity_ledger[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY][
                "record_sha256"
            ] = ""
            assert durable_value_contains_secret(
                malformed_identity_ledger,
                redactor=SecretRedactor(authenticated_identity),
            )

            malformed_provider_ledger = copy.deepcopy(checkpoint)
            malformed_provider_receipt = malformed_provider_ledger[
                INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY
            ]["receipts"][0]
            malformed_provider_receipt["result_session"]["provider_name"] = (
                "malformed-provider-secret"
            )
            assert durable_value_contains_secret(
                malformed_provider_ledger,
                redactor=SecretRedactor("malformed-provider-secret"),
            )

            def receipt(identity: str, *, malformed: bool) -> dict[str, Any]:
                material = copy.deepcopy(raw_receipt)
                material["command_identity"] = identity
                material["record_sha256"] = ""
                material = _InvocationLifecycleCommandReceipt.model_validate(material).model_dump(
                    mode="json"
                )
                if malformed:
                    profile_record = material["result_session"]["metadata"][
                        EXECUTION_PROFILE_METADATA_KEY
                    ]
                    profile_record["caller_collision"] = False
                return material

            layouts = (
                (receipt("a-valid", malformed=False), receipt("b-malformed", malformed=True)),
                (receipt("a-malformed", malformed=True), receipt("b-valid", malformed=False)),
                (receipt("a-malformed-only", malformed=True),),
            )
            for receipts in layouts:
                candidate = copy.deepcopy(checkpoint)
                candidate_ledger = copy.deepcopy(raw_ledger)
                candidate_ledger["receipts"] = list(receipts)
                candidate[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = candidate_ledger
                assert durable_value_contains_secret(
                    candidate,
                    redactor=SecretRedactor("direct_tools"),
                )

            incomplete_ledgers = []
            for field_name in (
                "record_type",
                "schema_version",
                "release_capacity_command_identity",
                "record_sha256",
            ):
                incomplete = copy.deepcopy(raw_ledger)
                incomplete.pop(field_name)
                incomplete_ledgers.append(incomplete)
            blank_ledger_digest = copy.deepcopy(raw_ledger)
            blank_ledger_digest["record_sha256"] = ""
            incomplete_ledgers.append(blank_ledger_digest)
            for field_name in ("record_type", "schema_version", "record_sha256"):
                incomplete = copy.deepcopy(raw_ledger)
                incomplete["receipts"][0].pop(field_name)
                incomplete_ledgers.append(incomplete)
            blank_receipt_digest = copy.deepcopy(raw_ledger)
            blank_receipt_digest["receipts"][0]["record_sha256"] = ""
            incomplete_ledgers.append(blank_receipt_digest)

            for raw_incomplete_ledger in incomplete_ledgers:
                candidate = copy.deepcopy(checkpoint)
                candidate[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = raw_incomplete_ledger
                assert durable_value_contains_secret(
                    candidate,
                    redactor=SecretRedactor("direct_tools"),
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_lifecycle_receipt_replay_rejects_digest_valid_forged_result() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        command = _create_command(
            session_id=f"forged-result-replay-{uuid4().hex}",
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
        try:
            await store.apply_invocation_lifecycle_command(command)
            checkpoint = await store.load_checkpoint(command.session_id)
            assert checkpoint is not None
            raw_ledger = checkpoint[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
            raw_receipt = copy.deepcopy(raw_ledger["receipts"][0])

            forged_status = copy.deepcopy(raw_receipt)
            forged_status["result_session"]["status"] = "completed"
            forged_status["record_sha256"] = ""
            with pytest.raises(ValueError, match="must return a running session"):
                _InvocationLifecycleCommandReceipt.model_validate(forged_status)

            forged_profile = copy.deepcopy(raw_receipt)
            forged_profile["result_session"]["metadata"][EXECUTION_PROFILE_METADATA_KEY][
                "expected"
            ] = _profile(tool_name="forged_tool").model_dump(mode="json")
            forged_profile["record_sha256"] = ""
            with pytest.raises(ValueError, match="conflicts with its active profile"):
                _InvocationLifecycleCommandReceipt.model_validate(forged_profile)

            forged_ceiling = copy.deepcopy(raw_receipt)
            forged_ceiling["result_session"]["metadata"][TOOL_CAPABILITY_CEILING_METADATA_KEY] = (
                ToolCapabilityCeiling(tool_names=("forged_tool",)).model_dump(mode="json")
            )
            forged_ceiling["record_sha256"] = ""
            with pytest.raises(ValueError, match="conflicts with its tool ceiling"):
                _InvocationLifecycleCommandReceipt.model_validate(forged_ceiling)

            raw_receipt["result_session"]["invocation"]["root_invocation_id"] = str(uuid4())
            raw_receipt["record_sha256"] = ""
            forged_receipt = _InvocationLifecycleCommandReceipt.model_validate(raw_receipt)
            forged_ledger = _InvocationLifecycleReceiptLedger(
                receipts=(forged_receipt,),
                release_capacity_command_identity=forged_receipt.command_identity,
            )

            def install_forged_receipt(_session, current):
                assert current is not None
                updated = copy.deepcopy(current)
                updated[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = forged_ledger.model_dump(
                    mode="json"
                )
                return updated

            with sessions_module._invocation_lifecycle_authority_mutation_scope():
                await store.transform_checkpoint(
                    command.session_id,
                    install_forged_receipt,
                )
            with pytest.raises(RuntimeError, match="forged session authority"):
                await store.apply_invocation_lifecycle_command(command)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_lifecycle_receipt_ledger_rolls_oldest_epoch_before_item_limit(monkeypatch) -> None:
    async def run() -> None:
        assert INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS == 128
        test_limit = 2
        monkeypatch.setattr(
            invocation_lifecycle_module,
            "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS",
            test_limit,
        )
        store = InMemorySessionStore()
        command = _create_command(
            session_id=f"receipt-item-limit-{uuid4().hex}",
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
        try:
            created = await store.apply_invocation_lifecycle_command(command)
            checkpoint = await store.load_checkpoint(command.session_id)
            assert checkpoint is not None
            session_before = await store.load(command.session_id)
            checkpoint_before = await store.load_checkpoint(command.session_id)
            assert session_before is not None
            assert checkpoint_before is not None

            def rebind_checkpoint(current_session, current):
                return checkpoint_with_active_invocation_execution_profile(
                    current,
                    session_id=current_session.id,
                    interaction_id=created.active_profile.interaction_id,
                    run_epoch=current_session.run_epoch + 1,
                    profile=created.active_profile.profile,
                    expected=created.active_profile,
                )

            rebind = prepare_rebind_invocation_command(
                session_before,
                checkpoint_before,
                expected_statuses={SessionStatus.RUNNING},
                checkpoint_transform=rebind_checkpoint,
            )
            rebound = await store.apply_invocation_lifecycle_command(rebind)
            assert rebound.session.run_epoch == rebind.target_active_profile.run_epoch
            compacted_checkpoint = await store.load_checkpoint(command.session_id)
            assert compacted_checkpoint is not None
            compacted_ledger = _InvocationLifecycleReceiptLedger.model_validate(
                compacted_checkpoint[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
            )
            rebind_identity = (
                f"rebind:{command.session_id}:{command.expected_session_instance_id}:"
                f"{rebind.target_active_profile.run_epoch}"
            )
            assert tuple(item.command_identity for item in compacted_ledger.receipts) == (
                rebind_identity,
            )
            assert compacted_ledger.release_capacity_command_identity == rebind_identity
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_lifecycle_receipt_ledger_reserves_release_capacity_for_rebind(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setattr(
            invocation_lifecycle_module,
            "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS",
            4,
        )
        store = InMemorySessionStore()
        command = _create_command(
            session_id=f"receipt-release-reserve-{uuid4().hex}",
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
        try:
            created = await store.apply_invocation_lifecycle_command(command)
            checkpoint = await store.load_checkpoint(command.session_id)
            assert checkpoint is not None
            active_ledger = _InvocationLifecycleReceiptLedger.model_validate(
                checkpoint[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
            )
            assert active_ledger.release_capacity_command_identity == (
                active_ledger.receipts[0].command_identity
            )

            await store.materialize_deferred_interaction_input(
                command.session_id,
                interaction_id=command.active_profile.interaction_id,
            )
            first_settlement = SettleInvocationCommand(
                session_id=command.session_id,
                expected_session_instance_id=command.expected_session_instance_id,
                expected_run_epoch=1,
                expected_active_profile=created.active_profile,
                transition=InteractionTransitionSpec(
                    event=_transition_event(
                        session_id=command.session_id,
                        interaction_id=command.active_profile.interaction_id,
                        status=SessionStatus.INTERRUPTED,
                    ),
                    from_statuses=(SessionStatus.RUNNING,),
                    to_status=SessionStatus.INTERRUPTED,
                ),
            )
            await store.apply_invocation_lifecycle_command(first_settlement)
            first_release = _release_invocation_command_with_cleanup_authority(
                ReleaseInvocationCommand(
                    session_id=command.session_id,
                    expected_session_instance_id=command.expected_session_instance_id,
                    expected_run_epoch=1,
                    expected_active_profile=created.active_profile,
                    settlement_transition=first_settlement.transition,
                )
            )
            released = await store.apply_invocation_lifecycle_command(first_release)
            checkpoint = await store.load_checkpoint(command.session_id)
            assert checkpoint is not None
            released_ledger = _InvocationLifecycleReceiptLedger.model_validate(
                checkpoint[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
            )
            assert len(released_ledger.receipts) == 2
            assert released_ledger.release_capacity_command_identity is None

            def rebind_checkpoint(current_session, current):
                return checkpoint_with_active_invocation_execution_profile(
                    current,
                    session_id=current_session.id,
                    interaction_id=created.active_profile.interaction_id,
                    run_epoch=current_session.run_epoch + 1,
                    profile=created.active_profile.profile,
                    expected=created.active_profile,
                )

            rebind = prepare_rebind_invocation_command(
                released.session,
                checkpoint,
                expected_statuses={SessionStatus.INTERRUPTED},
                checkpoint_transform=rebind_checkpoint,
            )
            rebound = await store.apply_invocation_lifecycle_command(rebind)
            checkpoint = await store.load_checkpoint(command.session_id)
            assert checkpoint is not None
            rebound_ledger = _InvocationLifecycleReceiptLedger.model_validate(
                checkpoint[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
            )
            assert len(rebound_ledger.receipts) == 3
            assert rebound_ledger.release_capacity_command_identity == (
                f"rebind:{command.session_id}:{command.expected_session_instance_id}:3"
            )

            second_settlement = SettleInvocationCommand(
                session_id=command.session_id,
                expected_session_instance_id=command.expected_session_instance_id,
                expected_run_epoch=3,
                expected_active_profile=rebound.active_profile,
                transition=InteractionTransitionSpec(
                    event=_transition_event(
                        session_id=command.session_id,
                        interaction_id=command.active_profile.interaction_id,
                        status=SessionStatus.COMPLETED,
                    ),
                    from_statuses=(SessionStatus.INTERRUPTED,),
                    to_status=SessionStatus.COMPLETED,
                ),
            )
            await store.apply_invocation_lifecycle_command(second_settlement)
            second_release = _release_invocation_command_with_cleanup_authority(
                ReleaseInvocationCommand(
                    session_id=command.session_id,
                    expected_session_instance_id=command.expected_session_instance_id,
                    expected_run_epoch=3,
                    expected_active_profile=rebound.active_profile,
                    settlement_transition=second_settlement.transition,
                )
            )
            final_release = await store.apply_invocation_lifecycle_command(second_release)
            assert final_release.session.run_epoch == 4
            final_checkpoint = await store.load_checkpoint(command.session_id)
            assert final_checkpoint is not None
            final_ledger = _InvocationLifecycleReceiptLedger.model_validate(
                final_checkpoint[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
            )
            assert len(final_ledger.receipts) == 4
            assert final_ledger.release_capacity_command_identity is None
        finally:
            await _close_store(store)

    asyncio.run(run())


async def _assert_active_session_metadata_cannot_invalidate_release_capacity(
    store,
    suffix: str,
    monkeypatch,
) -> None:
    command = _create_command(
        session_id=f"receipt-metadata-reserve-{suffix}-{uuid4().hex}",
        session_instance_id=str(uuid4()),
        interaction_id=f"interaction-{uuid4().hex}",
        profile=_profile(),
    )
    created = await store.apply_invocation_lifecycle_command(command)
    before = await store.load(command.session_id)
    assert before is not None
    monkeypatch.setattr(
        invocation_lifecycle_module,
        "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES",
        256_000,
    )
    with pytest.raises(ValueError, match="encoded JSON byte limit"):
        await store.update_metadata(
            command.session_id,
            {"oversized-for-release": "x" * 300_000},
        )
    assert await store.load(command.session_id) == before

    await store.materialize_deferred_interaction_input(
        command.session_id,
        interaction_id=command.active_profile.interaction_id,
    )
    settlement = SettleInvocationCommand(
        session_id=command.session_id,
        expected_session_instance_id=command.expected_session_instance_id,
        expected_run_epoch=1,
        expected_active_profile=created.active_profile,
        transition=InteractionTransitionSpec(
            event=_transition_event(
                session_id=command.session_id,
                interaction_id=command.active_profile.interaction_id,
                status=SessionStatus.COMPLETED,
            ),
            from_statuses=(SessionStatus.RUNNING,),
            to_status=SessionStatus.COMPLETED,
        ),
    )
    await store.apply_invocation_lifecycle_command(settlement)
    released = await store.apply_invocation_lifecycle_command(
        _release_invocation_command_with_cleanup_authority(
            ReleaseInvocationCommand(
                session_id=command.session_id,
                expected_session_instance_id=command.expected_session_instance_id,
                expected_run_epoch=1,
                expected_active_profile=created.active_profile,
                settlement_transition=settlement.transition,
            )
        )
    )
    assert released.session.run_epoch == 2


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_active_session_metadata_cannot_invalidate_release_capacity(
    store_kind: str,
    tmp_path,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "release-capacity-metadata.sqlite")
        )
        try:
            await _assert_active_session_metadata_cannot_invalidate_release_capacity(
                store,
                store_kind,
                monkeypatch,
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_lifecycle_receipt_ledger_enforces_exact_encoded_byte_limit(monkeypatch) -> None:
    command = _create_command(
        session_id=f"receipt-byte-limit-{uuid4().hex}",
        session_instance_id=str(uuid4()),
        interaction_id=f"interaction-{uuid4().hex}",
        profile=_profile(),
    )
    store = InMemorySessionStore()

    async def create_receipt_material() -> dict[str, Any]:
        try:
            await store.apply_invocation_lifecycle_command(command)
            checkpoint = await store.load_checkpoint(command.session_id)
            assert checkpoint is not None
            return copy.deepcopy(
                checkpoint[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]["receipts"][0]
            )
        finally:
            await _close_store(store)

    raw_receipt = asyncio.run(create_receipt_material())

    def ledger_with_padding(size: int) -> _InvocationLifecycleReceiptLedger:
        material = copy.deepcopy(raw_receipt)
        material["result_session"]["metadata"]["bounded_padding"] = "x" * size
        material["record_sha256"] = ""
        receipt = _InvocationLifecycleCommandReceipt.model_validate(material)
        return _InvocationLifecycleReceiptLedger(
            receipts=(receipt,),
            release_capacity_command_identity=receipt.command_identity,
        )

    def projected_size(ledger: _InvocationLifecycleReceiptLedger) -> int:
        receipt = ledger.receipts[0]
        projected_release = _projected_invocation_release_receipt(
            receipt,
            result_session=receipt.result_session,
        )
        projected_receipts = sorted(
            (*ledger.receipts, projected_release),
            key=lambda item: item.command_identity,
        )
        return len(
            canonical_durable_json_bytes(
                {
                    "record_type": ledger.record_type,
                    "schema_version": ledger.schema_version,
                    "receipts": [item.model_dump(mode="json") for item in projected_receipts],
                    "release_capacity_command_identity": None,
                    "record_sha256": "f" * 64,
                },
                "invocation lifecycle receipt ledger release capacity",
            )
        )

    empty = ledger_with_padding(0)
    empty_size = projected_size(empty)
    test_limit = empty_size + 4096
    monkeypatch.setattr(
        invocation_lifecycle_module,
        "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES",
        test_limit,
    )
    exact = ledger_with_padding((test_limit - empty_size) // 2)
    assert projected_size(exact) == test_limit
    with pytest.raises(ValueError, match="encoded JSON byte limit"):
        ledger_with_padding((test_limit - empty_size) // 2 + 1)


async def _assert_v4_active_profile_migration_fails_closed(raw_store, suffix: str) -> None:
    store = runtime_checkpoint_session_store(raw_store)
    profile = _profile()
    session_id = f"v4-ambiguous-admission-{suffix}-{uuid4().hex}"
    session_instance_id = str(uuid4())
    interaction_id = f"interaction-{uuid4().hex}"
    create = _create_command(
        session_id=session_id,
        session_instance_id=session_instance_id,
        interaction_id=interaction_id,
        profile=profile,
    )
    created = await store.apply_invocation_lifecycle_command(create)
    assert type(created) is InvocationMutationResult
    await store.materialize_deferred_interaction_input(
        session_id,
        interaction_id=interaction_id,
    )
    settlement = SettleInvocationCommand(
        session_id=session_id,
        expected_session_instance_id=session_instance_id,
        expected_run_epoch=1,
        expected_active_profile=created.active_profile,
        transition=InteractionTransitionSpec(
            event=_transition_event(
                session_id=session_id,
                interaction_id=interaction_id,
                status=SessionStatus.INTERRUPTED,
            ),
            from_statuses=(SessionStatus.RUNNING,),
            to_status=SessionStatus.INTERRUPTED,
        ),
    )
    await store.apply_invocation_lifecycle_command(settlement)
    release = _release_invocation_command_with_cleanup_authority(
        ReleaseInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            expected_run_epoch=1,
            expected_active_profile=created.active_profile,
            settlement_transition=settlement.transition,
        )
    )
    released = await store.apply_invocation_lifecycle_command(release)
    assert type(released) is InvocationReleaseResult
    assert released.session.run_epoch == 2

    ambiguous_profile = created.active_profile.model_copy(
        update={"interaction_id": f"caller-authored-{uuid4().hex}"}
    )

    def replace_with_ambiguous_v4(_session, current):
        assert current is not None
        updated = dict(current)
        updated[CHECKPOINT_SCHEMA_VERSION_KEY] = 4
        updated[ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY] = ambiguous_profile.model_dump(
            mode="json"
        )
        updated.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY, None)
        return updated

    # Version 4 exposed this root to generic writers. Seed that historical
    # representation directly and prove that migration does not reinterpret it
    # as positive release authority.
    with sessions_module._invocation_lifecycle_authority_mutation_scope():
        await raw_store.transform_checkpoint(session_id, replace_with_ambiguous_v4)

    migrated = await store.load_checkpoint(session_id)
    assert migrated is not None
    assert migrated[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    assert active_invocation_execution_profile_from_checkpoint(migrated) is None
    migrated_receipts = _InvocationLifecycleReceiptLedger.model_validate(
        migrated[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
    )
    assert migrated_receipts.receipts == ()

    def delete_v4_lifecycle_roots(_session, current):
        assert current is not None
        updated = dict(current)
        updated[CHECKPOINT_SCHEMA_VERSION_KEY] = 4
        updated.pop(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY, None)
        updated.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY, None)
        return updated

    with sessions_module._invocation_lifecycle_authority_mutation_scope():
        await raw_store.transform_checkpoint(session_id, delete_v4_lifecycle_roots)
    migrated = await store.load_checkpoint(session_id)
    assert migrated is not None
    assert active_invocation_execution_profile_from_checkpoint(migrated) is None
    assert (
        _InvocationLifecycleReceiptLedger.model_validate(
            migrated[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
        ).receipts
        == ()
    )

    callback_called = False

    def mutate_caller_checkpoint(_session, current):
        nonlocal callback_called
        callback_called = True
        assert current is not None
        assert ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY not in current
        assert INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY not in current
        return {**current, "generic-v4-migration": suffix}

    await store.transform_checkpoint(session_id, mutate_caller_checkpoint)
    assert callback_called is True
    migrated = await store.load_checkpoint(session_id)
    assert migrated is not None
    assert migrated["generic-v4-migration"] == suffix
    assert active_invocation_execution_profile_from_checkpoint(migrated) is None
    assert (
        _InvocationLifecycleReceiptLedger.model_validate(
            migrated[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
        )
        == migrated_receipts
    )

    new_interaction_id = f"interaction-{uuid4().hex}"
    admission = AdmitInvocationCommand(
        session_id=session_id,
        expected_session_instance_id=session_instance_id,
        expected_statuses=(SessionStatus.INTERRUPTED,),
        expected_run_epoch=2,
        expected_checkpoint_sha256=invocation_checkpoint_state_sha256(migrated),
        target_active_profile=ActiveInvocationExecutionProfile(
            session_id=session_id,
            interaction_id=new_interaction_id,
            run_epoch=3,
            profile=profile,
        ),
        interaction_started_event=Event(
            type=EventType.INTERACTION_STARTED,
            session_id=session_id,
            interaction_id=new_interaction_id,
            agent_name="assistant",
        ),
        interaction_source_messages=(),
        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
        expected_active_profile=ambiguous_profile,
    )
    session_before = await raw_store.load(session_id)
    events_before = await raw_store.load_events(session_id)
    with pytest.raises(
        SessionRunFenced,
        match="lost its active profile authority",
    ):
        await store.apply_invocation_lifecycle_command(admission)
    assert await raw_store.load(session_id) == session_before
    assert await raw_store.load_events(session_id) == events_before
    assert await store.load_checkpoint(session_id) == migrated

    recovery_app = CayuApp(session_store=raw_store, enable_logging=False)
    with pytest.raises(
        RuntimeError,
        match="Incomplete-session recovery lost durable invocation profile authority",
    ):
        await recovery_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
    assert await raw_store.load(session_id) == session_before
    assert await raw_store.load_events(session_id) == events_before
    assert await store.load_checkpoint(session_id) == migrated


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_invocation_lifecycle_commands_are_atomic_in_local_stores(
    store_kind: str,
    tmp_path,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = (
            _FailCreateResultPreparationMemoryStore()
            if store_kind == "memory"
            else _FailCreateResultPreparationSQLiteStore(tmp_path / "invocation-lifecycle.sqlite")
        )
        try:
            await _assert_create_result_preparation_failure_is_atomic(store, store_kind)
            if store_kind == "memory":
                from cayu.runtime import sessions as receipt_module
            else:
                from cayu.storage import sqlite as receipt_module

            await _assert_settlement_receipt_preparation_failure_is_atomic(
                store,
                store_kind,
                monkeypatch=monkeypatch,
                receipt_module=receipt_module,
            )
            await _assert_invocation_command_conformance(store, store_kind)
            await _assert_generic_checkpoint_replacements_preserve_authority(
                store,
                store_kind,
            )
            await _assert_v4_active_profile_migration_fails_closed(store, store_kind)
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_public_resume_rejects_ambiguous_v4_release_state(
    store_kind: str,
    tmp_path,
) -> None:
    class CompletingProvider(ModelProvider):
        name = "fake"

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:v4-public-resume-provider",
                behavior_version="1",
                implementation_version="1",
            )

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> None:
        raw_store = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "v4-public-resume.sqlite")
        )
        session_id = f"v4-public-resume-{store_kind}-{uuid4().hex}"
        provider = CompletingProvider()
        app = CayuApp(session_store=raw_store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        try:
            first_events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "first")],
                    )
                )
            ]
            assert first_events[-1].type is EventType.SESSION_COMPLETED
            released_session = await raw_store.load(session_id)
            assert released_session is not None
            assert released_session.status is SessionStatus.COMPLETED

            def replace_with_v4(_session, current):
                assert current is not None
                updated = dict(current)
                updated[CHECKPOINT_SCHEMA_VERSION_KEY] = 4
                # Historical generic replacement could remove both private
                # roots. Absence is not proof that this released invocation
                # never carried runtime lifecycle authority.
                updated.pop(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY, None)
                updated.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY, None)
                return updated

            with sessions_module._invocation_lifecycle_authority_mutation_scope():
                await raw_store.transform_checkpoint(session_id, replace_with_v4)
            versioned_store = runtime_checkpoint_session_store(raw_store)
            migrated = await versioned_store.load_checkpoint(session_id)
            assert migrated is not None
            assert active_invocation_execution_profile_from_checkpoint(migrated) is None
            assert (
                _InvocationLifecycleReceiptLedger.model_validate(
                    migrated[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
                ).receipts
                == ()
            )

            replacement_app = CayuApp(session_store=raw_store, enable_logging=False)
            replacement_app.register_provider(provider, default=True)
            replacement_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            session_before_resume = await raw_store.load(session_id)
            assert session_before_resume is not None
            with pytest.raises(
                SessionRunFenced,
                match="lost its durable predecessor invocation authority",
            ):
                async for _event in replacement_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "second")],
                    )
                ):
                    pass
            final_session = await raw_store.load(session_id)
            assert final_session == session_before_resume
            final_checkpoint = await versioned_store.load_checkpoint(session_id)
            assert final_checkpoint == migrated
        finally:
            await _close_store(raw_store)

    asyncio.run(run())


def test_invocation_lifecycle_commands_are_atomic_in_postgres(
    postgres_dsn: str,
    monkeypatch,
) -> None:
    async def run() -> None:
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        class FailCreateResultPreparationPostgresStore(
            _FailCreateResultPreparationMixin,
            PostgresSessionStore,
        ):
            invocation_lifecycle_command_version = 1

        store = FailCreateResultPreparationPostgresStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await _assert_create_result_preparation_failure_is_atomic(store, "postgres")
            from cayu.storage import postgres as receipt_module

            await _assert_settlement_receipt_preparation_failure_is_atomic(
                store,
                "postgres",
                monkeypatch=monkeypatch,
                receipt_module=receipt_module,
            )
            await _assert_invocation_command_conformance(store, "postgres")
            await _assert_generic_checkpoint_replacements_preserve_authority(
                store,
                "postgres",
            )
            await _assert_v4_active_profile_migration_fails_closed(store, "postgres")
            await _assert_active_session_metadata_cannot_invalidate_release_capacity(
                store,
                "postgres",
                monkeypatch,
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_incomplete_recovery_cleanup_rejects_missing_profile_after_claim() -> None:
    async def run() -> None:
        store = _MissingProfileAfterRecoveryClaimStore()
        session_id = f"recovery-cleanup-missing-profile-{uuid4().hex}"
        command = _create_command(
            session_id=session_id,
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
        created = await store.apply_invocation_lifecycle_command(command)
        assert type(created) is InvocationMutationResult
        await store.update_status(session_id, SessionStatus.COMPLETED)
        app = CayuApp(session_store=store, enable_logging=False)

        with pytest.raises(
            RuntimeError,
            match="recovery cleanup lost durable invocation profile authority",
        ):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )

        assert store.corrupted_after_claim_renewal is True
        assert store.generic_run_fence_release_calls == 0
        current = await store.load(session_id)
        assert current is not None
        assert current.run_epoch == created.session.run_epoch + 1
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY in checkpoint
        assert ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY not in checkpoint
        assert _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY not in checkpoint

        with pytest.raises(
            RuntimeError,
            match="Incomplete-session recovery lost durable invocation profile authority",
        ):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
        after_rejection = await store.load(session_id)
        assert after_rejection is not None
        assert after_rejection.run_epoch == current.run_epoch
        assert store.generic_run_fence_release_calls == 0

    asyncio.run(run())


def test_sqlite_create_replay_survives_store_restart(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "invocation-lifecycle-restart.sqlite"
        command = _create_command(
            session_id=f"restart-{uuid4().hex}",
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
        first = SQLiteSessionStore(path)
        try:
            created = await first.apply_invocation_lifecycle_command(command)
            assert created.replayed is False
        finally:
            await first.close()
        reopened = SQLiteSessionStore(path)
        try:
            replay = await reopened.apply_invocation_lifecycle_command(command)
            assert replay.replayed is True
            assert replay.session == created.session
        finally:
            await reopened.close()

    asyncio.run(run())


def test_runtime_checkpoint_adapter_owns_command_codec_and_rejects_v4_collision() -> None:
    async def run() -> None:
        raw_store = InMemorySessionStore()
        store = runtime_checkpoint_session_store(raw_store)
        fresh = _create_command(
            session_id=f"versioned-command-{uuid4().hex}",
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
        created = await store.apply_invocation_lifecycle_command(fresh)
        assert type(created) is InvocationMutationResult
        checkpoint = await raw_store.load_checkpoint(fresh.session_id)
        assert checkpoint is not None
        assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == (CURRENT_CHECKPOINT_SCHEMA_VERSION)

        collision = _create_command(
            session_id=f"v4-collision-{uuid4().hex}",
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
        await raw_store.apply_invocation_lifecycle_command(collision)

        def replace_with_v4_collision(_session, current):
            assert current is not None
            updated = dict(current)
            updated[CHECKPOINT_SCHEMA_VERSION_KEY] = 4
            updated[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = {
                "record_type": "cayu.invocation-lifecycle-command-receipt-ledger",
                "schema_version": 1,
                "receipts": [],
            }
            return updated

        with sessions_module._invocation_lifecycle_authority_mutation_scope():
            await raw_store.transform_checkpoint(
                collision.session_id,
                replace_with_v4_collision,
            )

        callback_observed = False

        def mutate_visible_checkpoint(_session, current):
            nonlocal callback_observed
            callback_observed = True
            assert current is not None
            assert ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY not in current
            assert INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY not in current
            return {**current, "generic-collision-migration": True}

        await store.transform_checkpoint(
            collision.session_id,
            mutate_visible_checkpoint,
        )
        assert callback_observed is True
        migrated = await store.load_checkpoint(collision.session_id)
        assert migrated is not None
        assert migrated[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
        assert migrated["generic-collision-migration"] is True
        assert active_invocation_execution_profile_from_checkpoint(migrated) is None
        assert (
            _InvocationLifecycleReceiptLedger.model_validate(
                migrated[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
            ).receipts
            == ()
        )
        with pytest.raises(ValueError, match="Session already exists"):
            await store.apply_invocation_lifecycle_command(collision)

    asyncio.run(run())


def test_runtime_checkpoint_adapter_withholds_authority_from_literal_custom_store() -> None:
    async def run() -> None:
        raw_store = _LiteralCheckpointTransformStore()
        store = runtime_checkpoint_session_store(raw_store)
        command = _create_command(
            session_id=f"literal-custom-store-{uuid4().hex}",
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
        try:
            created = await store.apply_invocation_lifecycle_command(command)
            before = await raw_store.load_checkpoint(command.session_id)
            assert before is not None
            callback_called = False

            def replace_literal_checkpoint(_session, current):
                nonlocal callback_called
                callback_called = True
                assert current is not None
                assert ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY not in current
                assert INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY not in current
                return {
                    **current,
                    "custom-store-visible": True,
                    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY: {"caller": "forged"},
                    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY: {"caller": "forged"},
                }

            await store.transform_checkpoint(command.session_id, replace_literal_checkpoint)
            assert callback_called is True
            after = await raw_store.load_checkpoint(command.session_id)
            assert after is not None
            assert after["custom-store-visible"] is True
            assert active_invocation_execution_profile_from_checkpoint(after) == (
                created.active_profile
            )
            assert (
                after[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
                == before[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]
            )

            snapshot_session, snapshot_checkpoint = await load_runtime_session_checkpoint_snapshot(
                store,
                command.session_id,
            )
            assert snapshot_session.id == command.session_id
            assert snapshot_checkpoint == after
        finally:
            await _close_store(raw_store)

    asyncio.run(run())


def test_runtime_atomic_snapshot_persists_only_an_actual_schema_migration() -> None:
    async def run() -> None:
        raw_store = InMemorySessionStore()
        store = runtime_checkpoint_session_store(raw_store)
        command = _create_command(
            session_id=f"snapshot-migration-{uuid4().hex}",
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
        try:
            await store.apply_invocation_lifecycle_command(command)

            def install_v4_checkpoint(_session, current):
                assert current is not None
                return {
                    CHECKPOINT_SCHEMA_VERSION_KEY: 4,
                    "ordinary": "preserved",
                }

            with sessions_module._invocation_lifecycle_authority_mutation_scope():
                await raw_store.transform_checkpoint(
                    command.session_id,
                    install_v4_checkpoint,
                )
            raw_v4 = await raw_store.load_checkpoint(command.session_id)
            assert raw_v4 is not None
            assert raw_v4[CHECKPOINT_SCHEMA_VERSION_KEY] == 4

            snapshot_session, snapshot_checkpoint = await load_runtime_session_checkpoint_snapshot(
                store,
                command.session_id,
            )
            assert snapshot_session.id == command.session_id
            assert snapshot_checkpoint is not None
            assert (
                snapshot_checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY]
                == CURRENT_CHECKPOINT_SCHEMA_VERSION
            )
            assert snapshot_checkpoint["ordinary"] == "preserved"
            assert await raw_store.load_checkpoint(command.session_id) == snapshot_checkpoint
        finally:
            await _close_store(raw_store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "store_type",
    [
        _UnknownInvocationCommandVersionStore,
        _OverriddenReleaseWithoutProofStore,
        _OverriddenAdmissionDependencyWithoutProofStore,
        _OverriddenRejectionWrapperWithoutProofStore,
        _OverriddenTransformWithoutProofStore,
        _InheritedInvocationCommandStore,
    ],
)
def test_custom_store_command_capability_fails_closed(store_type) -> None:
    async def run() -> None:
        store = store_type()
        try:
            command = _create_command(
                session_id=f"unsupported-{uuid4().hex}",
                session_instance_id=str(uuid4()),
                interaction_id=f"interaction-{uuid4().hex}",
                profile=_profile(),
            )
            with pytest.raises(NotImplementedError, match="command version 1"):
                await store.apply_invocation_lifecycle_command(command)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_explicit_subclass_command_capability_remains_valid() -> None:
    async def run() -> None:
        store = _ExplicitInheritedInvocationCommandStore()
        try:
            command = _create_command(
                session_id=f"inherited-{uuid4().hex}",
                session_instance_id=str(uuid4()),
                interaction_id=f"interaction-{uuid4().hex}",
                profile=_profile(),
            )
            result = await store.apply_invocation_lifecycle_command(command)
            assert type(result) is InvocationMutationResult
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_create_reconciles_commit_acknowledgement_loss() -> None:
    async def run() -> None:
        store = _CommitThenRaiseCreateStore()
        command = _create_command(
            session_id=f"commit-lost-{uuid4().hex}",
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
        result = await store.apply_invocation_lifecycle_command(command)
        assert type(result) is InvocationMutationResult
        assert result.replayed is True
        assert (await store.apply_invocation_lifecycle_command(command)).replayed is True

    asyncio.run(run())


def test_create_cancellation_after_commit_is_replayable() -> None:
    async def run() -> None:
        store = _CancelAfterCommitCreateStore()
        command = _create_command(
            session_id=f"cancel-after-commit-{uuid4().hex}",
            session_instance_id=str(uuid4()),
            interaction_id=f"interaction-{uuid4().hex}",
            profile=_profile(),
        )
        owner = asyncio.create_task(store.apply_invocation_lifecycle_command(command))
        await store.committed.wait()
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1
        replay = await store.apply_invocation_lifecycle_command(command)
        assert replay.replayed is True

    asyncio.run(run())


@pytest.mark.parametrize(
    "reserved_key",
    [
        ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    ],
)
def test_invocation_checkpoint_patch_rejects_active_authority(reserved_key: str) -> None:
    with pytest.raises(ValueError, match="cannot mutate lifecycle authority"):
        InvocationCheckpointPatch(
            mutation=RuntimePublicationMutation(
                operations=(
                    RuntimePublicationCheckpointOperation(
                        key=reserved_key,
                        expected_value_digest=None,
                        action="set",
                        value={"caller": "shaped"},
                    ),
                )
            )
        )


def test_invocation_commands_reject_invalid_authority_before_store_dispatch() -> None:
    session_id = f"invalid-command-{uuid4().hex}"
    session_instance_id = str(uuid4())
    interaction_id = f"interaction-{uuid4().hex}"
    profile = _profile()
    valid = _create_command(
        session_id=session_id,
        session_instance_id=session_instance_id,
        interaction_id=interaction_id,
        profile=profile,
    )

    with pytest.raises(ValueError, match="authenticated session incarnation"):
        CreateInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            request=RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
            ),
            identity=valid.identity,
            active_profile=valid.active_profile,
            interaction_started_event=valid.interaction_started_event,
            interaction_source_messages=valid.interaction_source_messages,
            checkpoint_patch=valid.checkpoint_patch,
            tool_discovery_initialization=valid.tool_discovery_initialization,
        )

    request_message = Message.text("user", "authoritative request input")
    mismatched_request = run_request_with_runtime_session_instance_authority(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[request_message],
            tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
        ),
        session_instance_id=session_instance_id,
    )
    with pytest.raises(ValueError, match="source messages conflict with the request"):
        CreateInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            request=mismatched_request,
            identity=valid.identity,
            active_profile=valid.active_profile,
            interaction_started_event=valid.interaction_started_event,
            interaction_source_messages=(Message.text("user", "different durable input"),),
            checkpoint_patch=valid.checkpoint_patch,
            tool_discovery_initialization=valid.tool_discovery_initialization,
        )

    mismatched_target_request = run_request_with_runtime_session_instance_authority(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[],
            target=ModelTarget(provider_name="other", model="other-model"),
            tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
        ),
        session_instance_id=session_instance_id,
    )
    with pytest.raises(ValueError, match="request target conflicts"):
        CreateInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            request=mismatched_target_request,
            identity=valid.identity,
            active_profile=valid.active_profile,
            interaction_started_event=valid.interaction_started_event,
            interaction_source_messages=(),
            checkpoint_patch=valid.checkpoint_patch,
        )

    with pytest.raises(ValueError, match="provider/model identity"):
        CreateInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            request=valid.request,
            identity=valid.identity.model_copy(update={"provider_name": "other"}),
            active_profile=valid.active_profile,
            interaction_started_event=valid.interaction_started_event,
            interaction_source_messages=(),
            checkpoint_patch=valid.checkpoint_patch,
        )

    with pytest.raises(ValueError, match="runtime identity"):
        CreateInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            request=valid.request,
            identity=valid.identity.model_copy(update={"runtime_version": "other"}),
            active_profile=valid.active_profile,
            interaction_started_event=valid.interaction_started_event,
            interaction_source_messages=(),
            checkpoint_patch=valid.checkpoint_patch,
        )

    with pytest.raises(ValueError, match="changed candidate"):
        RejectInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            expected_statuses=(SessionStatus.INTERRUPTED,),
            expected_run_epoch=1,
            expected_profile=profile,
            candidate_profile=profile,
            event=_rejection_event(
                session_id=session_id,
                expected=profile,
                candidate=_profile(tool_name="different"),
            ),
        )

    release_fields = {
        "session_id": session_id,
        "expected_session_instance_id": session_instance_id,
        "expected_run_epoch": 1,
        "expected_active_profile": valid.active_profile,
    }
    with pytest.raises(ValueError, match="exactly one interaction settlement"):
        ReleaseInvocationCommand(**release_fields)
    with pytest.raises(ValueError, match="exactly one interaction settlement"):
        ReleaseInvocationCommand(
            **release_fields,
            settlement_transition=InteractionTransitionSpec(
                event=_transition_event(
                    session_id=session_id,
                    interaction_id=interaction_id,
                    status=SessionStatus.INTERRUPTED,
                ),
                from_statuses=(SessionStatus.RUNNING,),
                to_status=SessionStatus.INTERRUPTED,
            ),
            recovery_claim_id=str(uuid4()),
        )


def test_invocation_context_preserves_exact_live_authority_references() -> None:
    policy = LoopPolicy()
    app = CayuApp(
        budget_policy=BudgetPolicy(),
        loop_policies=(policy,),
        enable_logging=False,
    )
    provider = _NeverCalledProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    session_id = f"invocation-context-{uuid4().hex}"
    session_instance_id = str(uuid4())
    interaction_id = f"interaction-{uuid4().hex}"
    active = ActiveInvocationExecutionProfile(
        session_id=session_id,
        interaction_id=interaction_id,
        run_epoch=1,
        profile=_profile(),
    )
    registered_agent = app._agents["assistant"]
    registered_provider = app._providers[provider.name]
    with pytest.raises(TypeError, match="runtime authority boundary"):
        InvocationContext(
            active_profile=active,
            binding=PreparedInvocationBinding(
                session_id=session_id,
                session_instance_id=session_instance_id,
                interaction_id=interaction_id,
                run_epoch=1,
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                runtime_name="cayu",
                runtime_version="test",
                environment_name=None,
            ),
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=None,
            runtime_hooks=app._runtime_hooks,
            loop_policies=app._loop_policies,
            request_loop_policies=(),
            budget_policy=app.budget_policy,
            tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
        )

    context = _authenticated_invocation_context(
        active_profile=active,
        binding=PreparedInvocationBinding(
            session_id=session_id,
            session_instance_id=session_instance_id,
            interaction_id=interaction_id,
            run_epoch=1,
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            runtime_name="cayu",
            runtime_version="test",
            environment_name=None,
        ),
        registered_agent=registered_agent,
        registered_provider=registered_provider,
        registered_environment=None,
        validated_profile=active.profile,
        runtime_hooks=app._runtime_hooks,
        loop_policies=app._loop_policies,
        request_loop_policies=(),
        budget_policy=app.budget_policy,
        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
    )

    assert context.active_profile is active
    assert context.profile is active.profile
    assert context.registered_agent is registered_agent
    assert context.registered_provider is registered_provider
    assert context.loop_policies[0] is policy
    assert context.budget_policy is app.budget_policy
    assert repr(context) == "InvocationContext(<authenticated>)"
    assert repr(provider) not in repr(context)
    assert copy.copy(context) is context
    assert copy.deepcopy(context) is context
    with pytest.raises(TypeError, match="no serialization form"):
        pickle.dumps(context)
    with pytest.raises(TypeError, match="dataclass"):
        asdict(context)
    with pytest.raises(ValueError, match="exact validated profile object"):
        _authenticated_invocation_context(
            active_profile=active,
            binding=context.binding,
            validated_profile=ExecutionProfileIdentity.model_validate(
                active.profile.model_dump(mode="python")
            ),
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=None,
            runtime_hooks=app._runtime_hooks,
            loop_policies=app._loop_policies,
            request_loop_policies=(),
            budget_policy=app.budget_policy,
            tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
        )
    with pytest.raises(FrozenInstanceError):
        context.__setattr__("registered_provider", registered_provider)

    store = InMemorySessionStore()
    created = asyncio.run(
        store.apply_invocation_lifecycle_command(
            _create_command(
                session_id=session_id,
                session_instance_id=session_instance_id,
                interaction_id=interaction_id,
                profile=active.profile,
            )
        )
    )
    admitted = context.with_admitted_session(created.session)
    assert admitted.active_profile is active
    assert admitted.registered_agent is registered_agent
    assert admitted.registered_provider is registered_provider
    assert not hasattr(admitted.binding, "session")
    assert admitted.with_admitted_session(created.session) is admitted
    rebound_profile = active.model_copy(update={"run_epoch": 2})
    rebound = admitted.with_rebound_session(
        created.session.model_copy(update={"run_epoch": 2}),
        active_profile=rebound_profile,
    )
    assert rebound.profile is active.profile
    assert rebound.registered_agent is registered_agent
    assert rebound.registered_provider is registered_provider
    assert rebound.runtime_hooks is admitted.runtime_hooks
    assert rebound.loop_policies is admitted.loop_policies
    assert rebound.budget_policy is admitted.budget_policy
    assert rebound.binding.run_epoch == 2
    equal_distinct_profile = ExecutionProfileIdentity.model_validate(
        rebound_profile.profile.model_dump(mode="python")
    )
    assert equal_distinct_profile == rebound_profile.profile
    assert equal_distinct_profile is not rebound_profile.profile
    with pytest.raises(ValueError, match="Rebound session conflicts"):
        admitted.with_rebound_session(
            created.session.model_copy(update={"run_epoch": 2}),
            active_profile=rebound_profile.model_copy(update={"profile": equal_distinct_profile}),
        )
    with pytest.raises(ValueError, match="Rebound session conflicts"):
        admitted.with_rebound_session(
            created.session.model_copy(update={"run_epoch": 2}),
            active_profile=rebound_profile.model_copy(
                update={"profile": _profile(tool_name="changed")}
            ),
        )
    with pytest.raises(ValueError, match="prepared invocation authority"):
        context.with_admitted_session(
            created.session.model_copy(update={"instance_id": str(uuid4())})
        )
    with pytest.raises(TypeError, match="dataclass"):
        replace(
            context,
            binding=replace(context.binding, provider_name="another-provider"),
        )

    environment = Environment(EnvironmentSpec(name="unexpected-environment"))
    app.register_environment(environment)
    registered_environment = app._environments[environment.spec.name]
    with pytest.raises(ValueError, match="does not permit an environment"):
        context.with_registered_environment(
            registered_environment,
            validated_profile=active.profile,
        )
    environment_context = _authenticated_invocation_context(
        active_profile=active,
        binding=replace(
            context.binding,
            environment_name=registered_environment.spec.name,
        ),
        validated_profile=active.profile,
        registered_agent=registered_agent,
        registered_provider=registered_provider,
        registered_environment=registered_environment,
        runtime_hooks=app._runtime_hooks,
        loop_policies=app._loop_policies,
        request_loop_policies=(),
        budget_policy=app.budget_policy,
        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
    )
    assert (
        environment_context.with_registered_environment(
            registered_environment,
            validated_profile=active.profile,
        )
        is environment_context
    )
    with pytest.raises(ValueError, match="invalid environment-owner settlement"):
        environment_context.with_registered_environment(
            replace(
                registered_environment,
                environment=Environment(registered_environment.spec),
            ),
            validated_profile=active.profile,
        )
