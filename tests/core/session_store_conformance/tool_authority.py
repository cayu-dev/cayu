"""Shared session-store tool authority contract cases."""

from __future__ import annotations

import asyncio
import base64
import io
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr
from tests.core._workload_secret_support import FakeProvider
from tests.core.session_store_conformance.support import (
    _CeilingConformanceTool,
    _close_store,
    _identity,
    _open_store,
    _PreviousProfileConformancePolicy,
    _public_authority_alias_codec,
    _reopen_store,
    _targeted_use_request,
)

import cayu.sessions.base as sessions_module
import cayu.tools.shared_artifacts as shared_artifacts_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.context.footprints import RequestFootprint
from cayu.events import (
    Event,
    EventType,
    event_payload_authority_is_runtime_generated,
)
from cayu.messages import (
    Message,
)
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_profiles import (
    execution_profile_from_session_metadata,
)
from cayu.sessions._operation_records import (
    SessionOperationPublication,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.invocation import (
    SessionExecutionSource,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import (
    ForkSessionRequest,
    ResumeRequest,
    RunRequest,
)
from cayu.storage.jsonl_export import export_sessions, import_sessions
from cayu.tools.exposure import (
    ToolCapabilityCeiling,
    ToolExposure,
)
from cayu.tools.grants import (
    TargetedToolGrant,
    TargetedToolUseDisposition,
    TargetedToolUseRejectionReason,
    TargetedToolUseRequest,
)
from cayu.tools.shared_artifacts import (
    SharedArtifactAudience,
    SharedArtifactGrant,
    SharedArtifactPolicy,
    SharedArtifactRef,
)

pytestmark = pytest.mark.postgres


def test_session_store_conformance_reconstructs_tool_exposure_evidence(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            provider = FakeProvider(
                [
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            initial_policy = _PreviousProfileConformancePolicy()
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_CeilingConformanceTool("inspect")],
                tool_exposure_policy=initial_policy,
            )
            session_id = f"tool-exposure-evidence-{session_store_case[0]}"
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "inspect")],
                    )
                )
            ]
            [public_event] = [
                event for event in events if event.type is EventType.TOOL_EXPOSURE_RECORDED
            ]
            public_evidence = ToolExposure.model_validate(public_event.payload)
            [public_footprint_event] = [
                event for event in events if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
            ]
            public_footprint = RequestFootprint.model_validate(public_footprint_event.payload)
            assert [request.previous_profile_id for request in initial_policy.requests] == [None]
            [before_reopen] = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=(EventType.TOOL_EXPOSURE_RECORDED,),
                )
            )
            [footprint_before_reopen] = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=(EventType.REQUEST_FOOTPRINT_RECORDED,),
                )
            )

            store = await _reopen_store(session_store_case, store)
            [after_reopen] = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=(EventType.TOOL_EXPOSURE_RECORDED,),
                )
            )
            [footprint_after_reopen] = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=(EventType.REQUEST_FOOTPRINT_RECORDED,),
                )
            )
            reconstructed = ToolExposure.model_validate(after_reopen.event.payload)
            reconstructed_footprint = RequestFootprint.model_validate(
                footprint_after_reopen.event.payload
            )

            assert after_reopen.sequence == before_reopen.sequence
            assert after_reopen.event.payload == before_reopen.event.payload
            assert (
                reconstructed.execution_profile_fingerprint,
                reconstructed.profile_id,
                reconstructed.catalogue_revision,
                reconstructed.exposure_fingerprint,
                reconstructed.registered_count,
                reconstructed.ceiling_count,
                reconstructed.exposed_count,
                reconstructed.profile_changed,
                reconstructed.step,
                reconstructed.provider_name,
                reconstructed.model,
            ) == (
                public_evidence.execution_profile_fingerprint,
                public_evidence.profile_id,
                public_evidence.catalogue_revision,
                public_evidence.exposure_fingerprint,
                1,
                1,
                1,
                False,
                1,
                "fake",
                "fake-model",
            )
            assert set(after_reopen.event.payload) == {
                "schema_version",
                "catalogue_revision",
                "execution_profile_fingerprint",
                "profile_id",
                "exposure_fingerprint",
                "registered_count",
                "ceiling_count",
                "exposed_count",
                "profile_changed",
                "step",
                "provider_name",
                "model",
                "model_step_id",
            }
            assert footprint_after_reopen.sequence == footprint_before_reopen.sequence
            assert footprint_after_reopen.event.payload == footprint_before_reopen.event.payload
            assert reconstructed_footprint.tool_exposure == public_footprint.tool_exposure
            footprint_profile = reconstructed_footprint.execution_profile_fingerprint
            assert footprint_profile is not None
            assert event_payload_authority_is_runtime_generated(
                footprint_after_reopen.event,
                field_name="execution_profile_fingerprint",
                value=footprint_profile,
            )

            exported = io.StringIO()
            assert await export_sessions(store, stream=exported) == 1
            [imported] = list(import_sessions(io.StringIO(exported.getvalue())))
            [imported_exposure] = [
                event for event in imported.events if event.type is EventType.TOOL_EXPOSURE_RECORDED
            ]
            [imported_footprint] = [
                event
                for event in imported.events
                if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
            ]
            for field_name in (
                "catalogue_revision",
                "execution_profile_fingerprint",
                "exposure_fingerprint",
                "model_step_id",
                "profile_id",
            ):
                value = imported_exposure.payload[field_name]
                assert event_payload_authority_is_runtime_generated(
                    imported_exposure,
                    field_name=field_name,
                    value=value,
                )
            assert event_payload_authority_is_runtime_generated(
                imported_footprint,
                field_name="execution_profile_fingerprint",
                value=footprint_profile,
            )

            restored_store = InMemorySessionStore()
            await restored_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await restored_store.append_events(
                session_id,
                [imported_exposure, imported_footprint],
            )
            restored_events = await restored_store.load_events(session_id)
            assert [event.payload for event in restored_events] == [
                imported_exposure.payload,
                imported_footprint.payload,
            ]
            assert all(
                event_payload_authority_is_runtime_generated(
                    restored_events[0],
                    field_name=field_name,
                    value=restored_events[0].payload[field_name],
                )
                for field_name in (
                    "catalogue_revision",
                    "execution_profile_fingerprint",
                    "exposure_fingerprint",
                    "model_step_id",
                    "profile_id",
                )
            )
            assert event_payload_authority_is_runtime_generated(
                restored_events[1],
                field_name="execution_profile_fingerprint",
                value=footprint_profile,
            )
            await store.append_event(
                session_id,
                Event(
                    type=EventType.REQUEST_FOOTPRINT_RECORDED,
                    session_id=session_id,
                    payload=public_footprint_event.payload,
                ),
            )
            [untrusted_footprint] = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.REQUEST_FOOTPRINT_RECORDED,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            assert "execution_profile_fingerprint" not in untrusted_footprint.event.payload

            resumed_provider = FakeProvider(
                [
                    [
                        ModelStreamEvent.text_delta("resumed"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            )
            resumed_policy = _PreviousProfileConformancePolicy()
            resumed_app = CayuApp(session_store=store, enable_logging=False)
            resumed_app.register_provider(resumed_provider, default=True)
            resumed_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_CeilingConformanceTool("inspect")],
                tool_exposure_policy=resumed_policy,
            )
            resumed_events = [
                event
                async for event in resumed_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
            [resumed_event] = [
                event for event in resumed_events if event.type is EventType.TOOL_EXPOSURE_RECORDED
            ]
            resumed_evidence = ToolExposure.model_validate(resumed_event.payload)

            assert resumed_events[-1].type is EventType.SESSION_COMPLETED
            assert [request.previous_profile_id for request in resumed_policy.requests] == [
                "initial-phase"
            ]
            assert (resumed_evidence.profile_id, resumed_evidence.profile_changed) == (
                "resumed-phase",
                True,
            )

            await store.append_event(
                session_id,
                Event(
                    type=EventType.TOOL_EXPOSURE_RECORDED,
                    session_id=session_id,
                    payload={
                        **resumed_event.payload,
                        "profile_id": "caller-authored-profile",
                    },
                ),
            )
            [untrusted_latest] = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.TOOL_EXPOSURE_RECORDED,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            assert "profile_id" not in untrusted_latest.event.payload
            assert "catalogue_revision" not in untrusted_latest.event.payload
            with pytest.raises(
                RuntimeError,
                match="Latest durable tool-exposure evidence is malformed",
            ):
                _ = [
                    event
                    async for event in resumed_app.resume(
                        ResumeRequest(
                            session_id=session_id,
                            messages=[Message.text("user", "reject forged evidence")],
                        )
                    )
                ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_targeted_grant_lifecycle_survives_reopen(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        stream: AsyncIterator[Event] | None = None
        try:
            session_id = f"targeted-lifecycle-{session_store_case[0]}"
            provider = FakeProvider(
                [
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                targeted_tool_mode="call_tool",
                tools=(_CeilingConformanceTool("remember"),),
            )
            stream = app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "review")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="lifecycle-grant",
                            tool_id="cayu:remember",
                            max_calls=2,
                        ),
                    ),
                )
            )
            while True:
                event = await anext(stream)
                if event.type is EventType.TARGETED_TOOL_GRANT_ISSUED:
                    break

            [record] = await store.list_targeted_tool_grants(session_id)
            session = await store.load(session_id)
            assert session is not None
            premature = await store.bind_targeted_tool_grant_use(
                _targeted_use_request(
                    record,
                    run_epoch=session.run_epoch,
                    identity="before-issuance",
                ),
                observed_at=record.issued_at - timedelta(seconds=1),
            )
            assert premature.disposition is TargetedToolUseDisposition.REJECTED
            assert premature.reason is TargetedToolUseRejectionReason.NOT_YET_VALID
            reconstructed = await store.reconstruct_targeted_tool_grants(
                session_id,
                expected_run_epoch=session.run_epoch,
                interaction_id=record.interaction_id,
                generation_id=record.generation_id,
                agent_name=record.agent_name,
                task_id=record.task_id,
                environment_name=record.environment_name,
                principal=record.principal,
                tenant=record.tenant,
                catalogue_revision=record.catalogue_revision,
                descriptors_by_id={
                    record.tool_id: (
                        record.tool_name,
                        record.descriptor_version,
                        record.schema_fingerprint,
                    )
                },
                capability_ceiling_names=frozenset({record.tool_name}),
                observed_at=datetime.now(UTC),
            )
            assert reconstructed.valid == (record,)
            assert reconstructed.rejected == ()

            first_request = _targeted_use_request(
                record,
                run_epoch=session.run_epoch,
                identity="first",
            )
            first = await store.bind_targeted_tool_grant_use(
                first_request,
                observed_at=datetime.now(UTC),
            )
            assert first.disposition is TargetedToolUseDisposition.BOUND
            rejoined = await store.bind_targeted_tool_grant_use(
                first_request,
                observed_at=datetime.now(UTC),
            )
            assert rejoined.disposition is TargetedToolUseDisposition.REJOINED
            altered_scope_request = TargetedToolUseRequest.model_validate(
                {
                    **first_request.model_dump(mode="python"),
                    "principal": "different-principal",
                }
            )
            altered_scope = await store.bind_targeted_tool_grant_use(
                altered_scope_request,
                observed_at=datetime.now(UTC),
            )
            assert altered_scope.disposition is TargetedToolUseDisposition.REJECTED
            assert altered_scope.reason is TargetedToolUseRejectionReason.PRINCIPAL_MISMATCH
            altered = await store.bind_targeted_tool_grant_use(
                _targeted_use_request(
                    record,
                    run_epoch=session.run_epoch,
                    identity="first",
                    arguments="altered",
                ),
                observed_at=datetime.now(UTC),
            )
            assert altered.reason is TargetedToolUseRejectionReason.ALTERED_REPLAY

            second = await store.bind_targeted_tool_grant_use(
                _targeted_use_request(
                    record,
                    run_epoch=session.run_epoch,
                    identity="second",
                ),
                observed_at=datetime.now(UTC),
            )
            assert second.disposition is TargetedToolUseDisposition.BOUND
            late_rejoin = await store.bind_targeted_tool_grant_use(
                first_request,
                observed_at=datetime.now(UTC),
            )
            assert late_rejoin.disposition is TargetedToolUseDisposition.REJOINED
            reconstructed_after_uses = await store.reconstruct_targeted_tool_grants(
                session_id,
                expected_run_epoch=session.run_epoch,
                interaction_id=record.interaction_id,
                generation_id=record.generation_id,
                agent_name=record.agent_name,
                task_id=record.task_id,
                environment_name=record.environment_name,
                principal=record.principal,
                tenant=record.tenant,
                catalogue_revision=record.catalogue_revision,
                descriptors_by_id={
                    record.tool_id: (
                        record.tool_name,
                        record.descriptor_version,
                        record.schema_fingerprint,
                    )
                },
                capability_ceiling_names=frozenset({record.tool_name}),
                observed_at=datetime.now(UTC),
            )
            assert reconstructed_after_uses.valid[0].used_calls == 2
            exhausted = await store.bind_targeted_tool_grant_use(
                _targeted_use_request(
                    record,
                    run_epoch=session.run_epoch,
                    identity="third",
                ),
                observed_at=datetime.now(UTC),
            )
            assert exhausted.reason is TargetedToolUseRejectionReason.EXHAUSTED

            for invalid_epoch in (-1, True):
                with pytest.raises(
                    ValueError,
                    match="expected_run_epoch must be a non-negative integer",
                ):
                    await store.revoke_targeted_tool_grant(
                        record.tool_ref,
                        session_id=session_id,
                        expected_run_epoch=invalid_epoch,
                        reason="operator revoked",
                        revoked_at=datetime.now(UTC),
                    )

            with pytest.raises(
                ValueError,
                match="revoked_at cannot precede a bound targeted tool use",
            ):
                await store.revoke_targeted_tool_grant(
                    record.tool_ref,
                    session_id=session_id,
                    expected_run_epoch=session.run_epoch,
                    reason="operator revoked",
                    revoked_at=record.issued_at,
                )
            state_after_rejected_revocation = await store.load_targeted_tool_grant_state(session_id)
            assert state_after_rejected_revocation.records[0].revoked_at is None

            revoked = await store.revoke_targeted_tool_grant(
                record.tool_ref,
                session_id=session_id,
                expected_run_epoch=session.run_epoch,
                reason="operator revoked",
                revoked_at=datetime.now(UTC),
            )
            assert revoked is not None
            assert revoked.revocation_reason == "operator revoked"
            rejected_after_revocation = await store.bind_targeted_tool_grant_use(
                _targeted_use_request(
                    record,
                    run_epoch=session.run_epoch,
                    identity="after-revocation",
                ),
                observed_at=datetime.now(UTC),
            )
            assert rejected_after_revocation.reason is TargetedToolUseRejectionReason.REVOKED

            reconstructed_after_revocation = await store.reconstruct_targeted_tool_grants(
                session_id,
                expected_run_epoch=session.run_epoch,
                interaction_id=record.interaction_id,
                generation_id=record.generation_id,
                agent_name=record.agent_name,
                task_id=record.task_id,
                environment_name=record.environment_name,
                principal=record.principal,
                tenant=record.tenant,
                catalogue_revision=record.catalogue_revision,
                descriptors_by_id={
                    record.tool_id: (
                        record.tool_name,
                        record.descriptor_version,
                        record.schema_fingerprint,
                    )
                },
                capability_ceiling_names=frozenset({record.tool_name}),
                observed_at=datetime.now(UTC),
            )
            assert reconstructed_after_revocation.valid == ()
            assert reconstructed_after_revocation.rejected == (
                (record.grant_id, TargetedToolUseRejectionReason.REVOKED),
            )

            async for _event in stream:
                pass
            stream = None
            store = await _reopen_store(session_store_case, store)
            state = await store.load_targeted_tool_grant_state(session_id)
            assert len(state.records) == 1
            assert state.records[0].used_calls == 2
            assert state.records[0].revocation_reason == "operator revoked"
            assert len(state.uses) == 2

            exported = io.StringIO()
            assert await export_sessions(store, stream=exported) == 1
            [imported] = list(import_sessions(io.StringIO(exported.getvalue())))
            assert imported.targeted_tool_grant_state == state
        finally:
            if stream is not None:
                await stream.aclose()
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_targeted_grant_replay_identity_is_interaction_wide(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        stream: AsyncIterator[Event] | None = None
        try:
            session_id = f"targeted-cross-grant-replay-{session_store_case[0]}"
            provider = FakeProvider(
                [
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                targeted_tool_mode="call_tool",
                tools=(
                    _CeilingConformanceTool("remember"),
                    _CeilingConformanceTool("recall"),
                ),
            )
            stream = app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "review")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="remember-grant",
                            tool_id="cayu:remember",
                        ),
                        TargetedToolGrant(
                            request_id="recall-grant",
                            tool_id="cayu:recall",
                        ),
                    ),
                )
            )
            issued_count = 0
            while issued_count < 2:
                event = await anext(stream)
                issued_count += event.type is EventType.TARGETED_TOOL_GRANT_ISSUED

            records = await store.list_targeted_tool_grants(session_id)
            records_by_tool = {record.tool_name: record for record in records}
            remember = records_by_tool["remember"]
            recall = records_by_tool["recall"]
            session = await store.load(session_id)
            assert session is not None
            remember_request = _targeted_use_request(
                remember,
                run_epoch=session.run_epoch,
                identity="shared-call",
            )
            recall_request = _targeted_use_request(
                recall,
                run_epoch=session.run_epoch,
                identity="shared-call",
            )
            results = await asyncio.gather(
                store.bind_targeted_tool_grant_use(
                    remember_request,
                    observed_at=datetime.now(UTC),
                ),
                store.bind_targeted_tool_grant_use(
                    recall_request,
                    observed_at=datetime.now(UTC),
                ),
            )
            bound = [
                result
                for result in results
                if result.disposition is TargetedToolUseDisposition.BOUND
            ]
            rejected = [
                result
                for result in results
                if result.disposition is TargetedToolUseDisposition.REJECTED
            ]
            assert len(bound) == 1
            assert len(rejected) == 1
            assert rejected[0].reason is TargetedToolUseRejectionReason.ALTERED_REPLAY
            winning_request = (
                remember_request
                if bound[0].grant is not None and bound[0].grant.tool_name == "remember"
                else recall_request
            )
            exact_retry = await store.bind_targeted_tool_grant_use(
                winning_request,
                observed_at=datetime.now(UTC),
            )
            assert exact_retry.disposition is TargetedToolUseDisposition.REJOINED

            state = await store.load_targeted_tool_grant_state(session_id)
            state_by_tool = {record.tool_name: record for record in state.records}
            assert sorted(record.used_calls for record in state_by_tool.values()) == [0, 1]
            assert len(state.uses) == 1
        finally:
            if stream is not None:
                await stream.aclose()
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("max_calls", "request_count", "exact_retry"),
    [
        (1, 16, True),
        (4, 16, False),
    ],
)
def test_session_store_conformance_targeted_grant_contention_is_bounded(
    session_store_case,
    max_calls: int,
    request_count: int,
    exact_retry: bool,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"targeted-contention-{max_calls}-{exact_retry}"
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            targeted_tool_mode="call_tool",
            tools=(_CeilingConformanceTool("remember"),),
        )
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "review")],
                tool_grants=(
                    TargetedToolGrant(
                        request_id="contention-grant",
                        tool_id="cayu:remember",
                        max_calls=max_calls,
                    ),
                ),
            )
        )
        try:
            while True:
                event = await anext(stream)
                if event.type is EventType.TARGETED_TOOL_GRANT_ISSUED:
                    break
            [record] = await store.list_targeted_tool_grants(session_id)
            session = await store.load(session_id)
            assert session is not None

            def use_request(index: int) -> TargetedToolUseRequest:
                identity = 0 if exact_retry else index
                return TargetedToolUseRequest(
                    tool_ref=record.tool_ref,
                    session_id=session_id,
                    interaction_id=record.interaction_id,
                    generation_id=record.generation_id,
                    agent_name=record.agent_name,
                    task_id=record.task_id,
                    environment_name=record.environment_name,
                    principal=record.principal,
                    tenant=record.tenant,
                    catalogue_revision=record.catalogue_revision,
                    descriptor_version=record.descriptor_version,
                    schema_fingerprint=record.schema_fingerprint,
                    tool_id=record.tool_id,
                    tool_name=record.tool_name,
                    model_step_id=f"model-step-{identity}",
                    outer_tool_call_id=f"outer-call-{identity}",
                    arguments_sha256=f"sha256:{identity:064x}",
                    invocation_id=f"invocation-{identity}",
                    expected_run_epoch=session.run_epoch,
                )

            observed_at = datetime.now(UTC)
            results = await asyncio.gather(
                *(
                    store.bind_targeted_tool_grant_use(
                        use_request(index),
                        observed_at=observed_at,
                    )
                    for index in range(request_count)
                )
            )
            if exact_retry:
                assert (
                    sum(
                        result.disposition is TargetedToolUseDisposition.BOUND for result in results
                    )
                    == 1
                )
                assert (
                    sum(
                        result.disposition is TargetedToolUseDisposition.REJOINED
                        for result in results
                    )
                    == request_count - 1
                )
            else:
                assert (
                    sum(
                        result.disposition is TargetedToolUseDisposition.BOUND for result in results
                    )
                    == max_calls
                )
                assert (
                    sum(
                        result.reason is TargetedToolUseRejectionReason.EXHAUSTED
                        for result in results
                    )
                    == request_count - max_calls
                )
            state = await store.load_targeted_tool_grant_state(session_id)
            assert state.records[0].used_calls == max_calls
            assert len(state.uses) == max_calls
            assert state.records[0].remaining_calls == 0
            async for _event in stream:
                pass
        finally:
            await stream.aclose()
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_round_trips_and_atomically_narrows_tool_ceiling(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"tool-ceiling-conformance-{session_store_case[0]}"
        try:
            initial_provider = FakeProvider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})]]
            )
            initial_app = CayuApp(session_store=store, enable_logging=False)
            initial_app.register_provider(initial_provider, default=True)
            initial_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[
                    _CeilingConformanceTool("alpha"),
                    _CeilingConformanceTool("beta"),
                ],
            )
            _ = [
                event
                async for event in initial_app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "start")],
                        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("alpha",)),
                    )
                )
            ]

            store = await _reopen_store(session_store_case, store)
            reconstructed = await store.load(session_id)
            assert reconstructed is not None
            assert reconstructed.tool_capability_ceiling == ToolCapabilityCeiling(
                tool_names=("alpha",)
            )

            resumed_provider = FakeProvider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})]]
            )
            resumed_app = CayuApp(session_store=store, enable_logging=False)
            resumed_app.register_provider(resumed_provider, default=True)
            resumed_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[
                    _CeilingConformanceTool("alpha"),
                    _CeilingConformanceTool("beta"),
                ],
            )
            fork_session_id = f"{session_id}-fork"
            fork_events = [
                event
                async for event in resumed_app.fork_session(
                    ForkSessionRequest(
                        source_session_id=session_id,
                        session_id=fork_session_id,
                        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                    )
                )
            ]
            assert fork_events[-1].type is EventType.SESSION_FORKED
            _ = [
                event
                async for event in resumed_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "narrow")],
                        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                    )
                )
            ]

            store = await _reopen_store(session_store_case, store)
            narrowed = await store.load(session_id)
            forked = await store.load(fork_session_id)
            assert narrowed is not None
            assert forked is not None
            assert narrowed.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=())
            assert forked.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=())
            expected_profile = execution_profile_from_session_metadata(narrowed.metadata)
            before = narrowed.model_copy(deep=True)
            with pytest.raises(ValueError, match="never widened"):
                await store.transition_status_and_checkpoint(
                    session_id,
                    from_statuses={SessionStatus.COMPLETED},
                    to_status=SessionStatus.RUNNING,
                    checkpoint_transform=lambda _session, checkpoint: checkpoint,
                    execution_profile=expected_profile,
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("alpha",)),
                )
            assert await store.load(session_id) == before
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_in_memory_public_authority_aliases_reject_retired_keys() -> None:
    first = _public_authority_alias_codec()
    replacement_key = SecretStr(
        base64.urlsafe_b64encode(bytes([24]) * 32).decode("ascii").rstrip("=")
    )
    retained = first.rotated(
        active_key_id="replacement",
        key=replacement_key,
    )
    retired = retained.rotated(
        active_key_id="replacement",
        key=replacement_key,
        retire_key_ids=("conformance",),
    )
    private_session_id = "private-session"
    old_alias = first.encode(private_session_id, field_name="session_id")
    active_alias = retained.encode(private_session_id, field_name="session_id")
    store = InMemorySessionStore(public_authority_alias_codec=retained)

    async def run() -> None:
        await store.register_public_authority_alias(
            old_alias,
            field_name="session_id",
            private_value=private_session_id,
        )
        await store.register_public_authority_alias(
            active_alias,
            field_name="session_id",
            private_value=private_session_id,
        )
        assert (
            await store.resolve_public_authority_alias(
                old_alias,
                field_name="session_id",
            )
            == private_session_id
        )

        store._public_authority_alias_codec = retired

        assert (
            await store.resolve_public_authority_alias(
                old_alias,
                field_name="session_id",
            )
            is None
        )
        assert (
            await store.resolve_public_authority_alias(
                active_alias,
                field_name="session_id",
            )
            == private_session_id
        )

    asyncio.run(run())


def test_session_store_conformance_reconstructs_shared_artifact_lineage_grant(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            suffix = session_store_case[0]
            parent = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"shared-artifact-parent-{suffix}",
                    causal_budget_id=f"shared-artifact-budget-{suffix}",
                    messages=[],
                ),
                identity=_identity(),
            )
            parent = await store.transition_status(
                parent.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            child = await store.create(
                sessions_module.run_request_with_runtime_invocation(
                    RunRequest(
                        agent_name="assistant",
                        session_id=f"shared-artifact-child-{suffix}",
                        parent_session_id=parent.id,
                        causal_budget_id=parent.causal_budget_id,
                        messages=[],
                    ),
                    source=SessionExecutionSource.SUBAGENT,
                ),
                identity=_identity(),
            )
            policy = SharedArtifactPolicy(
                publish_path_prefixes=("handoff",),
                materialize_path_prefixes=("received",),
                allowed_content_types=("text/plain",),
            )
            reference = SharedArtifactRef(
                artifact_store_id="conformance-artifacts",
                artifact_id="art_" + "1" * 32,
                content_digest="sha256:" + "2" * 64,
                size_bytes=5,
                source_session_id=parent.id,
                access_grant_id="sag_" + "3" * 32,
            )
            published_at = datetime(2026, 8, 29, 12, tzinfo=UTC)
            grant = SharedArtifactGrant(
                reference=reference,
                source_session_instance_id=parent.instance_id,
                source_workspace_id="parent-workspace",
                source_path_sha256="4" * 64,
                root_invocation_id=parent.invocation.root_invocation_id,
                root_session_id=parent.invocation.root_session_id,
                invocation_origin_sha256=shared_artifacts_module._origin_sha256(
                    parent.invocation.origin
                ),
                causal_budget_id=parent.causal_budget_id,
                content_type="text/plain",
                policy_fingerprint=policy.fingerprint,
                audience=SharedArtifactAudience.DESCENDANT_FORK_OR_SUBAGENT,
                max_lineage_depth=4,
                retention_class="lineage_handoff",
                published_at=published_at,
                expires_at=published_at + timedelta(hours=1),
            )
            grant_key = shared_artifacts_module._grant_storage_key(reference.access_grant_id)

            def publish_grant(current_session, checkpoint, current):
                assert current_session.instance_id == parent.instance_id
                assert current is None
                return SessionOperationPublication(
                    checkpoint={} if checkpoint is None else checkpoint,
                    operation_records={grant_key: grant.model_dump(mode="json")},
                )

            await store.publish_session_operation(
                parent.id,
                idempotency_key=grant_key,
                operation_transform=publish_grant,
                events=[],
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=parent.run_epoch,
            )
            store = await _reopen_store(session_store_case, store)

            authorized = await shared_artifacts_module.authorize_shared_artifact_materialization(
                session_store=store,
                caller_session_id=child.id,
                caller_session_instance_id=child.instance_id,
                reference=reference.model_dump(mode="json"),
                policy_fingerprint=policy.fingerprint,
                observed_at=(published_at + timedelta(minutes=1)).isoformat(),
            )
            assert SharedArtifactGrant.model_validate(authorized) == grant
        finally:
            await _close_store(store)

    asyncio.run(run())
