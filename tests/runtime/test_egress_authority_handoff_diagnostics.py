from __future__ import annotations

import asyncio

import pytest
from tests.core.test_execution_profiles import (
    _adoption_handoff_factory,
    _AdoptionHandoffAdapter,
)
from tests.runtime.test_egress_authority_transitions import _create_session, _decision_between
from tests.runtime.test_runtime_egress_authority_handoff import (
    _ENVIRONMENT,
    _FINGERPRINT,
    _SESSION,
    _factory,
    _HandoffAdapter,
)

from cayu.egress.broker import EgressDecision
from cayu.egress.runtime import _EGRESS_AUTHORITY_PARKED_OUTCOME
from cayu.egress.transitions import (
    EgressAuthorityTransitionCoordinator,
    SessionCheckpointEgressAuthorityTransitionStore,
    authorized_egress_authority_transition,
    egress_authority_owner_fingerprint,
)
from cayu.environments.factory import EnvironmentFactoryRequest
from cayu.events import Event, EventType
from cayu.sessions._invocation_scope import _activate_session_interaction
from cayu.sessions.memory import InMemorySessionStore


def _record_request(broker) -> None:
    broker._record(
        EgressDecision(
            allowed=True,
            status_code=200,
            destination="api.example.com",
            method="GET",
            path="/v1/items",
            grant_id=None,
            policy_name="provider",
            reason=None,
        )
    )


async def _adopt(emit, *, interaction_id=None):
    adapter = _HandoffAdapter()
    source = _factory(adapter, generation=1, allow_post=False)
    target = _factory(adapter, generation=2, allow_post=True)
    result = await source.create(
        EnvironmentFactoryRequest(
            session_id=_SESSION,
            event_session_id="audit-session",
            interaction_id=interaction_id,
            agent_name="assistant",
            environment_name=_ENVIRONMENT,
        )
    )
    target._emitter = emit
    result.environment.binding._emitter = emit
    store = InMemorySessionStore()
    await _create_session(store, _SESSION)
    decision = _decision_between(
        source.egress_authority_identity,
        target.egress_authority_identity,
        session_id=_SESSION,
        idempotency_identity="diagnostic-handoff",
    )
    authorized = authorized_egress_authority_transition(
        decision=decision,
        transition_id="diagnostic-handoff",
        environment_name=_ENVIRONMENT,
        owner_fingerprint=egress_authority_owner_fingerprint("owner"),
        source_environment_fingerprint=_FINGERPRINT,
    )
    await target.adopt_authority(
        factory_result=result,
        authorized=authorized,
        coordinator=EgressAuthorityTransitionCoordinator(
            SessionCheckpointEgressAuthorityTransitionStore(store)
        ),
        owner_token="owner",
    )
    return result


@pytest.mark.parametrize("release_unbound", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("committed", [False, True])
def test_lifecycle_publication_is_best_effort_and_never_replays_uncertain_appends(
    release_unbound: bool, grouped: bool, committed: bool
) -> None:
    async def run() -> None:
        attempts: list[Event] = []
        persisted: dict[str, Event] = {}

        async def emit(event: Event) -> Event:
            attempts.append(event)
            if event.id in persisted:
                raise ValueError("duplicate event ID")
            if committed:
                persisted[event.id] = event
            error = RuntimeError("event store unavailable or acknowledgement lost")
            if grouped:
                raise ExceptionGroup("publication failed", [error])
            raise error

        result = await _adopt(emit)
        binding = result.environment.binding
        runner = result.environment.runner
        target_id = runner._egress_grants[0].grant_id
        if release_unbound:
            await binding.release_unbound(outcome="completed")
            await binding.release_unbound(outcome="completed")
        else:
            bound = await binding.bind(None, runner, session_id=_SESSION)
            await binding.finalize(bound, outcome="completed")
            assert not binding._bound_finalization_states
        await binding._emit_revoked()
        assert runner.is_closed
        keys = [(event.type, event.payload.get("grant_id")) for event in attempts]
        assert len(keys) == len(set(keys))
        assert (EventType.CREDENTIAL_MODE_SELECTED, None) in keys
        assert (EventType.EGRESS_GRANT_MINTED, target_id) in keys
        assert (EventType.EGRESS_GRANT_REVOKED, target_id) in keys
        assert not binding._pending_lifecycle_events

    asyncio.run(run())


def test_adopted_request_audit_preserves_event_session() -> None:
    async def run() -> None:
        events: list[Event] = []

        async def emit(event: Event) -> Event:
            events.append(event)
            return event

        result = await _adopt(emit)
        runner = result.environment.runner
        _record_request(runner._authority_revoker._broker)
        await result.environment.binding.release_unbound(outcome="completed")
        request_events = [
            event for event in events if event.type == EventType.EGRESS_REQUEST_AUTHORIZED
        ]
        assert len(request_events) == 1
        assert {event.session_id for event in events} == {"audit-session"}

    asyncio.run(run())


@pytest.mark.parametrize("interaction_id", [None, "target-interaction"])
def test_adopted_request_audit_resolves_current_event_session_interaction(
    interaction_id: str | None,
) -> None:
    async def run() -> None:
        events: list[Event] = []

        async def emit(event: Event) -> Event:
            events.append(event)
            return event

        _activate_session_interaction(_SESSION, "logical-session-interaction")
        if interaction_id is not None:
            _activate_session_interaction("audit-session", interaction_id)
        result = await _adopt(emit, interaction_id="source-interaction")
        _record_request(result.environment.runner._authority_revoker._broker)
        await result.environment.binding.release_unbound(outcome="completed")
        request_events = [
            event for event in events if event.type == EventType.EGRESS_REQUEST_AUTHORIZED
        ]
        assert len(request_events) == 1
        assert request_events[0].session_id == "audit-session"
        assert request_events[0].interaction_id == interaction_id

    asyncio.run(run())


@pytest.mark.parametrize("interaction_id", [None, "target-interaction"])
def test_renewal_keeps_draining_the_live_broker_audit_bridge(
    interaction_id: str | None,
) -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        request_events: list[Event] = []

        async def emit(event: Event) -> Event:
            if event.type == EventType.EGRESS_REQUEST_AUTHORIZED:
                started.set()
                await release.wait()
                request_events.append(event)
            return event

        adapter = _AdoptionHandoffAdapter()
        factory = _adoption_handoff_factory(adapter=adapter, generation=1, allow_post=False)
        factory._emitter = emit
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id="renewal-session",
                event_session_id="audit-session",
                interaction_id="source-interaction",
                agent_name="assistant",
                environment_name="egress",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        bound = await binding.bind(None, runner, session_id="renewal-session")
        await binding.finalize(bound, outcome=_EGRESS_AUTHORITY_PARKED_OUTCOME)
        original_audit = adapter.broker._audit
        _activate_session_interaction("renewal-session", "logical-session-interaction")
        if interaction_id is not None:
            _activate_session_interaction("audit-session", interaction_id)
        await factory.renew_parked_authority(
            factory_result=result,
            expected_authority=factory.egress_authority_identity,
            session_id="renewal-session",
            environment_name="egress",
            agent_name="renewed-agent",
            execution_profile_fingerprint=None,
        )
        bound = await binding.bind(None, runner, session_id="renewal-session")
        _record_request(adapter.broker)
        await asyncio.wait_for(started.wait(), timeout=2)
        draining = asyncio.Event()
        original_drain = original_audit.drain

        async def observed_drain() -> None:
            draining.set()
            await original_drain()

        original_audit.drain = observed_drain
        finalizing = asyncio.create_task(binding.finalize(bound, outcome="completed"))
        try:
            await asyncio.wait_for(draining.wait(), timeout=2)
            assert not finalizing.done()
        finally:
            release.set()
            await asyncio.wait_for(finalizing, timeout=2)
        assert runner._audit is binding._audit is adapter.broker._audit is original_audit
        assert len(request_events) == 1
        assert request_events[0].session_id == "audit-session"
        assert request_events[0].agent_name == "renewed-agent"
        assert request_events[0].interaction_id == interaction_id

    asyncio.run(run())
