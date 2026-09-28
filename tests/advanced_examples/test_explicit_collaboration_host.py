"""The runnable example traverses real owners and qualified adapter serialization."""

import asyncio
import json
from contextvars import Context
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from examples.collaboration import explicit_host, finite_host_patterns
from examples.collaboration.explicit_host import JOURNEY_TIMEOUT_S, run_demo
from examples.collaboration.finite_host_patterns import shared_specialist
from examples.collaboration.host_demo_authority import (
    PRINCIPAL,
    ExactDisclosure,
    QuestionInput,
    commitment,
    reference,
)
from examples.collaboration.host_demo_transport import OfflineProvider, OfflineResponses
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_budget_refusal import refusal_ledger as refusal_ledger

from cayu import CayuApp, CollaborationHost, ContinuationService, HostContinuationRule, HostWaitRule
from cayu.collaboration import _host_producer_maintenance
from cayu.collaboration._contracts import OwnerRef
from cayu.collaboration.exports import SessionExportDenied
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.peer_content import PeerContentUnavailable
from cayu.messages import Message
from cayu.sessions import ResumeRequest, RunRequest
from cayu.sessions.context_views import RecipientSessionCreationRequest
from cayu.sessions.invocation import InvocationOriginClaim


def test_example_provider_identity_does_not_qualify_arbitrary_transports():
    with pytest.raises(TypeError, match="only qualifies"):
        OfflineProvider(object())


@pytest.mark.anyio
@pytest.mark.parametrize("request_key", [None, "independent-one"])
async def test_example_agent_sends_its_output_bound_to_the_real_adapter(request_key):
    transport = OfflineResponses()
    app = CayuApp(enable_logging=False)
    app.register_provider(OfflineProvider(transport), default=True)
    app.register_agent(explicit_host.demo_agent(request_key=request_key))
    async for _ in app.run(
        RunRequest(agent_name="demo-agent", messages=[Message.text("user", "DEMO_TASK")])
    ):
        pass
    assert len(transport.requests) == 1
    assert transport.requests[0]["max_output_tokens"] == 256
    if request_key is not None:
        assert transport.requests[0]["instructions"] == "DEMO_REQUEST:" + request_key


@pytest.mark.anyio
@pytest.mark.parametrize("count", [2, 8])
async def test_example_team_registers_shared_owners_without_model_execution(count):
    team = await explicit_host.create_demo_team(roles=tuple(f"role-{i}" for i in range(count)))
    assert len(team.participants) == count
    assert len(set(team.participants)) == count
    assert team.initialized.binding.limits.retained_bytes == explicit_host.TEAM_RETAINED_BYTES
    assert team.initialized.binding.limits.events == explicit_host.TEAM_EVENTS
    first, second = team.application(), team.application()
    assert first.session_store is second.session_store is team.sessions
    assert first.budget_ledger is second.budget_ledger is team.ledger
    assert await first.initialize_collaboration() == team.initialized
    assert await second.initialize_collaboration() == team.initialized
    for participant in team.participants:
        assert (
            await first.inspect_participant(participant, context=explicit_host.ACCESS) is not None
        )
    assert team.transport.requests == []


@pytest.mark.anyio
async def test_example_question_creation_is_isolated_between_teams(native_stores, monkeypatch):
    class CreatedBarrier(Exception):
        pass

    created = []

    async def stop_before_execution(app, execution, *_args, **_kwargs):
        session = await app.session_store.load(execution.request.session_id)
        assert session is not None
        created.append(session)
        raise CreatedBarrier
        yield  # Keep the same asynchronous-stream entrance without dispatch.

    monkeypatch.setattr(CayuApp, "execute_participant_session_to_wait", stop_before_execution)
    teams = []
    for _ in range(2):
        team = await explicit_host.create_demo_team(
            collaboration=native_stores[0], sessions=native_stores[1]
        )
        teams.append(team)
        with pytest.raises(CreatedBarrier):
            await explicit_host.run_question(
                team,
                key="same-question",
                sender=team.participants[0],
                recipient=team.participants[1],
                task="Independent work sharing the same native store.",
            )
        assert team.transport.requests == []
    assert created[0].id != created[1].id
    for team, session in zip(teams, created, strict=True):
        receipt = await team.sessions.load_participant_session_creation_receipt(session.id)
        assert receipt is not None
        assert receipt.binding.creation_key == team.scope + ":same-question:coordinator"


@pytest.mark.anyio
async def test_shared_specialist_question_authority_does_not_accumulate_other_sources():
    from cayu.collaboration._contracts import ObjectRef
    from cayu.collaboration.mandates import MandateDenied, ResourceSelector

    team = await explicit_host.create_demo_team(roles=("first", "second", "specialist"))
    first, second, specialist = team.participants
    left = await team.mandates.question_context(specialist, "left")
    right = await team.mandates.question_context(specialist, "right")
    assert left != right and left.participant == right.participant == specialist
    assert await team.mandates.question_context(specialist, "left") == left
    resources = []
    for key, context, recipient in (("left", left, first), ("right", right, second)):
        resource = ResourceSelector(
            resource=ObjectRef(
                owner=team.initialized.owner,
                kind="session_transcript_row",
                object_id=key,
                incarnation=key + "-instance",
                revision=1,
            )
        )
        resources.append(resource)
        audience = OwnerRef(
            application_scope=team.scope,
            owner_id=recipient.participant_id,
            incarnation=recipient.incarnation,
        )
        await team.mandates.grant_source(context, resource, audience)
    for index, context in enumerate((left, right)):
        async with team.mandates.acquire(context) as resolution:
            entry = resolution.chain.entries[0]
            assert entry.resources == (resources[index],)
            assert entry.reference == entry.root == context.mandate
            assert len(entry.audiences) == 2
    base = team.mandates.contexts[specialist]
    async with team.mandates.acquire(base) as resolution:
        assert resolution.chain.entries[0].resources == ()
    with pytest.raises(MandateDenied):
        await team.mandates.grant_source(base, resources[0], team.initialized.owner)
    async with team.mandates.acquire(left) as held:
        # New question registration and an unrelated source grant do not need
        # to wait for this mandate's (possibly remote) in-flight user.
        async with asyncio.timeout(1):
            third = await team.mandates.question_context(specialist, "third")
            await team.mandates.grant_source(third, resources[1], team.initialized.owner)
        update = asyncio.create_task(
            team.mandates.grant_source(left, resources[1], team.initialized.owner)
        )
        await asyncio.sleep(0.01)
        assert not update.done()
        update.cancel()
        with pytest.raises(asyncio.CancelledError):
            await update
        assert update.cancelled() and update.cancelling() == 1
        assert held.chain.entries[0].resources == (resources[0],)
    assert not team.mandates._readers
    assert team.transport.requests == []


@pytest.mark.anyio
async def test_example_preparation_survives_application_reconstruction():
    team = await explicit_host.create_demo_team()
    first = team.application(request_key="reconstructed")
    second = team.application(request_key="reconstructed")
    await first.initialize_collaboration()
    await second.initialize_collaboration()
    creation = RecipientSessionCreationRequest(
        creation_key="reconstructed:specialist",
        recipient=team.participants[1],
        request=RunRequest(
            agent_name="demo-agent",
            messages=[Message.text("user", "DEMO_TASK: reconstruct the exact preparation.")],
            invocation_origin=InvocationOriginClaim(subject=PRINCIPAL),
        ),
    )
    preparation = await first.prepare_recipient_creation(creation, context=explicit_host.ACCESS)
    changed = team.application(request_key="different-question")
    await changed.initialize_collaboration()
    with pytest.raises(ValueError, match="differs from the retained proposal"):
        await changed.create_recipient_session(
            creation, context=explicit_host.ACCESS, preparation=preparation
        )
    inventory, _ = await first.list_participant_sessions(
        team.participants[1], context=explicit_host.ACCESS
    )
    assert not inventory
    created = await second.create_recipient_session(
        creation, context=explicit_host.ACCESS, preparation=preparation
    )
    replay = await first.create_recipient_session(
        creation, context=explicit_host.ACCESS, preparation=preparation
    )
    assert created[0].id == replay[0].id
    assert created[1] == replay[1]
    assert team.transport.requests == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "roles",
    [
        (),
        ("single",),
        ("same", "same"),
        ("valid", True),
        ("valid", "bad role"),
        tuple(f"role-{i}" for i in range(9)),
        ["one", "two"],
    ],
)
async def test_example_team_rejects_invalid_roles_before_using_a_store(roles):
    with pytest.raises(ValueError, match="two to eight distinct"):
        await explicit_host.create_demo_team(roles=roles, collaboration=object())


@pytest.mark.anyio
async def test_shared_specialist_keeps_independent_inputs():
    first, second = await shared_specialist()
    assert first["request"].intent.request.sender != second["request"].intent.request.sender
    for result in (first, second):
        assert result["provider_calls"] == 3
        assert result["producer"] == "settled"
        assert result["delivery"] == "appended"
        assert result["continuation"] == "consumed"


@pytest.mark.anyio
@pytest.mark.qualification
@pytest.mark.parametrize(
    "pattern",
    [
        "sequential_specialists",
        "supervisor_workers",
        "peer_discussion",
        "independent_parallel_candidates",
        "bounded_review_revision",
    ],
)
async def test_finite_host_application_patterns(pattern, monkeypatch):
    denied = []
    if pattern in {"peer_discussion", "sequential_specialists"}:
        allow = ExactDisclosure.allow_question_input
        create = finite_host_patterns.create_demo_team
        teams = []

        async def capture_team(**kwargs):
            team = await create(**kwargs)
            teams.append(team)
            return team

        async def require_explicit_grant(policy, **kwargs):
            if pattern == "sequential_specialists":
                without_forwarding = {
                    key: value for key, value in kwargs.items() if key != "forward_from"
                }
                with pytest.raises(SessionExportDenied):
                    await allow(policy, **without_forwarding)
            value = QuestionInput(
                kwargs["source_receipt"],
                commitment({"text": kwargs["text"], "artifact_commitments": []}),
                kwargs["text"],
                kwargs["question_key"],
                kwargs["recipient"],
            )
            team = teams[0]
            before = len(team.transport.requests)
            with pytest.raises(SessionExportDenied):
                await explicit_host.run_question(
                    team,
                    key=value.question_key,
                    sender=team.participants[1 if pattern == "peer_discussion" else 0],
                    recipient=value.recipient,
                    task="Respond to the preceding reply.",
                    input_handoff=value,
                )
            assert len(team.transport.requests) == before == 3
            denied.append(value.source_receipt)
            authorized = await allow(policy, **kwargs)
            object.__setattr__(authorized, "text", "Unselected replacement content.")
            with pytest.raises(SessionExportDenied):
                await explicit_host.run_question(
                    team,
                    key=value.question_key,
                    sender=team.participants[1 if pattern == "peer_discussion" else 0],
                    recipient=value.recipient,
                    task="Respond to the preceding reply.",
                    input_handoff=authorized,
                )
            assert len(team.transport.requests) == before
            return await allow(policy, **kwargs)

        monkeypatch.setattr(finite_host_patterns, "create_demo_team", capture_team)
        monkeypatch.setattr(ExactDisclosure, "allow_question_input", require_explicit_grant)
    result = await getattr(finite_host_patterns, pattern)()
    if pattern in {"peer_discussion", "sequential_specialists"}:
        assert denied == [result[0]["source_receipt"]]
        selected_input = json.dumps(result[1]["serialized_requests"][1]["input"])
        assert result[0]["answer"] in selected_input
        assert result[0]["source_receipt"] in selected_input
    results = result["results"] if pattern == "independent_parallel_candidates" else result
    assert len(results) == 2
    for outcome in results:
        assert outcome["provider_calls"] == 3
        assert outcome["producer"] == "settled"
        assert outcome["delivery"] == "appended"
        assert outcome["continuation"] == "consumed"


@pytest.mark.parametrize(
    "revision,text", [(True, "draft"), (0, "draft"), (3, "draft"), (1, ""), (1, "x" * 513)]
)
def test_example_review_revision_is_bounded(revision, text):
    with pytest.raises(ValueError, match="two bounded revisions"):
        finite_host_patterns.DraftRevision(revision, text)


@pytest.mark.anyio
async def test_review_response_uses_real_adapter_revision_input():
    draft = finite_host_patterns.DraftRevision(1, "A bounded draft.")
    transport = OfflineResponses()
    app = CayuApp(enable_logging=False)
    app.register_provider(OfflineProvider(transport), default=True)
    app.register_agent(explicit_host.demo_agent(request_key="review-adapter"))
    async for _ in app.run(
        RunRequest(agent_name="demo-agent", messages=[Message.text("user", draft.review_task)])
    ):
        pass
    assert len(transport.requests) == 1
    assert draft.commitment in json.dumps(transport.requests[0]["input"])
    assert transport.requests[0]["max_output_tokens"] == 256


@pytest.mark.anyio
async def test_example_exposure_distinguishes_participant_from_receiving_owner(monkeypatch):
    """Policy characterization; the backend journey below exercises real inputs."""
    owner = OwnerRef(application_scope="demo", owner_id="collaboration", incarnation="one")
    participant_audience = OwnerRef(
        application_scope="demo", owner_id="coordinator", incarnation="participant-one"
    )
    contract = reference(owner, "policy", "disclosure")
    policy = ExactDisclosure(contract, expires_at_ms=9_000_000_000_000)
    source = SimpleNamespace(session_id="source", session_instance_id="source-one")
    key = SimpleNamespace(
        target_session_id="target",
        target_session_instance_id="target-one",
        creation_target=None,
        consumer_id="coordinator",
        consumer_participant_incarnation="participant-one",
    )
    receipt = SimpleNamespace(
        event_id="export-one",
        expected=SimpleNamespace(
            intent=SimpleNamespace(
                request=SimpleNamespace(
                    ref=source, audience=participant_audience, projector=contract, policy=contract
                )
            )
        ),
    )
    destination = SimpleNamespace(
        recipient=SimpleNamespace(
            participant_id="coordinator", incarnation="participant-one", owner=owner
        ),
        projector=contract,
        attempt=SimpleNamespace(append_key=key),
    )
    payload = {"text": "Selected result."}
    await policy.allow_source(source.session_id, source.session_instance_id)
    await policy.allow_delivery(
        receipt=receipt, payload=payload, destination=destination, producer_key="producer"
    )
    from cayu.collaboration.participants import ParticipantRef

    recipient = ParticipantRef(
        owner=owner, participant_id="coordinator", incarnation="participant-one"
    )
    handoff = await policy.allow_question_input(
        source_receipt=receipt.event_id,
        text=payload["text"],
        question_key="followup",
        recipient=recipient,
    )
    assert payload["text"] in await policy.question_input(
        handoff, question_key="followup", recipient=recipient
    )
    object.__setattr__(handoff, "text", "Unselected content.")
    with pytest.raises(SessionExportDenied):
        await policy.question_input(handoff, question_key="followup", recipient=recipient)
    # Mutating the returned object cannot change the installed grant.
    recovered = await policy.allow_question_input(
        source_receipt=receipt.event_id,
        text=payload["text"],
        question_key="followup",
        recipient=recipient,
    )
    assert payload["text"] in await policy.question_input(
        recovered, question_key="followup", recipient=recipient
    )
    verifier = ParticipantRef(owner=owner, participant_id="verifier", incarnation="verifier-one")
    forwarding = dict(
        source_receipt=receipt.event_id,
        text=payload["text"],
        question_key="verify",
        recipient=verifier,
    )
    for original in (None, verifier, recipient.model_copy(update={"incarnation": "wrong"})):
        with pytest.raises(SessionExportDenied):
            await policy.allow_question_input(**forwarding, forward_from=original)
    forwarded = await policy.allow_question_input(**forwarding, forward_from=recipient)
    assert payload["text"] in await policy.question_input(
        forwarded, question_key="verify", recipient=verifier
    )
    with pytest.raises(SessionExportDenied):
        await policy.question_input(forwarded, question_key="verify", recipient=recipient)
    occurrence = SimpleNamespace(
        source_export_receipt_id=receipt.event_id,
        producer_receipt_id="producer",
        sender_session_id=source.session_id,
        sender_session_instance_id=source.session_instance_id,
        audience=(key.consumer_id,),
        payload=SimpleNamespace(content_sha256=commitment({**payload, "artifact_commitments": []})),
    )
    origin = SimpleNamespace(
        append_key=key,
        target_session_id=key.target_session_id,
        target_session_instance_id=key.target_session_instance_id,
        provider_name="offline",
        model="gpt-5.6",
        requester_principal=PRINCIPAL,
    )
    item = SimpleNamespace(origin=origin, occurrence=occurrence, audience=owner)
    async with policy.acquire_peer_exposures(None, items=(item,)) as result:
        assert result == (occurrence.payload,)
    for rejected in (
        participant_audience,
        owner.model_copy(update={"incarnation": "replacement"}),
        owner.model_copy(update={"application_scope": "other"}),
    ):
        with pytest.raises(SessionExportDenied):
            async with policy.acquire_peer_exposures(
                None,
                items=(SimpleNamespace(origin=origin, occurrence=occurrence, audience=rejected),),
            ):
                pytest.fail("A different receiving owner acquired the projection.")
    origin.requester_principal = "unrelated-principal"
    with pytest.raises(SessionExportDenied):
        async with policy.acquire_peer_exposures(None, items=(item,)):
            pytest.fail("An unrelated invocation subject acquired the projection.")
    origin.requester_principal = PRINCIPAL
    monkeypatch.setattr(
        "examples.collaboration.host_demo_authority.time.time_ns",
        lambda: 9_000_000_000_000 * 1_000_000 - 1,
    )
    async with policy.acquire_peer_exposures(None, items=(item,)) as result:
        assert result == (occurrence.payload,)
    monkeypatch.setattr(
        "examples.collaboration.host_demo_authority.time.time_ns",
        lambda: 9_000_000_000_000 * 1_000_000,
    )
    with pytest.raises(SessionExportDenied):
        async with policy.acquire_peer_exposures(None, items=(item,)):
            pytest.fail("Expired disclosure authority acquired the projection.")


@pytest.mark.anyio
@pytest.mark.parametrize("progress", [0, 1])
@pytest.mark.parametrize("unavailable_first", [False, True])
async def test_example_wait_observes_consecutive_equal_progress_counts(
    monkeypatch, progress, unavailable_first
):
    """A pass count is not an operation identity or cumulative generation."""
    observations = []
    observed_at = []

    class Host:
        def __init__(self, app, registration):
            self.passes = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def service_once(self):
            self.passes += 1
            assert self.passes <= 200, "Consecutive progress must not suppress readback."
            if not progress and self.passes >= 2:
                # An independent owner commits after our readback interval,
                # without this host reporting any locally completed work.
                await asyncio.sleep(1.01)
            return SimpleNamespace(
                serviced=progress,
                failed=0,
                source_failures=0,
                pending=False,
                active=0,
                uncertain=0,
                servicing_pending=False,
            )

        def inspect(self):
            return SimpleNamespace(pending=0)

    async def inspect():
        observations.append(len(observations) + 1)
        observed_at.append(asyncio.get_running_loop().time())
        if unavailable_first and len(observations) == 1:
            raise CollaborationUnavailable("Exact observation has not acknowledged yet.")
        return observations[-1]

    monkeypatch.setattr(explicit_host, "CollaborationHost", Host)
    result = await explicit_host.service_until(None, None, inspect, lambda value: value == 2)
    assert result == 2
    assert observations == [1, 2]
    assert observed_at[1] - observed_at[0] >= 1.0


@pytest.mark.anyio
@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), 901, "1"])
async def test_example_rejects_unbounded_observation_before_host_construction(monkeypatch, timeout):
    def unexpected_host(*_):
        pytest.fail("Invalid observation bound constructed a host")

    monkeypatch.setattr(explicit_host, "CollaborationHost", unexpected_host)
    with pytest.raises(ValueError, match="bounded journey interval"):
        await explicit_host.service_until(None, None, None, None, timeout_s=timeout)


@pytest.mark.anyio
@pytest.mark.parametrize("owned_turns", [0, 3, None])
async def test_example_keeps_servicing_during_business_readback(monkeypatch, owned_turns):
    serviced = []
    observed = []
    release = asyncio.Event()

    class Host:
        def __init__(self, *_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def service_once(self):
            serviced.append(True)
            if len(serviced) == 4:
                release.set()
            # There need never be an idle pass, even after this question is
            # durably complete. Native source discovery is independently due.
            return SimpleNamespace(
                serviced=1,
                failed=0,
                source_failures=0,
                pending=True,
                active=0,
                uncertain=int(owned_turns is None or len(serviced) <= owned_turns),
                servicing_pending=False,
            )

        def inspect(self):
            return SimpleNamespace(pending=False)

    async def inspect():
        observed.append(len(serviced))
        await release.wait()
        return True

    monkeypatch.setattr(explicit_host, "CollaborationHost", Host)
    assert await explicit_host.service_until(None, None, inspect, bool)
    assert observed == [1]
    assert len(serviced) >= 4


@pytest.mark.anyio
async def test_example_cancelled_readback_remains_owned_and_observable(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    original = RuntimeError("readback failed after observer cancellation")
    calls = []

    class Host:
        def __init__(self, *_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def service_once(self):
            return SimpleNamespace(
                serviced=0,
                failed=0,
                source_failures=0,
                pending=True,
                active=0,
                uncertain=0,
                servicing_pending=False,
            )

        def inspect(self):
            return SimpleNamespace(pending=False)

    async def inspect():
        calls.append(True)
        entered.set()
        await release.wait()
        raise original

    monkeypatch.setattr(explicit_host, "CollaborationHost", Host)
    running = asyncio.create_task(explicit_host.service_until(None, None, inspect, bool))
    observation = None
    try:
        await entered.wait()
        running.cancel()
        assert running.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await running
        assert running.cancelled() and running.cancelling() == 1
        observation = cancelled.value.__dict__["collaboration_observation"]
        assert observation.task is not None
        assert not observation.task.done()
        assert calls == [True]
    finally:
        release.set()
        if observation is not None:
            value, error = await observation.task
            assert value is None and error is original


@pytest.mark.anyio
async def test_example_cancellation_keeps_original_signal_and_host_handle(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    hosts = []

    class Host:
        def __init__(self, *_):
            self.worker = None
            hosts.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass  # Bounded close cannot prove opaque work stopped.

        async def service_once(self):
            self.worker = asyncio.create_task(release.wait())
            entered.set()
            await asyncio.shield(self.worker)

        def inspect(self):
            return SimpleNamespace(pending=self.worker is not None and not self.worker.done())

    monkeypatch.setattr(explicit_host, "CollaborationHost", Host)
    running = asyncio.create_task(explicit_host.service_until(None, None, None, None))
    try:
        await entered.wait()
        running.cancel()
        assert running.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await running
        assert running.cancelled() and running.cancelling() == 1
        retained = cancelled.value.__dict__["collaboration_host"]
        assert retained is hosts[0] and retained.inspect().pending
        assert not retained.worker.cancelled()
    finally:
        release.set()
        if hosts and hosts[0].worker is not None:
            await hosts[0].worker


@pytest.mark.anyio
@pytest.mark.host_journey
async def test_public_host_creation_bound_delivery_gates_continuation(
    native_stores, refusal_ledger, monkeypatch
):
    await test_public_host_example_delivers_and_continues_once(
        native_stores, refusal_ledger, monkeypatch, creation_bound=True
    )


@pytest.mark.anyio
@pytest.mark.host_journey
async def test_public_host_example_delivers_and_continues_once(
    native_stores, refusal_ledger, monkeypatch, creation_bound=False
):
    # Both target forms qualify the composed backend journey, not the example's
    # shorter per-phase observation deadline. Keep the original overall bound
    # and all native authority deadlines; this does not reset time per phase.
    monkeypatch.setattr(explicit_host, "SERVICE_TIMEOUT_S", JOURNEY_TIMEOUT_S)
    stages = []
    raw_rejections = []
    deferred_continuations = []
    append = CayuApp.append_peer_content
    publish = _host_producer_maintenance.publish_producer_outcome

    async def publish_before_delivery(app, registration, **kwargs):
        outcome = await publish(app, registration, **kwargs)
        destination = registration.destinations[0]
        access = explicit_host.ACCESS
        inventory, _ = await app.list_participant_sessions(destination.recipient, context=access)
        key = destination.attempt.append_key
        session_id = key.target_session_id
        if key.creation_target is not None:
            decision = await app.session_store.read_session_creation_decision(key.creation_target)
            session_id = decision.receipt.session_id
        target = next(item for item in inventory if item.session_id == session_id)
        continuation = (await app.list_session_continuations(target, context=access)).items[0]
        wait = (await app.list_collaboration_waits(context=access)).items[0].recovery
        mandate = app._request_coordinator._registration.mandates.contexts[destination.recipient]
        # An independent host receives the real final latch before delivery.
        record = await explicit_host.service_until(
            app,
            explicit_host.host_registration(wait_rules=(HostWaitRule(wait, mandate),)),
            lambda: app.recover_session_continuation(continuation, context=access),
            lambda value: value.latch is not None,
        )
        rule = HostContinuationRule(
            continuation,
            ResumeRequest(
                session_id=target.session_id,
                messages=[Message.text("user", "DEMO_RESUME: wait for exact input.")],
            ),
            ContinuationService(
                ticket=record.ticket,
                latch=record.latch,
                continuation_id="early-integrate",
                mode="inline",
                accepted_at=datetime.now(UTC).isoformat(),
            ),
            access,
        )
        host = CollaborationHost(app, explicit_host.host_registration(continuation_rules=(rule,)))
        async with host, asyncio.timeout(60):
            while True:
                observed = await host.service_once()
                assert observed.failed == observed.source_failures == 0
                # Read-only discovery may be pending; parked work must not
                # occupy either native execution or maintenance capacity.
                assert observed.active == observed.uncertain == 0
                if observed.observed_blocked:
                    break
                await asyncio.sleep(0.01)
        retained = await app.recover_session_continuation(continuation, context=access)
        assert retained.ticket.state == "WAITING"
        assert retained.consumption is None
        deferred_continuations.append(continuation.ticket_key)
        return outcome

    async def append_after_raw_attempt(app, request, *, context):
        # Identical public data and genuine disclosure permission do not carry
        # the private producer-owner provenance. Run the real public entrance
        # in an empty context, as an independent caller would.
        raw = asyncio.create_task(append(app, request, context=context), context=Context())
        with pytest.raises(PeerContentUnavailable):
            await raw
        assert await app.session_store.read_peer_content_attempt(request) is None
        raw_rejections.append(request.operation_key)
        return await append(app, request, context=context)

    monkeypatch.setattr(CayuApp, "append_peer_content", append_after_raw_attempt)
    monkeypatch.setattr(
        _host_producer_maintenance, "publish_producer_outcome", publish_before_delivery
    )
    try:
        async with asyncio.timeout(JOURNEY_TIMEOUT_S):
            result = await run_demo(
                collaboration=native_stores[0],
                sessions=native_stores[1],
                ledger=refusal_ledger[0],
                report=stages.append,
                creation_bound=creation_bound,
            )
    except TimeoutError as error:
        error.add_note("Completed public journey stages: " + ", ".join(stages))
        pending_host = error.__dict__.get("collaboration_host")
        if pending_host is not None:
            error.add_note("Retained host counters: " + str(pending_host.inspect()))
            for operation in pending_host._owned._operations.values():
                coroutine = operation.task.get_coro()
                names = []
                while coroutine is not None:
                    code = getattr(coroutine, "cr_code", None)
                    if code is not None:
                        names.append(code.co_qualname)
                    coroutine = getattr(coroutine, "cr_await", None)
                error.add_note("Retained await path: " + " -> ".join(names))
        raise
    assert stages == [
        "accepted",
        "coordinator parked",
        "specialist admitted",
        "producer attached",
        "native producer completed",
        "export published",
        "delivery appended and producer settled",
        "continuation consumed",
    ]
    assert result["producer"] == "settled"
    assert result["delivery"] == "appended"
    assert result["continuation"] == "consumed"
    assert result["provider_calls"] == 3
    assert len(raw_rejections) == 1
    assert len(deferred_continuations) == 1
    first, specialist, continuation = result["serialized_requests"]
    # These are the real OpenAI adapter's transport payloads, not ModelRequest
    # doubles or manually fabricated peer serialization.
    assert all(item["model"] == "gpt-5.6" for item in (first, specialist, continuation))
    assert all(not item.get("stream", False) for item in (first, specialist, continuation))
    assert all(item["max_output_tokens"] == 256 for item in (first, specialist, continuation))
    first_input = json.dumps(first["input"])
    specialist_input = json.dumps(specialist["input"])
    resumed_input = json.dumps(continuation["input"])
    assert "DEMO_PARK" in first_input and "DEMO_TASK" not in first_input
    assert "DEMO_TASK" in specialist_input and "DEMO_PARK" not in specialist_input
    assert "DEMO_RESUME" in resumed_input
    assert "Specialist analysis completed for the selected input." in resumed_input
    assert "offline-example-not-a-credential" not in json.dumps(result["serialized_requests"])
