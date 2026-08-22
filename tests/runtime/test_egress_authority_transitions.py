from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

import cayu.runtime.egress_authority_transitions as transition_module
from cayu.core.events import Event, EventType
from cayu.egress import (
    EgressAuthorityBindingIdentity,
    EgressAuthorityCutoverNeedsAttention,
    EgressAuthorityCutoverReceipt,
    EgressAuthorityCutoverRequest,
    EgressAuthorityCutoverResult,
    EgressAuthorityCutoverStrategy,
    EgressAuthorityTransitionState,
    EgressBinding,
    HttpEgressPolicy,
    SandboxEgressAdapter,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
    VirtualEgressRunnerRequest,
    build_egress_authority_cutover_receipt,
    build_egress_authority_identity,
)
from cayu.egress.authority import _build_adapter_verified_egress_authority_cutover_receipt
from cayu.environments.admission import ExecutionEnvironmentAuthority
from cayu.runners.base import ExecCommand, ExecResult, Runner
from cayu.runtime._event_projection import (
    prepare_new_runtime_event,
    project_persisted_runtime_event,
)
from cayu.runtime.approvals import ResolutionActor, ResolutionActorSource
from cayu.runtime.egress import (
    _EgressAuditBridge,
    _EgressAuthorityRevoker,
    _EgressManagedRunner,
)
from cayu.runtime.egress_authority_transitions import (
    EgressAuthorityTransitionConflict,
    EgressAuthorityTransitionCoordinator,
    SessionCheckpointEgressAuthorityTransitionStore,
    advance_egress_authority_transition,
    authorized_egress_authority_transition,
    egress_authority_owner_fingerprint,
    egress_authority_transition_events,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileAuthorityDecision,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    _with_runtime_execution_profile_decision_authority,
    build_execution_profile_identity,
    changed_execution_profile_components,
    execution_profile_decision_payload,
    execution_profile_egress_authority_change,
    execution_profile_with_egress_authority,
)
from cayu.runtime.sessions import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults import SecretRedactor, SecretRef, StaticVault


def _authority(generation: int, *, allow_post: bool, allow_delete: bool = False):
    endpoints = [("GET", "/v1/items")]
    if allow_post:
        endpoints.append(("POST", "/v1/items"))
    if allow_delete:
        endpoints.append(("DELETE", "/v1/items"))
    policy = HttpEgressPolicy(
        name="provider",
        allowed_hosts=("api.example.com",),
        allowed_endpoints=endpoints,
    )
    return build_egress_authority_identity(
        policies={policy.name: policy},
        bindings=(
            EgressAuthorityBindingIdentity(
                destination="api.example.com",
                policy_name=policy.name,
                credential_kind="opaque_bearer",
                credential_authority_fingerprint="1" * 64,
            ),
        ),
        generation=generation,
        authority_source="trusted-app",
        authority_scope="session",
        policy_version=f"v{generation}",
        runner_kind="docker",
        cutover_strategy=EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH,
    )


def _profile(authority):
    return build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="test",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt="system",
        direct_tools=(),
        tool_catalogue_revision=f"sha256:{'c' * 64}",
        egress_authority=authority,
    )


def _decision_between(
    expected_authority,
    target_authority,
    *,
    session_id: str,
    idempotency_identity: str,
    actor_claims: dict | None = None,
) -> ExecutionProfileDecision:
    expected = _profile(expected_authority)
    candidate = execution_profile_with_egress_authority(expected, target_authority)
    changed = changed_execution_profile_components(expected, candidate)
    egress_change = execution_profile_egress_authority_change(expected, candidate)
    actor = ResolutionActor(
        subject="deployment-policy",
        source=ResolutionActorSource.SYSTEM,
        claims={} if actor_claims is None else actor_claims,
    )
    reason = "Adopt the next egress generation."
    payload = execution_profile_decision_payload(
        kind=ExecutionProfileDecisionKind.ADOPTED,
        expected_profile=expected,
        candidate_profile=candidate,
        changed_component_classes=changed,
        policy_identity="trusted-egress-policy:v1",
        policy_reason="The deployment policy authorized the wider destination operations.",
        authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        egress_authority_change=egress_change,
        idempotency_identity=idempotency_identity,
        actor=actor,
        reason=reason,
    )
    event = Event(
        type=EventType.SESSION_EXECUTION_PROFILE_DECIDED,
        session_id=session_id,
        agent_name="assistant",
        environment_name="egress",
        payload=payload,
    )
    return _with_runtime_execution_profile_decision_authority(
        ExecutionProfileDecision(
            kind=ExecutionProfileDecisionKind.ADOPTED,
            expected_profile=expected,
            candidate_profile=candidate,
            changed_component_classes=changed,
            policy_identity="trusted-egress-policy:v1",
            policy_reason="The deployment policy authorized the wider destination operations.",
            authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            egress_authority_change=egress_change,
            idempotency_identity=idempotency_identity,
            actor=actor,
            reason=reason,
            event=event,
        )
    )


def _decision(session_id: str = "egress-transition-session") -> ExecutionProfileDecision:
    return _decision_between(
        _authority(1, allow_post=False),
        _authority(2, allow_post=True),
        session_id=session_id,
        idempotency_identity="egress-transition-1",
    )


async def _create_session(store, session_id: str) -> None:
    await store.create(
        RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )


def _authorized_record(session_id: str = "egress-transition-session"):
    owner_fingerprint = egress_authority_owner_fingerprint("private-owner-token")
    return authorized_egress_authority_transition(
        decision=_decision(session_id),
        transition_id="egress-transition-1",
        environment_name="egress",
        owner_fingerprint=owner_fingerprint,
        source_environment_fingerprint="d" * 64,
    )


class _NoopRunner(Runner):
    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        del command, kwargs
        raise AssertionError("The transition coordinator must not execute guest work.")


class _OutcomeAdapter(SandboxEgressAdapter):
    runner_kind = "docker"
    process_external_allocation = False
    egress_authority_cutover_strategy = EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH

    def __init__(self, cutover, *, reconcile_receipt=None) -> None:
        self._cutover = cutover
        self._reconcile_receipt = reconcile_receipt
        self.calls = 0
        self.reconcile_calls = 0

    async def prepare(self, **kwargs) -> EgressBinding:
        del kwargs
        raise AssertionError("The coordinator must use cutover_authority.")

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        del request
        raise AssertionError("The coordinator must not create a runner.")

    async def cutover_authority(
        self,
        request: EgressAuthorityCutoverRequest,
    ) -> EgressAuthorityCutoverResult:
        self.calls += 1
        return await self._cutover(request)

    async def egress_environment_fingerprint(self, runner: Runner) -> str:
        del runner
        return "d" * 64

    async def reconcile_authority_cutover(self, request):
        del request
        self.reconcile_calls += 1
        return self._reconcile_receipt


def _cutover_material(authorized):
    registry = VirtualCredentialRegistry()
    grant = registry.mint(
        session_id=authorized.session_id,
        env_name="TARGET_TOKEN",
        secret=SecretRef(name="target"),
        destination="api.example.com",
        credential_kind="stripe_bearer",
        policy_name="provider",
    )
    policy = HttpEgressPolicy(
        name="provider",
        allowed_hosts=("api.example.com",),
        allowed_endpoints=(("GET", "/v1/items"), ("POST", "/v1/items")),
    )
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"target": "sk_test_target"}),
        policies={policy.name: policy},
    )
    current = EgressBinding(runner_kind="docker", guest_ca_path="/etc/cayu/ca.pem")
    current.bind_authority(authorized.expected_authority)
    replacement = EgressBinding(runner_kind="docker", guest_ca_path="/etc/cayu/ca.pem")
    replacement.bind_authority(authorized.target_authority)
    revoked: list[bool] = []

    async def revoke_current() -> bool:
        revoked.append(True)
        return False

    request = EgressAuthorityCutoverRequest(
        session_id=authorized.session_id,
        environment_name=authorized.environment_name,
        owner_fingerprint=authorized.owner_fingerprint,
        environment_fingerprint="d" * 64,
        runner=_NoopRunner(),
        current_binding=current,
        expected_authority=authorized.expected_authority,
        target_authority=authorized.target_authority,
        target_broker=broker,
        target_grants=(grant,),
        target_env_overlay={"TARGET_TOKEN": grant.presented_value},
        target_egress_destinations=("api.example.com",),
        revoke_current_authority=revoke_current,
        ca_cert_host_path="/tmp/cayu-test-ca.pem",
        guest_ca_path="/etc/cayu/ca.pem",
        invocation_quiescent=True,
    )
    receipt = _build_adapter_verified_egress_authority_cutover_receipt(
        expected=authorized.expected_authority,
        target=authorized.target_authority,
        environment_fingerprint="d" * 64,
    )
    return request, replacement, receipt, revoked


def test_checkpoint_store_cas_fences_stale_transition_owner() -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        authorized = await store.compare_and_set(expected=None, replacement=_authorized_record())
        assert await store.compare_and_set(expected=None, replacement=authorized) == authorized
        initial_events = await session_store.query_events()
        assert [record.event.type for record in initial_events] == [
            EventType.EGRESS_AUTHORITY_REQUESTED,
            EventType.EGRESS_AUTHORITY_AUTHORIZED,
        ]
        assert initial_events[0].event.payload["revision"] == 1
        assert initial_events[0].event.payload["classification"] == "wider"
        assert initial_events[0].event.payload["actor"]["source"] == "system"
        assert initial_events[0].event.payload["actor"]["subject"] == "deployment-policy"
        assert "secret" not in str(initial_events[0].event.payload).lower()
        installing = advance_egress_authority_transition(
            authorized,
            state=EgressAuthorityTransitionState.INSTALLING,
            reason="Installing.",
            environment_fingerprint="d" * 64,
        )
        await store.compare_and_set(expected=authorized, replacement=installing)
        refused = advance_egress_authority_transition(
            installing,
            state=EgressAuthorityTransitionState.REFUSED,
            reason="Refused.",
        )
        ambiguous = advance_egress_authority_transition(
            installing,
            state=EgressAuthorityTransitionState.AMBIGUOUS,
            reason="Ambiguous.",
        )

        results = await asyncio.gather(
            store.compare_and_set(expected=installing, replacement=refused),
            store.compare_and_set(expected=installing, replacement=ambiguous),
            return_exceptions=True,
        )
        assert sum(isinstance(item, EgressAuthorityTransitionConflict) for item in results) == 1
        assert (await store.load("egress-transition-session")).state in {
            EgressAuthorityTransitionState.REFUSED,
            EgressAuthorityTransitionState.AMBIGUOUS,
        }
        events = await session_store.query_events()
        assert len(events) == 4
        assert events[-2].event.type is EventType.EGRESS_AUTHORITY_INSTALLING
        assert events[-1].event.type in {
            EventType.EGRESS_AUTHORITY_REFUSED,
            EventType.EGRESS_AUTHORITY_AMBIGUOUS,
        }

    asyncio.run(exercise())


def test_sqlite_restart_reconstructs_installing_and_reconciles_exact_receipt(tmp_path) -> None:
    async def exercise() -> None:
        database = tmp_path / "egress-authority.sqlite3"
        session_store = SQLiteSessionStore(database)
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        installing = advance_egress_authority_transition(
            authorized,
            state=EgressAuthorityTransitionState.INSTALLING,
            reason="Installing.",
            environment_fingerprint="d" * 64,
        )
        await store.compare_and_set(expected=authorized, replacement=installing)
        await session_store.close()

        restarted_session_store = SQLiteSessionStore(database)
        restarted_store = SessionCheckpointEgressAuthorityTransitionStore(restarted_session_store)
        restarted = await restarted_store.load("egress-transition-session")
        assert restarted == installing
        request, _replacement, receipt, _revoked = _cutover_material(installing)
        active = await EgressAuthorityTransitionCoordinator(restarted_store).reconcile(
            current=installing,
            adapter=_OutcomeAdapter(lambda _request: None, reconcile_receipt=receipt),
            request=request,
            owner_token="private-owner-token",
        )
        assert active.state is EgressAuthorityTransitionState.ACTIVE
        assert active.receipt == receipt
        await restarted_session_store.close()

    asyncio.run(exercise())


def test_recovery_without_exact_receipt_persists_ambiguous() -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        authorized = await store.compare_and_set(expected=None, replacement=_authorized_record())
        installing = advance_egress_authority_transition(
            authorized,
            state=EgressAuthorityTransitionState.INSTALLING,
            reason="Installing.",
            environment_fingerprint="d" * 64,
        )
        await store.compare_and_set(expected=authorized, replacement=installing)
        request, _replacement, _receipt, _revoked = _cutover_material(installing)
        ambiguous = await EgressAuthorityTransitionCoordinator(store).reconcile(
            current=installing,
            adapter=_OutcomeAdapter(lambda _request: None),
            request=request,
            owner_token="private-owner-token",
        )
        assert ambiguous.state is EgressAuthorityTransitionState.AMBIGUOUS
        assert ambiguous.receipt is None

    asyncio.run(exercise())


def test_wider_transition_cannot_be_built_from_copied_runtime_decision() -> None:
    decision = _decision().model_copy(
        update={"authority_decision": ExecutionProfileAuthorityDecision.NOT_REQUIRED}
    )
    with pytest.raises(ValueError, match="runtime-owned profile decision"):
        authorized_egress_authority_transition(
            decision=decision,
            transition_id="unauthorized",
            environment_name="egress",
            owner_fingerprint="a" * 64,
            source_environment_fingerprint="d" * 64,
        )


def test_transition_durable_actor_excludes_authorization_claims() -> None:
    decision = _decision_between(
        _authority(1, allow_post=False),
        _authority(2, allow_post=True),
        session_id="egress-transition-session",
        idempotency_identity="claims-transition",
        actor_claims={"scope": "secret-authorization-claim"},
    )
    record = authorized_egress_authority_transition(
        decision=decision,
        transition_id="claims-transition",
        environment_name="egress",
        owner_fingerprint=egress_authority_owner_fingerprint("private-owner-token"),
        source_environment_fingerprint="d" * 64,
    )

    assert record.actor.claims == {}
    assert "claims" not in egress_authority_transition_events(record)[0].payload["actor"]


def test_identically_shaped_caller_decision_cannot_authorize_cutover() -> None:
    trusted = _decision()
    untrusted = ExecutionProfileDecision.model_validate(trusted.model_dump(mode="json"))

    with pytest.raises(ValueError, match="runtime-owned profile decision"):
        authorized_egress_authority_transition(
            decision=untrusted,
            transition_id="untrusted-copy",
            environment_name="egress",
            owner_fingerprint="a" * 64,
            source_environment_fingerprint="d" * 64,
        )


def test_mutated_runtime_decision_cannot_retain_authorization() -> None:
    trusted = _decision()
    replacement = _decision_between(
        _authority(1, allow_post=False),
        _authority(3, allow_post=True, allow_delete=True),
        session_id="egress-transition-session",
        idempotency_identity="retargeted-transition",
    )
    untrusted_replacement = ExecutionProfileDecision.model_validate(
        replacement.model_dump(mode="json")
    )
    for field_name in ExecutionProfileDecision.model_fields:
        object.__setattr__(trusted, field_name, getattr(untrusted_replacement, field_name))

    with pytest.raises(ValueError, match="runtime-owned profile decision"):
        authorized_egress_authority_transition(
            decision=trusted,
            transition_id="retargeted-transition",
            environment_name="egress",
            owner_fingerprint=egress_authority_owner_fingerprint("private-owner-token"),
            source_environment_fingerprint="d" * 64,
        )


def test_failure_before_backend_commit_is_durably_refused_and_replay_safe() -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        request, _replacement, _receipt, revoked = _cutover_material(authorized)

        async def fail_before_commit(_request):
            raise RuntimeError("staging failed")

        adapter = _OutcomeAdapter(fail_before_commit)
        with pytest.raises(RuntimeError, match="staging failed"):
            await coordinator.install(
                authorized=authorized,
                adapter=adapter,
                request=request,
                owner_token="private-owner-token",
            )
        refused = await store.load(authorized.session_id)
        assert refused is not None
        assert refused.state is EgressAuthorityTransitionState.REFUSED
        assert refused.receipt is None
        assert revoked == []
        assert adapter.calls == 1
        assert (
            await store.compare_and_set(
                expected=authorized,
                replacement=refused,
            )
            == refused
        )

    asyncio.run(exercise())


def test_refused_transition_retry_cannot_substitute_source_allocation() -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        refused = advance_egress_authority_transition(
            authorized,
            state=EgressAuthorityTransitionState.REFUSED,
            reason="Refused before backend mutation.",
        )
        await store.compare_and_set(expected=authorized, replacement=refused)

        substituted = authorized_egress_authority_transition(
            decision=_decision(authorized.session_id),
            transition_id="egress-transition-source-substitution",
            environment_name=authorized.environment_name,
            owner_fingerprint=egress_authority_owner_fingerprint("replacement-owner-token"),
            source_environment_fingerprint="e" * 64,
        )
        with pytest.raises(
            EgressAuthorityTransitionConflict,
            match="authoritative generation",
        ):
            await coordinator.authorize(substituted)

        assert await store.load(authorized.session_id) == refused

    asyncio.run(exercise())


def test_failure_during_old_authority_revocation_is_durably_ambiguous() -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        request, replacement, _receipt, revoked = _cutover_material(authorized)

        async def lose_boundary(cutover_request):
            await cutover_request.revoke_current_authority()
            raise EgressAuthorityCutoverNeedsAttention(
                "revocation acknowledgement was lost",
                replacement_binding=replacement,
                environment_fingerprint="d" * 64,
            )

        with pytest.raises(EgressAuthorityCutoverNeedsAttention):
            await coordinator.install(
                authorized=authorized,
                adapter=_OutcomeAdapter(lose_boundary),
                request=request,
                owner_token="private-owner-token",
            )
        ambiguous = await store.load(authorized.session_id)
        assert ambiguous is not None
        assert ambiguous.state is EgressAuthorityTransitionState.AMBIGUOUS
        assert ambiguous.environment_fingerprint == "d" * 64
        assert revoked == [True]

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel_before_dispatch", (True, False))
def test_managed_cutover_redelivers_cancellation_after_durable_activation(
    cancel_before_dispatch: bool,
    tmp_path,
) -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        material, replacement, receipt, _revoked = _cutover_material(authorized)
        dispatched = asyncio.Event()
        release = asyncio.Event()

        async def activate(_request):
            dispatched.set()
            await release.wait()
            return EgressAuthorityCutoverResult(binding=replacement, receipt=receipt)

        adapter = _OutcomeAdapter(activate)
        current_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        target_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        audit = _EgressAuditBridge(
            loop=asyncio.get_running_loop(),
            emitter=None,
            session_id=authorized.session_id,
            agent_name="assistant",
            environment_name=authorized.environment_name,
            execution_profile_fingerprint=None,
        )
        managed = _EgressManagedRunner(
            runner=material.runner,
            adapter=adapter,
            execution_environment_authority=ExecutionEnvironmentAuthority(
                identity="process_test_egress_cutover",
            ),
            egress_binding=material.current_binding,
            ca_dir=str(tmp_path / "ca"),
            authority_revoker=current_revoker,
            egress_grants=material.target_grants,
            egress_destinations=material.target_egress_destinations,
            output_redactor=SecretRedactor(),
            session_id=authorized.session_id,
            environment_name=authorized.environment_name,
            environment_fingerprint=material.environment_fingerprint,
            audit=audit,
        )
        task = asyncio.create_task(
            managed._adopt_egress_authority(
                coordinator=coordinator,
                authorized=authorized,
                owner_token="private-owner-token",
                target_authority=authorized.target_authority,
                target_broker=material.target_broker,
                target_grants=material.target_grants,
                target_credential_env=dict(material.target_env_overlay),
                target_egress_destinations=material.target_egress_destinations,
                target_revoker=target_revoker,
                target_redactor=SecretRedactor(),
                target_audit=audit,
            )
        )
        if cancel_before_dispatch:
            task.cancel("stop-before-dispatch")
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError, match="stop-before-dispatch"):
                await task
            assert task.cancelled()
            assert adapter.calls == 0
            assert await store.load(authorized.session_id) == authorized
            return
        await dispatched.wait()
        task.cancel("stop-after-dispatch")
        assert task.cancelling() == 1
        release.set()

        with pytest.raises(asyncio.CancelledError, match="stop-after-dispatch"):
            await task
        assert task.cancelled()
        assert adapter.calls == 1
        active = await store.load(authorized.session_id)
        assert active is not None
        assert active.state is EgressAuthorityTransitionState.ACTIVE
        assert managed._egress_binding is replacement

    asyncio.run(exercise())


def test_managed_cutover_redelivers_adapter_consumed_cancellation(tmp_path) -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        material, replacement, receipt, _revoked = _cutover_material(authorized)

        async def activate(_request):
            return EgressAuthorityCutoverResult(
                binding=replacement,
                receipt=receipt,
                cancellation=asyncio.CancelledError("cancelled-during-revocation"),
                cancellation_requests_consumed=1,
            )

        adapter = _OutcomeAdapter(activate)
        current_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        target_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        audit = _EgressAuditBridge(
            loop=asyncio.get_running_loop(),
            emitter=None,
            session_id=authorized.session_id,
            agent_name="assistant",
            environment_name=authorized.environment_name,
            execution_profile_fingerprint=None,
        )
        managed = _EgressManagedRunner(
            runner=material.runner,
            adapter=adapter,
            execution_environment_authority=ExecutionEnvironmentAuthority(
                identity="process_test_adapter_cancellation",
            ),
            egress_binding=material.current_binding,
            ca_dir=str(tmp_path / "ca"),
            authority_revoker=current_revoker,
            egress_grants=material.target_grants,
            egress_destinations=material.target_egress_destinations,
            output_redactor=SecretRedactor(),
            session_id=authorized.session_id,
            environment_name=authorized.environment_name,
            environment_fingerprint=material.environment_fingerprint,
            audit=audit,
        )
        task = asyncio.create_task(
            managed._adopt_egress_authority(
                coordinator=coordinator,
                authorized=authorized,
                owner_token="private-owner-token",
                target_authority=authorized.target_authority,
                target_broker=material.target_broker,
                target_grants=material.target_grants,
                target_credential_env=dict(material.target_env_overlay),
                target_egress_destinations=material.target_egress_destinations,
                target_revoker=target_revoker,
                target_redactor=SecretRedactor(),
                target_audit=audit,
            )
        )

        with pytest.raises(asyncio.CancelledError, match="cancelled-during-revocation"):
            await task

        assert task.cancelled()
        assert task.cancelling() == 1
        active = await store.load(authorized.session_id)
        assert active is not None
        assert active.state is EgressAuthorityTransitionState.ACTIVE
        assert managed._egress_binding is replacement

    asyncio.run(exercise())


def test_managed_cutover_cancellation_during_identity_preflight_cleans_target(tmp_path) -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        material, _replacement, _receipt, _revoked = _cutover_material(authorized)
        preflight_started = asyncio.Event()
        never_complete = asyncio.Event()

        async def must_not_dispatch(_request):
            raise AssertionError("Cancelled preflight reached adapter mutation.")

        adapter = _OutcomeAdapter(must_not_dispatch)

        async def block_identity_readback(_runner):
            preflight_started.set()
            await never_complete.wait()
            return material.environment_fingerprint

        adapter.egress_environment_fingerprint = block_identity_readback  # type: ignore[method-assign]
        current_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        target_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        audit = _EgressAuditBridge(
            loop=asyncio.get_running_loop(),
            emitter=None,
            session_id=authorized.session_id,
            agent_name="assistant",
            environment_name=authorized.environment_name,
            execution_profile_fingerprint=None,
        )
        managed = _EgressManagedRunner(
            runner=material.runner,
            adapter=adapter,
            execution_environment_authority=ExecutionEnvironmentAuthority(
                identity="process_test_preflight_cancellation",
            ),
            egress_binding=material.current_binding,
            ca_dir=str(tmp_path / "ca"),
            authority_revoker=current_revoker,
            egress_grants=material.target_grants,
            egress_destinations=material.target_egress_destinations,
            output_redactor=SecretRedactor(),
            session_id=authorized.session_id,
            environment_name=authorized.environment_name,
            environment_fingerprint=material.environment_fingerprint,
            audit=audit,
        )
        task = asyncio.create_task(
            managed._adopt_egress_authority(
                coordinator=coordinator,
                authorized=authorized,
                owner_token="private-owner-token",
                target_authority=authorized.target_authority,
                target_broker=material.target_broker,
                target_grants=material.target_grants,
                target_credential_env=dict(material.target_env_overlay),
                target_egress_destinations=material.target_egress_destinations,
                target_revoker=target_revoker,
                target_redactor=SecretRedactor(),
                target_audit=audit,
            )
        )
        await preflight_started.wait()
        task.cancel("cancel-during-environment-readback")
        assert task.cancelling() == 1

        with pytest.raises(
            asyncio.CancelledError,
            match="cancel-during-environment-readback",
        ):
            await task

        assert task.cancelled()
        assert task.cancelling() == 1
        assert target_revoker._revoked is True
        assert adapter.calls == 0
        assert await store.load(authorized.session_id) == authorized
        assert managed._egress_cutover_needs_attention is False

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "fatal_cause",
    (
        KeyboardInterrupt("fatal-cutover-signal"),
        BaseExceptionGroup(
            "cutover control signals",
            [asyncio.CancelledError("child-cancellation"), SystemExit("fatal-cutover-signal")],
        ),
    ),
)
def test_managed_cutover_preserves_fatal_signal_after_durable_ambiguity(
    fatal_cause: BaseException,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        material, _replacement, _receipt, _revoked = _cutover_material(authorized)

        async def fail_after_dispatch(_request):
            attention = EgressAuthorityCutoverNeedsAttention(
                "backend mutation outcome is unknown",
                replacement_binding=material.current_binding,
                environment_fingerprint=material.environment_fingerprint,
                target_authority_installed=False,
            )
            raise attention from fatal_cause

        adapter = _OutcomeAdapter(fail_after_dispatch)
        current_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        target_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )

        async def fail_target_cleanup(**_kwargs) -> bool:
            raise RuntimeError("target cleanup failed")

        monkeypatch.setattr(target_revoker, "revoke", fail_target_cleanup)
        audit = _EgressAuditBridge(
            loop=asyncio.get_running_loop(),
            emitter=None,
            session_id=authorized.session_id,
            agent_name="assistant",
            environment_name=authorized.environment_name,
            execution_profile_fingerprint=None,
        )
        managed = _EgressManagedRunner(
            runner=material.runner,
            adapter=adapter,
            execution_environment_authority=ExecutionEnvironmentAuthority(
                identity="process_test_fatal_cutover_signal",
            ),
            egress_binding=material.current_binding,
            ca_dir=str(tmp_path / "ca"),
            authority_revoker=current_revoker,
            egress_grants=material.target_grants,
            egress_destinations=material.target_egress_destinations,
            output_redactor=SecretRedactor(),
            session_id=authorized.session_id,
            environment_name=authorized.environment_name,
            environment_fingerprint=material.environment_fingerprint,
            audit=audit,
        )

        expected_fatal_type = (
            KeyboardInterrupt if isinstance(fatal_cause, KeyboardInterrupt) else SystemExit
        )
        with pytest.raises(expected_fatal_type, match="fatal-cutover-signal"):
            await managed._adopt_egress_authority(
                coordinator=coordinator,
                authorized=authorized,
                owner_token="private-owner-token",
                target_authority=authorized.target_authority,
                target_broker=material.target_broker,
                target_grants=material.target_grants,
                target_credential_env=dict(material.target_env_overlay),
                target_egress_destinations=material.target_egress_destinations,
                target_revoker=target_revoker,
                target_redactor=SecretRedactor(),
                target_audit=audit,
            )

        ambiguous = await store.load(authorized.session_id)
        assert ambiguous is not None
        assert ambiguous.state is EgressAuthorityTransitionState.AMBIGUOUS
        assert managed._egress_cutover_needs_attention is True

    asyncio.run(exercise())


def test_installing_commit_cancellation_refuses_before_adapter_dispatch() -> None:
    class CommitThenBlockInstallingStore(SessionCheckpointEgressAuthorityTransitionStore):
        def __init__(self, session_store) -> None:
            super().__init__(session_store)
            self.committed = asyncio.Event()
            self.release = asyncio.Event()

        async def compare_and_set(self, *, expected, replacement):
            persisted = await super().compare_and_set(
                expected=expected,
                replacement=replacement,
            )
            if replacement.state is EgressAuthorityTransitionState.INSTALLING:
                self.committed.set()
                await self.release.wait()
            return persisted

    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = CommitThenBlockInstallingStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        request, _replacement, _receipt, _revoked = _cutover_material(authorized)

        async def must_not_dispatch(_request):
            raise AssertionError("Cancellation after INSTALLING reached the adapter.")

        adapter = _OutcomeAdapter(must_not_dispatch)
        task = asyncio.create_task(
            coordinator.install(
                authorized=authorized,
                adapter=adapter,
                request=request,
                owner_token="private-owner-token",
            )
        )
        await store.committed.wait()
        task.cancel("stop-after-installing-commit")
        assert task.cancelling() == 1
        store.release.set()

        with pytest.raises(asyncio.CancelledError, match="stop-after-installing-commit"):
            await task
        assert task.cancelled()
        assert adapter.calls == 0
        refused = await store.load(authorized.session_id)
        assert refused is not None
        assert refused.state is EgressAuthorityTransitionState.REFUSED
        assert refused.receipt is None

    asyncio.run(exercise())


def test_managed_cutover_preserves_cancellation_during_active_acknowledgement(tmp_path) -> None:
    class CommitThenBlockActiveStore(SessionCheckpointEgressAuthorityTransitionStore):
        def __init__(self, session_store) -> None:
            super().__init__(session_store)
            self.active_committed = asyncio.Event()
            self.release_acknowledgement = asyncio.Event()

        async def compare_and_set(self, *, expected, replacement):
            persisted = await super().compare_and_set(
                expected=expected,
                replacement=replacement,
            )
            if replacement.state is EgressAuthorityTransitionState.ACTIVE:
                self.active_committed.set()
                await self.release_acknowledgement.wait()
            return persisted

    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = CommitThenBlockActiveStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        material, replacement, receipt, _revoked = _cutover_material(authorized)

        async def activate(_request):
            return EgressAuthorityCutoverResult(binding=replacement, receipt=receipt)

        adapter = _OutcomeAdapter(activate)
        current_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        target_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        audit = _EgressAuditBridge(
            loop=asyncio.get_running_loop(),
            emitter=None,
            session_id=authorized.session_id,
            agent_name="assistant",
            environment_name=authorized.environment_name,
            execution_profile_fingerprint=None,
        )
        managed = _EgressManagedRunner(
            runner=material.runner,
            adapter=adapter,
            execution_environment_authority=ExecutionEnvironmentAuthority(
                identity="process_test_active_ack_cancellation",
            ),
            egress_binding=material.current_binding,
            ca_dir=str(tmp_path / "ca"),
            authority_revoker=current_revoker,
            egress_grants=material.target_grants,
            egress_destinations=material.target_egress_destinations,
            output_redactor=SecretRedactor(),
            session_id=authorized.session_id,
            environment_name=authorized.environment_name,
            environment_fingerprint=material.environment_fingerprint,
            audit=audit,
        )
        task = asyncio.create_task(
            managed._adopt_egress_authority(
                coordinator=coordinator,
                authorized=authorized,
                owner_token="private-owner-token",
                target_authority=authorized.target_authority,
                target_broker=material.target_broker,
                target_grants=material.target_grants,
                target_credential_env=dict(material.target_env_overlay),
                target_egress_destinations=material.target_egress_destinations,
                target_revoker=target_revoker,
                target_redactor=SecretRedactor(),
                target_audit=audit,
            )
        )
        await store.active_committed.wait()
        task.cancel("stop-during-active-acknowledgement")
        assert task.cancelling() == 1
        store.release_acknowledgement.set()

        with pytest.raises(
            asyncio.CancelledError,
            match="stop-during-active-acknowledgement",
        ):
            await task
        assert task.cancelled()
        active = await store.load(authorized.session_id)
        assert active is not None
        assert active.state is EgressAuthorityTransitionState.ACTIVE
        assert managed._egress_binding is replacement
        assert managed._egress_cutover_settlement is None

    asyncio.run(exercise())


def test_finalization_cannot_overlap_retained_cutover_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    async def exercise() -> None:
        monkeypatch.setattr(
            transition_module,
            "DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS",
            0.01,
        )
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        material, replacement, receipt, _revoked = _cutover_material(authorized)
        dispatched = asyncio.Event()
        release = asyncio.Event()

        async def activate(_request):
            dispatched.set()
            await release.wait()
            return EgressAuthorityCutoverResult(binding=replacement, receipt=receipt)

        adapter = _OutcomeAdapter(activate)
        current_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        target_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        audit = _EgressAuditBridge(
            loop=asyncio.get_running_loop(),
            emitter=None,
            session_id=authorized.session_id,
            agent_name="assistant",
            environment_name=authorized.environment_name,
            execution_profile_fingerprint=None,
        )
        managed = _EgressManagedRunner(
            runner=material.runner,
            adapter=adapter,
            execution_environment_authority=ExecutionEnvironmentAuthority(
                identity="process_test_retained_egress_cutover",
            ),
            egress_binding=material.current_binding,
            ca_dir=str(tmp_path / "ca"),
            authority_revoker=current_revoker,
            egress_grants=material.target_grants,
            egress_destinations=material.target_egress_destinations,
            output_redactor=SecretRedactor(),
            session_id=authorized.session_id,
            environment_name=authorized.environment_name,
            environment_fingerprint=material.environment_fingerprint,
            audit=audit,
        )

        with pytest.raises(EgressAuthorityCutoverNeedsAttention):
            await managed._adopt_egress_authority(
                coordinator=coordinator,
                authorized=authorized,
                owner_token="private-owner-token",
                target_authority=authorized.target_authority,
                target_broker=material.target_broker,
                target_grants=material.target_grants,
                target_credential_env=dict(material.target_env_overlay),
                target_egress_destinations=material.target_egress_destinations,
                target_revoker=target_revoker,
                target_redactor=SecretRedactor(),
                target_audit=audit,
            )
        await dispatched.wait()
        assert managed._egress_cutover_settlement is not None
        settlement_task = managed._egress_cutover_settlement.task
        managed._teardown_timeout_s = 0.01

        with pytest.raises(EgressAuthorityCutoverNeedsAttention, match="cleanup remains fenced"):
            await managed.finalize(outcome="failed")
        assert not managed.closed
        assert not settlement_task.done()

        release.set()
        await settlement_task
        managed._teardown_timeout_s = 1
        await managed.finalize(outcome="failed")
        assert managed.closed
        assert managed._egress_cutover_settlement is None

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "late_signal",
    ("none", "cancellation", "fatal", "cancellation_then_reconcile_fatal"),
)
def test_retry_harvests_late_cutover_settlement_without_finalization(
    monkeypatch: pytest.MonkeyPatch,
    late_signal: str,
    tmp_path,
) -> None:
    async def exercise() -> None:
        monkeypatch.setattr(
            transition_module,
            "DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS",
            0.01,
        )
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        material, replacement, receipt, _revoked = _cutover_material(authorized)
        dispatched = asyncio.Event()
        release = asyncio.Event()

        async def activate(_request):
            dispatched.set()
            await release.wait()
            if late_signal == "fatal":
                attention = EgressAuthorityCutoverNeedsAttention(
                    "late cutover ended with a fatal backend signal",
                    replacement_binding=replacement,
                    environment_fingerprint=material.environment_fingerprint,
                    target_authority_installed=True,
                )
                raise attention from KeyboardInterrupt("late-cutover-fatal")
            return EgressAuthorityCutoverResult(
                binding=replacement,
                receipt=receipt,
                cancellation=(
                    asyncio.CancelledError("late-cutover-cancellation")
                    if late_signal in {"cancellation", "cancellation_then_reconcile_fatal"}
                    else None
                ),
                cancellation_requests_consumed=(
                    1 if late_signal in {"cancellation", "cancellation_then_reconcile_fatal"} else 0
                ),
            )

        adapter = _OutcomeAdapter(activate, reconcile_receipt=receipt)
        current_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        target_revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        audit = _EgressAuditBridge(
            loop=asyncio.get_running_loop(),
            emitter=None,
            session_id=authorized.session_id,
            agent_name="assistant",
            environment_name=authorized.environment_name,
            execution_profile_fingerprint=None,
        )
        managed = _EgressManagedRunner(
            runner=material.runner,
            adapter=adapter,
            execution_environment_authority=ExecutionEnvironmentAuthority(
                identity="process_test_late_retry_settlement",
            ),
            egress_binding=material.current_binding,
            ca_dir=str(tmp_path / "ca"),
            authority_revoker=current_revoker,
            egress_grants=material.target_grants,
            egress_destinations=material.target_egress_destinations,
            output_redactor=SecretRedactor(),
            session_id=authorized.session_id,
            environment_name=authorized.environment_name,
            environment_fingerprint=material.environment_fingerprint,
            audit=audit,
        )

        with pytest.raises(EgressAuthorityCutoverNeedsAttention):
            await managed._adopt_egress_authority(
                coordinator=coordinator,
                authorized=authorized,
                owner_token="private-owner-token",
                target_authority=authorized.target_authority,
                target_broker=material.target_broker,
                target_grants=material.target_grants,
                target_credential_env=dict(material.target_env_overlay),
                target_egress_destinations=material.target_egress_destinations,
                target_revoker=target_revoker,
                target_redactor=SecretRedactor(),
                target_audit=audit,
            )
        await dispatched.wait()
        assert managed._egress_cutover_settlement is not None

        release.set()
        if late_signal == "fatal":
            with pytest.raises(EgressAuthorityCutoverNeedsAttention):
                await managed._egress_cutover_settlement.task
        else:
            await managed._egress_cutover_settlement.task
        current = await store.load(authorized.session_id)
        assert current is not None
        assert current.state is EgressAuthorityTransitionState.AMBIGUOUS

        if late_signal == "cancellation_then_reconcile_fatal":

            async def fatal_readback(_request):
                raise RuntimeError("backend readback failed") from SystemExit(
                    "late-reconciliation-fatal"
                )

            monkeypatch.setattr(
                adapter,
                "reconcile_authority_cutover",
                fatal_readback,
            )
            with pytest.raises(SystemExit, match="late-reconciliation-fatal"):
                await managed._reconcile_egress_authority(
                    coordinator=coordinator,
                    authorized=current,
                    owner_token="private-owner-token",
                    target_authority=authorized.target_authority,
                )
            assert managed._egress_cutover_needs_attention is True
            assert managed._egress_cutover_settlement is not None
            return
        if late_signal == "fatal":
            with pytest.raises(KeyboardInterrupt, match="late-cutover-fatal"):
                await managed._reconcile_egress_authority(
                    coordinator=coordinator,
                    authorized=current,
                    owner_token="private-owner-token",
                    target_authority=authorized.target_authority,
                )
            assert managed._egress_cutover_needs_attention is True
            assert managed._egress_cutover_settlement is not None
            return
        if late_signal == "cancellation":
            retry_task = asyncio.create_task(
                managed._reconcile_egress_authority(
                    coordinator=coordinator,
                    authorized=current,
                    owner_token="private-owner-token",
                    target_authority=authorized.target_authority,
                )
            )
            with pytest.raises(asyncio.CancelledError, match="late-cutover-cancellation"):
                await retry_task
            assert retry_task.cancelled()
            assert retry_task.cancelling() == 1
            active = await store.load(authorized.session_id)
            assert active is not None
        else:
            active = await managed._reconcile_egress_authority(
                coordinator=coordinator,
                authorized=current,
                owner_token="private-owner-token",
                target_authority=authorized.target_authority,
            )

        assert active.state is EgressAuthorityTransitionState.ACTIVE
        assert managed._egress_binding is replacement
        assert managed._egress_cutover_settlement is None
        assert managed._egress_cutover_needs_attention is False
        assert adapter.calls == 1
        assert adapter.reconcile_calls == 1

    asyncio.run(exercise())


def test_backend_install_with_lost_ack_reconciles_only_exact_receipt() -> None:
    class LoseFirstActiveAckStore(SessionCheckpointEgressAuthorityTransitionStore):
        def __init__(self, session_store) -> None:
            super().__init__(session_store)
            self.lost = False

        async def compare_and_set(self, *, expected, replacement):
            if replacement.state is EgressAuthorityTransitionState.ACTIVE and not self.lost:
                self.lost = True
                raise RuntimeError("simulated acknowledgement loss")
            return await super().compare_and_set(expected=expected, replacement=replacement)

    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = LoseFirstActiveAckStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        request, replacement, receipt, _revoked = _cutover_material(authorized)

        async def activate(_request):
            return EgressAuthorityCutoverResult(binding=replacement, receipt=receipt)

        with pytest.raises(EgressAuthorityCutoverNeedsAttention) as captured:
            await coordinator.install(
                authorized=authorized,
                adapter=_OutcomeAdapter(activate),
                request=request,
                owner_token="private-owner-token",
            )
        assert captured.value.replacement_binding is replacement
        ambiguous = await store.load(authorized.session_id)
        assert ambiguous is not None
        assert ambiguous.state is EgressAuthorityTransitionState.AMBIGUOUS
        wrong_environment_receipt = _build_adapter_verified_egress_authority_cutover_receipt(
            expected=authorized.expected_authority,
            target=authorized.target_authority,
            environment_fingerprint="e" * 64,
        )
        with pytest.raises(EgressAuthorityTransitionConflict, match="different backend"):
            await coordinator.reconcile(
                current=ambiguous,
                adapter=_OutcomeAdapter(
                    activate,
                    reconcile_receipt=wrong_environment_receipt,
                ),
                request=request,
                owner_token="private-owner-token",
            )
        active = await coordinator.reconcile(
            current=ambiguous,
            adapter=_OutcomeAdapter(activate, reconcile_receipt=receipt),
            request=request,
            owner_token="private-owner-token",
        )
        assert active.state is EgressAuthorityTransitionState.ACTIVE
        assert active.receipt == receipt

    asyncio.run(exercise())


def test_managed_runner_reconciles_persisted_installing_with_current_resources(tmp_path) -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        material, replacement, receipt, _revoked = _cutover_material(authorized)
        installing = advance_egress_authority_transition(
            authorized,
            state=EgressAuthorityTransitionState.INSTALLING,
            reason="The prior worker lost acknowledgement after backend installation.",
            environment_fingerprint=material.environment_fingerprint,
        )
        installing = await store.compare_and_set(
            expected=authorized,
            replacement=installing,
        )
        adapter = _OutcomeAdapter(
            lambda _request: None,
            reconcile_receipt=receipt,
        )
        revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        audit = _EgressAuditBridge(
            loop=asyncio.get_running_loop(),
            emitter=None,
            session_id=authorized.session_id,
            agent_name="assistant",
            environment_name=authorized.environment_name,
            execution_profile_fingerprint=None,
        )
        managed = _EgressManagedRunner(
            runner=material.runner,
            adapter=adapter,
            execution_environment_authority=ExecutionEnvironmentAuthority(
                identity="process_test_restart_reconciliation",
            ),
            egress_binding=replacement,
            ca_dir=str(tmp_path / "ca"),
            authority_revoker=revoker,
            egress_grants=material.target_grants,
            egress_destinations=material.target_egress_destinations,
            output_redactor=SecretRedactor(),
            session_id=authorized.session_id,
            environment_name=authorized.environment_name,
            environment_fingerprint=material.environment_fingerprint,
            audit=audit,
        )

        active = await managed._reconcile_egress_authority(
            coordinator=coordinator,
            authorized=installing,
            owner_token="private-owner-token",
            target_authority=authorized.target_authority,
        )

        assert active.state is EgressAuthorityTransitionState.ACTIVE
        assert adapter.reconcile_calls == 1
        assert await store.load(authorized.session_id) == active
        assert managed._egress_cutover_needs_attention is False

    asyncio.run(exercise())


def test_stale_owner_cannot_begin_or_acknowledge_authorized_cutover() -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        request, replacement, receipt, _revoked = _cutover_material(authorized)
        stale_request = EgressAuthorityCutoverRequest(
            **{
                **request.__dict__,
                "owner_fingerprint": egress_authority_owner_fingerprint("stale-owner"),
            }
        )

        async def activate(_request):
            return EgressAuthorityCutoverResult(binding=replacement, receipt=receipt)

        adapter = _OutcomeAdapter(activate)
        with pytest.raises(EgressAuthorityTransitionConflict, match="owner"):
            await coordinator.install(
                authorized=authorized,
                adapter=adapter,
                request=stale_request,
                owner_token="stale-owner",
            )
        assert adapter.calls == 0
        assert await store.load(authorized.session_id) == authorized

        installing = advance_egress_authority_transition(
            authorized,
            state=EgressAuthorityTransitionState.INSTALLING,
            reason="Installing.",
            environment_fingerprint=request.environment_fingerprint,
        )
        await store.compare_and_set(expected=authorized, replacement=installing)
        with pytest.raises(EgressAuthorityTransitionConflict, match="owner"):
            await coordinator.reconcile(
                current=installing,
                adapter=adapter,
                request=stale_request,
                owner_token="stale-owner",
            )
        assert adapter.reconcile_calls == 0
        assert await store.load(authorized.session_id) == installing

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("managed_session_id", "managed_environment_name"),
    (("another-session", "egress"), ("egress-transition-session", "another-environment")),
)
def test_managed_cutover_rejects_wrong_session_or_environment_owner(
    managed_session_id: str,
    managed_environment_name: str,
    tmp_path,
) -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        material, _replacement, _receipt, _revoked = _cutover_material(authorized)

        async def must_not_dispatch(_request):
            raise AssertionError("Cross-session cutover reached the adapter.")

        adapter = _OutcomeAdapter(must_not_dispatch)
        revoker = _EgressAuthorityRevoker(
            grants=material.target_grants,
            broker=material.target_broker,
        )
        audit = _EgressAuditBridge(
            loop=asyncio.get_running_loop(),
            emitter=None,
            session_id=managed_session_id,
            agent_name="assistant",
            environment_name=managed_environment_name,
            execution_profile_fingerprint=None,
        )
        managed = _EgressManagedRunner(
            runner=material.runner,
            adapter=adapter,
            execution_environment_authority=ExecutionEnvironmentAuthority(
                identity="process_cross_session_test",
            ),
            egress_binding=material.current_binding,
            ca_dir=str(tmp_path / "ca"),
            authority_revoker=revoker,
            egress_grants=material.target_grants,
            egress_destinations=material.target_egress_destinations,
            output_redactor=SecretRedactor(),
            session_id=managed_session_id,
            environment_name=managed_environment_name,
            environment_fingerprint=material.environment_fingerprint,
            audit=audit,
        )

        with pytest.raises(EgressAuthorityTransitionConflict, match="session environment"):
            await managed._adopt_egress_authority(
                coordinator=coordinator,
                authorized=authorized,
                owner_token="private-owner-token",
                target_authority=authorized.target_authority,
                target_broker=material.target_broker,
                target_grants=material.target_grants,
                target_credential_env=dict(material.target_env_overlay),
                target_egress_destinations=material.target_egress_destinations,
                target_revoker=revoker,
                target_redactor=SecretRedactor(),
                target_audit=audit,
            )
        assert adapter.calls == 0
        assert await store.load(authorized.session_id) == authorized

    asyncio.run(exercise())


def test_caller_constructed_receipt_cannot_activate_recovery() -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        request, _replacement, _receipt, _revoked = _cutover_material(authorized)
        installing = advance_egress_authority_transition(
            authorized,
            state=EgressAuthorityTransitionState.INSTALLING,
            reason="Installing.",
            environment_fingerprint=request.environment_fingerprint,
        )
        await store.compare_and_set(expected=authorized, replacement=installing)
        forged = build_egress_authority_cutover_receipt(
            expected=authorized.expected_authority,
            target=authorized.target_authority,
            environment_fingerprint=request.environment_fingerprint,
        )
        adapter = _OutcomeAdapter(lambda _request: None, reconcile_receipt=forged)

        with pytest.raises(EgressAuthorityTransitionConflict, match="adapter-owned"):
            await coordinator.reconcile(
                current=installing,
                adapter=adapter,
                request=request,
                owner_token="private-owner-token",
            )
        assert adapter.reconcile_calls == 1
        assert await store.load(authorized.session_id) == installing

    asyncio.run(exercise())


def test_mutated_adapter_receipt_cannot_retarget_recovery() -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        authorized = await coordinator.authorize(_authorized_record())
        request, _replacement, _receipt, _revoked = _cutover_material(authorized)
        installing = advance_egress_authority_transition(
            authorized,
            state=EgressAuthorityTransitionState.INSTALLING,
            reason="Installing.",
            environment_fingerprint=request.environment_fingerprint,
        )
        await store.compare_and_set(expected=authorized, replacement=installing)

        unrelated = _build_adapter_verified_egress_authority_cutover_receipt(
            expected=_authority(2, allow_post=True),
            target=_authority(3, allow_post=True, allow_delete=True),
            environment_fingerprint="e" * 64,
        )
        forged = build_egress_authority_cutover_receipt(
            expected=authorized.expected_authority,
            target=authorized.target_authority,
            environment_fingerprint=request.environment_fingerprint,
        )
        for field_name in EgressAuthorityCutoverReceipt.model_fields:
            object.__setattr__(unrelated, field_name, getattr(forged, field_name))

        with pytest.raises(EgressAuthorityTransitionConflict, match="adapter-owned"):
            await coordinator.reconcile(
                current=installing,
                adapter=_OutcomeAdapter(lambda _request: None, reconcile_receipt=unrelated),
                request=request,
                owner_token="private-owner-token",
            )
        assert await store.load(authorized.session_id) == installing

    asyncio.run(exercise())


def test_terminal_transition_can_be_superseded_by_exact_next_generation() -> None:
    async def exercise() -> None:
        session_store = InMemorySessionStore()
        await _create_session(session_store, "egress-transition-session")
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        coordinator = EgressAuthorityTransitionCoordinator(store)
        first = await coordinator.authorize(_authorized_record())
        first_request, first_binding, first_receipt, _revoked = _cutover_material(first)

        async def activate_first(_request):
            return EgressAuthorityCutoverResult(
                binding=first_binding,
                receipt=first_receipt,
            )

        first_active, _result = await coordinator.install(
            authorized=first,
            adapter=_OutcomeAdapter(activate_first),
            request=first_request,
            owner_token="private-owner-token",
        )
        assert first_active.state is EgressAuthorityTransitionState.ACTIVE

        second_decision = _decision_between(
            first.target_authority,
            _authority(3, allow_post=True, allow_delete=True),
            session_id=first.session_id,
            idempotency_identity="egress-transition-2",
        )
        second_requested = authorized_egress_authority_transition(
            decision=second_decision,
            transition_id="egress-transition-2",
            environment_name="egress",
            owner_fingerprint=egress_authority_owner_fingerprint("next-owner-token"),
            source_environment_fingerprint="d" * 64,
        )
        second = await coordinator.authorize(second_requested)
        assert second.revision == first_active.revision + 1
        assert second.expected_authority == first_active.target_authority
        assert await coordinator.authorize(second_requested) == second

        conflicting_replay = authorized_egress_authority_transition(
            decision=second_decision,
            transition_id="egress-transition-2",
            environment_name="egress",
            owner_fingerprint=egress_authority_owner_fingerprint("other-owner"),
            source_environment_fingerprint="d" * 64,
        )
        with pytest.raises(EgressAuthorityTransitionConflict):
            await coordinator.authorize(conflicting_replay)

        second_request, second_binding, second_receipt, _revoked = _cutover_material(second)

        async def activate_second(_request):
            return EgressAuthorityCutoverResult(
                binding=second_binding,
                receipt=second_receipt,
            )

        second_active, _result = await coordinator.install(
            authorized=second,
            adapter=_OutcomeAdapter(activate_second),
            request=second_request,
            owner_token="next-owner-token",
        )
        assert second_active.state is EgressAuthorityTransitionState.ACTIVE
        assert second_active.target_authority.generation == 3
        assert await coordinator.authorize(second_requested) == second_active

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "secret",
    (
        "active",
        "wider",
        "fresh_authority_path",
        "http",
        "exact",
        "system",
        "cayu.egress-authority-cutover",
    ),
)
def test_transition_controls_survive_short_secret_collisions(secret: str) -> None:
    authorized = _authorized_record()
    installing = advance_egress_authority_transition(
        authorized,
        state=EgressAuthorityTransitionState.INSTALLING,
        reason="Installing.",
        environment_fingerprint="d" * 64,
    )
    _request, _replacement, receipt, _revoked = _cutover_material(authorized)
    active = advance_egress_authority_transition(
        installing,
        state=EgressAuthorityTransitionState.ACTIVE,
        reason="Active.",
        receipt=receipt,
        environment_fingerprint="d" * 64,
    )
    event = egress_authority_transition_events(active)[0]
    prepared = prepare_new_runtime_event(event, redactor=SecretRedactor(secret))
    projected = project_persisted_runtime_event(
        prepared,
        sequence=1,
        redactor=SecretRedactor(secret),
    )

    assert prepared.payload["state"] == "active"
    assert prepared.payload["classification"] == "wider"
    assert prepared.payload["adapter_strategy"] == "fresh_authority_path"
    assert projected.payload["state"] == "active"
    assert projected.payload["classification"] == "wider"
    assert projected.payload["adapter_strategy"] == "fresh_authority_path"
    assert projected.payload["actor"]["source"] == "system"
    assert projected.payload["from_authority"]["policies"][0]["kind"] == "http"
    assert projected.payload["from_authority"]["policies"][0]["operations"][0]["match"] == "exact"
    assert projected.payload["receipt"]["record_type"] == "cayu.egress-authority-cutover"
    assert projected.payload["receipt"]["state"] == "active"
    assert projected.payload["receipt"]["strategy"] == "fresh_authority_path"


def test_postgres_restart_replays_and_recovers_exact_active_receipt(
    postgres_dsn: str,
) -> None:
    async def exercise() -> None:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        session_id = f"egress-authority-postgres-{uuid4().hex}"
        session_store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        await _create_session(session_store, session_id)
        store = SessionCheckpointEgressAuthorityTransitionStore(session_store)
        authorized = await store.compare_and_set(
            expected=None,
            replacement=_authorized_record(session_id),
        )
        installing = advance_egress_authority_transition(
            authorized,
            state=EgressAuthorityTransitionState.INSTALLING,
            reason="Installing in Postgres.",
            environment_fingerprint="d" * 64,
        )
        await store.compare_and_set(expected=authorized, replacement=installing)
        await session_store.close()

        restarted_session_store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        restarted_store = SessionCheckpointEgressAuthorityTransitionStore(restarted_session_store)
        assert await restarted_store.load(session_id) == installing
        request, _replacement, receipt, _revoked = _cutover_material(installing)
        active = await EgressAuthorityTransitionCoordinator(restarted_store).reconcile(
            current=installing,
            adapter=_OutcomeAdapter(lambda _request: None, reconcile_receipt=receipt),
            request=request,
            owner_token="private-owner-token",
        )
        assert active.state is EgressAuthorityTransitionState.ACTIVE
        assert (
            await restarted_store.compare_and_set(
                expected=installing,
                replacement=active,
            )
            == active
        )
        events = await restarted_session_store.query_events()
        assert [record.event.type for record in events][-1] is (
            EventType.EGRESS_AUTHORITY_ACTIVATED
        )
        await restarted_session_store.close()

    asyncio.run(exercise())
