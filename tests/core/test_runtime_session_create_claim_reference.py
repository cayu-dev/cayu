from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

import pytest

from cayu._validation import canonical_durable_json_bytes
from cayu.core import Event, EventType, Message
from cayu.runtime import InMemorySessionStore, RunRequest, SessionIdentity, SessionStatus
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime import sessions as sessions_module
from cayu.runtime.sessions import (
    SESSION_CREATE_CLAIM_METADATA_KEY,
    DeferredInteractionInput,
    RuntimeSessionCreateClaimAuthenticationDisposition,
    RuntimeSessionCreateClaimReference,
    RuntimeSessionCreateClaimReferenceKey,
    Session,
    SessionExecutionSource,
    authenticate_runtime_session_create_claim_reference,
    bind_runtime_session_create_claim,
    copy_run_request,
    run_request_with_runtime_generated_authority,
    run_request_with_runtime_invocation,
    run_request_with_runtime_session_create_claim_reference,
    runtime_session_create_claim_reference,
)
from cayu.vaults import SecretRedactor

_REFERENCE_KEY = RuntimeSessionCreateClaimReferenceKey(
    key_id="test-reference-key",
    secret=b"k" * 32,
)


def test_reference_key_is_bounded_and_does_not_render_secret_material() -> None:
    maximum = RuntimeSessionCreateClaimReferenceKey(
        key_id="k" * 256,
        secret=b"s" * 32,
    )

    assert maximum.key_id == "k" * 256
    assert maximum.secret.hex() not in repr(maximum)
    with pytest.raises(ValueError, match="character bound"):
        RuntimeSessionCreateClaimReferenceKey(
            key_id="k" * 257,
            secret=b"s" * 32,
        )
    with pytest.raises(ValueError, match="at least 32 bytes"):
        RuntimeSessionCreateClaimReferenceKey(
            key_id="short-secret",
            secret=b"s" * 31,
        )


@dataclass(frozen=True)
class _ClaimFixture:
    store: InMemorySessionStore
    source_request: RunRequest
    reference: RuntimeSessionCreateClaimReference
    reconstructed_request: RunRequest
    claim: object
    session: Session
    deferred_input: DeferredInteractionInput


async def _create_claimed_session() -> _ClaimFixture:
    store = InMemorySessionStore()
    source_request = run_request_with_runtime_generated_authority(
        RunRequest(
            agent_name="assistant",
            session_id="runtime-owned-session",
            causal_budget_id="runtime-owned-budget",
            messages=[Message.text("user", "recover this exact request")],
            metadata={"scope": "claim-reference-test"},
        ),
        "session_id",
        "causal_budget_id",
    )
    source_request = run_request_with_runtime_invocation(
        source_request,
        source=SessionExecutionSource.WORKFLOW_STEP,
    )
    reference = runtime_session_create_claim_reference(
        source_request,
        operation_id="workflow-operation-1",
        key=_REFERENCE_KEY,
    )
    reconstructed_request, claim = run_request_with_runtime_session_create_claim_reference(
        copy_run_request(source_request),
        reference,
        operation_id="workflow-operation-1",
        key=_REFERENCE_KEY,
    )
    prepared = session_request_boundary.prepare_run_request(
        reconstructed_request,
        redactor=SecretRedactor(),
    )
    identity = SessionIdentity(provider_name="fake", model="fake-model")
    started = Event(
        id="interaction-started-1",
        type=EventType.INTERACTION_STARTED,
        session_id=reference.session_id,
        interaction_id="interaction-1",
    )
    bind_runtime_session_create_claim(
        prepared,
        identity=identity,
        interaction_started_event=started,
    )
    session = await store.create(
        prepared,
        identity=identity,
        interaction_started_event=started,
        interaction_source_messages=prepared.messages,
    )
    deferred = await store.load_deferred_interaction_input(session.id)
    assert deferred is not None
    return _ClaimFixture(
        store=store,
        source_request=source_request,
        reference=reference,
        reconstructed_request=reconstructed_request,
        claim=claim,
        session=session,
        deferred_input=deferred,
    )


def _authenticate(
    fixture: _ClaimFixture,
    *,
    session: Session | None = None,
    deferred_input: DeferredInteractionInput | None | object = ...,
):
    return authenticate_runtime_session_create_claim_reference(
        fixture.session if session is None else session,
        (
            fixture.deferred_input if deferred_input is ... else deferred_input  # type: ignore[arg-type]
        ),
        fixture.claim,
        fixture.reference,
        request=fixture.reconstructed_request,
        operation_id=fixture.reference.operation_id,
        parent_session=None,
        key=_REFERENCE_KEY,
    )


def test_reference_is_bounded_durable_and_reconstructs_only_exact_request() -> None:
    async def run() -> None:
        fixture = await _create_claimed_session()
        serialized = fixture.reference.model_dump_json()
        restored = RuntimeSessionCreateClaimReference.model_validate_json(serialized)

        assert len(serialized.encode()) < 1_024
        assert "recover this exact request" not in serialized
        assert _REFERENCE_KEY.secret.hex() not in serialized
        assert "request_authority_sha256" not in serialized
        assert restored.request_authority_key_id == _REFERENCE_KEY.key_id
        assert restored == fixture.reference
        assert len(restored.claim_id) == 64

        copied = copy_run_request(fixture.source_request)
        copied._runtime_session_create_claim = None
        legacy_unkeyed_digest = hashlib.sha256(
            canonical_durable_json_bytes(
                {
                    "request": copied.model_dump(mode="json", warnings=False),
                    "lifecycle_authority_sha256": (
                        sessions_module._run_request_invocation_lifecycle_authority_sha256(copied)
                    ),
                },
                "runtime session create reference request",
            )
        ).hexdigest()
        assert restored.request_authority_hmac_sha256 != legacy_unkeyed_digest

        rotated_key = RuntimeSessionCreateClaimReferenceKey(
            key_id="rotated-reference-key",
            secret=b"r" * 32,
        )
        rotated = runtime_session_create_claim_reference(
            copy_run_request(fixture.source_request),
            operation_id=restored.operation_id,
            key=rotated_key,
        )
        assert rotated.request_authority_hmac_sha256 != (restored.request_authority_hmac_sha256)
        assert rotated.claim_id == restored.claim_id

        wrong_secret = RuntimeSessionCreateClaimReferenceKey(
            key_id=_REFERENCE_KEY.key_id,
            secret=b"w" * 32,
        )
        with pytest.raises(ValueError, match="conflicts with request authority"):
            run_request_with_runtime_session_create_claim_reference(
                copy_run_request(fixture.source_request),
                restored,
                operation_id=restored.operation_id,
                key=wrong_secret,
            )

        changed = copy_run_request(fixture.source_request).model_copy(
            update={"messages": [Message.text("user", "different request")]}
        )
        with pytest.raises(ValueError, match="conflicts with request authority"):
            run_request_with_runtime_session_create_claim_reference(
                changed,
                restored,
                operation_id=restored.operation_id,
                key=_REFERENCE_KEY,
            )
        with pytest.raises(ValueError, match="conflicts with request authority"):
            run_request_with_runtime_session_create_claim_reference(
                copy_run_request(fixture.source_request),
                restored,
                operation_id="another-operation",
                key=_REFERENCE_KEY,
            )

        rebound_operation_id = "workflow-operation-rebound"
        rebound_reference = restored.model_copy(
            update={
                "operation_id": rebound_operation_id,
                "claim_id": sessions_module._runtime_session_create_reference_claim_id(
                    session_id=restored.session_id,
                    operation_id=rebound_operation_id,
                ),
            }
        )
        with pytest.raises(ValueError, match="conflicts with request authority"):
            run_request_with_runtime_session_create_claim_reference(
                copy_run_request(fixture.source_request),
                rebound_reference,
                operation_id=rebound_operation_id,
                key=_REFERENCE_KEY,
            )

        different_invocation = run_request_with_runtime_invocation(
            copy_run_request(fixture.source_request),
            source=SessionExecutionSource.TASK,
        )
        with pytest.raises(ValueError, match="conflicts with request authority"):
            run_request_with_runtime_session_create_claim_reference(
                different_invocation,
                restored,
                operation_id=restored.operation_id,
                key=_REFERENCE_KEY,
            )

    asyncio.run(run())


def test_authentication_distinguishes_missing_live_and_cleaned_terminal_session() -> None:
    async def run() -> None:
        fixture = await _create_claimed_session()
        missing = authenticate_runtime_session_create_claim_reference(
            None,
            None,
            fixture.claim,
            fixture.reference,
            request=fixture.reconstructed_request,
            operation_id=fixture.reference.operation_id,
            parent_session=None,
            key=_REFERENCE_KEY,
        )
        assert missing.disposition is (
            RuntimeSessionCreateClaimAuthenticationDisposition.MISSING_SESSION
        )

        live = _authenticate(fixture)
        assert live.matches is True
        assert live.session_status is SessionStatus.RUNNING
        assert live.transient_input_authenticated is True

        await fixture.store.materialize_deferred_interaction_input(
            fixture.session.id,
            interaction_id=fixture.deferred_input.interaction_id,
        )
        await fixture.store.transition_status(
            fixture.session.id,
            from_statuses={SessionStatus.RUNNING},
            to_status=SessionStatus.COMPLETED,
        )
        terminal = await fixture.store.load(fixture.session.id)
        assert terminal is not None
        assert await fixture.store.load_deferred_interaction_input(terminal.id) is None

        authenticated = _authenticate(
            fixture,
            session=terminal,
            deferred_input=None,
        )
        assert authenticated.matches is True
        assert authenticated.session_status is SessionStatus.COMPLETED
        assert authenticated.transient_input_authenticated is False

        metadata = dict(terminal.metadata)
        claim_record = dict(metadata[SESSION_CREATE_CLAIM_METADATA_KEY])
        claim_record["messages_sha256"] = "0" * 64
        tampered = _authenticate(
            fixture,
            session=terminal.model_copy(
                update={
                    "metadata": {
                        **metadata,
                        SESSION_CREATE_CLAIM_METADATA_KEY: claim_record,
                    }
                }
            ),
            deferred_input=None,
        )
        assert tampered.disposition is (
            RuntimeSessionCreateClaimAuthenticationDisposition.TAMPERED_EVIDENCE
        )

    asyncio.run(run())


def test_authentication_classifies_foreign_partial_malformed_and_tampered_evidence() -> None:
    async def run() -> None:
        fixture = await _create_claimed_session()
        metadata = dict(fixture.session.metadata)

        foreign_metadata = dict(metadata)
        foreign_metadata.pop(SESSION_CREATE_CLAIM_METADATA_KEY)
        foreign = _authenticate(
            fixture,
            session=fixture.session.model_copy(update={"metadata": foreign_metadata}),
        )
        assert foreign.disposition is (
            RuntimeSessionCreateClaimAuthenticationDisposition.FOREIGN_SESSION
        )

        claim_record = dict(metadata[SESSION_CREATE_CLAIM_METADATA_KEY])
        partial_record = dict(claim_record)
        partial_record.pop("messages_sha256")
        partial = _authenticate(
            fixture,
            session=fixture.session.model_copy(
                update={
                    "metadata": {
                        **metadata,
                        SESSION_CREATE_CLAIM_METADATA_KEY: partial_record,
                    }
                }
            ),
        )
        assert partial.disposition is (
            RuntimeSessionCreateClaimAuthenticationDisposition.INCOMPLETE_EVIDENCE
        )

        unknown_schema = _authenticate(
            fixture,
            session=fixture.session.model_copy(
                update={
                    "metadata": {
                        **metadata,
                        SESSION_CREATE_CLAIM_METADATA_KEY: {
                            **claim_record,
                            "schema_version": 0,
                        },
                    }
                }
            ),
        )
        assert unknown_schema.disposition is (
            RuntimeSessionCreateClaimAuthenticationDisposition.MALFORMED_EVIDENCE
        )

        malformed = _authenticate(
            fixture,
            session=fixture.session.model_copy(
                update={
                    "metadata": {
                        **metadata,
                        SESSION_CREATE_CLAIM_METADATA_KEY: {
                            **claim_record,
                            "schema_version": "1",
                        },
                    }
                }
            ),
        )
        assert malformed.disposition is (
            RuntimeSessionCreateClaimAuthenticationDisposition.MALFORMED_EVIDENCE
        )

        tampered = _authenticate(
            fixture,
            session=fixture.session.model_copy(
                update={
                    "metadata": {
                        **metadata,
                        SESSION_CREATE_CLAIM_METADATA_KEY: {
                            **claim_record,
                            "messages_sha256": "0" * 64,
                        },
                    }
                }
            ),
        )
        assert tampered.disposition is (
            RuntimeSessionCreateClaimAuthenticationDisposition.TAMPERED_EVIDENCE
        )

        wrong_invocation = _authenticate(
            fixture,
            session=fixture.session.model_copy(
                update={
                    "invocation": fixture.session.invocation.model_copy(
                        update={"source": SessionExecutionSource.SDK_RUN}
                    )
                }
            ),
        )
        assert wrong_invocation.disposition is (
            RuntimeSessionCreateClaimAuthenticationDisposition.FOREIGN_SESSION
        )

        identity_conflict = authenticate_runtime_session_create_claim_reference(
            fixture.session,
            fixture.deferred_input,
            fixture.claim,
            fixture.reference,
            request=fixture.reconstructed_request,
            operation_id="another-operation",
            parent_session=None,
            key=_REFERENCE_KEY,
        )
        assert identity_conflict.disposition is (
            RuntimeSessionCreateClaimAuthenticationDisposition.IDENTITY_CONFLICT
        )

        crossed_request, crossed_claim = run_request_with_runtime_session_create_claim_reference(
            copy_run_request(fixture.source_request),
            fixture.reference,
            operation_id=fixture.reference.operation_id,
            key=_REFERENCE_KEY,
        )
        assert crossed_claim is not fixture.claim
        crossed_authority = authenticate_runtime_session_create_claim_reference(
            fixture.session,
            fixture.deferred_input,
            fixture.claim,
            fixture.reference,
            request=crossed_request,
            operation_id=fixture.reference.operation_id,
            parent_session=None,
            key=_REFERENCE_KEY,
        )
        assert crossed_authority.disposition is (
            RuntimeSessionCreateClaimAuthenticationDisposition.IDENTITY_CONFLICT
        )

    asyncio.run(run())


def test_live_session_requires_matching_transient_input() -> None:
    async def run() -> None:
        fixture = await _create_claimed_session()
        incomplete = _authenticate(fixture, deferred_input=None)
        assert incomplete.disposition is (
            RuntimeSessionCreateClaimAuthenticationDisposition.INCOMPLETE_EVIDENCE
        )

        tampered_input = fixture.deferred_input.model_copy(
            update={"source_messages": [Message.text("user", "different input")]}
        )
        tampered = _authenticate(fixture, deferred_input=tampered_input)
        assert tampered.disposition is (
            RuntimeSessionCreateClaimAuthenticationDisposition.TAMPERED_EVIDENCE
        )

    asyncio.run(run())


def test_authentication_hmac_work_is_constant_per_bounded_request(monkeypatch) -> None:
    async def run() -> None:
        fixture = await _create_claimed_session()
        request_hmac = sessions_module._runtime_session_create_reference_request_hmac_sha256
        claim_hash = sessions_module._runtime_session_create_reference_claim_id
        request_hmac_calls = 0
        claim_hash_calls = 0

        def count_request_hmac(request: RunRequest, *, operation_id: str, key) -> str:
            nonlocal request_hmac_calls
            request_hmac_calls += 1
            return request_hmac(request, operation_id=operation_id, key=key)

        def count_claim_hash(**kwargs) -> str:
            nonlocal claim_hash_calls
            claim_hash_calls += 1
            return claim_hash(**kwargs)

        monkeypatch.setattr(
            sessions_module,
            "_runtime_session_create_reference_request_hmac_sha256",
            count_request_hmac,
        )
        monkeypatch.setattr(
            sessions_module,
            "_runtime_session_create_reference_claim_id",
            count_claim_hash,
        )

        for _ in range(100):
            assert _authenticate(fixture).matches is True

        assert request_hmac_calls == 100
        assert claim_hash_calls == 100

    asyncio.run(run())
