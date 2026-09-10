from __future__ import annotations

import asyncio
import base64

import pytest
from pydantic import SecretStr
from tests.core.test_verified_work_contracts import _contract, _RecordingProvider
from tests.core.test_work_attempt_admission import _assert_secret_absent_from_work_attempt_error

from cayu import (
    AgentSpec,
    CayuApp,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ResumeRequest,
    RunRequest,
    SecretRedactor,
    SQLiteSessionStore,
    SQLiteTaskStore,
    TaskCreate,
    WorkAttemptExecutionRequest,
    WorkAttemptRecoveryRequest,
    WorkCompletionConflict,
)
from cayu._validation import canonical_durable_json_bytes
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.work_attempt_admission import (
    WorkAttemptAdmission,
    WorkAttemptAdmissionState,
)
from cayu.runtime.work_attempt_source import (
    WORK_ATTEMPT_SOURCE_MAX_BYTES,
    WORK_ATTEMPT_SOURCE_MAX_FIELDS,
    WORK_ATTEMPT_SOURCE_MAX_ITEMS,
    WorkAttemptSourceRequest,
)


@pytest.mark.parametrize("boundary", ["bytes", "items", "fields"])
def test_source_snapshot_limits_include_the_complete_envelope(boundary):
    def capture(request, fields):
        return WorkAttemptSourceRequest.capture(
            kind="initial", request=request, fields_set=fields, source_request_sha256="a" * 64
        )

    empty = capture({"payload": ""}, ("payload",))
    overhead = len(canonical_durable_json_bytes(empty.model_dump(mode="json"), "source"))
    for delta in (-1, 0, 1):
        if boundary == "bytes":
            request = {"payload": "x" * (WORK_ATTEMPT_SOURCE_MAX_BYTES - overhead + delta)}
            fields = ("payload",)
        elif boundary == "items":
            # Root, kind, request, list, fields list/name, and the two digests.
            request = {"items": [None] * (WORK_ATTEMPT_SOURCE_MAX_ITEMS - 8 + delta)}
            fields = ("items",)
        else:
            fields = tuple(
                f"f{index:03}" for index in range(WORK_ATTEMPT_SOURCE_MAX_FIELDS + delta)
            )
            request = dict.fromkeys(fields)
        if delta > 0:
            with pytest.raises(ValueError):
                capture(request, fields)
        else:
            snapshot = capture(request, fields)
            assert (
                WorkAttemptSourceRequest.model_validate_json(snapshot.model_dump_json()) == snapshot
            )


def test_source_snapshot_preserves_explicit_defaults_and_detaches_data() -> None:
    implicit = ResumeRequest(session_id="source-session", messages=[Message.text("user", "next")])
    explicit = implicit.model_copy(update={"max_steps": implicit.max_steps})
    snapshots = [
        WorkAttemptSourceRequest.capture(
            kind="continuation",
            request=request.model_dump(mode="json", warnings=False),
            fields_set=tuple(sorted(request.model_fields_set)),
            source_request_sha256="a" * 64,
        )
        for request in (implicit, explicit)
    ]
    assert snapshots[0].request == snapshots[1].request
    assert snapshots[0].content_sha256 != snapshots[1].content_sha256
    assert "max_steps" not in snapshots[0].fields_set
    restored = WorkAttemptSourceRequest.model_validate_json(snapshots[0].model_dump_json())
    snapshots[0].request["messages"].clear()
    assert restored.request["messages"]
    assert restored.fields_set == snapshots[0].fields_set


@pytest.mark.parametrize(
    "fields", [["task_id", "session_id"], ["session_id"] * 2, [True], ["missing"]]
)
def test_source_snapshot_rejects_ambiguous_field_names(fields) -> None:
    with pytest.raises(ValueError):
        WorkAttemptSourceRequest.capture(
            kind="initial",
            request={"session_id": "s"},
            fields_set=fields,
            source_request_sha256="a" * 64,
        )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_admission_retains_exact_source_before_session_creation(
    backend: str, tmp_path, monkeypatch
) -> None:
    async def scenario() -> None:
        codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(
                active_key_id="test",
                keys={
                    "test": SecretStr(
                        base64.urlsafe_b64encode(bytes([31]) * 32).decode().rstrip("=")
                    )
                },
            )
        )
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(
                tmp_path / "source-sessions.sqlite", public_authority_alias_codec=codec
            )
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "source-tasks.sqlite")
        )

        def make_app():
            app = CayuApp(
                session_store=sessions,
                task_store=tasks,
                enable_logging=False,
                secret_redactor=SecretRedactor(["source-secret-canary", "other-source-secret"]),
            )
            app.register_provider(_RecordingProvider(), default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            return app

        try:
            app = make_app()
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(task_id="source-task", type="work", work_contract=contract.reference())
            )
            source = RunRequest(
                agent_name="worker",
                task_id=task.id,
                session_id="source-session",
                messages=[Message.text("user", "Retain input, redact source-secret-canary.")],
                metadata={"job": {"revision": 7}},
                max_steps=7,
            )
            execution = WorkAttemptExecutionRequest(
                admission_id="source-admission",
                claim_id="source-claim",
                attempt_id="source-attempt",
                interaction_id="source-interaction",
                worker_id="source-worker",
                generation=1,
                lease_seconds=300,
            )
            store_type = type(tasks)
            original = store_type.prepare_work_attempt_admission

            async def lose_ack(store, request):
                await original(store, request)
                raise ConnectionError("preparation committed before session creation")

            with monkeypatch.context() as patch:
                patch.setattr(store_type, "prepare_work_attempt_admission", lose_ack)
                with pytest.raises(ConnectionError):
                    await app.admit_work_attempt(source, execution=execution)
            assert await sessions.load(source.session_id) is None
            prepared = await tasks.load_work_attempt_admission(execution.admission_id)
            assert prepared is not None and prepared.state is WorkAttemptAdmissionState.PREPARING
            snapshot = prepared.source_request
            assert snapshot is not None
            assert snapshot.source_request_sha256 == prepared.source_request_sha256
            assert "source-secret-canary" not in prepared.model_dump_json()
            assert snapshot.request["messages"] != source.model_dump(mode="json")["messages"]
            assert snapshot.request["execution_deadline"]["expires_at"] is None
            assert snapshot.request["metadata"] == {"job": {"revision": 7}}
            assert "max_steps" in snapshot.fields_set
            retained_json = prepared.model_dump_json()
            snapshot.request["metadata"]["job"]["revision"] = 8
            with pytest.raises(ValueError, match="source conflicts"):
                WorkAttemptAdmission.model_validate_json(prepared.model_dump_json())
            stored = await tasks.load_work_attempt_admission(execution.admission_id)
            assert stored is not None and stored.model_dump_json() == retained_json
            assert stored.source_request is not None
            forged_snapshot = WorkAttemptSourceRequest.capture(
                kind="initial",
                request={**stored.source_request.request, "metadata": {"different": True}},
                fields_set=stored.source_request.fields_set,
                source_request_sha256=stored.source_request_sha256,
            )
            forged = stored.model_copy(update={"source_request": forged_snapshot})
            assert WorkAttemptAdmission.model_validate_json(forged.model_dump_json()) == forged

            async def load_conflicting_source(store, admission_id):
                assert store is tasks and admission_id == execution.admission_id
                return forged

            with monkeypatch.context() as patch:
                patch.setattr(store_type, "load_work_attempt_admission", load_conflicting_source)
                with pytest.raises(WorkCompletionConflict):
                    await app.admit_work_attempt(source, execution=execution)
            assert await sessions.load(source.session_id) is None
            assert await tasks.load_work_attempt_admission(execution.admission_id) == stored
            malformed_source = WorkAttemptSourceRequest.capture(
                kind="initial",
                request={
                    **stored.source_request.request,
                    "max_steps": True,
                    "metadata": {"private": "recovery-source-private-canary"},
                },
                fields_set=stored.source_request.fields_set,
                source_request_sha256=stored.source_request_sha256,
            )
            for bad_source in (None, malformed_source):
                bad_admission = stored.model_copy(update={"source_request": bad_source})
                claim_calls = []

                async def load_invalid_recovery_source(
                    store, admission_id, *, bad_admission=bad_admission
                ):
                    assert store is tasks and admission_id == execution.admission_id
                    return bad_admission

                async def must_not_claim(store, request, *, claim_calls=claim_calls):
                    claim_calls.append(request)
                    raise AssertionError("Invalid recovery source reached claim mutation")

                with monkeypatch.context() as patch:
                    patch.setattr(
                        store_type, "load_work_attempt_admission", load_invalid_recovery_source
                    )
                    patch.setattr(store_type, "claim_work_attempt_recovery", must_not_claim)
                    with pytest.raises(RuntimeError) as failure:
                        await app.recover_work_attempt(
                            WorkAttemptRecoveryRequest(
                                admission_id=execution.admission_id,
                                claim_id="invalid-source-claim",
                                worker_id="replacement",
                                generation=2,
                                lease_seconds=300,
                            )
                        )
                _assert_secret_absent_from_work_attempt_error(
                    failure.value, "recovery-source-private-canary"
                )
                assert claim_calls == []
                assert await tasks.load_work_attempt_admission(execution.admission_id) == stored
            different_input = source.model_copy(
                update={
                    "messages": [Message.text("user", "Retain input, redact other-source-secret.")]
                }
            )
            assert app.redact_json(different_input.model_dump(mode="json")) == app.redact_json(
                source.model_dump(mode="json")
            )
            with pytest.raises(WorkCompletionConflict):
                await app.admit_work_attempt(different_input, execution=execution)
            assert await tasks.load_work_attempt_admission(execution.admission_id) == stored
            if backend == "sqlite":
                await sessions.close()
                await tasks.close()
                sessions = SQLiteSessionStore(
                    tmp_path / "source-sessions.sqlite", public_authority_alias_codec=codec
                )
                tasks = SQLiteTaskStore(tmp_path / "source-tasks.sqlite")
                restored = await tasks.load_work_attempt_admission(execution.admission_id)
                assert restored is not None and restored.model_dump_json() == retained_json
            # The original process identity is still required for same-generation
            # replay. This verifies source persistence, not replacement recovery.
            if backend == "memory":
                active = await app.admit_work_attempt(source, execution=execution)
                assert active.state is WorkAttemptAdmissionState.ACTIVE
                assert active.source_request == stored.source_request
        finally:
            if backend == "sqlite":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())


def test_public_admission_rejects_oversized_source_before_any_mutation() -> None:
    async def scenario() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        contract = _contract()
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(task_id="oversized-source", type="work", work_contract=contract.reference())
        )
        with pytest.raises(ValueError, match="portable limit") as failure:
            await app.admit_work_attempt(
                RunRequest(
                    agent_name="worker",
                    task_id=task.id,
                    session_id="oversized-session",
                    messages=[Message.text("user", "x" * WORK_ATTEMPT_SOURCE_MAX_BYTES)],
                    metadata={"private": "oversized-source-private-canary"},
                ),
                execution=WorkAttemptExecutionRequest(
                    admission_id="oversized-admission",
                    claim_id="oversized-claim",
                    attempt_id="oversized-attempt",
                    interaction_id="oversized-interaction",
                    worker_id="oversized-worker",
                    generation=1,
                    lease_seconds=300,
                ),
            )
        _assert_secret_absent_from_work_attempt_error(
            failure.value, "oversized-source-private-canary"
        )
        assert await tasks.load_task(task.id) == task
        assert await tasks.load_work_attempt_admission("oversized-admission") is None
        assert await sessions.load("oversized-session") is None

    asyncio.run(scenario())
