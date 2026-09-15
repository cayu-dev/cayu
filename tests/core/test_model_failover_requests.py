"""Policy propagation and ingress evidence, not successful failover acceptance."""

from __future__ import annotations

import asyncio
import warnings

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelFailoverPolicy,
    ModelTarget,
    ResumeRequest,
    RunRequest,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelProviderError, ModelStreamEvent
from cayu.runtime._session_request_boundary import prepare_resume_request, prepare_run_request
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.work_attempt_semantics import (
    WorkAttemptRunSemantics,
    copy_work_attempt_run_semantics,
)
from cayu.server.routes import ResumeBody, RunBody
from cayu.sessions.base import InMemorySessionStore, copy_resume_request, copy_run_request
from cayu.tasks.dispatch import (
    DispatchRequest,
    _queued_dispatch_request_payload,
    _queued_dispatch_request_sha256,
    copy_dispatch_request,
)
from cayu.vaults.redaction import SecretRedactor
from cayu.workflows.workflow import StepRunOptions


def _policy():
    return ModelFailoverPolicy(
        fallbacks=(ModelTarget(provider_name="backup", model="large"),), max_total_attempts=3
    )


@pytest.mark.parametrize(
    ("model_type", "fields", "copier"),
    [
        (
            RunRequest,
            {"agent_name": "agent", "messages": [Message.text("user", "hi")]},
            copy_run_request,
        ),
        (
            ResumeRequest,
            {"session_id": "session", "messages": [Message.text("user", "hi")]},
            copy_resume_request,
        ),
        (
            DispatchRequest,
            {"session_id": "session", "messages": [Message.text("user", "hi")]},
            copy_dispatch_request,
        ),
        (
            StepRunOptions,
            {},
            lambda value: StepRunOptions.model_validate_json(value.model_dump_json()),
        ),
        (WorkAttemptRunSemantics, {"max_steps": 3}, copy_work_attempt_run_semantics),
        (
            RunBody,
            {"prompt": "hi"},
            lambda value: RunBody.model_validate_json(value.model_dump_json()),
        ),
        (
            ResumeBody,
            {"session_id": "session", "prompt": "hi"},
            lambda value: ResumeBody.model_validate_json(value.model_dump_json()),
        ),
    ],
)
def test_policy_survives_request_copy_and_wire_reconstruction(model_type, fields, copier):
    policy = _policy()
    request = model_type(**fields, failover=policy)
    copied = copier(request)
    reconstructed = model_type.model_validate_json(copied.model_dump_json())
    assert copied.failover == reconstructed.failover == policy
    assert copied.failover is not policy
    assert copied.failover.fallbacks[0] is not policy.fallbacks[0]
    object.__setattr__(policy.fallbacks[0], "model", "caller-changed")
    assert copied.failover.fallbacks[0].model == "large"


@pytest.mark.parametrize(
    "model_type,fields,copier",
    [
        (RunRequest, {"agent_name": "agent", "messages": []}, copy_run_request),
        (
            ResumeRequest,
            {"session_id": "session", "messages": [Message.text("user", "hi")]},
            copy_resume_request,
        ),
        (
            DispatchRequest,
            {"session_id": "session", "messages": [Message.text("user", "hi")]},
            copy_dispatch_request,
        ),
        (WorkAttemptRunSemantics, {"max_steps": 3}, copy_work_attempt_run_semantics),
    ],
)
def test_post_construction_mutation_is_rejected_before_serialization(
    model_type, fields, copier, capsys, caplog
):
    canary = "SECRET_FAILOVER_SERIALIZER_CANARY"

    class Hostile:
        def __repr__(self):
            return canary

        __str__ = __repr__

    request = model_type(**fields, failover=_policy())
    object.__setattr__(request.failover.fallbacks[0], "model", Hostile())
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises((TypeError, ValueError)) as failure:
            copier(request)
    assert canary not in str(failure.value) + repr(failure.value)
    assert not recorded
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("kind", ["run", "resume"])
@pytest.mark.parametrize("field_name", ["provider_name", "model"])
def test_secret_target_is_rejected_not_redacted_to_another_target(kind, field_name):
    secret = "secret-provider-credential"
    redactor = SecretRedactor(secret)
    policy = ModelFailoverPolicy(
        fallbacks=(
            ModelTarget(**{"provider_name": "backup", "model": "large", field_name: secret}),
        )
    )
    with pytest.raises(ValueError, match="secret") as failure:
        if kind == "run":
            prepare_run_request(
                RunRequest(agent_name="agent", messages=[], failover=policy), redactor=redactor
            )
        else:
            prepare_resume_request(
                ResumeRequest(
                    session_id="session", messages=[Message.text("user", "hi")], failover=policy
                ),
                redactor=redactor,
            )
    assert secret not in str(failure.value)


def test_queued_exact_request_identity_binds_every_policy_field():
    request = DispatchRequest(
        session_id="session", messages=[Message.text("user", "hi")], failover=_policy()
    )
    original = _queued_dispatch_request_sha256(request, schema_version=4)
    payload = _queued_dispatch_request_payload(request, schema_version=4)
    assert DispatchRequest.model_validate(payload).failover == request.failover
    for policy in (
        ModelFailoverPolicy(fallbacks=_policy().fallbacks, max_total_attempts=4),
        ModelFailoverPolicy(
            fallbacks=(ModelTarget(provider_name="other", model="large"),), max_total_attempts=3
        ),
        ModelFailoverPolicy(
            fallbacks=(ModelTarget(provider_name="backup", model="other"),), max_total_attempts=3
        ),
    ):
        changed = request.model_copy(update={"failover": policy})
        assert _queued_dispatch_request_sha256(changed, schema_version=4) != original
    with pytest.raises(ValueError, match="queue writer"):
        _queued_dispatch_request_payload(request, schema_version=2)


@pytest.mark.parametrize("supported", [True, False])
def test_native_runtime_admission_requires_atomic_store_owners(supported):
    class UnattestedStore(InMemorySessionStore):
        async def _prepare_model_completion_stage_atomic(self, prepared):
            return await super()._prepare_model_completion_stage_atomic(prepared)

    async def scenario():
        def unavailable(_request):
            raise ModelProviderError(
                "unavailable", provider="primary", status_code=503, retryable=True
            )

        provider = ScriptedModelProvider(name="primary", response_factory=unavailable)
        backup = ScriptedModelProvider(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]], name="backup"
        )
        store = InMemorySessionStore() if supported else UnattestedStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        request = RunRequest(
            agent_name="agent",
            session_id="native-admission",
            messages=[Message.text("user", "hi")],
            failover=_policy(),
            retry_policy=RetryPolicy(max_attempts=1),
        )
        if supported:
            events = [event async for event in app.run(request)]
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(provider.requests) == len(backup.requests) == 1
        else:
            with pytest.raises(NotImplementedError, match="atomic model failover"):
                _ = [event async for event in app.run(request)]
            assert not provider.requests and not backup.requests
            assert request.session_id is not None
            assert await store.load(request.session_id) is None
            assert await store.load_checkpoint(request.session_id) is None
            with pytest.raises(KeyError):
                await store.load_events(request.session_id)

    asyncio.run(scenario())


def test_native_postgres_public_run_and_resume(postgres_dsn):
    from tests.core.test_model_failover_recovery import _RecoveryProvider

    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    def application(store):
        app = CayuApp(session_store=store, enable_logging=False)
        primary, backup = _RecoveryProvider("primary"), _RecoveryProvider("backup")
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        return app, primary, backup

    async def scenario():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            app, primary, backup = application(store)
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="native-postgres",
                        messages=[Message.text("user", "hi")],
                        failover=_policy(),
                        retry_policy=RetryPolicy(max_attempts=1),
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(primary.requests) == len(backup.requests) == 1
            await store.close()
            store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
            app, primary, backup = application(store)
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="native-postgres",
                        messages=[Message.text("user", "continue")],
                        retry_policy=RetryPolicy(max_attempts=1),
                    )
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            assert not primary.requests and len(backup.requests) == 1
            checkpoint = await store.load_checkpoint("native-postgres")
            assert checkpoint is not None
            assert checkpoint["model_failover"]["candidate_index"] == 1
            assert checkpoint["model_failover"]["attempts_used"] == 1
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("entrance", ["resume", "fork"])
def test_inherited_route_rejects_unattested_store_before_mutation(tmp_path, entrance):
    from tests.core.test_model_failover_recovery import _RecoveryProvider

    from cayu.sessions.base import ForkSessionRequest
    from cayu.storage.sqlite import SQLiteSessionStore

    class UnattestedStore(SQLiteSessionStore):
        async def _prepare_model_completion_stage_atomic(self, prepared):
            return await super()._prepare_model_completion_stage_atomic(prepared)

    def application(store):
        app = CayuApp(session_store=store, enable_logging=False)
        primary, backup = _RecoveryProvider("primary"), _RecoveryProvider("backup")
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        return app, primary, backup

    async def scenario():
        path = tmp_path / "unsupported-resume.sqlite"
        store = SQLiteSessionStore(path)
        try:
            app, _, _ = application(store)
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="source",
                        messages=[Message.text("user", "hi")],
                        failover=_policy(),
                        retry_policy=RetryPolicy(max_attempts=1),
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED
            checkpoint = await store.load_checkpoint("source")
            persisted = await store.load_events("source")
            source = await store.load("source")
            await store.close()
            store = UnattestedStore(path)
            app, primary, backup = application(store)
            with pytest.raises(NotImplementedError, match="atomic model failover"):
                if entrance == "resume":
                    _ = [
                        event
                        async for event in app.resume(
                            ResumeRequest(
                                session_id="source", messages=[Message.text("user", "continue")]
                            )
                        )
                    ]
                else:
                    _ = [
                        event
                        async for event in app.fork_session(
                            ForkSessionRequest(source_session_id="source", session_id="child")
                        )
                    ]
            assert not primary.requests and not backup.requests
            assert await store.load("source") == source
            assert await store.load_checkpoint("source") == checkpoint
            assert await store.load_events("source") == persisted
            assert await store.load("child") is None
        finally:
            await store.close()

    asyncio.run(scenario())
