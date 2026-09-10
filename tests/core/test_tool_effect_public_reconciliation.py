from __future__ import annotations

import asyncio
import warnings
from base64 import urlsafe_b64encode
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr
from tests.core._workload_secret_support import RequireApprovalPolicy
from tests.core.test_runtime import RecordingEnvironmentFactory
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.core.test_tool_effect_runtime_dispatch import _ObservingSQLiteStore, _ObservingStore
from tests.core.test_tool_round_execution_identities import _SequencedProvider, _tool_call_response

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    Message,
    ResumeRequest,
    RunRequest,
    RuntimeEvidenceRequest,
    Tool,
    ToolEffect,
    ToolEffectConflict,
    ToolResult,
    ToolSpec,
    runtime_evidence,
)
from cayu.core.tools import DurableToolRecoveryEvidence
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.factory import EnvironmentFactoryOperation
from cayu.providers import ModelStreamEvent
from cayu.runtime._event_projection import project_persisted_runtime_event
from cayu.runtime._tool_effect_reconciliation import ToolEffectReconciliationTimeout
from cayu.runtime._tool_effect_state import ToolEffectRecord
from cayu.runtime.approvals import ToolApprovalDecision, ToolApprovalRequest
from cayu.runtime.hooks import AfterToolCallDecision, RuntimeHook
from cayu.runtime.human_review import (
    HumanReviewConflict,
    HumanReviewContext,
    HumanReviewDenied,
    HumanReviewDisclosure,
    HumanReviewField,
    HumanReviewPolicy,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
    tool_effect_receipt_digest,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize(
    "outcome,fault_phase",
    [
        ("completed", None),
        ("failed", None),
        ("not_found", None),
        ("unsupported", None),
        ("conflict", None),
        pytest.param("conflict", "observation-error", id="conflict-postcommit"),
        pytest.param("conflict", "observation-abandon", id="conflict-abandon"),
        pytest.param("completed", "terminal", id="completed-postcommit"),
        pytest.param("failed", "terminal", id="failed-postcommit"),
        pytest.param("not_found", "observation-error", id="observation-postcommit"),
        pytest.param("not_found", "observation-abandon", id="observation-abandon"),
        pytest.param("not_found", "unlisted-resource", id="observation-unlisted-resource"),
        pytest.param("completed", "unlisted-resource", id="terminal-unlisted-resource"),
    ],
)
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("mode", ["lookup", "receipt"])
@pytest.mark.parametrize("approval_gate", [False, True], ids=["ordinary", "approval"])
def test_public_reconciliation_consumes_verified_outcome_without_external_replay(
    outcome,
    fault_phase,
    backend,
    mode,
    approval_gate,
    tmp_path,
    monkeypatch,
    caplog,
    capsys,
    control_secrets=None,
):
    observation_event_type = (
        "tool.effect.reconciliation.conflict"
        if outcome == "conflict"
        else "tool.effect.reconciliation.observed"
    )

    async def scenario(store):
        calls = []
        lookups = []
        hostile_formatting = []
        journal_reads = []
        native_disposition = (
            fault_phase.removeprefix("native-")
            if fault_phase is not None and fault_phase.startswith("native-")
            else None
        )
        callback_entered = asyncio.Event()
        callback_release = asyncio.Event()
        callback_settled = asyncio.Event()
        callback_signal = fault_phase in {
            "cancel",
            "repeated-cancel",
            "deadline",
            "cancel-cleanup-read",
        }
        lifecycle = fault_phase in {"lifecycle", "factory-failure", "preflight"}
        hook_observations = []

        async def assert_admitted(prior):
            current = ToolEffectRecord.model_validate(
                await store.load_session_operation("receipt-public", store.effect_keys[0])
            )
            assert current.revision == prior.revision + 1
            for name in (
                "intent",
                "state",
                "dispatch_id",
                "terminal",
                "observation",
                "resource_versions",
            ):
                assert getattr(current, name) == getattr(prior, name)
            assert current.reconciliation_attempt is not None
            assert current.reconciliation_attempt.source_revision == prior.revision
            assert current.reconciliation_attempt.source_run_epoch == request.expected_run_epoch
            assert current.reconciliation_attempt.lookup is (mode == "lookup")
            assert (
                sum(
                    event.id == current.reconciliation_attempt.event_id
                    and event.type.value == "tool.effect.reconciliation.started"
                    for event in await store.load_events("receipt-public")
                )
                == 1
            )
            return current

        class ObservingHook(RuntimeHook):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:receipt-hook", behavior_version="1", implementation_version="1"
                )

            async def after_tool_call(self, context):
                selected = ToolEffectRecord.model_validate(
                    await store.load_session_operation("receipt-public", store.effect_keys[0])
                )
                assert selected.state == "reconciled_completed"
                assert selected.terminal.event_id == context.tool_event.id
                hook_observations.append(context.result.content)
                return AfterToolCallDecision(
                    action="modify", modified_result=ToolResult(content="must not replace receipt")
                )

        class Factory(RecordingEnvironmentFactory):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:receipt-factory", behavior_version="1", implementation_version="1"
                )

        environment_spec = EnvironmentSpec(
            name="receipt-environment",
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:receipt-environment", behavior_version="1", implementation_version="1"
            ),
        )
        factory = Factory(Environment(environment_spec))
        review_context = HumanReviewContext(recipient="operator", purpose="effect-recovery")

        class ReviewPolicy(HumanReviewPolicy):
            version = "receipt-review-v1"
            binding_key = b"receipt-test-only-binding-key-32-bytes"
            can_decide = True

            def authorize(self, context, *, session_id, session_metadata, action):
                return (
                    context == review_context
                    and session_id == "receipt-public"
                    and (action == "inspect" or self.can_decide)
                )

            def project(self, context, source):
                return HumanReviewDisclosure(
                    status="permitted",
                    fields=(HumanReviewField(label="Effect", text="Record the reviewed effect."),),
                    sensitive_content="application_attested",
                )

        review_policy = ReviewPolicy() if fault_phase == "human-review" else None

        class External(Tool):
            spec = ToolSpec(
                name="record",
                effect=ToolEffect.EXTERNAL,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:receipt-external-tool",
                    behavior_version="1",
                    implementation_version="1",
                ),
            )

            async def run(self, ctx, args):
                calls.append(ctx.idempotency_key)
                raise RuntimeError("external acknowledgement lost")

        class Reconciler:
            async def reconcile(self, *, context, receipt):
                if mode == "lookup":
                    assert receipt is None
                else:
                    assert receipt is not None
                    assert receipt.receipt_id == "external-receipt"
                    assert receipt.idempotency_key == calls[0]
                assert context.idempotency_key == calls[0]
                starts = [
                    event
                    for event in await store.load_events("receipt-public")
                    if event.type.value == "tool.effect.reconciliation.started"
                ]
                assert starts
                started = starts[-1]
                assert started.tool_name == context.tool_name
                assert started.payload["schema_version"] == 1
                assert started.payload["intent_digest"] == context.intent_digest
                assert started.payload["expected_revision"] == context.record_revision
                assert started.payload["lookup"] is (mode == "lookup")
                assert len(started.payload["request_digest"]) == 64
                assert "receipt" not in started.payload
                assert "idempotency_key" not in started.payload
                if lifecycle:
                    assert factory.requests[-1].operation == EnvironmentFactoryOperation.RECONNECT
                lookups.append(context)
                if len(lookups) == 1 and fault_phase == "child-cancel":
                    child = asyncio.current_task()
                    assert child is not None
                    child.cancel()
                    assert child.cancelling() == 1
                    await asyncio.sleep(0)
                    raise AssertionError("Child cancellation was not delivered")
                if len(lookups) == 1 and fault_phase == "callback-timeout":
                    raise TimeoutError("application callback timeout")
                if callback_signal and len(lookups) == 1:
                    callback_entered.set()
                    try:
                        await callback_release.wait()
                    finally:
                        callback_settled.set()
                observed_resources = (
                    {"unlisted-field": "private-resource-canary"}
                    if fault_phase == "unlisted-resource"
                    else {"part-1": "v1"}
                    if len(lookups) == 1
                    else {}
                )
                if outcome in {"not_found", "unsupported", "conflict"}:
                    return ToolEffectReconciliationResult(
                        outcome=outcome,
                        observation="partial",
                        resource_versions=observed_resources,
                    )
                result = ToolEffectReconciliationResult(
                    outcome=outcome,
                    observation="sent",
                    resource_versions=observed_resources,
                    receipt=ToolEffectReceipt(
                        receipt_id="external-receipt",
                        receipt_schema="deployment",
                        receipt_schema_version=1,
                        tool_call_id=context.tool_call_id,
                        tool_name=context.tool_name,
                        idempotency_key=context.idempotency_key,
                        outcome=outcome,
                        message="verified external outcome",
                        resource_versions={"part-2": "v2"},
                        source="reconciler",
                        observed_at=datetime(2026, 9, 8, tzinfo=UTC),
                    ),
                )
                if fault_phase == "hostile-result":

                    class Hostile:
                        def __repr__(self):
                            hostile_formatting.append("repr")
                            return "private-callback-canary"

                        def __str__(self):
                            hostile_formatting.append("str")
                            return "private-callback-canary"

                    return result.model_copy(
                        update={
                            "receipt": result.receipt.model_copy(
                                update={
                                    "structured": {"unsafe": Hostile()},
                                    "message": "private-callback-canary",
                                }
                            )
                        }
                    )
                return result

        class JournalExternal(External):
            async def reconcile_durable_tool_call(self, **kwargs):
                assert kwargs["idempotency_key"] == calls[0]
                assert kwargs["arguments"] == {"value": 7}
                journal_reads.append(kwargs["tool_call_id"])
                if native_disposition == "missing":
                    return None
                return DurableToolRecoveryEvidence(
                    native_disposition, ToolResult(content="journal-confirmed outcome")
                )

        class VersionedProvider(_SequencedProvider):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:receipt-provider",
                    behavior_version="1",
                    implementation_version="1",
                )

        provider = VersionedProvider(
            [
                _tool_call_response(7),
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

        def build_app(*, secret_redactor=None):
            if secret_redactor is None and control_secrets is not None:
                secret_redactor = SecretRedactor(control_secrets)
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                human_review_policy=review_policy,
                secret_redactor=secret_redactor,
            )
            app.register_provider(provider, default=True)
            if lifecycle:
                app.register_environment_factory(environment_spec, factory, default=True)
            app.register_agent(
                AgentSpec(name="agent", model="test"),
                tools=[JournalExternal() if native_disposition is not None else External()],
                tool_policy=RequireApprovalPolicy() if approval_gate else None,
                runtime_hooks=[ObservingHook()] if lifecycle else None,
                tool_effect_reconcilers={
                    "record": ToolEffectReconciliationRegistration(
                        reconciler=Reconciler(),
                        spec=_spec(
                            supports_lookup=True,
                            timeout_seconds=0.1 if fault_phase == "deadline" else 30,
                        ),
                    )
                },
            )
            return app

        app = build_app()
        initial = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="receipt-public",
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        assert initial[-1].type.value == "session.interrupted"
        if approval_gate:
            approval_event = next(
                event for event in initial if event.type.value == "tool.call.approval_requested"
            )
            initial_review = (
                await app.inspect_human_review("receipt-public", context=review_context)
                if review_policy is not None
                else None
            )
            initial = [
                event
                async for event in app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id="receipt-public",
                        approval_id=approval_event.payload["approval"]["approval_id"],
                        tool_round_id=approval_event.payload["tool_round_id"],
                        tool_call_id=approval_event.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                        review_reference=None
                        if initial_review is None
                        else initial_review.reference,
                    )
                )
            ]
            assert initial[-1].type.value == "session.interrupted"
        record = ToolEffectRecord.model_validate(
            await store.load_session_operation("receipt-public", store.effect_keys[0])
        )
        assert record.state == "outcome_unknown"
        if native_disposition is not None:
            app = build_app()
            recovered = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="receipt-public", messages=[Message.text("user", "continue")]
                    )
                )
            ]
            current = ToolEffectRecord.model_validate(
                await store.load_session_operation("receipt-public", store.effect_keys[0])
            )
            terminals = [
                event
                for event in await store.load_events("receipt-public")
                if event.type.value in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(calls) == len(journal_reads) == 1
            assert lookups == []
            if native_disposition == "confirmed":
                assert recovered[-1].type.value == "session.completed"
                assert current.state == "completed"
                assert len(terminals) == 1
                assert terminals[0].id == current.terminal.event_id
                assert terminals[0].payload["result"]["content"] == "journal-confirmed outcome"
                assert len(provider.requests) == 2
            else:
                assert recovered[-1].type.value == "session.interrupted"
                assert current == record
                assert terminals == []
                assert len(provider.requests) == 1
            return
        session = await store.load("receipt-public")
        supplied = (
            None
            if mode == "lookup"
            else ToolEffectReceipt(
                receipt_id="external-receipt",
                receipt_schema="deployment",
                receipt_schema_version=1,
                tool_call_id=record.intent.tool_call_id,
                tool_name=record.intent.tool_name,
                idempotency_key=record.intent.idempotency_key,
                outcome="failed" if outcome == "failed" else "completed",
                message="verified external outcome",
                source="operator",
                observed_at=datetime(2026, 9, 8, tzinfo=UTC),
            )
        )
        request = ToolEffectReconciliationRequest(
            review_reference=(
                await app.inspect_human_review("receipt-public", context=review_context)
            ).reference
            if review_policy is not None
            else None,
            **{
                name: getattr(record.intent, name)
                for name in (
                    "session_id",
                    "session_instance_id",
                    "tool_round_id",
                    "tool_call_id",
                    "tool_name",
                    "idempotency_key",
                )
            },
            expected_run_epoch=session.run_epoch,
            expected_revision=record.revision,
            lookup=mode == "lookup",
            receipt=supplied,
        )
        if fault_phase in {"inspection", "inspection-no-key"}:
            started = next(event for event in initial if event.type.value == "tool.call.started")
            before_events = await store.load_events("receipt-public")
            before_session = await store.load("receipt-public")
            if fault_phase == "inspection-no-key":
                if store.public_authority_alias_codec is None:
                    with pytest.raises(RuntimeError, match="configured alias keyring"):
                        await app.inspect_tool_effect(
                            "receipt-public",
                            tool_round_id=started.payload["tool_round_id"],
                            tool_call_id=started.payload["tool_call_id"],
                        )
                else:
                    # Memory stores supply their own ephemeral codec even when
                    # the application did not configure a deployment keyring.
                    target = await app.inspect_tool_effect(
                        "receipt-public",
                        tool_round_id=started.payload["tool_round_id"],
                        tool_call_id=started.payload["tool_call_id"],
                    )
                    assert target.idempotency_key != record.intent.idempotency_key
                assert await store.load_events("receipt-public") == before_events
                assert await store.load("receipt-public") == before_session
                assert lookups == []
                assert len(calls) == 1
                return
            target = await app.inspect_tool_effect(
                "receipt-public",
                tool_round_id=started.payload["tool_round_id"],
                tool_call_id=started.payload["tool_call_id"],
            )
            assert target.idempotency_key != record.intent.idempotency_key
            assert target.session_instance_id != record.intent.session_instance_id
            assert target.expected_run_epoch == request.expected_run_epoch
            assert target.expected_revision == record.revision
            assert record.intent.idempotency_key not in target.model_dump_json()
            assert record.intent.session_instance_id not in target.model_dump_json()
            codec = store.public_authority_alias_codec
            assert codec is not None
            previous_codec = PublicAuthorityAliasCodec(
                PublicAuthorityAliasKeyring(
                    active_key_id="previous",
                    keys={
                        "previous": SecretStr(urlsafe_b64encode(b"\x01" * 32).decode().rstrip("="))
                    },
                )
            )
            retired_codec = PublicAuthorityAliasCodec(
                PublicAuthorityAliasKeyring(
                    active_key_id="retired", keys={"retired": SecretStr("A" * 43)}
                )
            )
            assert await store.load_events("receipt-public") == before_events
            assert await store.load("receipt-public") == before_session
            assert lookups == []
            request = ToolEffectReconciliationRequest(
                **target.model_dump(),
                lookup=mode == "lookup",
                receipt=None
                if supplied is None
                else supplied.model_copy(
                    update={
                        "tool_call_id": target.tool_call_id,
                        "idempotency_key": target.idempotency_key,
                    }
                ),
            )
            for field, wrong in (
                ("session_instance_id", target.idempotency_key),
                ("idempotency_key", target.session_instance_id),
                ("idempotency_key", target.idempotency_key[:-1] + "!"),
                (
                    "idempotency_key",
                    retired_codec.encode(
                        record.intent.idempotency_key,
                        field_name="idempotency_key",
                        session_id="receipt-public",
                    ),
                ),
                (
                    "idempotency_key",
                    codec.encode(
                        record.intent.idempotency_key,
                        field_name="idempotency_key",
                        session_id="different-session",
                    ),
                ),
                (
                    "session_instance_id",
                    codec.encode(
                        record.intent.session_instance_id,
                        field_name="session_instance_id",
                        session_id="different-session",
                    ),
                ),
            ):
                changes = {field: wrong}
                if field == "idempotency_key" and request.receipt is not None:
                    changes["receipt"] = request.receipt.model_copy(update={field: wrong})
                with pytest.raises(ToolEffectConflict):
                    _ = [
                        event
                        async for event in app.reconcile_tool_effect(
                            request.model_copy(update=changes)
                        )
                    ]
            assert lookups == []
            assert await store.load_events("receipt-public") == before_events
            canary = "inspection-private-canary"
            protected_app = build_app(secret_redactor=SecretRedactor(canary))
            with pytest.raises(ToolEffectConflict) as rejected:
                await protected_app.inspect_tool_effect(
                    "receipt-public", tool_round_id=canary, tool_call_id=target.tool_call_id
                )
            assert canary not in str(rejected.value)
            assert canary not in repr(rejected.value)
            protected_app = build_app(secret_redactor=SecretRedactor(record.intent.idempotency_key))
            protected_target = await protected_app.inspect_tool_effect(
                "receipt-public",
                tool_round_id=target.tool_round_id,
                tool_call_id=target.tool_call_id,
            )
            assert protected_target == target
            assert canary not in caplog.text
            captured = capsys.readouterr()
            assert canary not in captured.out + captured.err
            assert await store.load_events("receipt-public") == before_events
            assert await store.load("receipt-public") == before_session
            # Tokens issued under a retained previous key resolve to the same
            # private request, including after application reconstruction.
            previous_aliases = {
                name: previous_codec.encode(
                    getattr(record.intent, name), field_name=name, session_id="receipt-public"
                )
                for name in ("session_instance_id", "idempotency_key")
            }
            request = request.model_copy(
                update={
                    **previous_aliases,
                    "receipt": None
                    if request.receipt is None
                    else request.receipt.model_copy(
                        update={"idempotency_key": previous_aliases["idempotency_key"]}
                    ),
                }
            )
            app = build_app()
        if lifecycle:
            before_factory = len(factory.requests)
            if fault_phase == "preflight":
                before_events = await store.load_events("receipt-public")
                before_session = await store.load("receipt-public")
                conflicts = [
                    ("session_instance_id", "wrong-instance"),
                    ("tool_round_id", "wrong-round"),
                    ("tool_call_id", "wrong-call"),
                    ("tool_name", "wrong-tool"),
                    ("idempotency_key", "wrong-key"),
                    ("expected_run_epoch", request.expected_run_epoch + 1),
                    ("expected_revision", request.expected_revision + 1),
                ]
                for field, value in conflicts:
                    changes = {field: value}
                    if request.receipt is not None and field in {
                        "tool_call_id",
                        "tool_name",
                        "idempotency_key",
                    }:
                        changes["receipt"] = request.receipt.model_copy(update={field: value})
                    with pytest.raises(ToolEffectConflict):
                        _ = [
                            event
                            async for event in app.reconcile_tool_effect(
                                request.model_copy(update=changes)
                            )
                        ]
                if request.receipt is not None:
                    for changes in (
                        {"receipt_schema": "unregistered-schema"},
                        {"receipt_schema_version": 2},
                    ):
                        with pytest.raises(ToolEffectConflict):
                            _ = [
                                event
                                async for event in app.reconcile_tool_effect(
                                    request.model_copy(
                                        update={
                                            "receipt": request.receipt.model_copy(update=changes)
                                        }
                                    )
                                )
                            ]
                assert len(factory.requests) == before_factory
                assert lookups == hook_observations == []
                assert await store.load("receipt-public") == before_session
                assert await store.load_events("receipt-public") == before_events
                assert (
                    ToolEffectRecord.model_validate(
                        await store.load_session_operation("receipt-public", store.effect_keys[0])
                    )
                    == record
                )
            if fault_phase == "factory-failure":
                factory.fail_create = True
                rejected = [event async for event in app.reconcile_tool_effect(request)]
                assert rejected[-1].type.value == "session.interrupted"
                assert len(factory.requests) == before_factory + 1
                assert lookups == hook_observations == []
                assert (
                    ToolEffectRecord.model_validate(
                        await store.load_session_operation("receipt-public", store.effect_keys[0])
                    )
                    == record
                )
                factory.fail_create = False
                current = await store.load("receipt-public")
                request = request.model_copy(update={"expected_run_epoch": current.run_epoch})
            # Reconstruct the app to ensure the factory, not a leftover concrete
            # environment from the failed run, supplies the recovery environment.
            app = build_app()
        if fault_phase in {"start-precommit", "start-postcommit"}:
            publish = store.publish_session_operation
            start_failure = OSError("reconciliation start acknowledgement unavailable")
            fail_once = True

            async def fail_start(*args, **kwargs):
                nonlocal fail_once
                if fail_once and any(
                    event.type.value == "tool.effect.reconciliation.started"
                    for event in kwargs.get("events", ())
                ):
                    fail_once = False
                    if fault_phase == "start-postcommit":
                        await publish(*args, **kwargs)
                    raise start_failure
                return await publish(*args, **kwargs)

            monkeypatch.setattr(store, "publish_session_operation", fail_start)
            if fault_phase == "start-precommit":
                with pytest.raises(OSError) as caught:
                    _ = [event async for event in app.reconcile_tool_effect(request)]
                assert caught.value is start_failure
                assert not fail_once
                assert lookups == []
                assert len(calls) == len(provider.requests) == 1
                assert (
                    ToolEffectRecord.model_validate(
                        await store.load_session_operation("receipt-public", store.effect_keys[0])
                    )
                    == record
                )
                assert not any(
                    event.type.value
                    in {
                        "tool.effect.reconciliation.started",
                        "tool.call.completed",
                        "tool.call.failed",
                    }
                    for event in await store.load_events("receipt-public")
                )
                current = await store.load("receipt-public")
                assert current.status.value != "running"
                monkeypatch.setattr(store, "publish_session_operation", publish)
                request = request.model_copy(update={"expected_run_epoch": current.run_epoch})
                app = build_app()
            # A committed admission with a lost acknowledgement now has exact
            # record readback, so normal reconciliation may continue once.
        if fault_phase in {"start-readback", "start-fanout"}:
            publish = store.publish_session_operation
            read = store.load_session_operation
            writer = app._recovery_coordinator._event_writer
            fanout = writer.fan_out_persisted
            write_failure = OSError("start committed but acknowledgement lost")
            read_failure = OSError("start exact readback unavailable")
            fail_once = True
            readback_pending = False

            async def fail_start_readback(*args, **kwargs):
                nonlocal fail_once, readback_pending
                result = await publish(*args, **kwargs)
                if fail_once and any(
                    event.type.value == "tool.effect.reconciliation.started"
                    for event in kwargs.get("events", ())
                ):
                    fail_once = False
                    readback_pending = True
                    raise write_failure
                return result

            async def fail_exact_read(*args, **kwargs):
                nonlocal readback_pending
                if readback_pending:
                    readback_pending = False
                    raise read_failure
                return await read(*args, **kwargs)

            async def fail_start_fanout(events):
                nonlocal fail_once
                if fail_once and any(
                    event.type.value == "tool.effect.reconciliation.started" for event in events
                ):
                    fail_once = False
                    raise write_failure
                return await fanout(events)

            if fault_phase == "start-readback":
                monkeypatch.setattr(store, "publish_session_operation", fail_start_readback)
                monkeypatch.setattr(store, "load_session_operation", fail_exact_read)
                with pytest.raises(ExceptionGroup) as caught:
                    _ = [event async for event in app.reconcile_tool_effect(request)]
                assert caught.value.exceptions == (write_failure, read_failure)
            else:
                monkeypatch.setattr(writer, "fan_out_persisted", fail_start_fanout)
                with pytest.raises(OSError) as caught:
                    _ = [event async for event in app.reconcile_tool_effect(request)]
                assert caught.value is write_failure
            assert not fail_once
            assert lookups == []
            assert len(calls) == len(provider.requests) == 1
            admitted = await assert_admitted(record)
            current = await store.load("receipt-public")
            assert current.status.value != "running"
            assert not any(
                event.type.value in {"tool.call.completed", "tool.call.failed"}
                for event in await store.load_events("receipt-public")
            )
            record = admitted
            request = request.model_copy(
                update={
                    "expected_run_epoch": current.run_epoch,
                    "expected_revision": admitted.revision,
                }
            )
            app = build_app()
        if fault_phase == "start-abandon":
            stream = app.reconcile_tool_effect(request)
            try:
                async for started in stream:
                    if started.type.value == "tool.effect.reconciliation.started":
                        break
                else:
                    pytest.fail("Reconciliation stream did not expose its start.")
                assert lookups == []
                assert len(calls) == len(provider.requests) == 1
                assert (await store.load("receipt-public")).status.value == "running"
                admitted = await assert_admitted(record)
            finally:
                await stream.aclose()
            assert lookups == []
            assert len(calls) == len(provider.requests) == 1
            current = await store.load("receipt-public")
            assert current.status.value == "interrupted"
            assert (
                ToolEffectRecord.model_validate(
                    await store.load_session_operation("receipt-public", store.effect_keys[0])
                )
                == admitted
            )
            assert not any(
                event.type.value in {"tool.call.completed", "tool.call.failed"}
                for event in await store.load_events("receipt-public")
            )
            record = admitted
            request = request.model_copy(
                update={
                    "expected_run_epoch": current.run_epoch,
                    "expected_revision": record.revision,
                }
            )
            app = build_app()
        if fault_phase in {"validation-abandon", "validation-plan-repair"}:
            stream = app.reconcile_tool_effect(request)
            try:
                async for validated in stream:
                    if validated.type.value == "tool.effect.receipt.validated":
                        break
                else:
                    pytest.fail("Reconciliation did not expose receipt validation.")
                assert len(calls) == len(lookups) == len(provider.requests) == 1
                selected = ToolEffectRecord.model_validate(
                    await store.load_session_operation("receipt-public", store.effect_keys[0])
                )
                assert selected.state == "reconciled_completed"
                assert selected.terminal.receipt is not None
                assert selected.resource_versions == {"part-1": "v1", "part-2": "v2"}
                atomic_events = await store.load_events("receipt-public")
                validation_index = next(
                    index
                    for index, event in enumerate(atomic_events)
                    if event.type.value == "tool.effect.receipt.validated"
                )
                assert atomic_events[validation_index + 1].id == selected.terminal.event_id
                assert validated.payload["receipt_evidence"]["receipt_digest"] == (
                    tool_effect_receipt_digest(selected.terminal.receipt)
                )
            finally:
                await stream.aclose()
            assert (await store.load("receipt-public")).status.value == "interrupted"
            assert len(calls) == len(lookups) == len(provider.requests) == 1
            assert (
                ToolEffectRecord.model_validate(
                    await store.load_session_operation("receipt-public", store.effect_keys[0])
                )
                == selected
            )
            # Keep the exact original request: selected receipt replay is not a
            # fresh decision and must neither revalidate nor redispatch.
            app = build_app()
            if fault_phase == "validation-plan-repair":
                from cayu import (
                    IncompleteSessionRecoveryAction,
                    IncompleteSessionRecoveryRequest,
                    RecoveryBlockerCode,
                    RecoveryPlanAction,
                    RecoveryPlanRequest,
                    RecoveryPlanSelection,
                )

                plan = await app.plan_recovery(
                    RecoveryPlanRequest(
                        selection=RecoveryPlanSelection(session_ids=("receipt-public",))
                    )
                )
                assert plan.items[0].allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)
                expected_blocker = (
                    RecoveryBlockerCode.TOOL_APPROVAL_REQUIRED
                    if approval_gate
                    else RecoveryBlockerCode.TOOL_EFFECT_CONTINUATION_REQUIRED
                )
                assert expected_blocker in {blocker.code for blocker in plan.items[0].blockers}
                repaired = await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="receipt-public")
                )
                expected_action = (
                    IncompleteSessionRecoveryAction.PENDING_APPROVAL
                    if approval_gate
                    else IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT
                )
                assert expected_action in repaired.actions
                assert len(calls) == len(lookups) == len(provider.requests) == 1
                assert (
                    ToolEffectRecord.model_validate(
                        await store.load_session_operation("receipt-public", store.effect_keys[0])
                    )
                    == selected
                )
        if fault_phase == "receipt-postcommit":
            publish = store.publish_session_operation
            lose_receipt_ack = True

            async def lose_atomic_receipt_ack(session_id, **kwargs):
                nonlocal lose_receipt_ack
                is_receipt = any(
                    event.type.value == "tool.effect.receipt.validated"
                    for event in kwargs["events"]
                )
                if is_receipt:
                    assert [event.type.value for event in kwargs["events"]] == [
                        "tool.effect.receipt.validated",
                        "tool.call.failed" if outcome == "failed" else "tool.call.completed",
                    ]
                published = await publish(session_id, **kwargs)
                if is_receipt and lose_receipt_ack:
                    lose_receipt_ack = False
                    raise OSError("atomic receipt acknowledgement lost")
                return published

            monkeypatch.setattr(store, "publish_session_operation", lose_atomic_receipt_ack)
        if callback_signal:
            if fault_phase == "cancel-cleanup-read":
                from contextvars import ContextVar

                cleanup_read = ContextVar("receipt_cleanup_read", default=False)
                cleanup = app._recovery_coordinator._cleanup_recovery_handoff
                load = store.load
                cleanup_error = OSError("receipt cleanup authority read failed")
                read_failed = False

                async def scoped_cleanup(**kwargs):
                    token = cleanup_read.set(
                        kwargs.get("invocation_context") is not None
                        and isinstance(kwargs.get("authoritative_failure"), asyncio.CancelledError)
                    )
                    try:
                        if cleanup_read.get() and not read_failed:
                            with monkeypatch.context() as cleanup_patch:
                                cleanup_patch.setattr(type(store), "load", fail_cleanup_read)
                                return await cleanup(**kwargs)
                        return await cleanup(**kwargs)
                    finally:
                        cleanup_read.reset(token)

                async def fail_cleanup_read(_store, *args, **kwargs):
                    nonlocal read_failed
                    if cleanup_read.get() and not read_failed:
                        read_failed = True
                        raise cleanup_error
                    return await load(*args, **kwargs)

                monkeypatch.setattr(
                    app._recovery_coordinator, "_cleanup_recovery_handoff", scoped_cleanup
                )

            async def collect():
                return [event async for event in app.reconcile_tool_effect(request)]

            owner = app._recovery_coordinator._effect_reconciliation_owner
            task = asyncio.create_task(collect())
            try:
                entered = asyncio.create_task(callback_entered.wait())
                try:
                    done, _ = await asyncio.wait(
                        {entered, task}, timeout=10, return_when=asyncio.FIRST_COMPLETED
                    )
                    if task in done:
                        await task
                    assert entered in done, "Reconciliation callback did not start"
                finally:
                    entered.cancel()
                    await asyncio.gather(entered, return_exceptions=True)
                assert (await store.load("receipt-public")).status.value == "running"
                if fault_phase == "deadline":
                    with pytest.raises(ToolEffectReconciliationTimeout):
                        await task
                    assert not task.cancelled()
                    assert task.cancelling() == 0
                else:
                    count = 2 if fault_phase == "repeated-cancel" else 1
                    for _ in range(count):
                        task.cancel()
                    with pytest.raises(asyncio.CancelledError) as cancelled:
                        await task
                    assert task.cancelled()
                    assert task.cancelling() == count
                    if fault_phase == "cancel-cleanup-read":
                        assert read_failed
                        pending_errors = [cancelled.value]
                        seen_errors = set()
                        cleanup_failures = []
                        while pending_errors:
                            error = pending_errors.pop()
                            if id(error) in seen_errors:
                                continue
                            seen_errors.add(id(error))
                            if isinstance(error, OSError) and str(error) == str(cleanup_error):
                                cleanup_failures.append(error)
                            if error.__cause__ is not None:
                                pending_errors.append(error.__cause__)
                            if isinstance(error, BaseExceptionGroup):
                                pending_errors.extend(error.exceptions)
                        assert len(cleanup_failures) == 1
                assert owner.pending_operations == 1
                retained_operations = tuple(owner._operations._operations)
                assert len(retained_operations) == 1
                assert not callback_settled.is_set()
                assert (await store.load("receipt-public")).status.value == "interrupted"
                admitted = await assert_admitted(record)
                before_release = await store.load_events("receipt-public")
                callback_release.set()
                await asyncio.wait_for(callback_settled.wait(), 10)
                # The registry owns the invocation envelope, which settles after
                # the extension coroutine and its result-validation phase.
                await asyncio.wait_for(
                    asyncio.gather(*(asyncio.shield(item) for item in retained_operations)), 10
                )
                for _ in range(4):
                    await asyncio.sleep(0)
                assert owner.pending_operations == 0, (
                    owner._operations._reservations,
                    [(item.done(), item.cancelled()) for item in owner._operations._operations],
                    caplog.text,
                )
                # A late validated result has no publication authority after its
                # request stopped waiting. Only a new explicit claim may select it.
                assert await store.load_events("receipt-public") == before_release
                assert (
                    ToolEffectRecord.model_validate(
                        await store.load_session_operation("receipt-public", store.effect_keys[0])
                    )
                    == admitted
                )
                current = await store.load("receipt-public")
                request = request.model_copy(
                    update={
                        "expected_run_epoch": current.run_epoch,
                        "expected_revision": admitted.revision,
                    }
                )
                recovered = [event async for event in app.reconcile_tool_effect(request)]
                assert recovered[-1].type.value == "session.completed"
                assert len(calls) == 1 and len(lookups) == 2
                assert len(provider.requests) == 2
                terminals = [
                    event
                    for event in await store.load_events("receipt-public")
                    if event.type.value in {"tool.call.completed", "tool.call.failed"}
                ]
                assert len(terminals) == 1
            finally:
                callback_release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await owner.aclose(timeout_seconds=10)
            return
        if fault_phase in {"child-cancel", "callback-timeout"}:

            async def consume_callback_failure():
                return [event async for event in app.reconcile_tool_effect(request)]

            caller = asyncio.create_task(consume_callback_failure())
            expected_error = RuntimeError if fault_phase == "child-cancel" else TimeoutError
            with pytest.raises(expected_error) as caught:
                await caller
            assert not caller.cancelled()
            assert caller.cancelling() == 0
            assert type(caught.value) is expected_error
            if fault_phase == "callback-timeout":
                assert str(caught.value) == "application callback timeout"
            admitted = await assert_admitted(record)
            assert (await store.load("receipt-public")).status.value == "interrupted"
            assert len(calls) == len(lookups) == len(provider.requests) == 1
            assert not any(
                event.type.value
                in {
                    "tool.call.completed",
                    "tool.call.failed",
                    "tool.effect.receipt.validated",
                }
                for event in await store.load_events("receipt-public")
            )
            current = await store.load("receipt-public")
            request = request.model_copy(
                update={
                    "expected_run_epoch": current.run_epoch,
                    "expected_revision": admitted.revision,
                }
            )
            recovered = [event async for event in app.reconcile_tool_effect(request)]
            assert recovered[-1].type.value == "session.completed"
            assert len(calls) == 1 and len(lookups) == 2 and len(provider.requests) == 2
            return
        if review_policy is not None:
            assert request.review_reference is not None
            before_denials = ToolEffectRecord.model_validate(
                await store.load_session_operation("receipt-public", store.effect_keys[0])
            )
            for reference, rejection in (
                (None, HumanReviewDenied),
                (
                    request.review_reference.model_copy(update={"content_tag": "0" * 64}),
                    HumanReviewConflict,
                ),
                (
                    request.review_reference.model_copy(
                        update={
                            "context": review_context.model_copy(
                                update={"recipient": "another-operator"}
                            )
                        }
                    ),
                    HumanReviewDenied,
                ),
            ):
                with pytest.raises(rejection):
                    _ = [
                        event
                        async for event in app.reconcile_tool_effect(
                            request.model_copy(update={"review_reference": reference})
                        )
                    ]
                assert len(calls) == 1 and lookups == []
                assert (
                    ToolEffectRecord.model_validate(
                        await store.load_session_operation("receipt-public", store.effect_keys[0])
                    )
                    == before_denials
                )
            review_policy.can_decide = False
            with pytest.raises(HumanReviewDenied):
                _ = [event async for event in app.reconcile_tool_effect(request)]
            assert lookups == []
            review_policy.can_decide = True
        if fault_phase in {"unlisted-resource", "hostile-result"}:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                with pytest.raises(ValueError) as rejected:
                    _ = [event async for event in app.reconcile_tool_effect(request)]
            if fault_phase == "unlisted-resource":
                assert "registered allow-list" in str(rejected.value)
            else:
                assert hostile_formatting == []
                assert "private-callback-canary" not in str(rejected.value)
                assert "private-callback-canary" not in repr(rejected.value)
                assert not caught_warnings
            await assert_admitted(record)
            durable_events = await store.load_events("receipt-public")
            assert not any(
                event.type.value
                in {
                    "tool.call.completed",
                    "tool.call.failed",
                    "tool.effect.reconciliation.observed",
                    "tool.effect.reconciliation.conflict",
                    "tool.effect.receipt.validated",
                }
                for event in durable_events
            )
            assert "private-resource-canary" not in repr(
                [event.model_dump() for event in durable_events]
            )
            assert "private-resource-canary" not in caplog.text
            assert "private-callback-canary" not in repr(
                [event.model_dump() for event in durable_events]
            )
            assert "private-callback-canary" not in caplog.text
            captured = capsys.readouterr()
            assert "private-resource-canary" not in captured.out + captured.err
            assert "private-callback-canary" not in captured.out + captured.err
            assert (await store.load("receipt-public")).status.value == "interrupted"
            assert len(calls) == len(lookups) == len(provider.requests) == 1
            return
        if fault_phase == "terminal":
            transcript_reader = "load_transcript_snapshot" if approval_gate else "load_transcript"
            read_transcript = getattr(store, transcript_reader)
            fail_once = True

            async def fail_after_receipt(*args, **kwargs):
                nonlocal fail_once
                effect = await store.load_session_operation("receipt-public", store.effect_keys[0])
                if fail_once and effect["state"].startswith("reconciled_"):
                    fail_once = False
                    raise OSError("post-receipt transcript read unavailable")
                return await read_transcript(*args, **kwargs)

            monkeypatch.setattr(store, transcript_reader, fail_after_receipt)
            if approval_gate:
                # The existing approval closure owner publishes pre-close failure
                # as a resumable interruption instead of raising it to the caller.
                interrupted = [event async for event in app.reconcile_tool_effect(request)]
                assert interrupted[-1].type.value == "session.interrupted"
                assert "post-receipt transcript" in str(interrupted[-1].payload)
            else:
                with pytest.raises(OSError, match="post-receipt transcript"):
                    _ = [event async for event in app.reconcile_tool_effect(request)]
            assert not fail_once
            assert len(calls) == len(lookups) == 1
            assert len(provider.requests) == 1
            assert (await store.load("receipt-public")).status.value == "interrupted"
            committed = ToolEffectRecord.model_validate(
                await store.load_session_operation("receipt-public", store.effect_keys[0])
            )
            assert committed.state == f"reconciled_{outcome}"
            assert (
                await store.load_runtime_publication_receipt(
                    "receipt-public",
                    f"approval-close:{record.intent.approval_id}"
                    if approval_gate
                    else f"tool-round:{record.intent.tool_round_id}",
                )
                is None
            )
            app = build_app()
        interrupted_observations = []
        if fault_phase in {"observation-error", "observation-abandon"}:
            transition_status = store.transition_status
            reached = asyncio.Event()
            fail_once = True

            async def stop_after_observation(*args, **kwargs):
                nonlocal fail_once
                effect = await store.load_session_operation("receipt-public", store.effect_keys[0])
                if (
                    fail_once
                    and effect.get("observation") is not None
                    and kwargs.get("to_status").value == "interrupted"
                ):
                    fail_once = False
                    reached.set()
                    raise OSError("post-observation status write unavailable")
                return await transition_status(*args, **kwargs)

            if fault_phase == "observation-error":
                monkeypatch.setattr(store, "transition_status", stop_after_observation)
            stream = app.reconcile_tool_effect(request)
            try:
                if fault_phase == "observation-error":
                    with pytest.raises(OSError, match="post-observation status"):
                        async for event in stream:
                            interrupted_observations.append(event)
                else:
                    async for event in stream:
                        interrupted_observations.append(event)
                        if event.type.value == observation_event_type:
                            break
                    # Delivery is backpressured: closing at this yield exercises
                    # the committed observation before normal interruption resumes.
                    assert (await store.load("receipt-public")).status.value == "running"
                    await asyncio.wait_for(stream.aclose(), timeout=10)
            finally:
                await stream.aclose()
            if fault_phase == "observation-error":
                assert reached.is_set()
            assert (await store.load("receipt-public")).status.value == "interrupted"
            committed = ToolEffectRecord.model_validate(
                await store.load_session_operation("receipt-public", store.effect_keys[0])
            )
            assert committed.state == "outcome_unknown"
            assert committed.observation is not None
            assert len(calls) == len(lookups) == len(provider.requests) == 1
            app = build_app()
        events = []
        if outcome == "conflict":
            with pytest.raises(ToolEffectConflict, match="rejected the receipt"):
                async for event in app.reconcile_tool_effect(request):
                    events.append(event)
        else:
            events = [event async for event in app.reconcile_tool_effect(request)]
        assert len(calls) == 1
        assert len(lookups) == 1
        if fault_phase == "start-postcommit":
            assert not fail_once
        if fault_phase == "receipt-postcommit":
            assert not lose_receipt_ack
        settled = ToolEffectRecord.model_validate(
            await store.load_session_operation("receipt-public", store.effect_keys[0])
        )
        if fault_phase is None:
            public_starts = [
                event
                for event in events
                if event.type.value == "tool.effect.reconciliation.started"
            ]
            assert len(public_starts) == 1
            assert public_starts[0].payload["schema_version"] == 1
            assert public_starts[0].payload["lookup"] is (mode == "lookup")
            assert public_starts[0].payload["expected_revision"] == record.revision
            assert len(public_starts[0].payload["request_digest"]) == 64
            assert len(public_starts[0].payload["intent_digest"]) == 64
            assert "receipt" not in public_starts[0].payload
            assert "idempotency_key" not in public_starts[0].payload
        terminal_events = [
            event
            for event in await store.load_events("receipt-public")
            if event.type.value in {"tool.call.completed", "tool.call.failed"}
        ]
        if outcome in {"not_found", "unsupported", "conflict"}:
            # An accepted nonterminal decision advances evidence, not effect state.
            assert settled.state == record.state == "outcome_unknown"
            assert settled.intent == record.intent
            assert settled.dispatch_id == record.dispatch_id
            assert settled.revision == record.revision + 2
            assert settled.reconciliation_attempt is None
            assert settled.observation.result.outcome == outcome
            assert len(provider.requests) == 1
            assert terminal_events == []
            assert not any(
                event.type.value == "tool.effect.receipt.validated"
                for event in await store.load_events("receipt-public")
            )
            if fault_phase in {"observation-error", "observation-abandon"}:
                assert events == [
                    event
                    for event in interrupted_observations
                    if event.type.value == observation_event_type
                ]
            else:
                assert events[-1].type.value == "session.interrupted"
            before_replay = await store.load_events("receipt-public")
            app = build_app()
            replayed = []
            if outcome == "conflict":
                with pytest.raises(ToolEffectConflict, match="rejected the receipt"):
                    async for event in app.reconcile_tool_effect(request):
                        replayed.append(event)
            else:
                replayed = [event async for event in app.reconcile_tool_effect(request)]
            original_observation = next(
                event for event in events if event.type.value == observation_event_type
            )
            if outcome == "conflict":
                assert original_observation.payload["kind"] == "validator_rejected"
                assert original_observation.payload["result"]["outcome"] == "conflict"
                durable_conflict = next(
                    event for event in before_replay if event.id == settled.observation.event_id
                )
                projected = project_persisted_runtime_event(
                    durable_conflict, sequence=1, redactor=SecretRedactor()
                )
                assert projected.payload["kind"] == "validator_rejected"
                assert projected.payload["result"]["outcome"] == "conflict"
                assert projected.payload["resource_versions"] == {"part-1": "v1"}
                assert projected.payload["idempotency_key"] == "[PRIVATE_EVENT_AUTHORITY]"
                assert record.intent.idempotency_key not in repr(projected.payload)
                assert not any(
                    event.type.value == "tool.effect.reconciliation.observed"
                    for event in before_replay
                )
            assert replayed == [original_observation]
            assert settled.resource_versions == {"part-1": "v1"}
            assert await store.load_events("receipt-public") == before_replay
            assert len(calls) == len(lookups) == len(provider.requests) == 1
            current_session = await store.load("receipt-public")
            next_request = request.model_copy(
                update={
                    "expected_run_epoch": current_session.run_epoch,
                    "expected_revision": settled.revision,
                }
            )
            if fault_phase == "new-start-abandon":
                stream = app.reconcile_tool_effect(next_request)
                try:
                    async for event in stream:
                        if event.type.value == "tool.effect.reconciliation.started":
                            break
                    else:
                        pytest.fail("New explicit decision did not expose admission.")
                    assert len(lookups) == 1
                finally:
                    await stream.aclose()
                admitted = ToolEffectRecord.model_validate(
                    await store.load_session_operation("receipt-public", store.effect_keys[0])
                )
                assert admitted.reconciliation_attempt is not None
                assert admitted.observation == settled.observation
                assert admitted.revision == settled.revision + 1
                current_session = await store.load("receipt-public")
                assert current_session.status.value == "interrupted"
                before_old_replay = await store.load_events("receipt-public")
                with pytest.raises(ToolEffectConflict, match="stale session authority"):
                    _ = [event async for event in app.reconcile_tool_effect(request)]
                assert await store.load_events("receipt-public") == before_old_replay
                assert await store.load("receipt-public") == current_session
                assert len(lookups) == 1
                settled = admitted
                next_request = next_request.model_copy(
                    update={
                        "expected_run_epoch": current_session.run_epoch,
                        "expected_revision": admitted.revision,
                    }
                )
            if outcome == "conflict":
                with pytest.raises(ToolEffectConflict, match="rejected the receipt"):
                    _ = [event async for event in app.reconcile_tool_effect(next_request)]
            else:
                _ = [event async for event in app.reconcile_tool_effect(next_request)]
            later = ToolEffectRecord.model_validate(
                await store.load_session_operation("receipt-public", store.effect_keys[0])
            )
            assert later.state == "outcome_unknown"
            assert later.resource_versions == {"part-1": "v1"}
            assert later.observation.result.resource_versions == {}
            assert later.revision == settled.revision + 2
            assert later.reconciliation_attempt is None
            assert len(calls) == len(provider.requests) == 1
            assert len(lookups) == 2
            # A new explicit version supersedes the old unresolved observation.
            # An old request cannot reselect it or invoke the callback again.
            before_stale_replay = await store.load_events("receipt-public")
            with pytest.raises(ToolEffectConflict, match="stale session authority"):
                _ = [event async for event in app.reconcile_tool_effect(request)]
            assert await store.load_events("receipt-public") == before_stale_replay
            assert len(calls) == len(provider.requests) == 1
            assert len(lookups) == 2
        else:
            assert settled.state == f"reconciled_{outcome}"
            assert settled.terminal.receipt.receipt_id == "external-receipt"
            assert terminal_events[0].payload["reconciliation_state"] == "reconciled"
            if fault_phase is None and mode == "lookup":
                report = await runtime_evidence(
                    app,
                    RuntimeEvidenceRequest(
                        root_session_id="receipt-public", max_sessions=10, max_events=1000
                    ),
                )
                receipts = report.sessions[0].receipts
                assert len(receipts) == 1
                assert receipts[0].receipt_id == "external-receipt"
                assert receipts[0].reconciliation_state == "reconciled"
                evidence = receipts[0].receipt_evidence
                assert evidence is not None
                durable_receipt = settled.terminal.receipt
                assert evidence.outcome == outcome
                assert evidence.source == durable_receipt.source
                assert evidence.receipt_schema == durable_receipt.receipt_schema
                assert evidence.receipt_schema_version == durable_receipt.receipt_schema_version
                assert evidence.receipt_digest == tool_effect_receipt_digest(durable_receipt)
                assert evidence.observed_at == durable_receipt.observed_at
                assert evidence.integrity == durable_receipt.integrity
                assert evidence.resource_versions == durable_receipt.resource_versions
            assert settled.resource_versions == {"part-1": "v1", "part-2": "v2"}
            assert settled.terminal.receipt.resource_versions == settled.resource_versions
            assert len(terminal_events) == 1
            assert terminal_events[0].id == settled.terminal.event_id
            public_terminal = next(
                event
                for event in events
                if event.type.value in {"tool.call.completed", "tool.call.failed"}
            )
            receipt = settled.terminal.receipt
            assert receipt is not None
            assert public_terminal.payload["effect_reconciled"] is True
            validations = [
                event
                for event in await store.load_events("receipt-public")
                if event.type.value == "tool.effect.receipt.validated"
            ]
            assert len(validations) == 1
            assert (
                validations[0].payload["receipt_evidence"]
                == terminal_events[0].payload["receipt_evidence"]
            )
            assert "result" not in validations[0].payload
            assert "idempotency_key" not in validations[0].payload
            if fault_phase is None:
                public_validation = next(
                    event for event in events if event.type.value == "tool.effect.receipt.validated"
                )
                assert (
                    public_validation.payload["receipt_evidence"]
                    == public_terminal.payload["receipt_evidence"]
                )
                assert events.index(public_validation) < events.index(public_terminal)
            assert public_terminal.payload["receipt_evidence"] == {
                "schema_version": 1,
                "receipt_id": receipt.receipt_id,
                "receipt_schema": receipt.receipt_schema,
                "receipt_schema_version": receipt.receipt_schema_version,
                "outcome": receipt.outcome,
                "source": receipt.source,
                "observed_at": receipt.observed_at.isoformat(),
                "receipt_digest": tool_effect_receipt_digest(receipt),
                "integrity": dict(receipt.integrity),
                "resource_versions": dict(receipt.resource_versions),
            }
            assert len(provider.requests) == 2, [
                (event.type.value, event.payload.get("error"))
                for event in events
                if event.type.value in {"session.failed", "session.interrupted"}
            ]
            assert events[-1].type.value == "session.completed"
            if lifecycle:
                assert len(factory.requests) > before_factory
                assert hook_observations == ["verified external outcome"]
            transcript = await store.load_transcript("receipt-public")
            assert transcript[2].content[0].content == "verified external outcome"
            assert transcript[2].content[0].is_error is (outcome == "failed")
            before_replay = await store.load_events("receipt-public")
            replayed = [event async for event in app.reconcile_tool_effect(request)]
            assert len(replayed) == 1
            assert replayed[0].id == next(
                event.id
                for event in events
                if event.type.value in {"tool.call.completed", "tool.call.failed"}
            )
            assert await store.load_events("receipt-public") == before_replay
            assert len(calls) == 1
            assert len(lookups) == 1
            assert len(provider.requests) == 2
            if lifecycle:
                assert hook_observations == ["verified external outcome"]
            if fault_phase == "inspection":
                active_request = request.model_copy(
                    update={
                        "session_instance_id": target.session_instance_id,
                        "idempotency_key": target.idempotency_key,
                        "receipt": None
                        if request.receipt is None
                        else request.receipt.model_copy(
                            update={"idempotency_key": target.idempotency_key}
                        ),
                    }
                )
                active_replay = [event async for event in app.reconcile_tool_effect(active_request)]
                assert [event.id for event in active_replay] == [event.id for event in replayed]
                assert await store.load_events("receipt-public") == before_replay
                assert len(calls) == len(lookups) == 1
                assert len(provider.requests) == 2
            for field, value in (
                ("session_instance_id", "different-incarnation"),
                ("tool_round_id", "different-round"),
                ("tool_call_id", "different-call"),
                ("tool_name", "different-tool"),
                ("idempotency_key", "different-key"),
                ("expected_run_epoch", request.expected_run_epoch + 1),
                ("expected_revision", request.expected_revision + 1),
                ("max_steps", 1),
            ):
                changes = {field: value}
                if request.receipt is not None and field in {
                    "tool_call_id",
                    "tool_name",
                    "idempotency_key",
                }:
                    changes["receipt"] = request.receipt.model_copy(update={field: value})
                conflicting = request.model_copy(update=changes)
                with pytest.raises(ToolEffectConflict):
                    _ = [event async for event in app.reconcile_tool_effect(conflicting)]
            if request.receipt is not None:
                for field, value in (
                    ("receipt_id", "different-receipt"),
                    ("receipt_schema", "different-schema"),
                    ("receipt_schema_version", 2),
                    ("outcome", "completed" if outcome == "failed" else "failed"),
                    ("message", "different output"),
                    ("structured", {"version": 9}),
                    ("resource_versions", {"deployment": "different-version"}),
                    ("observed_at", datetime(2026, 9, 9, tzinfo=UTC)),
                    ("source", "adapter"),
                    ("integrity", {"proof": "different-proof"}),
                ):
                    conflicting = request.model_copy(
                        update={"receipt": request.receipt.model_copy(update={field: value})}
                    )
                    with pytest.raises(ToolEffectConflict):
                        _ = [event async for event in app.reconcile_tool_effect(conflicting)]
            assert await store.load_events("receipt-public") == before_replay
            assert len(calls) == len(lookups) == 1
            assert len(provider.requests) == 2

    async def run_backend():
        codec = (
            PublicAuthorityAliasCodec(
                PublicAuthorityAliasKeyring(
                    active_key_id="test",
                    keys={
                        "test": SecretStr("A" * 43),
                        "previous": SecretStr(urlsafe_b64encode(b"\x01" * 32).decode().rstrip("=")),
                    },
                )
            )
            if fault_phase == "inspection" or control_secrets is not None
            else None
        )
        store = (
            _ObservingStore(public_authority_alias_codec=codec)
            if backend == "memory"
            else _ObservingSQLiteStore(
                str(tmp_path / "public-receipts.db"), public_authority_alias_codec=codec
            )
        )
        try:
            await scenario(store)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run_backend())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True])
@pytest.mark.parametrize("outcome", ["not_found", "unsupported", "conflict"])
def test_public_observation_controls_survive_matching_secret_redaction(
    outcome, backend, approval_gate, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        outcome=outcome,
        fault_phase=None,
        backend=backend,
        mode="lookup",
        approval_gate=approval_gate,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        caplog=caplog,
        capsys=capsys,
        control_secrets=[outcome, "partial", "outcome_unknown", "validator_rejected"],
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", ["completed", "failed"])
@pytest.mark.parametrize("approval_gate", [False, True])
def test_public_receipt_controls_survive_matching_secret_redaction(
    outcome, backend, approval_gate, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        outcome=outcome,
        fault_phase=None,
        backend=backend,
        mode="lookup",
        approval_gate=approval_gate,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        caplog=caplog,
        capsys=capsys,
        control_secrets=["reconciled", outcome],
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True])
@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_public_receipt_validation_lost_ack_is_reconciled_atomically(
    backend, approval_gate, outcome, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        outcome,
        "receipt-postcommit",
        backend,
        "receipt",
        approval_gate,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True])
@pytest.mark.parametrize("mode", ["lookup", "receipt"])
def test_public_receipt_validation_abandonment_retains_atomic_selection(
    backend, approval_gate, mode, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        "validation-abandon",
        backend,
        mode,
        approval_gate,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True], ids=["ordinary", "approval"])
def test_public_receipt_selection_survives_recovery_plan_repair(
    backend, approval_gate, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        "validation-plan-repair",
        backend,
        "lookup",
        approval_gate,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True])
@pytest.mark.parametrize("mode", ["lookup", "receipt"])
def test_public_reconciliation_start_abandonment_prevents_callback(
    backend, approval_gate, mode, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        "start-abandon",
        backend,
        mode,
        approval_gate,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True])
@pytest.mark.parametrize(
    "fault_phase",
    [
        "start-precommit",
        "start-postcommit",
        "start-readback",
        "start-fanout",
    ],
)
def test_public_reconciliation_start_requires_proven_durable_admission(
    backend, approval_gate, fault_phase, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        fault_phase,
        backend,
        "receipt",
        approval_gate,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True])
def test_new_admission_fences_prior_observation_replay(
    backend, approval_gate, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "not_found",
        "new-start-abandon",
        backend,
        "lookup",
        approval_gate,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("mode", ["lookup", "receipt"])
@pytest.mark.parametrize("approval_gate", [False, True])
def test_public_inspection_supplies_safe_exact_reconciliation_target(
    backend, mode, approval_gate, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        "inspection",
        backend,
        mode,
        approval_gate,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_inspection_without_explicit_keyring_follows_store_authority(
    backend, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        "inspection-no-key",
        backend,
        "lookup",
        False,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("mode", ["lookup", "receipt"])
def test_approval_reconciliation_preserves_configured_human_review(
    backend, mode, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        "human-review",
        backend,
        mode,
        True,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True], ids=["ordinary", "approval"])
@pytest.mark.parametrize("signal", ["cancel", "repeated-cancel", "deadline"])
def test_public_reconciliation_retains_late_callback_without_settlement(
    backend, approval_gate, signal, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        signal,
        backend,
        "lookup",
        approval_gate,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True], ids=["ordinary", "approval"])
@pytest.mark.parametrize("fault_phase", ["child-cancel", "callback-timeout"])
def test_public_reconciliation_preserves_callback_failure_classification(
    backend, approval_gate, fault_phase, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        fault_phase,
        backend,
        "lookup",
        approval_gate,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True], ids=["ordinary", "approval"])
@pytest.mark.parametrize("fault_phase", ["lifecycle", "factory-failure"])
def test_public_reconciliation_uses_environment_and_observe_only_hooks(
    backend, approval_gate, fault_phase, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        fault_phase,
        backend,
        "lookup",
        approval_gate,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True], ids=["ordinary", "approval"])
@pytest.mark.parametrize("mode", ["lookup", "receipt"])
def test_public_reconciliation_rejects_invalid_authority_before_factory(
    backend, approval_gate, mode, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        "preflight",
        backend,
        mode,
        approval_gate,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("disposition", ["confirmed", "unresolved", "not_started", "missing"])
def test_public_resume_requires_positive_native_journal_evidence(
    backend, disposition, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        f"native-{disposition}",
        backend,
        "lookup",
        False,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True], ids=["ordinary", "approval"])
def test_public_reconciliation_rejects_hostile_callback_without_diagnostics(
    backend, approval_gate, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        "hostile-result",
        backend,
        "lookup",
        approval_gate,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_reconciliation_cancellation_survives_cleanup_authority_read_failure(
    backend, tmp_path, monkeypatch, caplog, capsys
):
    test_public_reconciliation_consumes_verified_outcome_without_external_replay(
        "completed",
        "cancel-cleanup-read",
        backend,
        "lookup",
        False,
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
    )
