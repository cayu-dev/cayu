"""Native material-bound preparation reaches ordinary inert recipient creation."""

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from examples.collaboration.planning import fork_policy, fresh_policy
from pydantic import ValidationError
from tests.artifacts.test_resource_transfer_templates import registered_template
from tests.artifacts.test_resources import preparation_permit
from tests.core.test_budget_binding import _binding
from tests.core.test_builtin_tools import TINY_PNG_BYTES
from tests.core.test_collaboration_request_foundation import setup
from tests.core.test_context_selection_exclusion import reopened_store
from tests.core.test_explicit_session_compaction import RecordingCompactor
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_prepared_admission_public import PreparationResolver
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_request_planning_contracts import _policy
from tests.core.test_request_planning_fresh import declined_successor
from tests.core.test_request_planning_public import complete_plan, lookup_plan
from tests.core.test_request_planning_resource_owner import assert_stage_envelope
from tests.core.test_request_planning_view_owner import source_scenario

from cayu.agents import AgentSpec
from cayu.artifacts import ArtifactScope, LocalArtifactStore
from cayu.artifacts.attachments import file_attachment
from cayu.artifacts.resources import (
    LocalArtifactResourceOwner,
    MandateResourcePreparationReader,
    ResourceOwnerError,
    ResourceOwnerUnavailable,
)
from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ExactMatch,
    ExactNotFound,
    ObjectRef,
)
from cayu.collaboration._planning_creation_types import (
    RequestCreationStageCommand,
    creation_stage_command,
)
from cayu.collaboration._planning_resource_types import (
    RequestResourceStageAdoption,
    RequestResourceTransferStageCommand,
)
from cayu.collaboration._planning_stages import creation_stage_intent, resource_stage_intent
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration.mandates import MandateDenied
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import (
    RequestPlanningControl,
    RequestPlanningFork,
    RequestPlanningFresh,
    RequestPlanningRequest,
    planning_policy_commitment,
    preparation_stage_count,
)
from cayu.collaboration.prepared_admission import created_admission_target
from cayu.collaboration.recipient_preparation import ResourceRecipientCreationPreparation
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration
from cayu.collaboration.requests import RequestAdmissionCommand
from cayu.collaboration.resource_preparation import RequestPlanningResource
from cayu.context import CheckpointCompactionContextPolicy
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import FilePart, Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import CompactSessionRequest, ResumeRequest, RunRequest
from cayu.sessions._planning_creation_owner import NativePlanningCreationOwner
from cayu.sessions._planning_resource_owner import NativePlanningResourceOwner
from cayu.sessions._recipient_preparation import (
    resolve_resource_recipient,
    resolved_material_creation_request,
)
from cayu.sessions.context_views import RecipientSessionCreationRequest
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio

# These journeys qualify reconstruction and ownership, not deadline timing.
# Revalidating every retained FORK/resource frontier under parallel backend
# qualification can exceed six minutes. Keep a finite scenario window with a
# narrower resource deadline; dedicated native deadline tests cover expiry.
JOURNEY_TTL_MS = 900_000
RESOURCE_LIFETIME_MS = JOURNEY_TTL_MS - 60_000


def resource_environment(artifacts):
    # Restart reconstructs this same static behavior. An undeclared environment
    # intentionally has process-local identity and cannot prove that equality.
    return Environment(
        EnvironmentSpec(
            name="resource",
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:planning-resource-environment",
                behavior_version="1",
                implementation_version="1",
            ),
        ),
        artifact_store=artifacts,
    )


@pytest.fixture
def artifact_store(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    artifact = asyncio.run(
        store.put_bytes(
            TINY_PNG_BYTES,
            artifact_id="art_" + "1" * 32,
            filename="input.png",
            content_type="image/png",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
    )
    return store, artifact


@pytest.mark.parametrize(
    "interruption",
    [
        "none",
        "cancel",
        "ack",
        "planner",
        "planner_restart",
        "planner_fork",
        "planner_fork_restart",
        "planner_fork_parent",
        "planner_cancel_acquire",
        "planner_cancel_transfer",
        "planner_cancel_second",
        "planner_ack_acquire",
        "planner_ack_transfer",
        "planner_expired",
        "planner_fork_expired",
    ],
)
async def test_exact_resource_preparation_creates_one_inert_child(
    tmp_path, artifact_store, monkeypatch, native_stores, interruption
):
    await resource_creation_journey(
        tmp_path, artifact_store, monkeypatch, native_stores, interruption
    )


@pytest.mark.parametrize("native_stores", ["sqlite", "postgres"], indirect=True)
async def test_resource_transfer_process_loss_recovers_through_public_planning(
    tmp_path, artifact_store, monkeypatch, native_stores
):
    await resource_creation_journey(
        tmp_path, artifact_store, monkeypatch, native_stores, "planner_process"
    )


@pytest.mark.parametrize("event_capacity", [7, 9])
async def test_optional_resource_capacity_cannot_strand_planning_cleanup(
    tmp_path, artifact_store, monkeypatch, native_stores, event_capacity
):
    await resource_creation_journey(
        tmp_path, artifact_store, monkeypatch, native_stores, f"planner_capacity_{event_capacity}"
    )


async def resource_process_journey(application, planning, policy, reader, destination, stores):
    """Die after native acceptance, then reconcile in a genuinely fresh process."""
    from cayu.collaboration._planning_records import RequestPlanningRecord

    backend, address = stores[3]
    assert backend in {"sqlite", "postgres"}
    session_address = str(stores[1].path) if backend == "sqlite" else address
    material = {
        "backend": backend,
        "address": address,
        "session_address": session_address,
        "expected": planning.model_dump(mode="json"),
        "policy": policy.model_dump(mode="json"),
        "resource": {
            "artifact_root": str(destination._store.root),
            "owner_journal": str(destination._journal.root),
            "selector_journal": str(reader._owner._journal.root),
            "resolution": reader._resolver.resolution.model_dump(mode="json"),
            "context": reader._context.model_dump(mode="json"),
            "registration": reader._registration.model_dump(mode="json"),
            "initialized": reader._initialized.model_dump(mode="json"),
        },
    }

    async def run(mode):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tests.recovery.request_planning_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(json.dumps({**material, "mode": mode}).encode()),
                JOURNEY_TTL_MS / 1000,
            )
            assert process.returncode == (19 if mode == "crash_after_resource_transfer" else 0), (
                stderr.decode()
            )
            return stdout
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    await run("crash_after_resource_transfer")
    retained = await lookup_plan(
        application,
        planning,
        application._request_coordinator._registration.mandates.recipient.context,
    )
    assert isinstance(retained, ExactMatch)
    assert retained.receipt.state == "preparing"
    assert retained.receipt.stage_count == retained.receipt.pending_stages == 2
    result = RequestPlanningRecord.model_validate_json(await run("recover_resource"))
    assert result.receipt.command == planning
    return result


async def resource_creation_journey(
    tmp_path, artifact_store, monkeypatch, native_stores, interruption
):
    artifacts, artifact = artifact_store
    (
        source,
        command,
        permit,
        destination,
        template,
        reader,
        resolver,
        _source_ledger,
        ledger,
    ) = await registered_template(
        tmp_path,
        artifacts,
        artifact,
        planning_deadline=int(time.time() * 1000) + RESOURCE_LIFETIME_MS,
        collaboration_store=native_stores[0],
    )
    recipe = RequestPlanningResource(
        acquisition_permit=permit,
        transfer=template,
        transfer_permit=reader._responsibilities[-1][1],
    )
    scope = destination.owner.application_scope
    reg = registration(scope=scope)
    _, initialized, _, _, request, _ = await setup(ledger, reg=reg)
    recipient = recipe.transfer_permit.intent.request.participant
    recipes = (recipe,)
    if interruption == "planner_cancel_second":
        # Two independent operations over the same immutable material are not
        # one pin. A partial batch must preserve and settle both exact owners.
        second_command = command.model_copy(
            update={"operation": initialized.operation("acquisition-second")}
        )
        second_template = template.model_copy(
            update={
                "operation": initialized.operation("planned-transfer-second"),
                "acquisition": second_command,
            }
        )

        def qualified_permit(native, settlement):
            raw = preparation_permit(native)
            return raw.model_copy(
                update={
                    "intent": raw.intent.model_copy(
                        update={
                            "limits": initialized.binding.limits,
                            "request": raw.intent.request.model_copy(
                                update={
                                    "participant": recipient,
                                    "expected_configuration_revision": 1,
                                    "required_settlement": settlement,
                                }
                            ),
                        }
                    )
                }
            )

        recipes += (
            RequestPlanningResource(
                acquisition_permit=qualified_permit(second_command, "exclusion"),
                transfer=second_template,
                transfer_permit=qualified_permit(second_template, "quiescence"),
            ),
        )
        reader = MandateResourcePreparationReader(
            owner=reader._owner,
            resolver=resolver,
            context=reader._context,
            redactor=SecretRedactor(),
            registration=reader._registration,
            policy=reader._policy,
            artifact_store=artifacts,
            collaboration_store=ledger,
            initialized=initialized,
            responsibilities=tuple((item.acquisition, item.acquisition_permit) for item in recipes),
            transfer_templates=tuple((item.transfer, item.transfer_permit) for item in recipes),
        )
        destination = LocalArtifactResourceOwner(
            destination._journal.root,
            owner=destination.owner,
            artifact_store=artifacts,
            preparation_reader=reader,
        )
        source = destination
    request = request.model_copy(update={"target": recipient, "ttl_ms": JOURNEY_TTL_MS})
    request_resolver = PreparationResolver(request, recipient)
    # The parent-compaction journey includes a non-provider deterministic
    # compactor. Its finite shared sponsor budget must allow both operations,
    # rather than falsely requiring every operation to identify a model.
    binding = _binding(
        application_scope=scope,
        **({"provider_name": None, "model": None} if interruption == "planner_fork_parent" else {}),
    )

    class BudgetReceiver:
        async def resolve_budget_binding(self, *, request):
            return binding

    application = app(
        ledger,
        reg,
        session_store=native_stores[1],
        collaboration_requests=RequestRegistration(
            mandates=request_resolver,
            max_ttl_ms=JOURNEY_TTL_MS,
            resource_owners=(destination,),
            prepared_admission=PreparedAdmissionRegistration(
                receiver=ObjectRef(
                    owner=initialized.owner,
                    kind="request_receiver",
                    object_id="resource-recipient",
                    incarnation="one",
                    revision=1,
                )
            ),
        ),
        budget_binding_receiver=BudgetReceiver(),
        enable_common_root_budget_binding=True,
    )
    fork = "fork" in interruption
    advance_parent = interruption == "planner_fork_parent"
    context_policy = (
        CheckpointCompactionContextPolicy(
            compactor=RecordingCompactor(), max_user_turns=1, compact_after_messages=100
        )
        if advance_parent
        else None
    )
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("historical resource context"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
        + (
            [
                [
                    ModelStreamEvent.text_delta("later parent text outside selected boundary"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
            if advance_parent
            else []
        )
        if fork
        else [],
        name="provider",
    )
    application.register_provider(provider, default=True)
    application.register_agent(
        AgentSpec(name="reviewer", model="model", system_prompt="system"),
        context_policy=context_policy,
    )
    application.register_environment(resource_environment(artifacts), default=True)
    await application.initialize_collaboration()
    reopened_sessions = native_stores[1]
    reopened_ledger = ledger

    async def reconstruct():
        nonlocal application, reader, source, destination, reopened_sessions, reopened_ledger
        await source.drain()
        reopened_ledger = native_stores[2]()
        if native_stores[3][0] != "memory":
            reopened_sessions = reopened_store(native_stores, tmp_path)
        reader = MandateResourcePreparationReader(
            owner=reader._owner,
            resolver=resolver,
            context=reader._context,
            redactor=SecretRedactor(),
            registration=reader._registration,
            policy=reader._policy,
            artifact_store=artifacts,
            collaboration_store=reopened_ledger,
            initialized=initialized,
            responsibilities=tuple((item.acquisition, item.acquisition_permit) for item in recipes),
            transfer_templates=tuple((item.transfer, item.transfer_permit) for item in recipes),
        )
        destination = LocalArtifactResourceOwner(
            destination._journal.root,
            owner=destination.owner,
            artifact_store=artifacts,
            preparation_reader=reader,
        )
        source = destination
        application = app(
            reopened_ledger,
            reg,
            session_store=reopened_sessions,
            # Session/resource reconstruction does not replace the independent
            # budget owner or erase reservations from the source model turn.
            budget_ledger=application.budget_ledger,
            collaboration_requests=replace(
                application._request_coordinator._registration,
                receiving_owner=None,
                resource_owners=(destination,),
            ),
            budget_binding_receiver=BudgetReceiver(),
            enable_common_root_budget_binding=True,
        )
        application.register_provider(provider, default=True)
        application.register_agent(
            AgentSpec(name="reviewer", model="model", system_prompt="system"),
            context_policy=context_policy,
        )
        application.register_environment(resource_environment(artifacts), default=True)
        await application.initialize_collaboration()

    try:
        accepted = await application.accept_collaboration_request(
            request, context=request_resolver.sender.context
        )
        base_request = RecipientSessionCreationRequest(
            request=RunRequest(
                agent_name="reviewer",
                messages=[
                    Message(
                        role="user",
                        content=(
                            FilePart(
                                attachment=file_attachment(
                                    artifact_id=artifact.id,
                                    kind="image",
                                    filename=artifact.filename,
                                    content_type=artifact.content_type,
                                    size_bytes=artifact.size_bytes,
                                )
                            ),
                        ),
                    )
                ],
            ),
            creation_key="resource-material-child:" + scope,
            recipient=recipient,
        )
        base = await application.prepare_recipient_creation(base_request, context=CONTEXT)
        assert base.creation_request.input_artifact_ids == (artifact.id,)
        with pytest.raises(ValidationError, match="explicit retained-resource"):
            RequestPlanningFresh(preparation=base)
        with pytest.raises(PermissionError, match="without exact retained transfers"):
            await application.create_recipient_session(
                base_request, context=CONTEXT, preparation=base
            )
        assert isinstance(
            await application.session_store.read_session_creation_decision(base.creation),
            ExactNotFound,
        )
        assert provider.requests == []
        blueprint = None
        if fork:
            _, source_created = await create(application, initialized, key="resource-fork-source")
            source_participant = source_created.participants[0].reference
            _, _, _, selection, _ = await source_scenario(
                native_stores,
                configured=(application, initialized, source_participant),
                provider=provider,
            )
            blueprint = await application.prepare_recipient_fork(
                base_request,
                selection,
                source_participant=source_participant,
                context=CONTEXT,
                deadline_at_ms=accepted.expected.intent.selection.expires_at_ms,
            )
            proposal = RequestPlanningFork(preparation=blueprint, resources=recipes)
        else:
            proposal = RequestPlanningFresh(preparation=base, resources=recipes)
        stage_count = 2 + 2 * len(recipes) + int(fork)
        assert preparation_stage_count(proposal) == stage_count
        assert type(proposal).model_validate_json(proposal.model_dump_json()) == proposal
        with pytest.raises(ValidationError):
            RequestPlanningFresh(preparation=base, resources=(recipe, recipe))
        policy = (fork_policy if fork else fresh_policy)(
            _policy().reference.model_copy(update={"owner": initialized.owner}),
            _policy().limits.model_copy(
                update={"max_resources": len(recipes), "max_stages": stage_count}
            ),
            blueprint if fork else base,
            resources=recipes,
        )
        assert policy.default == proposal
        assert len(planning_policy_commitment(policy, redactor=SecretRedactor())) == 64
        for narrowed in ({"max_resources": len(recipes) - 1}, {"max_stages": stage_count - 1}):
            with pytest.raises(CollaborationContractError):
                planning_policy_commitment(
                    policy.model_copy(update={"limits": policy.limits.model_copy(update=narrowed)}),
                    redactor=SecretRedactor(),
                )
        planning = RequestPlanningRequest(
            operation=initialized.operation("resource-plan"),
            expected=accepted.expected,
            expected_revision=1,
            expected_input_revision=0,
            expected_input_sha256=clarification_commitment(accepted.expected, SecretRedactor()),
            planning_generation=1,
            admission_operation=initialized.operation("resource-final-admission"),
            admission_generation=1,
            initiator=_initiator(request_resolver.recipient.context),
            policy=policy.reference,
            policy_sha256=planning_policy_commitment(policy, redactor=SecretRedactor()),
            limits=policy.limits,
            deadline_at_ms=accepted.expected.intent.selection.expires_at_ms,
            predecessor=None,
        )
        if interruption.startswith("planner"):
            restart = interruption.endswith("_restart") or advance_parent
            if restart:
                planning = planning.model_copy(
                    update={"limits": planning.limits.model_copy(update={"max_recovery_items": 1})}
                )
            application = app(
                ledger,
                reg,
                session_store=native_stores[1],
                budget_ledger=application.budget_ledger,
                collaboration_requests=replace(
                    application._request_coordinator._registration,
                    receiving_owner=None,
                    planning_policies=(policy,),
                ),
                budget_binding_receiver=BudgetReceiver(),
                enable_common_root_budget_binding=True,
            )
            application.register_provider(provider, default=True)
            application.register_agent(
                AgentSpec(name="reviewer", model="model", system_prompt="system"),
                context_policy=context_policy,
            )
            application.register_environment(resource_environment(artifacts), default=True)
            await application.initialize_collaboration()
            refreshed = await application.prepare_recipient_creation(base_request, context=CONTEXT)
            assert refreshed.execution_profile_json == base.execution_profile_json
            if interruption.endswith("_expired"):
                creation_target = None

                async def interrupted_before_creation(owner, expected, *, context):
                    nonlocal creation_target
                    creation_target = expected.preparation.creation
                    raise OSError("interrupted before resource-bearing creation")

                monkeypatch.setattr(
                    NativePlanningCreationOwner, "create", interrupted_before_creation
                )
                application._request_coordinator._owners.observation_timeout = 180
                with pytest.raises(CollaborationUnavailable):
                    await application.plan_collaboration_request(
                        planning, context=request_resolver.recipient.context
                    )
                assert not application._request_coordinator._owners.pending
                retained = await lookup_plan(
                    application, planning, request_resolver.recipient.context
                )
                assert isinstance(retained, ExactMatch)
                assert retained.receipt.state == "preparing" and retained.receipt.pending_stages
                await reconstruct()
                transaction = reopened_ledger._transaction

                @asynccontextmanager
                async def expired_owner_time(scope, *, write):
                    async with transaction(scope, write=write) as tx:

                        async def now_ms():
                            return planning.deadline_at_ms

                        tx.now_ms = now_ms
                        yield tx

                monkeypatch.setattr(reopened_ledger, "_transaction", expired_owner_time)
                terminal = await complete_plan(
                    application, planning, request_resolver.recipient.context, reconcile=True
                )
                assert terminal.state == "expired" and terminal.pending_stages == 0
                assert terminal.reserved_bytes == terminal.reserved_events == 0
                assert creation_target is not None
                decision = await reopened_sessions.read_session_creation_decision(creation_target)
                assert isinstance(decision, ExactMatch) and decision.receipt.state == "excluded"
                assert decision.receipt.settlement_acknowledged
                assert (
                    await complete_plan(
                        application, planning, request_resolver.recipient.context, reconcile=True
                    )
                    == terminal
                )
                assert len(provider.requests) == (1 if fork else 0)
                # All source/destination pins must be discharged, not merely
                # hidden behind the terminal planning status.
                await artifacts.delete(artifact.id)
                return
            if interruption.startswith("planner_capacity_"):
                import cayu.artifacts.resources as resource_contract

                # Nine slots admit the source while preserving two for native
                # transfer exclusion. Seven refuse source acquisition itself.
                # Neither failure may consume another recipe's cleanup room.
                capacity = int(interruption.rsplit("_", 1)[1])
                monkeypatch.setattr(resource_contract, "RESOURCE_MAX_EVENTS", capacity)
                with pytest.raises(CollaborationUnavailable) as rejected:
                    await complete_plan(application, planning, request_resolver.recipient.context)
                assert "event reservation" in str(rejected.value.__cause__)
                retained = await lookup_plan(
                    application, planning, request_resolver.recipient.context
                )
                assert isinstance(retained, ExactMatch)
                assert retained.receipt.state == "preparing"
                assert (
                    retained.receipt.pending_stages
                    == retained.receipt.stage_count
                    == (1 if capacity == 7 else 2)
                )
                # Reconstruct every owner before cleanup: capacity is durable,
                # not a promise retained only by the failed planner instance.
                await reconstruct()
                control = RequestPlanningControl(
                    expected=planning,
                    expected_revision=retained.receipt.revision,
                    kind="cancelled",
                    initiator=planning.initiator,
                )
                settled = await complete_plan(
                    application, control, request_resolver.recipient.context, control=True
                )
                assert settled.state == "cancelled" and settled.pending_stages == 0
                assert settled.reserved_bytes == settled.reserved_events == 0
                assert provider.requests == []
                await artifacts.delete(artifact.id)
                return
            if interruption.startswith("planner_cancel_"):
                boundary = interruption.removeprefix("planner_cancel_")
                method_name = "acquire" if boundary == "second" else boundary
                method = getattr(NativePlanningResourceOwner, method_name)
                reached, release = asyncio.Event(), asyncio.Event()

                async def pause_after_native_commit(owner, *args, **kwargs):
                    found = await method(owner, *args, **kwargs)
                    if boundary == "second" and args[0] != recipes[1]:
                        return found
                    reached.set()
                    await release.wait()
                    return found

                monkeypatch.setattr(
                    NativePlanningResourceOwner, method_name, pause_after_native_commit
                )
                original_app = application
                original_app._request_coordinator._owners.observation_timeout = 180
                observer = asyncio.create_task(
                    original_app.plan_collaboration_request(
                        planning, context=request_resolver.recipient.context
                    )
                )
                try:
                    async with asyncio.timeout(180):
                        await reached.wait()
                    observer.cancel()
                    observer.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await observer
                    assert observer.cancelled() and observer.cancelling() == 2
                    retained = await lookup_plan(
                        original_app, planning, request_resolver.recipient.context
                    )
                    assert isinstance(retained, ExactMatch)
                    count = {"acquire": 1, "transfer": 2, "second": 3}[boundary]
                    assert retained.receipt.stage_count == retained.receipt.pending_stages == count
                    assert original_app._request_coordinator._owners.pending
                    successor_app, successor = await declined_successor(
                        native_stores, original_app, planning, revision=retained.receipt.revision
                    )
                    # A different worker cannot supersede resource responsibility
                    # merely because the original observer was cancelled.
                    with pytest.raises(CollaborationConflict):
                        await complete_plan(
                            successor_app, successor, request_resolver.recipient.context
                        )
                    # Revoke acquisition/disclosure while retaining independent
                    # cleanup authority; revoking release too must still refuse.
                    resolution = resolver.resolution
                    resolver.resolution = resolution.model_copy(
                        update={
                            "principal": resolution.principal.model_copy(
                                update={"actions": ("release",)}
                            ),
                            "chain": resolution.chain.model_copy(
                                update={
                                    "entries": tuple(
                                        entry.model_copy(update={"actions": ("release",)})
                                        for entry in resolution.chain.entries
                                    )
                                }
                            ),
                        }
                    )
                    await reconstruct()
                    control = RequestPlanningControl(
                        expected=planning,
                        expected_revision=retained.receipt.revision,
                        kind="cancelled",
                        initiator=_initiator(request_resolver.recipient.context),
                    )
                    terminal = await complete_plan(
                        application, control, request_resolver.recipient.context, control=True
                    )
                    assert terminal.state == "cancelled" and terminal.pending_stages == 0
                    assert terminal.reserved_bytes == terminal.reserved_events == 0
                    successor = successor.model_copy(
                        update={
                            "predecessor": successor.predecessor.model_copy(
                                update={"revision": terminal.revision}
                            )
                        }
                    )
                    replacement = await complete_plan(
                        successor_app, successor, request_resolver.recipient.context
                    )
                    assert replacement.state == "declined"
                    pending = tuple(original_app._request_coordinator._owners.pending)
                    release.set()
                    if pending:
                        _, remaining = await asyncio.wait(pending, timeout=180)
                        assert not remaining, "Original native handoff still owns late progress."
                    final = await lookup_plan(
                        application, planning, request_resolver.recipient.context
                    )
                    assert isinstance(final, ExactMatch) and final.receipt == terminal
                    successor_final = await lookup_plan(
                        successor_app, successor, request_resolver.recipient.context
                    )
                    assert isinstance(successor_final, ExactMatch)
                    assert successor_final.receipt == replacement
                    assert isinstance(
                        await application.session_store.read_session_creation_decision(
                            base.creation
                        ),
                        ExactNotFound,
                    )
                    with pytest.raises((ResourceOwnerError, MandateDenied)):
                        await source.acquire(
                            command, preparation=await source.authorize(command, permit=permit)
                        )
                    assert provider.requests == []
                    await artifacts.delete(artifact.id)
                finally:
                    release.set()
                    if not observer.done():
                        observer.cancel()
                    await asyncio.gather(observer, return_exceptions=True)
                    pending = tuple(original_app._request_coordinator._owners.pending)
                    if pending:
                        _, remaining = await asyncio.wait(pending, timeout=180)
                        assert not remaining
                return
            if interruption.startswith("planner_ack_"):
                boundary = interruption.removeprefix("planner_ack_")
                method = getattr(NativePlanningResourceOwner, boundary)
                committed = None

                async def lose_native_ack(owner, *args, **kwargs):
                    nonlocal committed
                    found = await method(owner, *args, **kwargs)
                    if committed is None:
                        committed = found
                        raise OSError("native resource commit acknowledged by no observer")
                    assert found == committed
                    return found

                monkeypatch.setattr(NativePlanningResourceOwner, boundary, lose_native_ack)
                with pytest.raises((CollaborationUnavailable, OSError)):
                    await complete_plan(application, planning, request_resolver.recipient.context)
                assert committed is not None
                retained = await lookup_plan(
                    application, planning, request_resolver.recipient.context
                )
                assert isinstance(retained, ExactMatch)
                assert (
                    retained.receipt.stage_count
                    == retained.receipt.pending_stages
                    == (1 if boundary == "acquire" else 2)
                )
                await reconstruct()
            result = (
                await resource_process_journey(
                    application, planning, policy, reader, destination, native_stores
                )
                if interruption == "planner_process"
                else await complete_plan(application, planning, request_resolver.recipient.context)
            )
            if restart:
                assert result.state == "preparing" and result.pending_stages == 2
                assert result.stage_count == 2
                assert len(provider.requests) == int(fork)
                if advance_parent:
                    retention = await application.session_store._read_context_view_retention(
                        blueprint.view
                    )
                    assert isinstance(retention, ExactMatch)
                    assert retention.receipt.state == "adopted"
                    view_id = retention.receipt.selection.view_id
                    source_id = selection.source_session_id
                    historical = await application.read_context_view(
                        view_id,
                        source_session_id=source_id,
                        participant=source_participant,
                        context=CONTEXT,
                    )
                    continued = [
                        event
                        async for event in application.resume(
                            ResumeRequest(
                                session_id=source_id,
                                messages=[Message.text("user", "advance the source")],
                            ),
                            context=CONTEXT,
                        )
                    ]
                    assert any(event.type.value == "session.completed" for event in continued), (
                        "\n".join(event.model_dump_json() for event in continued)
                    )
                    current = await application.session_store.load(source_id)
                    compacted = [
                        event
                        async for event in application.compact_session(
                            CompactSessionRequest(
                                session_id=source_id,
                                idempotency_key="resource-fork-parent-compaction",
                                expected_run_epoch=current.run_epoch,
                                expected_transcript_cursor=(
                                    await application.session_store.load_transcript_cursor(
                                        source_id
                                    )
                                ),
                            ),
                            context=CONTEXT,
                        )
                    ]
                    assert any(event.type.value == "session.checkpointed" for event in compacted), (
                        "\n".join(event.model_dump_json() for event in compacted)
                    )
                    assert (
                        await application.session_store._read_context_view_retention(blueprint.view)
                        == retention
                    )
                    assert (
                        await application.read_context_view(
                            view_id,
                            source_session_id=source_id,
                            participant=source_participant,
                            context=CONTEXT,
                        )
                    ).view == historical.view
                await reconstruct()
                if advance_parent:
                    assert (
                        await application.read_context_view(
                            view_id,
                            source_session_id=source_id,
                            participant=source_participant,
                            context=CONTEXT,
                        )
                    ).view == historical.view
                for _ in range(8):
                    result = await complete_plan(
                        application, planning, request_resolver.recipient.context, reconcile=True
                    )
                    if result.state == "admitted":
                        break
            assert result.state == "admitted" and result.stage_count == stage_count
            assert result.pending_stages == 0
            found = await lookup_plan(application, planning, request_resolver.recipient.context)
            assert isinstance(found, ExactMatch) and found.receipt == result
            from cayu.collaboration._planning_resources import resource_stages

            _, stages = await resource_stages(application._request_coordinator, result)
            if fork:
                assert stages[0].receipt.state == "released"
                retention = await application.session_store._read_context_view_retention(
                    blueprint.view
                )
                assert isinstance(retention, ExactMatch) and retention.receipt.state == "released"
                stages = stages[1:]
            assert stages[0].receipt.receiving.proves_exclusion
            assert isinstance(stages[1].receipt, RequestResourceStageAdoption)
            child = stages[2].receipt
            assert child.prepared.target.resources == (stages[1].receipt.material,)
            if interruption == "planner_ack_acquire":
                assert stages[1].intent.command.acquisition == committed.receipt
            elif interruption == "planner_ack_transfer":
                assert stages[1].receipt.material == committed.reference
            session = await application.session_store.load(child.decision.session_id)
            assert session.status == "pending" and session.run_epoch == 0
            assert len(provider.requests) == int(fork) + int(advance_parent)
            if fork:
                transcript = await application.session_store.load_transcript(session.id)
                assert "historical resource context" in "\n".join(
                    message.model_dump_json() for message in transcript
                )
                assert "later parent text outside selected boundary" not in "\n".join(
                    message.model_dump_json() for message in transcript
                )
            transfer_stage = stages[1].intent.command
            adoption = await NativePlanningResourceOwner(application).read_adoption(
                transfer_stage, stages[2].intent.command, context=CONTEXT
            )
            assert adoption.receipt == stages[1].receipt
            await application.session_store.delete_session(session.id)
            released = await destination.settle_preparation(
                transfer_stage.native_command, permit=recipe.transfer_permit, source_owner=source
            )
            assert released.proves_exclusion
            await artifacts.delete(artifact.id)
            return
        resources = NativePlanningResourceOwner(application)
        acquired = await resources.acquire(recipe, context=CONTEXT)
        transferred = await resources.transfer(recipe, acquired.receipt, context=CONTEXT)
        resolved = await resolve_resource_recipient(
            application, base, (transferred.reference,), context=CONTEXT
        )
        assert isinstance(resolved, ResourceRecipientCreationPreparation)
        assert resolved.creation.request_commitment != base.creation.request_commitment
        assert resolved.creation.permit.operation == base.creation.permit.operation
        assert isinstance(
            await application.session_store.read_session_creation_decision(resolved.creation),
            ExactNotFound,
        )
        stage = RequestCreationStageCommand(
            operation=resolved.creation.permit.operation,
            expected=accepted.expected,
            preparation=resolved,
        )
        stage = RequestCreationStageCommand.model_validate_json(stage.model_dump_json())
        # Pure schedule projections are not receiving authority. The native
        # creation owner below still authenticates the actual material.
        projection = SimpleNamespace(decision=proposal, receipt=SimpleNamespace(command=planning))
        assert creation_stage_command(projection, preparation=resolved) == stage
        assert (
            creation_stage_intent(projection, SecretRedactor(), preparation=resolved).ordinal == 3
        )
        first = resource_stage_intent(projection, 0, SecretRedactor())
        second = resource_stage_intent(
            projection, 0, SecretRedactor(), acquisition=acquired.receipt
        )
        assert (first.ordinal, second.ordinal) == (1, 2)
        assert first.command.resource == second.command.resource == recipe
        assert (
            len(
                {
                    first.operation,
                    second.operation,
                    first.command.operation,
                    second.command.operation,
                }
            )
            == 4
        )
        for invalid_index in (True, -1, 1):
            with pytest.raises(ValueError):
                resource_stage_intent(projection, invalid_index, SecretRedactor())
        owner = NativePlanningCreationOwner(application)
        created = await owner.create(stage, context=CONTEXT)
        assert created.receipt.prepared.target.resources == (transferred.reference,)
        assert_stage_envelope(initialized, created.receipt, expected_plan=planning, ordinal=3)
        transfer_stage = RequestResourceTransferStageCommand(
            operation=initialized.operation("retained-resource-transfer"),
            expected=accepted.expected,
            resource=recipe,
            acquisition=acquired.receipt,
        )
        adopted = await resources.read_adoption(transfer_stage, stage, context=CONTEXT)
        assert adopted.receipt.material == transferred.reference
        assert adopted.receipt == RequestResourceStageAdoption.model_validate_json(
            adopted.receipt.model_dump_json()
        )
        assert_stage_envelope(initialized, adopted.receipt)
        before_adoption_rejection = destination._journal.path.read_bytes()
        with pytest.raises(CollaborationUnavailable):
            await resources.read_adoption(
                transfer_stage.model_copy(
                    update={
                        "acquisition": acquired.receipt.model_copy(
                            update={"receipt_id": "different-source-receipt"}
                        )
                    }
                ),
                stage,
                context=CONTEXT,
            )
        assert destination._journal.path.read_bytes() == before_adoption_rejection
        assert (await destination.read_transfer(transferred.receipt.command)).receipt.stage == (
            "accepted"
        )
        creation = resolved_material_creation_request(
            resolved, None, (transferred.receipt,), (transferred.preparation,)
        )
        session, receipt = await application.lookup_recipient_session(creation, context=CONTEXT)
        assert session.status == "pending" and session.run_epoch == 0
        assert created_admission_target(resolved.creation, receipt.participant_receipt) == (
            created.receipt.prepared.target
        )
        admission = RequestAdmissionCommand(
            operation=initialized.operation("resource-admission"),
            expected=accepted.expected,
            expected_revision=1,
            expected_input_revision=0,
            expected_input_sha256=clarification_commitment(accepted.expected, SecretRedactor()),
            generation=1,
            decision="fresh",
            prepared=created.receipt.prepared,
            evidence=(),
            initiator=_initiator(request_resolver.recipient.context),
        )
        before = await application.inspect_collaboration_request(
            accepted.expected, context=request_resolver.sender.context
        )

        def unqualified_receiver(*args, **kwargs):
            pytest.fail("A resource-free receiver cannot consume resource-bearing admission.")

        with monkeypatch.context() as patch:
            patch.setattr(
                application._request_coordinator._registration.receiving_owner,
                "prepared_admission_version",
                3,
            )
            patch.setattr(
                application._request_coordinator._registration.receiving_owner,
                "acquire",
                unqualified_receiver,
            )
            with pytest.raises(CollaborationUnavailable):
                await application.admit_collaboration_request(
                    admission, context=request_resolver.recipient.context
                )
        assert (
            await application.inspect_collaboration_request(
                accepted.expected, context=request_resolver.sender.context
            )
            == before
        )
        await application.settle_recipient_resource_handoff(
            receipt, resource_owner=destination, source_owners=(source,), context=CONTEXT
        )
        await reconstruct()
        # A committed child replays from its own durable evidence, without
        # renewing disclosure or attempting source acquisition after handoff.
        resolver.revoked = True
        assert (
            await NativePlanningCreationOwner(application).create(stage, context=CONTEXT) == created
        )
        assert await application.prepare_recipient_admission(creation, context=CONTEXT) == (
            created.receipt.prepared
        )
        source_before = source._journal.path.read_bytes()
        destination_before = destination._journal.path.read_bytes()
        assert (
            await NativePlanningResourceOwner(application).read_adoption(
                transfer_stage, stage, context=CONTEXT
            )
            == adopted
        )
        assert destination._journal.path.read_bytes() == destination_before
        if interruption == "none":
            # The child receipt is necessary but does not replace current
            # native acceptance evidence at the resource owner.
            digest = transferred.receipt.operation_digest
            async with destination._journal.transaction() as journal:
                original_events = list(journal["events"])
                journal["events"] = [
                    event
                    for event in original_events
                    if not (event.get("transfer") == digest and event.get("stage") == "accepted")
                ]
            try:
                with pytest.raises(ResourceOwnerUnavailable):
                    await NativePlanningResourceOwner(application).read_adoption(
                        transfer_stage, stage, context=CONTEXT
                    )
                with pytest.raises(CollaborationUnavailable):
                    await application.admit_collaboration_request(
                        admission, context=request_resolver.recipient.context
                    )
                assert (
                    await application.inspect_collaboration_request(
                        accepted.expected, context=request_resolver.sender.context
                    )
                    == before
                )
            finally:
                async with destination._journal.transaction() as journal:
                    journal["events"] = original_events
        if interruption == "cancel":
            entered, proceed = asyncio.Event(), asyncio.Event()
            original_guard = destination._hold_adopted_recipient_material

            @asynccontextmanager
            async def paused_guard(references, *, recipient):
                async with original_guard(references, recipient=recipient):
                    entered.set()
                    await proceed.wait()
                    yield

            with monkeypatch.context() as patch:
                patch.setattr(destination, "_hold_adopted_recipient_material", paused_guard)
                observer = asyncio.create_task(
                    application.admit_collaboration_request(
                        admission, context=request_resolver.recipient.context
                    )
                )
                try:
                    async with asyncio.timeout(30):
                        await entered.wait()
                    observer.cancel()
                    observer.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await observer
                    assert observer.cancelled() and observer.cancelling() == 2
                    competing = LocalArtifactResourceOwner(
                        destination._journal.root,
                        owner=destination.owner,
                        artifact_store=artifacts,
                        preparation_reader=reader,
                    )
                    try:
                        with pytest.raises(
                            ResourceOwnerUnavailable, match="mutation remains owned"
                        ):
                            await competing.release_transfer(transferred.receipt)
                    finally:
                        await competing.drain()
                finally:
                    proceed.set()
                    pending = tuple(application._request_coordinator._owners.pending)
                    async with asyncio.timeout(30):
                        await asyncio.gather(*pending)
        elif interruption == "ack":
            original_transaction = reopened_ledger._transaction
            key = (
                admission.operation.namespace_incarnation,
                admission.operation.generation,
                admission.operation.caller_key,
            )
            lost = False

            @asynccontextmanager
            async def lost_ack(scope, *, write):
                nonlocal lost
                committed = False
                async with original_transaction(scope, write=write) as tx:
                    yield tx
                    if write:
                        committed = await tx.get("operations", key) is not None
                if committed:
                    lost = True
                    raise OSError("resource admission acknowledgement lost")

            with monkeypatch.context() as patch:
                patch.setattr(reopened_ledger, "_transaction", lost_ack)
                with pytest.raises(CollaborationUnavailable):
                    await application.admit_collaboration_request(
                        admission, context=request_resolver.recipient.context
                    )
            assert lost
        admitted = await application.admit_collaboration_request(
            admission, context=request_resolver.recipient.context
        )
        assert admitted.state == "admitted"
        assert (
            await application.admit_collaboration_request(
                admission, context=request_resolver.recipient.context
            )
            == admitted
        )
        assert source._journal.path.read_bytes() == source_before
        assert destination._journal.path.read_bytes() == destination_before
        assert provider.requests == []
        resolver.revoked = False
        await application.session_store.delete_session(session.id)
        await destination.release_transfer(transferred.receipt)
        await artifacts.delete(artifact.id)
    finally:
        resolver.revoked = False
        await source.drain()
        await destination.drain()
        if reopened_sessions is not native_stores[1]:
            await reopened_sessions.close()
