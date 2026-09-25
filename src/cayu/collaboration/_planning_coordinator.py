"""Explicit planning under existing mandate and request transaction owners."""

from __future__ import annotations

from cayu.collaboration._contracts import (
    CollaborationConflict,
    ContractValue,
    ExactConflict,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
    ObjectRef,
)
from cayu.collaboration._mandate_validation import MandateUse, validate_mandate_resolution
from cayu.collaboration._planning_prerequisite import _PrerequisiteRead, read_prerequisite
from cayu.collaboration._planning_records import RequestPlanningReceipt, RequestPlanningRecord
from cayu.collaboration._planning_store import (
    apply_local_decision_in_transaction,
    read_plan_in_transaction,
    retain_decision_in_transaction,
    retain_plan_in_transaction,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_coordinator import _initiator, _safe_request_failure
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.base import PLANNING_FAMILY, _stored_mode
from cayu.collaboration.mandates import MandateAccessContext, MandateResolution
from cayu.collaboration.participants import CollaborationNotInitialized, CollaborationUnavailable
from cayu.collaboration.planning import (
    RequestPlanningContinue,
    RequestPlanningControl,
    RequestPlanningDefer,
    RequestPlanningFork,
    RequestPlanningFresh,
    RequestPlanningPrerequisite,
    RequestPlanningRequest,
    evaluate_configured_request_policy,
)


class _Submission(ContractValue):
    request: RequestPlanningRequest
    context: MandateAccessContext
    control: RequestPlanningControl | None = None


async def plan_request(
    requests,
    request,
    *,
    context,
    read_only=False,
    clarifications=None,
    control=None,
    require_retained=False,
    recipient_creation=None,
    recipient_fork=None,
    recipient_resources=None,
):
    """Retained mutation ownership outlives a cancelled foreground observer."""
    value = prepare_contract(
        _Submission,
        {"request": request, "context": context, "control": control},
        redactor=requests._redactor,
    )
    if value.control is not None:
        require_exact_contract(value.request, value.control.expected, redactor=requests._redactor)

    async def owned():
        async def progress():
            record = await _held(
                requests, value, read_only=read_only, require_retained=require_retained
            )
            if isinstance(record, _PrerequisiteRead):
                # The registered admission reader may use the same non-reentrant
                # mandate guard. Resolve outside it, then reacquire current
                # planning authority and CAS the unchanged predecessor.
                evidence = await read_prerequisite(requests, record, value.context)
                record = await _held(requests, value, read_only=read_only, prerequisite=evidence)
            if (
                not read_only
                and isinstance(record.decision, (RequestPlanningFresh, RequestPlanningFork))
                and record.decision.resources
                and (record.state == "preparing" or record.pending_stages)
            ):
                from cayu.collaboration._planning_resources import progress_resources

                return await progress_resources(
                    requests,
                    value,
                    record,
                    recipient_resources,
                    recipient_creation,
                    recipient_fork,
                    max_items=value.request.limits.max_recovery_items if require_retained else None,
                )
            if (
                not read_only
                and isinstance(record.decision, RequestPlanningFork)
                and (record.state == "preparing" or record.pending_stages)
            ):
                from cayu.collaboration._planning_fork import progress_fork

                return await progress_fork(
                    requests,
                    value,
                    record,
                    recipient_fork,
                    recipient_creation,
                    max_items=value.request.limits.max_recovery_items if require_retained else None,
                )
            if (
                not read_only
                and isinstance(record.decision, RequestPlanningFresh)
                and (record.state == "preparing" or record.pending_stages)
            ):
                from cayu.collaboration._planning_creation import progress_creation

                return await progress_creation(
                    requests,
                    value,
                    record,
                    recipient_creation,
                    max_items=value.request.limits.max_recovery_items if require_retained else None,
                )
            if (
                not read_only
                and record.state == "preparing"
                and isinstance(record.decision, RequestPlanningContinue)
            ):
                from cayu.collaboration._planning_recipient import admit_prepared_plan

                return await admit_prepared_plan(requests, record, context=value.context)
            if read_only or record.state != "clarifying" or not record.pending_stages:
                return record
            if clarifications is None:
                raise CollaborationUnavailable("Planning clarification receiver is unavailable.")
            from cayu.collaboration._planning_stages import (
                _PlannedStage,
                clarification_stage_intent,
            )
            from cayu.collaboration.exports import SessionExportAccessContext

            # The planning mandate guard has exited. The existing export owner
            # acquires its own guard, avoiding recursive non-reentrant admission.
            await clarifications._open_planned(
                record.decision.opening,
                record.decision.source,
                context=SessionExportAccessContext(
                    principal=value.context.principal, mandate=value.context
                ),
                planned=_PlannedStage(
                    record.receipt.command, clarification_stage_intent(record, requests._redactor)
                ),
            )
            result = await _held(requests, value, read_only=True)
            if not isinstance(result, ExactMatch):
                raise CollaborationUnavailable("Planning clarification readback is unavailable.")
            return result.receipt

        return await requests._dependency(progress)

    return await requests._observe(
        requests._owners.run(
            owned,
            key=("request-planning-read" if read_only else "request-planning", object()),
            expectation=contract_bytes(value, redactor=requests._redactor),
            redactor=requests._redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
        )
    )


async def _held(
    requests,
    value,
    *,
    read_only,
    prerequisite=None,
    require_retained=False,
    creation_readback=None,
    fork_preparation=None,
    view_readback=None,
    view_reservation=None,
    resource_readback=None,
):
    registration = requests._registration
    if registration is None or requests._resolver_ref is None:
        raise CollaborationNotInitialized("Request planning registration is unavailable.")
    redactor = requests._redactor
    command, context = value.request, value.context
    participants = requests._participants
    store, initialized = participants._ready()
    participants._capability(store, initialized, mutation=not read_only, family=PLANNING_FAMILY)
    selected = command.expected.intent.selection
    _, grant = participants._authorize(
        CollaborationAccessContext(principal=context.principal), "request_readback"
    )
    participants._require_refs(grant, (selected.sender.reference, selected.recipient.reference))
    require_exact_contract(
        requests._resolver_ref,
        prepare_contract(ObjectRef, registration.mandates.ref, redactor=redactor),
        redactor=redactor,
    )
    async with registration.mandates.acquire(context) as raw:
        resolution = prepare_contract(MandateResolution, raw, redactor=redactor)
        deadline = min(
            resolution.principal.expires_at_ms,
            *(entry.expires_at_ms for entry in resolution.chain.entries),
        )

        async def validate(actions):
            async with store._transaction(initialized.binding.application_scope, write=False) as tx:
                now = await tx.now_ms()
            validate_mandate_resolution(
                resolution,
                context=context,
                resolver=requests._resolver_ref,
                use=MandateUse(
                    audience=initialized.owner,
                    scope=initialized.binding.application_scope,
                    actions=actions,
                    resources=(),
                    inputs=(),
                ),
                now_ms=now,
                resource_owners=requests._resource_owners,
                redactor=redactor,
            )

        await validate(("readback",))
        async with store._transaction(initialized.binding.application_scope, write=False) as tx:
            now = await tx.now_ms()
            if now >= deadline:
                raise CollaborationAccessDenied("Planning read authority expired.")
            raw_receipt = await tx.get("operations", operation_key(command.operation))
            if raw_receipt is not None and _stored_mode(raw_receipt) == "request_plan":
                receipt = prepare_contract(RequestPlanningReceipt, raw_receipt, redactor=redactor)
                actual = receipt.command.expected.intent.selection
                participants._require_refs(
                    grant, (actual.sender.reference, actual.recipient.reference)
                )
            elif raw_receipt is not None:
                await requests._require_retained_read_grant(tx, command.operation, grant)
            try:
                found = await read_plan_in_transaction(
                    store, tx, initialized, command, redactor=redactor
                )
            except (CollaborationUnavailable, ValueError):
                if read_only:
                    return ExactUnavailable()
                raise
            now = await tx.now_ms()
        if read_only:
            return found
        if require_retained and not isinstance(found, ExactMatch):
            raise CollaborationUnavailable(
                "Planning recovery requires an exact retained operation."
            )
        if isinstance(found, ExactConflict):
            raise CollaborationConflict("Planning operation conflicts with retained evidence.")
        if isinstance(found, ExactUnavailable):
            raise CollaborationUnavailable("Planning history is unavailable.")
        if (
            value.control is None
            and isinstance(found, ExactMatch)
            and found.receipt.state == "preparing"
            and isinstance(found.receipt.decision, (RequestPlanningFresh, RequestPlanningFork))
            and now >= command.deadline_at_ms
        ):
            from cayu.collaboration._planning_control import control_plan_in_transaction

            # A persisted preparing state is not evidence that its deadline is
            # still open. Reuse the exact administrative control owner; expiry
            # retains foreign responsibility until receiving readback settles it.
            _, cleanup_grant = participants._authorize(
                CollaborationAccessContext(principal=context.principal), "request_control"
            )
            participants._require_refs(
                cleanup_grant, (selected.sender.reference, selected.recipient.reference)
            )
            await validate(("readback", "administer"))
            async with store._transaction(initialized.binding.application_scope, write=True) as tx:
                if await tx.now_ms() >= deadline:
                    raise CollaborationAccessDenied("Planning cleanup authority expired.")
                current = await read_plan_in_transaction(
                    store, tx, initialized, command, redactor=redactor
                )
                if not isinstance(current, ExactMatch):
                    raise CollaborationUnavailable(
                        "Planning expiry lost its exact retained intent."
                    )
                if current.receipt.state == "preparing":
                    expired = await control_plan_in_transaction(
                        store,
                        tx,
                        initialized,
                        RequestPlanningControl(
                            expected=command,
                            expected_revision=current.receipt.revision,
                            kind="expired",
                            initiator=_initiator(context),
                        ),
                        redactor=redactor,
                    )
                    found = ExactMatch[RequestPlanningRecord](receipt=expired)
                else:
                    found = current
        if (
            creation_readback is not None
            or fork_preparation is not None
            or view_readback is not None
            or view_reservation is not None
            or resource_readback is not None
        ):
            from cayu.collaboration._planning_creation import adopt_creation_stage
            from cayu.collaboration._planning_fork import (
                adopt_fork_preparation,
                adopt_view_settlement,
            )

            if not isinstance(found, ExactMatch):
                raise CollaborationUnavailable("Creation settlement lacks its retained plan.")
            cleanup = found.receipt.state in {"cancelled", "expired"}
            if (
                fork_preparation is not None or view_reservation is not None
            ) and found.receipt.state != "preparing":
                raise CollaborationConflict("FORK plan no longer admits a child preparation.")
            _, settlement_grant = participants._authorize(
                CollaborationAccessContext(principal=context.principal),
                "request_control" if cleanup else "request_accept",
            )
            participants._require_refs(
                settlement_grant, (selected.sender.reference, selected.recipient.reference)
            )
            await validate(("readback", "administer" if cleanup else "prepare"))
            async with store._transaction(initialized.binding.application_scope, write=True) as tx:
                if await tx.now_ms() >= deadline:
                    raise CollaborationAccessDenied("Planning settlement authority expired.")
                current = await read_plan_in_transaction(
                    store, tx, initialized, command, redactor=redactor
                )
                if not isinstance(current, ExactMatch):
                    raise CollaborationUnavailable("Creation settlement lost its exact plan.")
                if view_reservation is not None:
                    from cayu.collaboration._planning_view_reservation import register_reserved_view

                    return await register_reserved_view(
                        store, tx, initialized, current.receipt, view_reservation, redactor=redactor
                    )
                if resource_readback is not None:
                    from cayu.collaboration._planning_resources import adopt_resource_progress

                    return await adopt_resource_progress(
                        store,
                        tx,
                        initialized,
                        current.receipt,
                        resource_readback,
                        redactor=redactor,
                    )
                if fork_preparation is not None:
                    if current.receipt.state != "preparing":
                        raise CollaborationConflict(
                            "FORK plan was sealed before child preparation."
                        )
                    return await adopt_fork_preparation(
                        store, tx, initialized, current.receipt, fork_preparation, redactor=redactor
                    )
                if view_readback is not None:
                    return await adopt_view_settlement(
                        store, tx, initialized, current.receipt, view_readback, redactor=redactor
                    )
                return await adopt_creation_stage(
                    store, tx, initialized, current.receipt, creation_readback, redactor=redactor
                )
        if value.control is not None:
            from cayu.collaboration._planning_control import control_plan_in_transaction

            require_exact_contract(value.control.initiator, _initiator(context), redactor=redactor)
            _, control_grant = participants._authorize(
                CollaborationAccessContext(principal=context.principal), "request_control"
            )
            participants._require_refs(
                control_grant, (selected.sender.reference, selected.recipient.reference)
            )
            await validate(("readback", "administer"))
            async with store._transaction(initialized.binding.application_scope, write=True) as tx:
                if await tx.now_ms() >= deadline:
                    raise CollaborationAccessDenied("Planning cleanup authority expired.")
                controlled = await control_plan_in_transaction(
                    store, tx, initialized, value.control, redactor=redactor
                )
            return controlled
        if (
            isinstance(found, ExactMatch)
            and found.receipt.state
            in {
                "deferred",
                "declined",
                "admitted",
                "clarifying",
                "preparing",
                "cancelled",
                "expired",
                "superseded",
            }
            and not (
                found.receipt.state == "preparing"
                and isinstance(found.receipt.decision, (RequestPlanningFresh, RequestPlanningFork))
                and not found.receipt.pending_stages
            )
        ):
            return found.receipt
        require_exact_contract(command.initiator, _initiator(context), redactor=redactor)
        if context.participant != selected.recipient.reference:
            raise CollaborationAccessDenied("Planning requires current recipient authority.")
        _, mutation_grant = participants._authorize(
            CollaborationAccessContext(principal=context.principal), "request_accept"
        )
        participants._require_refs(
            mutation_grant, (selected.sender.reference, selected.recipient.reference)
        )
        await validate(("readback", "prepare"))
        policy = (
            found.receipt.receipt.policy
            if isinstance(found, ExactMatch)
            else requests._planning_policies.get(command.policy)
        )
        if policy is None:
            raise CollaborationUnavailable("Exact planning policy is not registered.")
        if (
            isinstance(found, ExactNotFound)
            and command.predecessor is not None
            and prerequisite is None
        ):
            async with store._transaction(initialized.binding.application_scope, write=False) as tx:
                raw = await tx.get("request_plans", operation_key(command.predecessor.operation))
                if raw is None:
                    raise CollaborationConflict("Planning predecessor is unavailable.")
                predecessor = prepare_contract(RequestPlanningRecord, raw, redactor=redactor)
                require_exact_contract(
                    command.expected, predecessor.receipt.command.expected, redactor=redactor
                )
                if isinstance(predecessor.decision, RequestPlanningDefer) and isinstance(
                    predecessor.decision.prerequisite, RequestPlanningPrerequisite
                ):
                    return _PrerequisiteRead(predecessor.decision.prerequisite)
        if isinstance(found, ExactNotFound):
            async with store._transaction(initialized.binding.application_scope, write=True) as tx:
                if await tx.now_ms() >= deadline:
                    raise CollaborationAccessDenied("Planning authority expired before retention.")
                record = await retain_plan_in_transaction(
                    store,
                    tx,
                    initialized,
                    command,
                    policy,
                    redactor=redactor,
                    prerequisite=prerequisite,
                )
        else:
            assert isinstance(found, ExactMatch)
            record = found.receipt
        if record.decision is None:
            proposal = evaluate_configured_request_policy(
                record.receipt.policy,
                input_revision=command.expected_input_revision,
                redactor=redactor,
            )
            async with store._transaction(initialized.binding.application_scope, write=True) as tx:
                if await tx.now_ms() >= deadline:
                    raise CollaborationAccessDenied("Planning authority expired before decision.")
                record = await retain_decision_in_transaction(
                    store, tx, initialized, command, proposal, redactor=redactor
                )
        if isinstance(record.decision, (RequestPlanningFresh, RequestPlanningFork)):
            _, creation_grant = participants._authorize(
                CollaborationAccessContext(principal=context.principal), "administration"
            )
            participants._require_refs(creation_grant, (selected.recipient.reference,))
            if isinstance(record.decision, RequestPlanningFork):
                participants._require_refs(
                    creation_grant,
                    (record.decision.preparation.view.permit.intent.request.participant,),
                )
        async with store._transaction(initialized.binding.application_scope, write=True) as tx:
            if await tx.now_ms() >= deadline:
                raise CollaborationAccessDenied("Planning authority expired before admission.")
            return await apply_local_decision_in_transaction(
                store, tx, initialized, command, redactor=redactor
            )
