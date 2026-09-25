"""Prove mandatory control fits before retaining/growing native planning debt."""

from cayu.collaboration._contracts import CollaborationContractError, InitiatorBinding, ObjectRef
from cayu.collaboration._planning_records import (
    MAX_REQUEST_PLAN_EVENTS,
    RequestPlanningRecord,
    planning_disposition,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.planning import RequestPlanningControl
from cayu.collaboration.requests import MAX_CONTROL_INITIATOR_BYTES
from cayu.vaults.redaction import SecretRedactor


def preflight_stage_terminal(stage, initialized, *, ceiling, redactor, expected_plan=None):
    """Bound individual terminal rows before retaining their responsibility.

    These maximal-size projections are never stored or accepted as receipts.
    Sequence/time/position use the portable integer ceiling and native event IDs
    use their fixed UUID-hex width. Aggregate reserves do not replace this check.
    """
    from hashlib import sha256

    from cayu.collaboration._permits import PermitReceipt
    from cayu.collaboration._planning_creation_types import (
        RequestCreationStageCommand,
        RequestCreationStageReceipt,
    )
    from cayu.collaboration._planning_fork_types import (
        RequestViewStageCommand,
        RequestViewStageReceipt,
    )
    from cayu.collaboration._planning_resource_types import (
        RequestResourceStageAdoption,
        RequestResourceStageRelease,
        RequestResourceTransferStageCommand,
        ResourceStageCommand,
    )
    from cayu.collaboration._prepared_admission_store import prepared_admission_permit
    from cayu.collaboration.participants import ParticipantEvent
    from cayu.collaboration.requests import (
        RequestAdmissionCommand,
        RequestAdmissionReceipt,
        RequestEvent,
    )

    maximum = 2**53 - 1
    event = stage.registration_event.model_copy(
        update={
            "id": "f" * 32,
            "sequence": maximum,
            "type": "plan_stage_excluded",
        }
    )
    excluded = stage.model_copy(
        update={
            "state": "excluded",
            "settled_at_ms": maximum,
            "settlement_event": event,
            "reserved_bytes": 0,
        }
    )
    command = stage.intent.command
    projected = (
        []
        if isinstance(command, RequestCreationStageCommand | ResourceStageCommand)
        else [excluded]
    )
    followup_required = 0
    if isinstance(command, ResourceStageCommand):
        from cayu.artifacts.resources import resource_operation_digest
        from cayu.collaboration._permits import ReceivingSettlementReceipt

        terminal = RequestResourceStageRelease(
            command=command,
            receiving=ReceivingSettlementReceipt(
                expected=command.permit,
                receiving_owner=command.permit.intent.request.target.owner,
                receipt_id="resource-settled-" + resource_operation_digest(command.native_command),
                outcome="quiescent",
                admission_excluded=True,
            ),
        )
        projected.append(
            excluded.model_copy(
                update={
                    "state": "settled",
                    "receipt": terminal,
                    "settlement_event": event.model_copy(
                        update={
                            "type": "plan_stage_settled",
                            "commitment": sha256(
                                contract_bytes(terminal, redactor=redactor)
                            ).hexdigest(),
                        }
                    ),
                }
            )
        )
        if isinstance(command, RequestResourceTransferStageCommand):
            from cayu.artifacts._resource_material import material_commitment
            from cayu.artifacts._resource_material_types import ResourceMaterialReference
            from cayu.collaboration._contracts import MAX_ID_BYTES

            adoption = RequestResourceStageAdoption(
                command=command,
                creation_operation=command.operation.model_copy(
                    update={"caller_key": '"' * MAX_ID_BYTES}
                ),
                creation_sha256="f" * 64,
                session_id='"' * MAX_ID_BYTES,
                session_instance_id='"' * MAX_ID_BYTES,
                creation_receipt_commitment="sha256:" + "f" * 64,
                material=ResourceMaterialReference(
                    owner=command.resource.transfer.destination,
                    operation=command.resource.transfer.operation,
                    template_commitment=material_commitment(command.resource.transfer),
                    transfer_commitment="sha256:" + "f" * 64,
                    preparation_commitment="sha256:" + "f" * 64,
                ),
            )
            projected.append(
                excluded.model_copy(
                    update={
                        "state": "settled",
                        "receipt": adoption,
                        "settlement_event": event.model_copy(
                            update={
                                "type": "plan_stage_settled",
                                "commitment": sha256(
                                    contract_bytes(adoption, redactor=redactor)
                                ).hexdigest(),
                            }
                        ),
                    }
                )
            )
    if isinstance(command, RequestViewStageCommand):
        from cayu.collaboration._contracts import MAX_ID_BYTES

        if expected_plan is None or expected_plan.limits.max_stages < 3:
            raise CollaborationContractError("FORK requires view, creation and admission stages.")
        for state in ("excluded", "released", "expired"):
            terminal = RequestViewStageReceipt(
                command=command,
                state=state,
                view_id=None if state == "excluded" else '"' * MAX_ID_BYTES,
                manifest_commitment=None if state == "excluded" else "sha256:" + "f" * 64,
                pin_commitment=None if state == "excluded" else "sha256:" + "f" * 64,
            )
            projected.append(
                excluded.model_copy(
                    update={
                        "state": "settled",
                        "receipt": terminal,
                        "settlement_event": event.model_copy(
                            update={
                                "type": "plan_stage_settled",
                                "commitment": sha256(
                                    contract_bytes(terminal, redactor=redactor)
                                ).hexdigest(),
                            }
                        ),
                    }
                )
            )
    if isinstance(command, RequestCreationStageCommand):
        from cayu.collaboration._contracts import MAX_ID_BYTES
        from cayu.collaboration.prepared_admission import (
            ForkRecipientAdmissionTarget,
            FreshRecipientAdmissionTarget,
            PreparedRecipientAdmission,
        )
        from cayu.collaboration.recipient_preparation import (
            ForkRecipientCreationPreparation,
            MaterialRecipientCreationPreparation,
        )
        from cayu.sessions.creation_fence import SessionCreationDecision

        proposal = command.preparation
        fork = isinstance(proposal, ForkRecipientCreationPreparation)
        target = proposal.creation
        registration = target.permit.intent.request
        # Use the complete bounded identifier envelope (including JSON escaping),
        # not an optimistic assumption about a native store's UUID formatting.
        maximum_identity = '"' * MAX_ID_BYTES
        session_id = target.requested_session_id or maximum_identity
        prepared_target_values = dict(
            creation=target,
            session_id=session_id,
            session_instance_id=maximum_identity,
            creation_receipt_commitment="sha256:" + "f" * 64,
            initial_input_commitment=target.material_commitment,
            definition_commitment="sha256:" + "f" * 64,
            resources=proposal.resources
            if isinstance(proposal, MaterialRecipientCreationPreparation)
            else (),
        )
        if fork:
            selection = proposal.selection
            prepared_target = ForkRecipientAdmissionTarget.model_validate(
                {
                    **prepared_target_values,
                    "selected_view_commitment": "sha256:" + "f" * 64,
                    "manifest_commitment": selection.manifest_commitment,
                    "view_id": selection.view_id,
                    "source_session_id": selection.source_session_id,
                    "source_session_instance_id": selection.source_session_instance_id,
                }
            )
        else:
            prepared_target = FreshRecipientAdmissionTarget.model_validate(prepared_target_values)
        prepared = PreparedRecipientAdmission(
            receiver=proposal.receiver,
            recipient=registration.participant,
            lifecycle_revision=registration.expected_lifecycle_revision,
            configuration_revision=registration.expected_configuration_revision,
            admission_generation=registration.admission_generation,
            target=prepared_target,
            execution_profile_json=proposal.execution_profile_json,
            budget_binding_json=proposal.budget_binding_json,
        )
        resource_count = (
            len(proposal.resources)
            if isinstance(proposal, MaterialRecipientCreationPreparation)
            else 0
        )
        final_ordinal = (3 if fork else 2) + 2 * resource_count
        if (
            expected_plan is None
            or expected_plan.limits.max_stages < final_ordinal
            or expected_plan.limits.max_resources < resource_count
        ):
            raise CollaborationContractError(
                "Creation planning requires bounded resource, creation and admission stages."
            )
        from cayu.collaboration._planning_records import RequestPlanningStageIntent

        # Probe the exact second-stage envelope too, before the first stage can
        # commit permission to create. These are size projections, not receipts.
        followup = RequestAdmissionCommand(
            operation=expected_plan.admission_operation,
            expected=expected_plan.expected,
            expected_revision=expected_plan.expected_revision,
            expected_input_revision=expected_plan.expected_input_revision,
            expected_input_sha256=expected_plan.expected_input_sha256,
            generation=expected_plan.admission_generation,
            decision="fork" if fork else "fresh",
            # A fixed-width size probe, never published as decision evidence.
            # Material-bound creation is not itself the frozen policy proposal.
            proposal_commitment="f" * 64,
            evidence=(),
            initiator=expected_plan.initiator,
            prepared=prepared,
        )
        plan_hash = sha256(contract_bytes(expected_plan, redactor=redactor)).hexdigest()
        followup_intent = RequestPlanningStageIntent(
            operation=expected_plan.operation.model_copy(
                update={"caller_key": "plan-stage-" + plan_hash}
            ),
            plan=expected_plan.operation,
            plan_sha256=plan_hash,
            ordinal=final_ordinal,
            command=followup,
        )
        followup_stage = stage.model_copy(
            update={
                "intent": followup_intent,
                "registration_event": stage.registration_event.model_copy(
                    update={
                        "operation": followup_intent.operation,
                        "commitment": sha256(
                            contract_bytes(followup_intent, redactor=redactor)
                        ).hexdigest(),
                    }
                ),
            }
        )
        followup_required = preflight_stage_terminal(
            followup_stage, initialized, ceiling=ceiling, redactor=redactor
        )
        for created in (False, True):
            receipt = RequestCreationStageReceipt(
                command=command,
                decision=SessionCreationDecision(
                    target=target,
                    state="created" if created else "excluded",
                    responsibility_registered=True,
                    settlement_acknowledged=True,
                    session_id=session_id if created else None,
                    session_instance_id=maximum_identity if created else None,
                    creation_receipt_commitment="sha256:" + "f" * 64 if created else None,
                ),
                definition_commitment=prepared_target.definition_commitment if created else None,
            )
            projected.append(
                excluded.model_copy(
                    update={
                        "state": "settled",
                        "receipt": receipt,
                        "settlement_event": event.model_copy(
                            update={
                                "type": "plan_stage_settled",
                                "commitment": sha256(
                                    contract_bytes(receipt, redactor=redactor)
                                ).hexdigest(),
                            }
                        ),
                    }
                )
            )
    if isinstance(command, RequestAdmissionCommand) and command.prepared is not None:
        permit = prepared_admission_permit(initialized, command, redactor=redactor)
        receipt = RequestAdmissionReceipt(
            command=command,
            state="admitted",
            revision=command.expected_revision + 1,
            decided_at_ms=maximum,
            event=RequestEvent(
                id="f" * 32,
                sequence=maximum - 1,
                operation=command.operation,
                request=command.expected.intent.selection.reference,
                type="request_admission",
                participants=event.participants,
            ),
            admission_permit=PermitReceipt(
                expected=permit,
                position=maximum,
                event=ParticipantEvent(
                    id="f" * 32,
                    sequence=maximum - 2,
                    operation=permit.operation,
                    type="permit_registered",
                    participants=(command.prepared.recipient,),
                ),
            ),
        )
        projected.append(
            excluded.model_copy(
                update={
                    "state": "settled",
                    "receipt": receipt,
                    "settlement_event": event.model_copy(
                        update={
                            "type": "plan_stage_settled",
                            "commitment": sha256(
                                contract_bytes(receipt, redactor=redactor)
                            ).hexdigest(),
                        }
                    ),
                }
            )
        )
    required = max(
        followup_required,
        *(len(contract_bytes(terminal, redactor=redactor)) for terminal in projected),
    )
    if required > ceiling:
        raise CollaborationContractError("Planning stage lacks room for terminal evidence.")
    return required


def control_record_ceiling(record: RequestPlanningRecord) -> int:
    """Size/shape probe only; these synthetic values are never stored authority."""
    command = record.receipt.command
    owner = command.expected.destination
    reference = ObjectRef(
        owner=owner, kind="control", object_id="control", incarnation="control", revision=2**53 - 1
    )
    initiator = InitiatorBinding(
        issuer=owner,
        principal="control",
        participant=reference,
        mandate=reference,
        invocation_id="control",
        interaction_id="control",
    )
    # Include optional-reference structure, then reserve all remaining allowed
    # canonical bytes for the actual administrator, including escaped strings.
    redactor = SecretRedactor()
    slack = MAX_CONTROL_INITIATOR_BYTES - len(contract_bytes(initiator, redactor=redactor))
    control = RequestPlanningControl(
        expected=command,
        expected_revision=MAX_REQUEST_PLAN_EVENTS - 1,
        kind="cancelled",
        initiator=initiator,
    )
    sequences = (
        record.receipt.event.sequence,
        *range(2**53 - MAX_REQUEST_PLAN_EVENTS + 1, 2**53),
    )
    terminal = prepare_contract(
        RequestPlanningRecord,
        record.model_copy(
            update={
                "state": "cancelled",
                "revision": MAX_REQUEST_PLAN_EVENTS,
                "event_sequences": sequences,
                "reserved_bytes": 0,
                "reserved_events": 0,
                "disposition": planning_disposition(control),
                "successor": None,
                "next_due_at_ms": command.deadline_at_ms,
                "stage_count": command.limits.max_stages if record.decision is not None else 0,
                "pending_stages": command.limits.max_stages if record.decision is not None else 0,
            }
        ),
        redactor=redactor,
    )
    return len(contract_bytes(terminal, redactor=redactor)) + max(slack, 0)


def preflight_plan_control(record: RequestPlanningRecord) -> None:
    required = control_record_ceiling(record)
    maximum = record.receipt.command.limits.max_record_bytes
    if required > maximum:
        raise CollaborationContractError(
            f"Planning lacks room for mandatory bounded cleanup ({required} > {maximum} bytes)."
        )
