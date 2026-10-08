"""Original-invocation accounting evidence, not external-work quiescence."""

from hashlib import sha256

from pydantic import Field, StrictInt, model_validator

from cayu._validation import MAX_PORTABLE_JSON_INTEGER, canonical_durable_json_bytes
from cayu.budgets.base import budget_settlement_id
from cayu.collaboration._contracts import ContractValue, OperationRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import ProducerNativeFailure, ProducerOutputRegistration
from cayu.collaboration.prepared_admission import NativeCommitment, prepared_budget
from cayu.events import EventType
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
)
from cayu.sessions.event_queries import EventQuery
from cayu.vaults.redaction import SecretRedactor


class ProducerBudgetSettlement(ContractValue):
    """Verified original accounting; never a grant to stop or erase native work."""

    registration: OperationRef
    native_release_commitment: NativeCommitment
    binding_commitment: NativeCommitment
    reservation_count: StrictInt = Field(ge=0, le=MAX_PORTABLE_JSON_INTEGER)
    inventory_commitment: NativeCommitment
    native_failure_commitment: NativeCommitment | None = None

    @model_validator(mode="after")
    def require_empty_inventory_evidence(self):
        if (self.reservation_count == 0) != (self.native_failure_commitment is not None):
            raise ValueError("Empty producer accounting requires exact native failure evidence.")
        return self


async def read_producer_budget_settlement(controller, command):
    command = prepare_contract(ProducerOutputRegistration, command, redactor=SecretRedactor())
    ledger = controller._budget_ledger
    if not ledger._supports_producer_budget_readback():
        raise NotImplementedError("Producer accounting readback is not qualified.")
    native = await controller._session_store._read_native_producer_release(command)
    prepared = command.admission.prepared
    assert prepared is not None
    binding = prepared_budget(prepared.budget_binding_json)
    await controller._verify_retained_budget_binding(binding)
    profile = ExecutionProfileIdentity.model_validate_json(prepared.execution_profile_json)
    expected_payload = {
        "interaction_id": native.interaction_id,
        "session_instance_id": native.session_instance_id,
        "execution_profile_fingerprint": profile.fingerprint,
        "budget_binding_id": binding.binding_id,
        "budget_binding_authority_sha256": binding.authority_digest,
        "budget_root_id": binding.root_budget_id,
    }
    required_keys = {binding.root_budget_id, *binding.ancestor_budget_ids}
    attempts = {}
    seen = set()
    previous_id = None
    digest = sha256(b"cayu.producer-budget-settlement.v1\0")
    after = None
    while page := await ledger._scan_reservation_records(session_id=native.session_id, after=after):
        for record in page:
            payload = record.settlement_event_payload
            # Only positive, owner-persisted incarnation evidence can identify
            # unrelated history. Its invocation and settlement must also agree
            # below; an interaction ID alone need not be unique across incarnations.
            historical = (
                type(payload.get("session_instance_id")) is str
                and bool(payload["session_instance_id"])
                and payload["session_instance_id"] != expected_payload["session_instance_id"]
                and type(payload.get("interaction_id")) is str
                and bool(payload["interaction_id"])
            )
            if (
                (previous_id is not None and record.reservation_id <= previous_id)
                or record.session_id != native.session_id
                or any(
                    type(payload.get(key)) is not str or not payload[key]
                    for key in (
                        ("session_instance_id", "interaction_id")
                        if historical
                        else expected_payload
                    )
                )
                or (
                    not historical
                    and any(payload.get(key) != value for key, value in expected_payload.items())
                )
            ):
                raise ValueError(
                    "Producer reservation inventory conflicts with original authority."
                )
            previous_id = record.reservation_id
            comparison_payload = payload if historical else expected_payload
            # Atomic binding admission allows at most sixteen ceilings per
            # consumed attempt. This also bounds retained readback bookkeeping.
            if not historical and len(seen) >= binding.allowance * 16:
                raise ValueError("Producer reservation inventory exceeds its original allowance.")
            if record.status not in {"released", "reconciled"}:
                raise ValueError("Original producer accounting remains unsettled.")
            settlement = await ledger.load_settlement(budget_settlement_id(record.reservation_id))
            if settlement is None or settlement.event_published is not True:
                raise ValueError("Original producer settlement evidence is unavailable.")
            reconciliation = settlement.reconciliation
            if (
                settlement.session_id != native.session_id
                or settlement.event.interaction_id != comparison_payload["interaction_id"]
                or any(
                    settlement.event.payload.get(key) != value
                    for key, value in comparison_payload.items()
                    if key != "interaction_id"
                )
                or any(
                    getattr(reconciliation, key) != getattr(record, key)
                    for key in (
                        "reservation_id",
                        "budget_limit_id",
                        "model_step_id",
                        "model_attempt_id",
                        "status",
                        "reserved_amount",
                        "actual_amount",
                        "reason",
                        "billing_identity",
                    )
                )
                or reconciliation.settled_at != record.updated_at
            ):
                raise ValueError("Original producer settlement conflicts with its reservation.")
            if historical:
                continue
            seen.add(record.reservation_id)
            step, limits, keys = attempts.setdefault(
                record.model_attempt_id, (record.model_step_id, set(), set())
            )
            if (
                step != record.model_step_id
                or record.budget_limit_id in limits
                or len(limits) >= 16
                or len(attempts) > binding.allowance
            ):
                raise ValueError("Producer accounting attempt membership conflicts.")
            limits.add(record.budget_limit_id)
            if record.scope == "causal":
                keys.add(record.key)
            material = canonical_durable_json_bytes(
                {
                    "reservation": record.model_dump(mode="json"),
                    "settlement": settlement.model_dump(mode="json"),
                },
                "producer budget inventory",
            )
            digest.update(len(material).to_bytes(8, "big"))
            digest.update(material)
        after = page[-1].reservation_id
    if any(not required_keys.issubset(keys) for _, _, keys in attempts.values()):
        raise ValueError("Original producer accounting lacks complete ceiling evidence.")
    native_failure_commitment = None
    if not seen:
        # Retained native evidence can contradict a supposedly empty inventory.
        # Presence is a refusal only; the absence of these events never grants
        # settlement without all of the owner evidence below and above.
        if await controller._session_store.query_events(
            EventQuery(
                session_id=native.session_id,
                interaction_id=native.interaction_id,
                event_types=(EventType.BUDGET_RESERVED, EventType.MODEL_STARTED),
                limit=1,
            )
        ):
            raise ValueError("Empty producer inventory contradicts native accounting evidence.")
        # The qualified ledger's complete inventory is interpreted only after
        # exact native release and original binding readback above. A refusal
        # before the first reservation has no accepted accounting to reconcile.
        # Require positive native failure evidence too: an answer with missing
        # reservations, mutable terminal status, or absent output is not proof.
        failed = await controller._session_store._read_retained_native_producer_output(command)
        if type(failed) is not ProducerNativeFailure:
            raise ValueError("Empty producer accounting lacks exact native failure evidence.")
        redactor = SecretRedactor()
        failed = prepare_contract(ProducerNativeFailure, failed, redactor=redactor)
        require_exact_contract(command, failed.registration, redactor=redactor)
        if failed.interaction_id != native.interaction_id:
            raise ValueError("Producer native failure belongs to another interaction.")
        native_failure_commitment = (
            "sha256:" + sha256(contract_bytes(failed, redactor=redactor)).hexdigest()
        )
    # This receipt says nothing about descendants or outstanding external work;
    # final cleanup must independently prove those responsibilities quiescent.
    return ProducerBudgetSettlement(
        registration=command.operation,
        native_release_commitment=native.release_commitment,
        binding_commitment="sha256:" + binding.authority_digest,
        reservation_count=len(seen),
        inventory_commitment="sha256:" + digest.hexdigest(),
        native_failure_commitment=native_failure_commitment,
    )
