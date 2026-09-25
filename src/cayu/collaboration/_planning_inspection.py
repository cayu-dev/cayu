"""Scope-authorized bounded discovery of durable planning responsibility."""

from pydantic import Field, StrictInt

from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES, ContractValue, ExactMatch, ObjectRef
from cayu.collaboration._mandate_validation import MandateUse, validate_mandate_resolution
from cayu.collaboration._planning_records import (
    RequestPlanningCursor,
    RequestPlanningPage,
    RequestPlanningRecord,
)
from cayu.collaboration._planning_store import read_plan_in_transaction
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_coordinator import _safe_request_failure
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.base import PLANNING_FAMILY
from cayu.collaboration.mandates import MandateAccessContext, MandateResolution
from cayu.collaboration.participants import CollaborationNotInitialized, CollaborationUnavailable
from cayu.collaboration.planning import MAX_REQUEST_PLANNING_PAGE


class _Query(ContractValue):
    context: MandateAccessContext
    after: RequestPlanningCursor | None
    limit: StrictInt = Field(ge=1, le=MAX_REQUEST_PLANNING_PAGE)


async def list_pending_plans(requests, *, context, after=None, limit=MAX_REQUEST_PLANNING_PAGE):
    query = prepare_contract(
        _Query, {"context": context, "after": after, "limit": limit}, redactor=requests._redactor
    )

    async def owned():
        return await requests._dependency(lambda: _inspect(requests, query))

    return await requests._observe(
        requests._owners.run(
            owned,
            key=("planning-discovery", object()),
            expectation=contract_bytes(query, redactor=requests._redactor),
            redactor=requests._redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
        )
    )


async def _inspect(requests, query):
    participants, redactor = requests._participants, requests._redactor
    store, initialized = participants._ready()
    participants._capability(store, initialized, mutation=False, family=PLANNING_FAMILY)
    registration = requests._registration
    if registration is None or requests._resolver_ref is None:
        raise CollaborationNotInitialized("Planning discovery is not registered.")
    _, grant = participants._authorize(
        CollaborationAccessContext(principal=query.context.principal), "request_readback"
    )
    # Match existing request-due maintenance: cursors must not disclose another
    # participant's operations through a partially authorized scope scan.
    participants._require_refs(grant, (), create=True)
    require_exact_contract(
        requests._resolver_ref,
        prepare_contract(ObjectRef, registration.mandates.ref, redactor=redactor),
        redactor=redactor,
    )
    async with registration.mandates.acquire(query.context) as raw:
        resolution = prepare_contract(MandateResolution, raw, redactor=redactor)
        async with store._transaction(initialized.binding.application_scope, write=False) as tx:
            now = await tx.now_ms()
        validate_mandate_resolution(
            resolution,
            context=query.context,
            resolver=requests._resolver_ref,
            use=MandateUse(
                audience=initialized.owner,
                scope=initialized.binding.application_scope,
                actions=("readback",),
                resources=(),
                inputs=(),
            ),
            now_ms=now,
            resource_owners=requests._resource_owners,
            redactor=redactor,
        )
        deadline = min(
            resolution.principal.expires_at_ms,
            *(entry.expires_at_ms for entry in resolution.chain.entries),
        )
        async with store._transaction(initialized.binding.application_scope, write=False) as tx:
            await store._anchor(tx, initialized, redactor)
            now = await tx.now_ms()
            if now >= deadline:
                raise CollaborationAccessDenied("Planning discovery authority expired.")
            rows = await tx.scan_pending_request_plans(after=query.after, limit=query.limit)
            records = []
            # Bound bytes independently of item count. Each admitted record is
            # <=64KiB; reserve half the envelope for page structure.
            used = 0
            for raw in rows:
                candidate = prepare_contract(RequestPlanningRecord, raw, redactor=redactor)
                size = len(contract_bytes(candidate, redactor=redactor))
                if records and used + size > MAX_ENVELOPE_BYTES // 2:
                    break
                exact = await read_plan_in_transaction(
                    store, tx, initialized, candidate.receipt.command, redactor=redactor
                )
                if not isinstance(exact, ExactMatch) or exact.receipt != candidate:
                    raise CollaborationUnavailable(
                        "Planning discovery lacks exact retained evidence."
                    )
                records.append(candidate)
                used += size
            cursor = None
            if records and (len(records) < len(rows) or len(rows) == query.limit):
                last = records[-1]
                cursor = RequestPlanningCursor(
                    next_due_at_ms=last.next_due_at_ms, operation=last.receipt.command.operation
                )
            return prepare_contract(
                RequestPlanningPage,
                RequestPlanningPage(
                    records=tuple(records),
                    next_cursor=cursor,
                    observed_at_ms=now,
                ),
                redactor=redactor,
            )
