"""One native transaction boundary for export publication and owned cleanup.

This private entrance accepts only owner-prepared values and an owner-owned
commit guard. Public authorization is resolved before entering it; passing an
authorization-shaped value to a caller-facing export API cannot reach this seam.
"""

from collections.abc import Callable
from datetime import datetime

from cayu.collaboration._session_export_store import (
    ROOT_KEY,
    ExportMutation,
    encoded,
    mutation_scope,
    source_digest,
)
from cayu.collaboration.exports import SessionExportConflict
from cayu.sessions.base import SessionOperationPublication


async def publish_export_mutation(
    store,
    session,
    before,
    after,
    key,
    record,
    events,
    *,
    commit_guard: Callable[[datetime], None],
    source=(),
    additional_records=None,
    expected_old=None,
):
    """Bind the exact mutation and run its guard at the store's commit time.

    Both the export-root comparison and operation comparison are owned by the
    native transaction. Cancellation cannot turn a late worker into an unfenced
    publisher: its captured mutation still names the original incarnation and
    exact before/after values.
    """
    records = {key: record, **(additional_records or {})}
    mutation = ExportMutation(
        session.id,
        session.instance_id,
        None if before is None else encoded(before.model_dump(mode="json")),
        encoded(after.model_dump(mode="json")),
        tuple((k, encoded(v)) for k, v in records.items()),
        tuple(r.index for r in source),
        source_digest(source) if source else None,
        tuple(encoded(e.model_dump(mode="json")) for e in events),
    )

    def transform(current, checkpoint, old, now):
        if current.instance_id != session.instance_id or old != expected_old:
            raise SessionExportConflict()
        commit_guard(now)
        updated = {} if checkpoint is None else dict(checkpoint)
        updated[ROOT_KEY] = after.model_dump(mode="json")
        return SessionOperationPublication(checkpoint=updated, operation_records=records)

    with mutation_scope(mutation):
        await store.publish_session_operation_guarded_with_store_time(
            session.id,
            idempotency_key=key,
            operation_transform=transform,
            commit_guard=lambda: None,
            commit_time_guard=commit_guard,
            events=events,
        )
