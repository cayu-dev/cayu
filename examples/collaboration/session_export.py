"""Export approved metadata from an ordinary session; no model or participant required."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from cayu import (
    CayuApp,
    ExportLimits,
    InMemorySessionStore,
    Message,
    RunRequest,
    SessionExportAccessContext,
    SessionExportAuthorization,
    SessionExportDenied,
    SessionExportPolicy,
    SessionExportProjector,
    SessionExportRef,
    SessionExportRegistration,
    SessionExportRequest,
    SessionExportSettlementRequest,
    SessionIdentity,
)


class LocalPolicy(SessionExportPolicy):
    """Single-process host policy: the lock also serializes revocation.

    A distributed deployment needs its own cross-process guard. Context must
    come from host authentication, never from a model or untrusted request body.
    """

    def __init__(self, session):
        self.session = session
        self.lock = asyncio.Lock()
        self.principals = {"example-operator"}
        owner = {"application_scope": "example", "owner_id": "exports", "incarnation": "v1"}
        self.identity = SessionExportAuthorization.model_validate(
            {
                "issuer": owner,
                "principal": "example-operator",
                "policy": {
                    "owner": owner,
                    "kind": "policy",
                    "object_id": "metadata-only",
                    "incarnation": "v1",
                    "revision": 1,
                },
                "revision": 1,
                "expires_at_ms": 1,
            }
        )

    @property
    def ref(self):
        return self.identity.policy

    @asynccontextmanager
    async def acquire(self, context, *, session_id, session_instance_id, actions, audience=None):
        async with self.lock:
            allowed = {"initialize", "source", "export", "readback", "expose", "retire"}
            if (
                context.principal not in self.principals
                or session_id != self.session.id
                or session_instance_id != self.session.instance_id
                or not actions
                or any(action not in allowed for action in actions)
                or (actions == ("initialize",) and audience is not None)
                or (actions != ("initialize",) and audience != self.identity.issuer)
            ):
                raise SessionExportDenied()
            expires = datetime.now(UTC) + timedelta(minutes=5)
            yield self.identity.model_copy(
                update={
                    "principal": context.principal,
                    "expires_at_ms": int(expires.timestamp() * 1000),
                }
            )

    async def revoke(self, principal):
        async with self.lock:
            self.principals.discard(principal)


class RowCountProjector(SessionExportProjector):
    """Approve only a row count, never transcript text or other source fields."""

    def __init__(self, policy):
        self._ref = policy.ref.model_copy(update={"kind": "projector", "object_id": "row-count"})
        self.audience = policy.identity.issuer

    @property
    def ref(self):
        return self._ref

    def project(self, source):
        return {"record_count": len(source)}

    def validate(self, source, output, audience):
        return (
            audience == self.audience
            and set(output) == {"record_count"}
            and type(output["record_count"]) is int
            and output["record_count"] == len(source)
        )


async def main() -> None:
    store = InMemorySessionStore()
    session = await store.create(
        RunRequest(session_id="ordinary-session", agent_name="example", messages=[]),
        identity=SessionIdentity(provider_name="example", model="unused"),
    )
    await store.append_transcript_messages(session.id, [Message.text("user", "Example input")])
    policy = LocalPolicy(session)
    projector = RowCountProjector(policy)
    registration = SessionExportRegistration(
        owner=policy.identity.issuer,
        policy=policy,
        projectors=(projector,),
        limits=ExportLimits(max_exports=4, max_pending=2, max_retained_bytes=4 * 65536),
    )
    app = CayuApp(session_store=store, session_exports=registration, enable_logging=False)
    # In this local example the host explicitly selects its authenticated operator.
    context = SessionExportAccessContext(principal="example-operator")
    try:
        namespace = await app.initialize_session_exports(session.id, context=context)
        operation = {
            "application_scope": namespace.owner.application_scope,
            "namespace_incarnation": namespace.namespace_incarnation,
            "generation": namespace.generation,
            "caller_key": "export-row-count",
        }
        request = SessionExportRequest(
            ref=SessionExportRef.model_validate(
                {
                    "session_id": namespace.session_id,
                    "session_instance_id": namespace.session_instance_id,
                    "operation": operation,
                }
            ),
            source_indices=(0,),
            audience=namespace.owner,
            projector=projector.ref,
            policy=policy.ref,
        )
        receipt = await app.export_session(request, context=context)
        assert await app.export_session(request, context=context) == receipt
        lookup = await app.lookup_session_export(request, context=context)
        assert lookup.status == "match" and lookup.receipt == receipt
        assert await app.read_session_export(request, context=context) == {"record_count": 1}
        settlement = SessionExportSettlementRequest.model_validate(
            {
                "request": request,
                "operation": {**operation, "caller_key": "retire-row-count"},
                "mode": "retire",
            }
        )
        retired = await app.settle_session_export(settlement, context=context)
        assert await app.settle_session_export(settlement, context=context) == retired
        # Historical evidence remains replayable; retirement is not pruning.
        assert await app.export_session(request, context=context) == receipt
        await policy.revoke(context.principal)
        try:
            await app.lookup_session_export(request, context=context)
        except SessionExportDenied:
            pass
        else:
            raise AssertionError("Revocation must deny even historical readback")
        print("Export, exact replay, authorized read, retirement, and revocation succeeded.")
    finally:
        await app.drain_session_exports()


if __name__ == "__main__":
    asyncio.run(main())
