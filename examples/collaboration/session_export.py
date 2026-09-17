"""Export approved metadata from an ordinary session; no model or participant required."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from cayu import (
    CayuApp,
    ContentReleaseExpectation,
    ContentReleaseReader,
    ContentReleaseRequest,
    ExportLimits,
    InMemorySessionStore,
    Message,
    ReleasedContent,
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


class ApprovedReportProjector(SessionExportProjector):
    """Project one host-approved immutable report, not arbitrary report-shaped text.

    The application obtains this exact report/revision from its own trusted report
    owner before registration. New approved material needs a new configuration
    reference. This small example has no remote report store or implicit retrieval.
    Source tool names and matching shapes are not authentication.
    """

    def __init__(self, *, reference, audience, report_reference, passed):
        if type(passed) is not bool or report_reference.revision is None:
            raise ValueError("A pinned report and strict boolean decision are required.")
        self._ref = reference
        self.audience = audience
        self._approved = json.dumps(
            {"report": report_reference.model_dump(mode="json"), "passed": passed},
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def ref(self):
        return self._ref

    def approved_source(self):
        return json.loads(self._approved)

    def _selected(self, source):
        if len(source) != 1 or len(source[0].message.content) != 1:
            raise SessionExportDenied()
        part = source[0].message.content[0]
        if (
            part.type != "tool_result"
            or part.is_error
            or part.content
            or part.artifacts
            or type(part.structured) is not dict
            or set(part.structured) != {"report", "passed"}
            or type(part.structured["passed"]) is not bool
            or json.dumps(part.structured, sort_keys=True, separators=(",", ":")) != self._approved
        ):
            raise SessionExportDenied()
        # Return the detached owner-approved record, never copy untrusted titles,
        # URLs, attachment material or unselected private source into the output.
        return self.approved_source()

    def project(self, source):
        return self._selected(source)

    def validate(self, source, output, audience):
        return (
            audience == self.audience
            and type(output.get("passed")) is bool
            and json.dumps(output, sort_keys=True, separators=(",", ":"))
            == json.dumps(self._selected(source), sort_keys=True, separators=(",", ":"))
        )


class LocalReviewOwner(ContentReleaseReader):
    """One explicit host-reviewed announcement, with a held revocation guard.

    This local example keeps review evidence in memory. A deployment must load
    its real durable review decision; reconstructing a caller proposal is not
    approval. No source content is selected for this standalone announcement.
    """

    def __init__(self, policy):
        self.policy = policy
        self.lock = asyncio.Lock()
        self.approved = None

    @property
    def ref(self):
        return self.policy.ref.model_copy(update={"kind": "review", "object_id": "announcement"})

    def approve(self, request, text):
        expectation = ContentReleaseExpectation(
            request=request.release,
            source_owner=self.policy.identity.issuer,
            session_id=request.ref.session_id,
            session_instance_id=request.ref.session_instance_id,
            source_indices=request.source_indices,
            audience=request.audience,
            validator=request.projector,
            policy=request.policy,
        )
        self.approved = ReleasedContent.model_validate(
            {
                "receipt": {
                    "expected": expectation,
                    "reviewer": {
                        "issuer": self.policy.identity.issuer,
                        "principal": "example-reviewer",
                        "participant": None,
                        "mandate": None,
                        "invocation_id": None,
                        "interaction_id": None,
                    },
                    "authorization_revision": 1,
                    "expires_at_ms": int(
                        (datetime.now(UTC) + timedelta(minutes=5)).timestamp() * 1000
                    ),
                },
                "text": text,
            }
        )

    @asynccontextmanager
    async def acquire(self, expected):
        async with self.lock:
            if self.approved is None or self.approved.receipt.expected != expected:
                raise SessionExportDenied()
            yield self.approved


async def main() -> None:
    store = InMemorySessionStore()
    session = await store.create(
        RunRequest(session_id="ordinary-session", agent_name="example", messages=[]),
        identity=SessionIdentity(provider_name="example", model="unused"),
    )
    await store.append_transcript_messages(session.id, [Message.text("user", "Example input")])
    policy = LocalPolicy(session)
    projector = RowCountProjector(policy)
    review = LocalReviewOwner(policy)
    report = ApprovedReportProjector(
        reference=policy.ref.model_copy(update={"kind": "projector", "object_id": "report-v1"}),
        audience=policy.identity.issuer,
        report_reference=policy.ref.model_copy(update={"kind": "report", "object_id": "checks"}),
        passed=True,
    )
    await store.append_transcript_messages(
        session.id,
        [
            Message.tool_result(
                tool_call_id="checks", tool_name="checks", structured=report.approved_source()
            )
        ],
    )
    registration = SessionExportRegistration(
        owner=policy.identity.issuer,
        policy=policy,
        projectors=(projector, report),
        release_readers=(review,),
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
        announcement = "The report is ready for authorized inspection."
        announcement_request = request.model_copy(
            update={
                "ref": request.ref.model_copy(
                    update={
                        "operation": request.ref.operation.model_copy(
                            update={"caller_key": "announcement"}
                        )
                    }
                ),
                "source_indices": (),
                "mode": "reviewed_prose",
                "projector": review.ref,
                "release": ContentReleaseRequest(
                    decision=review.ref.model_copy(update={"kind": "review-decision"}),
                    source_commitment=sha256(b"[]").hexdigest(),
                    text_commitment=sha256(announcement.encode("utf-8")).hexdigest(),
                    exposure=(),
                ),
            }
        )
        # Explicit trusted-host review precedes the export call. An agent cannot
        # obtain approval merely by submitting these same bytes or reference.
        review.approve(announcement_request, announcement)
        await app.export_session(announcement_request, context=context)
        assert await app.read_session_export(announcement_request, context=context) == {
            "text": announcement
        }
        await app.settle_session_export(
            SessionExportSettlementRequest(
                request=announcement_request,
                operation=request.ref.operation.model_copy(
                    update={"caller_key": "retire-announcement"}
                ),
                mode="retire",
            ),
            context=context,
        )
        report_request = request.model_copy(
            update={
                "ref": request.ref.model_copy(
                    update={
                        "operation": request.ref.operation.model_copy(
                            update={"caller_key": "report"}
                        )
                    }
                ),
                "source_indices": (1,),
                "projector": report.ref,
            }
        )
        await app.export_session(report_request, context=context)
        assert (
            await app.read_session_export(report_request, context=context)
            == report.approved_source()
        )
        await app.settle_session_export(
            SessionExportSettlementRequest(
                request=report_request,
                operation=request.ref.operation.model_copy(update={"caller_key": "retire-report"}),
                mode="retire",
            ),
            context=context,
        )
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
