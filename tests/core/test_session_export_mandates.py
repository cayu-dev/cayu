"""Registered authority is enforced through the real session export receiver."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_session_export_content_release import run_async
from tests.core.test_session_exports import AUDIENCE, CONTEXT, OWNER, _ref, harness, published
from tests.core.test_session_exports import backend as backend

from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration._session_export_authority import selected_resources
from cayu.collaboration.exports import (
    SessionExportAccessContext,
    SessionExportConflict,
    SessionExportDenied,
    SessionExportSettlementRequest,
    SessionExportUnavailable,
)
from cayu.collaboration.mandates import (
    CollaborationMandate,
    MandateAccessContext,
    MandateChain,
    MandateResolution,
    MandateResolver,
    MandateRestrictions,
    PrincipalResolution,
    ResourceSelector,
    ResourceSelectorOwner,
)


class Resolver(MandateResolver):
    def __init__(self, request):
        self.lock = asyncio.Lock()
        self.revoked = False
        root = CollaborationMandate(
            reference=_ref("mandate"),
            root=_ref("mandate"),
            parent=None,
            issuer=OWNER,
            principal=CONTEXT.principal,
            participant=None,
            audiences=(AUDIENCE, OWNER),
            scopes=(OWNER.application_scope,),
            actions=("source", "publish", "readback", "expose", "retire", "release", "administer"),
            resources=selected_resources(OWNER, request),
            remaining_delegations=2,
            sponsor=_ref("sponsor"),
            budgets=(_ref("common-budget"),),
            restrictions=MandateRestrictions(
                channels=("source",),
                excluded_sources=(),
                independence_policy=_ref("independence"),
                disclosure_policy=_ref("disclosure"),
            ),
            expires_at_ms=4102444800000,
            revocation_generation=1,
        )
        self.resolution = MandateResolution(
            principal=PrincipalResolution(
                resolver=self.ref,
                issuer=OWNER,
                principal=CONTEXT.principal,
                participants=(),
                audiences=root.audiences,
                scopes=root.scopes,
                actions=root.actions,
                expires_at_ms=root.expires_at_ms,
            ),
            chain=MandateChain(entries=(root,)),
        )
        self.context = SessionExportAccessContext(
            principal=CONTEXT.principal,
            mandate=MandateAccessContext(
                issuer=OWNER,
                principal=CONTEXT.principal,
                mandate=root.reference,
            ),
        )

    @property
    def ref(self):
        return _ref("mandate-resolver")

    @asynccontextmanager
    async def acquire(self, context):
        async with self.lock:
            if self.revoked or context != self.context.mandate:
                raise SessionExportDenied()
            yield self.resolution


@pytest.mark.parametrize("replacement", [False, True])
@run_async
async def test_resource_owner_registration_cannot_move_to_another_identity(backend, replacement):
    class Owner(ResourceSelectorOwner):
        identity = OWNER
        calls = 0

        @property
        def owner(self):
            return self.identity

        def canonicalize(self, selector):
            self.calls += 1
            return selector

        def contains(self, parent, child):
            return parent == child

    async with harness(backend) as case:
        plain, store, _, _ = case.app()
        await case.create(store)
        request = await case.request(plain)
        resolver = Resolver(request)
        owner = Owner()
        app, _, _, projector = case.app(mandates=resolver, resource_owners=(owner,))
        if replacement:
            owner.identity = OWNER.model_copy(update={"incarnation": "replacement"})
            with pytest.raises(SessionExportDenied):
                await app.export_session(request, context=resolver.context)
            assert owner.calls == 0
            assert projector.calls == 0
            assert not await published(store, case.session_id)
        else:
            await app.export_session(request, context=resolver.context)
            assert owner.calls > 0
            assert projector.calls == 1


@run_async
async def test_registered_mandate_publication_replay_and_current_exposure(backend):
    async with harness(backend) as case:
        original, store, _, _ = case.app()
        await case.create(store)
        request = await case.request(original)
        resolver = Resolver(request)
        app, _, _, projector = case.app(mandates=resolver)
        receipt = await app.export_session(request, context=resolver.context)
        assert receipt.expected.initiator.mandate == resolver.context.mandate.mandate
        assert receipt.expected.intent.authorization.mandate == resolver.resolution
        assert projector.calls == 1
        assert await app.read_session_export(request, context=resolver.context) == {"count": 5}
        reopened, _, _, _ = case.app(mandates=resolver, projectors=())
        assert await reopened.export_session(request, context=resolver.context) == receipt
        resolver.revoked = True
        with pytest.raises(SessionExportDenied):
            await app.read_session_export(request, context=resolver.context)
        assert len(await published(store, case.session_id)) == 1


@pytest.mark.parametrize("callback", ["canonicalize", "contains"])
@run_async
async def test_owner_callback_cannot_rewrite_comparison_authority(backend, callback):
    class RewritingOwner(ResourceSelectorOwner):
        @property
        def owner(self):
            return OWNER

        def canonicalize(self, selector):
            if callback == "canonicalize":
                object.__setattr__(selector.resource, "object_id", "rewritten")
            return selector

        def contains(self, parent, child):
            object.__setattr__(child.resource, "object_id", "rewritten")
            return True

    async with harness(backend) as case:
        plain, store, _, _ = case.app()
        await case.create(store)
        request = await case.request(plain)
        resolver = Resolver(request)
        root = resolver.resolution.chain.entries[0].model_copy(
            update={"resources": (ResourceSelector(resource=_ref("namespace"), mode="subtree"),)}
        )
        resolver.resolution = resolver.resolution.model_copy(
            update={"chain": MandateChain(entries=(root,))}
        )
        original = resolver.resolution.model_dump_json()
        app, _, _, projector = case.app(mandates=resolver, resource_owners=(RewritingOwner(),))
        with pytest.raises(SessionExportDenied):
            await app.export_session(request, context=resolver.context)
        assert resolver.resolution.model_dump_json() == original
        assert projector.calls == 0
        assert not await published(store, case.session_id)


@pytest.mark.parametrize(
    "changed", ["action", "audience", "resource", "channel", "issuer", "expiry"]
)
@run_async
async def test_unauthorized_mandate_refuses_before_projection_or_publication(backend, changed):
    async with harness(backend) as case:
        original, store, _, _ = case.app()
        await case.create(store)
        request = await case.request(original)
        resolver = Resolver(request)
        root = resolver.resolution.chain.entries[0]
        if changed == "action":
            root = root.model_copy(update={"actions": ("readback",)})
        elif changed == "audience":
            root = root.model_copy(update={"audiences": (OWNER,)})
        elif changed == "resource":
            root = root.model_copy(update={"resources": ()})
        elif changed == "channel":
            root = root.model_copy(
                update={"restrictions": root.restrictions.model_copy(update={"channels": ()})}
            )
        elif changed == "issuer":
            resolver.resolution = resolver.resolution.model_copy(
                update={
                    "principal": resolver.resolution.principal.model_copy(
                        update={"resolver": _ref("replacement-resolver")}
                    )
                }
            )
        else:
            root = root.model_copy(update={"expires_at_ms": 1})
        resolver.resolution = resolver.resolution.model_copy(
            update={"chain": MandateChain(entries=(root,))}
        )
        app, _, _, projector = case.app(mandates=resolver)
        with pytest.raises(SessionExportDenied):
            await app.export_session(request, context=resolver.context)
        assert projector.calls == 0
        assert not await published(store, case.session_id)


@pytest.mark.parametrize(
    "changed",
    [
        "sponsor",
        "budgets",
        "actions",
        "audiences",
        "scopes",
        "resources",
        "remaining_delegations",
        "expires_at_ms",
        "revocation_generation",
        "channels",
        "excluded_sources",
        "independence_policy",
        "disclosure_policy",
    ],
)
@run_async
async def test_same_mandate_identity_with_changed_claims_conflicts_on_replay(backend, changed):
    async with harness(backend) as case:
        original, store, _, _ = case.app()
        await case.create(store)
        request = await case.request(original)
        resolver = Resolver(request)
        app, _, _, projector = case.app(mandates=resolver)
        await app.export_session(request, context=resolver.context)
        root = resolver.resolution.chain.entries[0]
        replacements = {
            "sponsor": _ref("new-sponsor"),
            "budgets": (*root.budgets, _ref("extra-budget")),
            "actions": ("readback",),
            "audiences": (AUDIENCE,),
            "scopes": (*root.scopes, "extra-scope"),
            "resources": (),
            "remaining_delegations": 1,
            "expires_at_ms": root.expires_at_ms - 1,
            "revocation_generation": 2,
            "channels": (),
            "excluded_sources": (_ref("excluded-report"),),
            "independence_policy": _ref("changed-independence"),
            "disclosure_policy": _ref("changed-disclosure"),
        }
        if changed in {"channels", "excluded_sources", "independence_policy", "disclosure_policy"}:
            root = root.model_copy(
                update={
                    "restrictions": root.restrictions.model_copy(
                        update={changed: replacements[changed]}
                    )
                }
            )
        else:
            root = root.model_copy(update={changed: replacements[changed]})
        resolver.resolution = resolver.resolution.model_copy(
            update={"chain": MandateChain(entries=(root,))}
        )
        with pytest.raises(SessionExportConflict):
            await app.export_session(request, context=resolver.context)
        assert projector.calls == 1
        # Still-authorized historical inspection is separate from mutation replay:
        # narrowing current rights cannot rewrite the earlier receipt.
        assert (
            await app.lookup_session_export(request, context=resolver.context)
        ).status == "match"
        assert len(await published(store, case.session_id)) == 1


@run_async
async def test_settlement_replay_binds_full_mandate_not_only_reference(backend):
    async with harness(backend) as case:
        original, store, _, _ = case.app()
        await case.create(store)
        request = await case.request(original)
        resolver = Resolver(request)
        app, _, _, _ = case.app(mandates=resolver)
        await app.export_session(request, context=resolver.context)
        settlement = SessionExportSettlementRequest(
            request=request,
            operation=request.ref.operation.model_copy(update={"caller_key": "retire"}),
            mode="retire",
        )
        receipt = await app.settle_session_export(settlement, context=resolver.context)
        assert receipt.mandate_commitment is not None
        assert await app.settle_session_export(settlement, context=resolver.context) == receipt
        root = resolver.resolution.chain.entries[0].model_copy(
            update={"sponsor": _ref("replacement")}
        )
        resolver.resolution = resolver.resolution.model_copy(
            update={"chain": MandateChain(entries=(root,))}
        )
        with pytest.raises(SessionExportConflict):
            await app.settle_session_export(settlement, context=resolver.context)


@run_async
async def test_raw_request_authority_and_missing_resolver_never_grant_permission(backend):
    async with harness(backend) as case:
        app, store, _, projector = case.app()
        await case.create(store)
        request = await case.request(app)
        resolver = Resolver(request)
        raw = request.model_dump(mode="json")
        raw["mandate"] = resolver.resolution.model_dump(mode="json")
        with pytest.raises(CollaborationContractError):
            await app.export_session(raw, context=CONTEXT)
        with pytest.raises(SessionExportDenied):
            await app.export_session(request, context=resolver.context)
        assert projector.calls == 0
        assert not await published(store, case.session_id)


@pytest.mark.parametrize(
    "channel", ["prompt", "context", "tool", "retrieval", "artifact", "source"]
)
@pytest.mark.parametrize("empty_source", [False, True])
@run_async
async def test_reviewed_manifest_channels_are_checked_at_export_and_exposure(
    backend, channel, empty_source
):
    from tests.core.test_session_export_content_release import ReviewOwner, reviewed_request

    from cayu.collaboration._session_export_store import source_digest
    from cayu.collaboration.releases import ContentExposure

    async with harness(backend) as case:
        reader = ReviewOwner()
        initial, store, _, _ = case.app(release_readers=(reader,))
        await case.create(store)
        request, _ = await reviewed_request(case, initial, store, reader)
        exposure = ContentExposure(
            source=_ref("review-input"), channel=channel, commitment="a" * 64
        )
        release = request.release.model_copy(update={"exposure": (exposure,)})
        if empty_source:
            release = release.model_copy(update={"source_commitment": source_digest(())})
        request = request.model_copy(
            update={
                "release": release,
                "source_indices": () if empty_source else request.source_indices,
            }
        )
        reader.approve(request, "Reviewed advice.")
        resolver = Resolver(request)
        root = resolver.resolution.chain.entries[0]
        root = root.model_copy(
            update={
                "resources": (*root.resources, ResourceSelector(resource=exposure.source)),
                "restrictions": root.restrictions.model_copy(
                    update={"channels": tuple(dict.fromkeys(("source", channel)))}
                ),
            }
        )
        resolver.resolution = resolver.resolution.model_copy(
            update={"chain": MandateChain(entries=(root,))}
        )
        app, _, _, _ = case.app(mandates=resolver, release_readers=(reader,))
        allowed = resolver.resolution
        restricted = root.model_copy(
            update={
                "restrictions": root.restrictions.model_copy(
                    update={
                        "channels": tuple(
                            item for item in root.restrictions.channels if item != channel
                        ),
                    }
                ),
            }
        )
        resolver.resolution = allowed.model_copy(
            update={"chain": MandateChain(entries=(restricted,))}
        )
        with pytest.raises(SessionExportDenied):
            await app.export_session(request, context=resolver.context)
        assert not reader.calls and not await published(store, case.session_id)
        resolver.resolution = allowed
        receipt = await app.export_session(request, context=resolver.context)
        assert await app.read_session_export(request, context=resolver.context) == {
            "text": "Reviewed advice."
        }
        resolver.resolution = allowed.model_copy(
            update={"chain": MandateChain(entries=(restricted,))}
        )
        # Current disclosure restrictions do not rewrite historical receipt identity.
        assert (
            await app.lookup_session_export(request, context=resolver.context)
        ).receipt == receipt
        with pytest.raises(SessionExportDenied):
            await app.read_session_export(request, context=resolver.context)
        excluded = root.model_copy(
            update={
                "restrictions": root.restrictions.model_copy(
                    update={"excluded_sources": (exposure.source,)}
                ),
            }
        )
        resolver.resolution = allowed.model_copy(
            update={"chain": MandateChain(entries=(excluded,))}
        )
        with pytest.raises(SessionExportDenied):
            await app.read_session_export(request, context=resolver.context)


@pytest.mark.parametrize(
    "outcome",
    [
        "published",
        "registration_failed",
        "projection_failed",
        "cancelled_before_registration",
        "cancelled_after_registration",
    ],
)
@run_async
async def test_participant_export_settles_its_registered_responsibility(
    backend, outcome, monkeypatch, participant_backend
):
    from tests.core.test_participant_identity import configuration, registration

    from cayu.collaboration.access import (
        CollaborationAccessContext,
        CollaborationAccessGrant,
        CollaborationAccessPolicy,
    )
    from cayu.collaboration.participants import ParticipantCreate

    class Access(CollaborationAccessPolicy):
        def authorize(self, context, *, application_scope, action):
            assert context.principal == CONTEXT.principal
            return CollaborationAccessGrant(application_scope=application_scope, participants=None)

    participant_store = participant_backend
    try:
        async with harness(backend) as case:
            initial, store, _, _ = case.app()
            await case.create(store)
            request = await case.request(initial)
            resolver = Resolver(request)
            participant_registration = registration(policy=Access(), scope=OWNER.application_scope)
            app, _, _, projector = case.app(
                mandates=resolver,
                collaboration=participant_registration,
                collaboration_store=participant_store,
            )
            namespace = await app.initialize_collaboration()
            access = CollaborationAccessContext(principal=CONTEXT.principal)
            created = await app.create_participant(
                ParticipantCreate(
                    operation=namespace.operation(case.session_id), configuration=configuration()
                ),
                context=access,
            )
            participant = created.participants[0].reference
            root = resolver.resolution.chain.entries[0].model_copy(
                update={"participant": participant}
            )
            resolver.resolution = resolver.resolution.model_copy(
                update={
                    "principal": resolver.resolution.principal.model_copy(
                        update={"participants": (participant,)}
                    ),
                    "chain": MandateChain(entries=(root,)),
                }
            )
            resolver.context = resolver.context.model_copy(
                update={
                    "mandate": resolver.context.mandate.model_copy(
                        update={"participant": participant}
                    ),
                }
            )
            if outcome == "published":
                receipt = await app.export_session(request, context=resolver.context)
                assert receipt.expected.initiator.participant is not None
                assert await app.export_session(request, context=resolver.context) == receipt
                assert projector.calls == 1
            else:
                entered, release = asyncio.Event(), asyncio.Event()
                with monkeypatch.context() as patch:
                    if outcome.startswith("cancelled"):
                        original_register = participant_store._register_permit

                        async def delayed(*args, **kwargs):
                            result = None
                            if outcome == "cancelled_after_registration":
                                result = await original_register(*args, **kwargs)
                            entered.set()
                            await asyncio.wait_for(release.wait(), 10)
                            if result is None:
                                result = await original_register(*args, **kwargs)
                            return result

                        patch.setattr(participant_store, "_register_permit", delayed)
                        observer = asyncio.create_task(
                            app.export_session(request, context=resolver.context)
                        )
                        await asyncio.wait_for(entered.wait(), 5)
                        observer.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await observer
                        assert observer.cancelled() and observer.cancelling() == 1
                    elif outcome == "registration_failed":

                        async def unavailable(*args, **kwargs):
                            raise RuntimeError("registration unavailable")

                        patch.setattr(participant_store, "_register_permit", unavailable)
                    else:

                        def failed(*args, **kwargs):
                            raise RuntimeError("projection failed")

                        patch.setattr(projector, "project", failed)
                    if not outcome.startswith("cancelled"):
                        with pytest.raises(SessionExportUnavailable):
                            await app.export_session(request, context=resolver.context)
                # Administration need not impersonate the original delegated actor.
                administrator, _, _, _ = case.app(
                    collaboration=participant_registration,
                    collaboration_store=participant_store,
                )
                await administrator.initialize_collaboration()
                settled = await administrator.reconcile_session_export(request, context=CONTEXT)
                assert settled.state == "excluded" and settled.receipt is None
                assert (
                    await administrator.reconcile_session_export(request, context=CONTEXT)
                    == settled
                )
                release.set()
                if outcome.startswith("cancelled"):
                    await app.drain_session_exports()
                    assert projector.calls == 0
                with pytest.raises(SessionExportUnavailable):
                    await app.export_session(request, context=resolver.context)
                assert not await published(store, case.session_id)
            inspected = await app.inspect_participant(participant, context=access)
            assert inspected.outstanding_obligations == 0
            assert inspected.issued_permit_frontier == int(
                outcome
                not in {
                    "registration_failed",
                    "cancelled_before_registration",
                }
            )
            if outcome != "published":
                await store.delete_session(case.session_id)
    finally:
        await participant_store.close()


@pytest.fixture
def participant_backend(backend, request, tmp_path):
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.storage.collaboration_postgres import PostgresCollaborationStore
    from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
    from cayu.storage.migrations import SchemaMode

    kind = backend[0]
    if kind == "memory":
        return InMemoryCollaborationStore()
    if kind == "sqlite":
        return SQLiteCollaborationStore(tmp_path / "participants.sqlite")
    return PostgresCollaborationStore(
        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
    )
