"""Canonical restrictions apply before source projection and payload exposure."""

import pytest
from tests.core.test_session_export_content_release import ReviewOwner, reviewed_request, run_async
from tests.core.test_session_export_mandates import Resolver
from tests.core.test_session_exports import CONTEXT, OWNER, _ref, harness, published
from tests.core.test_session_exports import backend as backend

from cayu.collaboration.exports import SessionExportDenied
from cayu.collaboration.mandates import MandateChain, ResourceSelector, ResourceSelectorOwner
from cayu.collaboration.releases import ContentExposure


@pytest.mark.parametrize("mode", ["deterministic", "reviewed_prose"])
@pytest.mark.parametrize("entrance", ["export", "read"])
@pytest.mark.parametrize("exclusion", ["alias", "canonical", "unrelated"])
@run_async
async def test_exclusions_require_canonical_owner_identity(backend, mode, entrance, exclusion):
    async with harness(backend) as case:
        reader = ReviewOwner()
        plain, store, _, _ = case.app(release_readers=(reader,))
        await case.create(store)
        request = await case.request(plain)
        if mode == "reviewed_prose":
            request, _ = await reviewed_request(case, plain, store, reader)
            canonical = _ref("reviewed-input")
            request = request.model_copy(
                update={
                    "release": request.release.model_copy(
                        update={
                            "exposure": (
                                ContentExposure(
                                    source=canonical, channel="source", commitment="a" * 64
                                ),
                            )
                        }
                    )
                }
            )
            reader.approve(request, "Reviewed advice.")
        resolver = Resolver(request)
        root = resolver.resolution.chain.entries[0]
        if mode == "deterministic":
            canonical = root.resources[0].resource
        else:
            root = root.model_copy(
                update={"resources": (*root.resources, ResourceSelector(resource=canonical))}
            )
        alias = canonical.model_copy(update={"object_id": "alias-to-selected-resource"})

        class Owner(ResourceSelectorOwner):
            @property
            def owner(self):
                return OWNER

            def canonicalize(self, selector):
                return (
                    selector.model_copy(update={"resource": canonical})
                    if selector.resource == alias
                    else selector
                )

            def contains(self, parent, child):
                return parent == child

        child = root.model_copy(
            update={
                "reference": _ref("child-mandate"),
                "parent": root.reference,
                "remaining_delegations": root.remaining_delegations - 1,
            }
        )
        resolver.resolution = resolver.resolution.model_copy(
            update={"chain": MandateChain(entries=(root, child))}
        )
        resolver.context = resolver.context.model_copy(
            update={
                "mandate": resolver.context.mandate.model_copy(update={"mandate": child.reference})
            }
        )
        app, _, _, projector = case.app(
            mandates=resolver, resource_owners=(Owner(),), release_readers=(reader,)
        )
        if entrance == "read":
            await app.export_session(request, context=resolver.context)
        blocked = {"alias": alias, "canonical": canonical, "unrelated": _ref("other-resource")}[
            exclusion
        ]
        resolver.resolution = resolver.resolution.model_copy(
            update={
                "chain": MandateChain(
                    entries=tuple(
                        entry.model_copy(
                            update={
                                "restrictions": entry.restrictions.model_copy(
                                    update={"excluded_sources": (blocked,)}
                                )
                            }
                        )
                        for entry in (root, child)
                    )
                )
            }
        )
        original = resolver.resolution.model_dump_json()
        call = app.export_session if entrance == "export" else app.read_session_export
        if exclusion == "unrelated":
            await call(request, context=resolver.context)
            assert await app.read_session_export(request, context=resolver.context) == (
                {"count": 5} if mode == "deterministic" else {"text": "Reviewed advice."}
            )
        else:
            with pytest.raises(SessionExportDenied):
                await call(request, context=resolver.context)
            if entrance == "export":
                assert projector.calls == 0 and not reader.calls
                assert not await published(store, case.session_id)
                assert (
                    await plain.lookup_session_export(request, context=CONTEXT)
                ).status == "not_found"
            else:
                assert len(await published(store, case.session_id)) == 1
        assert resolver.resolution.model_dump_json() == original
