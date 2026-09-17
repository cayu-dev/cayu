"""Domain authorization refusals retain the public export denial contract."""

from contextlib import asynccontextmanager

import pytest
from tests.core.test_participant_identity import configuration, registration
from tests.core.test_session_export_content_release import run_async
from tests.core.test_session_export_mandates import Resolver
from tests.core.test_session_export_mandates import participant_backend as participant_backend
from tests.core.test_session_exports import CONTEXT, OWNER, SECRET, harness, published
from tests.core.test_session_exports import backend as backend

from cayu.collaboration.access import (
    CollaborationAccessContext,
    CollaborationAccessDenied,
    CollaborationAccessGrant,
    CollaborationAccessPolicy,
)
from cayu.collaboration.exports import SessionExportDenied
from cayu.collaboration.mandates import MandateChain, MandateDenied
from cayu.collaboration.participants import ParticipantCreate
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("entrance", ["export", "read", "lookup"])
@pytest.mark.parametrize("denier", ["resolver", "participant"])
@run_async
async def test_domain_denials_are_public_export_denials(
    backend, participant_backend, entrance, denier
):
    class Access(CollaborationAccessPolicy):
        denied = False

        def authorize(self, context, *, application_scope, action):
            if self.denied:
                raise CollaborationAccessDenied(SECRET)
            return CollaborationAccessGrant(application_scope=application_scope, participants=None)

    class DenyingResolver(Resolver):
        denied = False

        @asynccontextmanager
        async def acquire(self, context):
            if self.denied:
                raise MandateDenied() from RuntimeError(SECRET)
            async with super().acquire(context) as value:
                yield value

    try:
        async with harness(backend) as case:
            plain, store, _, _ = case.app()
            await case.create(store)
            request = await case.request(plain)
            resolver = DenyingResolver(request)
            access = Access()
            app, _, _, projector = case.app(
                mandates=resolver,
                redactor=SecretRedactor((SECRET,)),
                collaboration=registration(scope=OWNER.application_scope, policy=access),
                collaboration_store=participant_backend,
            )
            initialized = await app.initialize_collaboration()
            created = await app.create_participant(
                ParticipantCreate(
                    operation=initialized.operation(case.session_id), configuration=configuration()
                ),
                context=CollaborationAccessContext(principal=CONTEXT.principal),
            )
            participant = created.participants[0].reference
            resolver.resolution = resolver.resolution.model_copy(
                update={
                    "principal": resolver.resolution.principal.model_copy(
                        update={"participants": (participant,)}
                    ),
                    "chain": MandateChain(
                        entries=(
                            resolver.resolution.chain.entries[0].model_copy(
                                update={"participant": participant}
                            ),
                        )
                    ),
                }
            )
            resolver.context = resolver.context.model_copy(
                update={
                    "mandate": resolver.context.mandate.model_copy(
                        update={"participant": participant}
                    )
                }
            )
            if entrance != "export":
                await app.export_session(request, context=resolver.context)
            resolver.denied = denier == "resolver"
            access.denied = denier == "participant"
            method = {
                "export": app.export_session,
                "read": app.read_session_export,
                "lookup": app.lookup_session_export,
            }[entrance]
            with pytest.raises(SessionExportDenied) as caught:
                await method(request, context=resolver.context)
            assert isinstance(caught.value, PermissionError)
            seen = set()

            def check(error):
                if error is None or id(error) in seen:
                    return
                seen.add(id(error))
                assert SECRET not in str(error) and SECRET not in repr(error)
                check(error.__cause__)
                check(error.__context__)

            check(caught.value)
            assert projector.calls == (0 if entrance == "export" else 1)
            assert len(await published(store, case.session_id)) == projector.calls
    finally:
        await participant_backend.close()
