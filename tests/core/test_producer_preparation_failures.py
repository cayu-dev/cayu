"""Public preparation preserves primary and authority-release failures in order."""

from contextlib import asynccontextmanager

import pytest
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu import ProducerOutputProposal
from cayu.collaboration import _producer_preparation as preparation
from cayu.collaboration.participants import CollaborationUnavailable


@pytest.mark.anyio
async def test_public_preparation_preserves_primary_and_nested_guard_cleanup(
    native_stores, monkeypatch
):
    app, resolver, _, provider, session, _, command, execution = await output_scenario(
        native_stores
    )
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    proposal = ProducerOutputProposal(
        operation=command.operation,
        admission=command.admission,
        binding_incarnation=command.binding_incarnation,
        limits=command.limits,
        destinations=command.destinations,
    )
    receiver = app._request_coordinator._registration.receiving_owner
    acquire = receiver.acquire
    observed = []

    async def failed_read(*args):
        observed.append("primary")
        raise LookupError("native-observation-primary")

    @asynccontextmanager
    async def failed_release(*args, **kwargs):
        async with acquire(*args, **kwargs) as authority:
            try:
                yield authority
            finally:
                observed.extend(("release", "close"))
                raise ExceptionGroup(
                    "authority-release",
                    [
                        OSError("release-secondary"),
                        ExceptionGroup("nested", [RuntimeError("close-third")]),
                    ],
                )

    monkeypatch.setattr(preparation, "_native_execution_commitment", failed_read)
    monkeypatch.setattr(receiver, "acquire", failed_release)
    with pytest.raises(CollaborationUnavailable) as raised:
        await app.prepare_producer_output(proposal, execution, context=resolver.recipient.context)
    assert observed == ["primary", "release", "close"]
    seen = set()
    leaves = []

    def visit(error):
        if error is None or id(error) in seen:
            return
        seen.add(id(error))
        visit(error.__cause__)
        visit(error.__context__)
        if isinstance(error, BaseExceptionGroup):
            for child in error.exceptions:
                visit(child)
        else:
            leaves.append(str(error))

    visit(raised.value)
    markers = [
        marker
        for text in leaves
        for marker in ("native-observation-primary", "release-secondary", "close-third")
        if marker in text
    ]
    assert markers == ["native-observation-primary", "release-secondary", "close-third"]
    assert not provider.requests
    assert await app.session_store.load(session.id) == session
    assert not app._request_coordinator._owners.pending
