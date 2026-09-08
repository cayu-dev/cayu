"""Final permission refusal does not revoke the shared native input owner."""

import asyncio

import pytest
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_authorization import (
    BrowserControlInputRejected,
    BrowserControlPermissionDenied,
)
from cayu.runtime._browser_control_channel import BoundBrowserGuest, BrowserGuestCommandOwner
from cayu.runtime._browser_control_input_tickets import BrowserInputTickets
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlPrincipal,
    BrowserHandbackIntent,
    BrowserSensitiveEntryIntent,
    BrowserTextInputIntent,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "publication_failure", [None, BrowserControlPermissionDenied, BrowserControlInputRejected]
)
def test_final_input_denial_preserves_control_and_allows_handback(
    backend, tmp_path, monkeypatch, publication_failure
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            await BrowserControlPublisher(store).publish(bootstrap)
            policy = Policy(True)
            owner = coordinator(store, policy)
            principal = BrowserControlPrincipal(subject="operator")
            pending = await owner.request_takeover(
                principal=principal, operator_session_id="continuity", intent=intent_for(bootstrap)
            )
            acquired = await owner._publish_guest_acquisition(expected=pending, lease_until_ms=2000)
            sensitive = await owner.request_sensitive_entry(
                principal=principal,
                operator_session_id="continuity",
                intent=BrowserSensitiveEntryIntent(
                    identity=acquired.identity,
                    expected_record_revision=acquired.revision,
                    expected_control_epoch=acquired.control_epoch,
                    request_id=acquired.request.request_id,
                ),
            )
            settled = await owner._publish_guest_sensitive_entry(expected=sensitive)
            intent = BrowserTextInputIntent(
                identity=settled.identity,
                expected_record_revision=settled.revision,
                expected_control_epoch=settled.control_epoch,
                request_id=settled.request.request_id,
                input_sequence=1,
                page=settled.request.pages[0],
            )
            tickets = BrowserInputTickets(owner)
            ticket = await tickets.issue(
                principal=principal, operator_session_id="continuity", intent=intent
            )
            authenticated = tickets.consume(ticket)
            await owner._prepare_text_input(
                principal=principal, operator_session_id="continuity", intent=intent
            )
            calls = []

            class Connection:
                async def send(self, value):
                    calls.append(value)
                    pytest.fail("Denied input reached the native channel")

            commands = BrowserGuestCommandOwner(
                coordinator=owner,
                connection=Connection(),
                bound=BoundBrowserGuest(settled, "channel", "b" * 64),
            )
            before = await store.load_checkpoint("session")
            task = asyncio.create_task(
                commands.request_text_input(
                    principal=authenticated.principal,
                    operator_session_id=authenticated.operator_session_id,
                    intent=authenticated.intent,
                    text="private-input-canary",
                )
            )
            await asyncio.sleep(0)
            entry = commands._input
            assert entry is not None
            if publication_failure:
                publish = owner._publish_text_input_admission

                async def commit_then_deny(publication):
                    await publish(publication)
                    raise publication_failure()

                monkeypatch.setattr(owner, "_publish_text_input_admission", commit_then_deny)
                with pytest.raises(publication_failure):
                    await commands.step()
                with pytest.raises(BrowserControlConflict):
                    await task
                assert commands.publication_transition is not None
                _, current = await owner._load(settled.identity)
                assert current.pending_input_sequence == 1
                assert current.manual_mutation_uncertain
                assert calls == [] and entry.payload == bytearray()
                assert await owner.drain()
                return
            policy.allowed = False
            await commands.step()
            with pytest.raises(BrowserControlPermissionDenied):
                await task
            assert not commands._closed and commands._input is None
            assert commands.publication_transition is None
            assert commands.bound.record == settled
            assert calls == [] and entry.payload == bytearray()
            assert await store.load_checkpoint("session") == before
            policy.allowed = True
            handed = await owner.request_handback(
                principal=principal,
                operator_session_id="continuity",
                intent=BrowserHandbackIntent(
                    identity=settled.identity,
                    expected_record_revision=settled.revision,
                    expected_control_epoch=settled.control_epoch,
                    request_id=settled.request.request_id,
                ),
            )
            assert handed.state == "handback_pending"
            assert await owner.drain()

    asyncio.run(scenario())
