"""Native cleanup-proof schema keys are not secret-bearing application data."""

import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import SecretStr
from tests.core.test_tool_completion import FinalTool, call
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import AgentSpec, CayuApp, Message, ModelStreamEvent, RunRequest, ScriptedModelProvider
from cayu.external_waits import ExternalEventWaits
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions._checkpoint_secret_validation import require_secret_free_durable_object
from cayu.sessions._invocation_lifecycle import _invocation_lifecycle_receipt_ledger_from_checkpoint
from cayu.sessions.external_waits import ExternalEventDelivery
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("external", [False, True])
def test_public_tool_round_preserves_cleanup_schema_keys(backend, external, tmp_path, request):
    async def scenario():
        codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr("A" * 43)})
        )
        async with stores(
            backend, tmp_path, request, [datetime.now(UTC)], public_authority_alias_codec=codec
        ) as (store, reopen):
            redactor = SecretRedactor(["external_execution_origin", "admission_epoch"])
            tool = FinalTool()
            provider = ScriptedModelProvider(
                [
                    call(),
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                ]
            )

            def app_for(native):
                app = CayuApp(session_store=native, secret_redactor=redactor, enable_logging=False)
                app.register_provider(provider, default=True)
                app.register_agent(AgentSpec(name="root", model="model"), tools=[tool])
                return app

            app = app_for(store)
            session_id = "key-collision-" + uuid4().hex
            run = RunRequest(
                agent_name="root", session_id=session_id, messages=[Message.text("user", "Help")]
            )
            waits = ExternalEventWaits(store=store, access_policy=Policy(), redactor=redactor)
            if external:
                correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
                registered = registration(correlation)
                await waits.register(registered, context=CONTEXT)
                parked = await SessionExternalWaitAdapter(app, waits).run_to_wait(
                    run, registered, context=CONTEXT
                )
                assert parked.wait.pending_handoff
            else:
                events = [event async for event in app.run(run)]
                assert (await store.load(session_id)).status.value == "completed", json.dumps(
                    [event.payload for event in events if event.type.value == "session.failed"]
                )
            assert tool.calls == 1 and len(provider.requests) == 2
            await app.aclose()
            await waits.aclose()

            native = reopen()
            checkpoint = await runtime_checkpoint_session_store(native).load_checkpoint(session_id)
            ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
            assert bool(ledger.receipts[0].external_execution_origin) is external
            assert (
                require_secret_free_durable_object(
                    checkpoint, redactor=redactor, field_name="checkpoint"
                )
                == checkpoint
            )
            # The same spellings in ordinary data are not schema authority.
            for payload in (
                {"external_execution_origin": None},
                {"external_execution_origin": {"admission_epoch": 1}},
                {"admission_epoch": 1},
                {"value": "external_execution_origin"},
            ):
                with pytest.raises(ValueError, match="workload secret"):
                    require_secret_free_durable_object(
                        {**checkpoint, "caller_data": payload},
                        redactor=redactor,
                        field_name="checkpoint",
                    )
            if external:
                restored_app = app_for(native)
                restored_waits = ExternalEventWaits(
                    store=native, access_policy=Policy(), redactor=redactor
                )
                await restored_waits.deliver(
                    ExternalEventDelivery(
                        correlation=correlation, delivery_id="done", payload_json='{"done":true}'
                    ),
                    context=CONTEXT,
                )
                result = await SessionExternalWaitAdapter(
                    restored_app, restored_waits
                ).service_wait(registered, context=CONTEXT)
                assert not result.wait.pending_handoff
                assert tool.calls == 1 and len(provider.requests) == 3
                await restored_app.aclose()
                await restored_waits.aclose()

    asyncio.run(scenario())
