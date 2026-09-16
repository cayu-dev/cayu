"""Authorized model-only restoration must replace suppressed input markers."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from tests.core.test_tool_argument_continuity import _private_fixture, _PrivateReadProbe

from cayu.messages import ProviderStatePart
from cayu.providers.base import ModelRequest
from cayu.providers.openai import build_openai_payload
from cayu.runtime._argument_continuity import append_record, materialize
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("redact", [False, True])
@pytest.mark.parametrize("envelope", ["legacy", "unavailable", "different"])
def test_model_only_restoration_preserves_audit_and_envelope_authority(retain, redact, envelope):
    async def run():
        session = SimpleNamespace(id="session", instance_id="incarnation")
        message, continuity = _private_fixture()
        call = message.content[0].model_copy(update={"arguments_state": "unavailable"})
        arguments = (
            {}
            if envelope == "legacy"
            else call.continuation_arguments()
            if envelope == "unavailable"
            else {"text": "a different submitted object"}
        )
        native = ProviderStatePart(
            provider="openai",
            state={
                "type": "function_call",
                "id": "fc_test",
                "call_id": call.tool_call_id,
                "name": call.tool_name,
                "arguments": json.dumps(arguments, sort_keys=True, separators=(",", ":")),
                "status": "completed",
            },
        )
        message = message.model_copy(update={"content": (call, native)})
        raw = append_record(
            None,
            continuity=continuity,
            session=session,
            messages=[message],
            request_digest="d" * 64,
        )
        before = json.dumps(
            {"message": message.model_dump(mode="json"), "raw": raw}, sort_keys=True
        )
        restored = await materialize(
            store=_PrivateReadProbe(raw),
            session=session,
            profile=continuity.profile,
            messages=[message],
            names=frozenset({"private_tool"}) if retain else frozenset(),
            redactor=SecretRedactor("private-0") if redact else SecretRedactor(),
        )
        model_call = restored[0].content[0]
        assert model_call.arguments_state == ("finalized" if retain else "unavailable")
        private_arguments = {"text": REDACTED_SECRET if redact else "private-0"}
        assert model_call.arguments == (private_arguments if retain else {})
        payload = build_openai_payload(
            ModelRequest(model="test", messages=restored), stream=True, reasoning_state="inline"
        )
        native_call = next(item for item in payload["input"] if item["type"] == "function_call")
        assert json.loads(native_call["arguments"]) == (
            private_arguments if retain and envelope != "different" else arguments
        )
        assert native_call["id"] == "fc_test"
        assert (
            json.dumps({"message": message.model_dump(mode="json"), "raw": raw}, sort_keys=True)
            == before
        )
        assert message.content[0].arguments == {}
        assert message.content[0].arguments_state == "unavailable"

    asyncio.run(run())
