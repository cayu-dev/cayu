"""Accepted resolution projections retain authority without persisting secrets."""

import asyncio
import json

import pytest
from tests.core import test_foreground_child_parent_gate as gate_tests

from cayu import CayuApp
from cayu.vaults import SecretRedactor


@pytest.mark.parametrize("kind", ["approval", "input"])
def test_gate_continuation_uses_redacted_accepted_projection(tmp_path, monkeypatch, kind):
    secret = "gate-resolution-secret-canary"
    applications = []
    redactor = SecretRedactor(secret)

    def build(**kwargs):
        app = CayuApp(secret_redactor=redactor, **kwargs)
        applications.append(app)
        return app

    original_approval = CayuApp.resolve_tool_approval
    original_input = CayuApp.resolve_user_input

    async def approve(self, request):
        if request.session_id == "parent":
            request = request.model_copy(update={"metadata": {"accepted": secret}})
        async for event in original_approval(self, request):
            yield event

    async def answer(self, response):
        if response.session_id == "parent":
            response = response.model_copy(
                update={"metadata": {"accepted": secret}, "answer": f"yes {secret}"}
            )
        async for event in original_input(self, response):
            yield event

    monkeypatch.setattr(gate_tests, "CayuApp", build)
    monkeypatch.setattr(CayuApp, "resolve_tool_approval", approve)
    monkeypatch.setattr(CayuApp, "resolve_user_input", answer)
    gate_tests._run_case(tmp_path, "memory", kind, "approval")

    async def inspect():
        store = applications[0].session_store
        checkpoint = await store.load_checkpoint("parent")
        continuation = checkpoint["foreground_parent_continuation"]
        action_id = continuation["publication_id"].split(":", 1)[1]
        receipt = await store.load_runtime_publication_receipt(
            "parent", continuation["publication_id"]
        )
        digest = receipt.intent["resolution_request_digest"]
        accepted = await store.load_session_operation(
            "parent", f"foreground-gate:{kind}:{action_id}:{digest}"
        )
        assert accepted["request"]["metadata"] == continuation["request"]["metadata"]
        assert accepted["request"]["metadata"]["accepted"] != secret
        assert secret not in json.dumps(accepted)
        assert secret not in json.dumps(checkpoint)
        assert secret not in json.dumps(
            [e.model_dump(mode="json") for e in await store.load_events("parent")]
        )

    asyncio.run(inspect())
