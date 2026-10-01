"""Generated projects opt in explicitly, including when stores are injected."""

import asyncio
import importlib
import json
from pathlib import Path

import pytest

from cayu import InMemorySessionStore, ScriptedModelProvider
from cayu.cli import main
from cayu.cli.project import project_context
from cayu.model_policy import configured_model_policy, model_policy_enabled


@pytest.mark.parametrize("injected", [False, True])
def test_generated_project_has_explicit_policy_storage(tmp_path, monkeypatch, capsys, injected):
    for name in ("CAYU_DATABASE_URL", "CAYU_REQUIRE_POSTGRES", "CAYU_MODEL_POLICY_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    assert main(["new", "agent", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    vectors = json.loads(
        (Path(__file__).parents[1] / "fixtures/model_policy/contract.json").read_text()
    )
    config = tmp_path / "policy.json"
    config.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "agent_name": "agent",
                        "provider_name": "scripted",
                        "origin": "https://policy.example",
                        "scope": vectors["effective"]["scope"],
                        "incarnation_id": "incarnation-1",
                        "incarnation_epoch": 1,
                        "credential_env": "TEST_POLICY_CREDENTIAL",
                    }
                ]
            }
        )
    )
    monkeypatch.setenv("CAYU_MODEL_POLICY_CONFIG", str(config))
    monkeypatch.setenv("TEST_POLICY_CREDENTIAL", "controlled-policy-credential")
    with project_context(tmp_path / "agent"):
        storage = importlib.import_module("configuration.storage")
        stores = storage.build_stores(
            session_store=InMemorySessionStore() if injected else None,
            knowledge_scope=importlib.import_module("knowledge.retrieval").build_knowledge_scope(),
        )
        assert stores.configured is not None
        try:
            assert stores.configured.model_policy_store is not None
            policy = configured_model_policy(stores.configured.model_policy_store)
            assert policy is not None and len(policy.controllers) == 1
            # Construction must not connect to management or require inference auth.
            monkeypatch.setattr(storage, "build_stores", lambda **kwargs: stores)
            app = importlib.import_module("app").build_app(provider=ScriptedModelProvider([]))
            assert app.model_policy is not None
        finally:
            asyncio.run(stores.configured.close())


def test_inference_credentials_do_not_enable_policy(monkeypatch):
    monkeypatch.delenv("CAYU_MODEL_POLICY_CONFIG", raising=False)
    monkeypatch.setenv("CAYU_GATEWAY_API_KEY", "controlled-inference-credential")
    assert not model_policy_enabled()
    assert configured_model_policy(None) is None


def test_generated_gateway_entrypoint_adopts_reports_and_runs(tmp_path, monkeypatch, capsys):
    from tests.core.test_gateway_provider import chunk, provider_for, response, wire
    from tests.core.test_model_policy_runtime import Channel

    from cayu.entrypoint import run_project_entrypoint
    from cayu.model_policy import HttpPolicyChannel

    for name in ("CAYU_DATABASE_URL", "CAYU_REQUIRE_POSTGRES", "CAYU_MODEL_POLICY_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    assert main(["new", "agent", "--provider", "cayu-gateway", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    channel = Channel()
    config = tmp_path / "policy.json"
    config.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "agent_name": "agent",
                        "provider_name": "cayu_gateway",
                        "origin": "https://policy.example",
                        "scope": channel.scope,
                        "incarnation_id": "incarnation-1",
                        "incarnation_epoch": 1,
                        "credential_env": "TEST_POLICY_CREDENTIAL",
                    }
                ]
            }
        )
    )
    monkeypatch.setenv("CAYU_MODEL_POLICY_CONFIG", str(config))
    monkeypatch.setenv("TEST_POLICY_CREDENTIAL", "controlled-policy-credential")

    async def read(self):
        return await channel.read_snapshot()

    async def report(self, body):
        return await channel.report(body)

    monkeypatch.setattr(HttpPolicyChannel, "read_snapshot", read)
    monkeypatch.setattr(HttpPolicyChannel, "report", report)
    calls = []

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            assert request.url.path == "/v1/models"
            return response(b'{"data":[{"id":"model-a"}]}')
        assert json.loads(request.content)["model"] == "model-a"
        return response(
            wire(chunk({"content": "adopted"}), chunk(finish="stop")),
            content_type="text/event-stream",
        )

    provider = provider_for(handle)
    with project_context(tmp_path / "agent"):
        storage = importlib.import_module("configuration.storage")
        stores = storage.build_stores(
            knowledge_scope=importlib.import_module("knowledge.retrieval").build_knowledge_scope()
        )
        monkeypatch.setattr(storage, "build_stores", lambda **kwargs: stores)
        app = importlib.import_module("app").build_app(provider=provider)
        try:
            assert run_project_entrypoint(lambda: app, ["--message", "hello"]) == 0
            assert "adopted" in capsys.readouterr().out
            assert len(channel.receipts) == 1
            assert [request.method for request in calls] == ["GET", "POST"]
            assert app.model_policy is not None and not app.model_policy._tasks
        finally:
            asyncio.run(provider.aclose())
            asyncio.run(stores.configured.close())
