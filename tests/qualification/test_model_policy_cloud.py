"""Optional paired-tree qualification against Cloud's actual authenticated HTTP owner.

Run with the reviewed Cloud source on PYTHONPATH. Only the desired-model source
is controlled; grants, enrollment, snapshot publication, reports, HTTP parsing,
Runtime installation, and session execution use production implementations.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest

from cayu import AgentSpec, CayuApp, Message, ModelStreamEvent, RunRequest
from cayu.model_policy import HttpPolicyChannel, ModelPolicy, ModelPolicyController
from cayu.sessions.outcomes import run_to_completion
from tests.core.test_model_policy_runtime import VECTORS, CatalogProvider
from tests.core.test_model_policy_runtime import store_factory as store_factory

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_cloud_http_to_runtime_session_and_refusal(store_factory, monkeypatch):
    pytest.importorskip("cayu_cloud.gateway.model_policy_refusals")
    from cayu_cloud.gateway.model_policy_api import register_model_policy_routes
    from cayu_cloud.gateway.model_policy_grants import PolicyGrant, prepare_policy_credential
    from cayu_cloud.gateway.model_policy_lifecycle import ModelPolicyLifecycle
    from cayu_cloud.gateway.model_policy_memory import MemoryModelPolicyStore
    from cayu_cloud.gateway.model_policy_service import ModelPolicyService
    from cayu_cloud.gateway.model_policy_snapshots import issue_model_policy_snapshot
    from cayu_cloud.gateway.model_policy_status import read_application_report_status
    from fastapi import FastAPI

    from cayu.runtime._policy_wire import canonical

    scope = {**VECTORS["effective"]["scope"], "instance_id": uuid4().hex}
    cloud = MemoryModelPolicyStore(now=lambda: datetime.now(UTC))
    issued = []
    for kind, permissions, incarnation, epoch in (
        ("controller", ("policy:enroll",), None, None),
        ("instance", ("policy:read", "policy:report", "policy:report-read"), "incarnation-1", 1),
    ):
        credential = prepare_policy_credential(
            PolicyGrant(
                credential_id=uuid4().hex,
                revision=1,
                scope=scope,
                kind=kind,
                permissions=permissions,
                incarnation_id=incarnation,
                incarnation_epoch=epoch,
                expires_at=datetime.now(UTC) + timedelta(hours=1),
                revoked=False,
            )
        )
        await cloud.publish_grant(
            operation_id=uuid4().hex,
            actor_id="test-admin",
            expected=None,
            replacement=credential.stored,
        )
        issued.append(credential.secret.get_secret_value())
    async with cloud.transaction(issued[0], scope) as records:
        await ModelPolicyLifecycle(records).enroll(
            canonical(
                {
                    "schema_version": 1,
                    "kind": "enrollment_request",
                    "scope": scope,
                    "operation_id": "enroll-1",
                    "expected_epoch": 0,
                    "process_nonce": "process-1",
                }
            ),
            incarnation_id="incarnation-1",
            receipt_id="enrolled-1",
            accepted_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        )

    class Source:
        revision = 1
        model = "model-a"

        async def read(self, token, *, resource, snapshot_id, issued_at):
            async with cloud.credential_transaction(token) as (grant, records):
                assert resource == scope["instance_id"] + "/snapshot"
                projection = {
                    kind: {
                        "scope_id": scope[field],
                        "model_revision": self.revision,
                        "allowed_models": ["model-a", "model-b", "unknown-model"],
                    }
                    for kind, field in (
                        ("organization", "organization_id"),
                        ("agent", "cloud_agent_id"),
                        ("key", "inference_key_id"),
                    )
                }
                projection["default_model"] = self.model
                return await issue_model_policy_snapshot(
                    records,
                    scope=grant.scope,
                    incarnation_id=grant.incarnation_id,
                    incarnation_epoch=grant.incarnation_epoch,
                    projection=projection,
                    snapshot_id=snapshot_id,
                    issued_at=issued_at,
                )

    source = Source()
    server = FastAPI()
    register_model_policy_routes(server, service=ModelPolicyService(cloud, snapshots=source))
    monkeypatch.setenv("TEST_PAIRED_POLICY_KEY", issued[1])
    channel = HttpPolicyChannel(
        origin="https://cloud.test",
        scope=scope,
        incarnation=("incarnation-1", 1),
        credential_env="TEST_PAIRED_POLICY_KEY",
    )
    channel._client._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server))
    store = store_factory()
    provider = CatalogProvider(
        [
            ModelStreamEvent.text_delta("paired-ok"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    controller = ModelPolicyController(
        store=store,
        agent_name="assistant",
        provider_name=provider.name,
        scope=scope,
        incarnation=channel.incarnation,
        channel=channel,
    )
    app = CayuApp(model_policy=ModelPolicy([controller], poll_interval=0.05), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="spec-model"))
    await app.start_model_policy()
    try:
        assert controller.status == "ready"
        result = await run_to_completion(
            app, RunRequest(agent_name="assistant", messages=[Message.text("user", "hello")])
        )
        assert result.ok, result.error
        assert (await app.session_store.load(result.session_id)).model == "model-a"
        source.model, source.revision = "model-b", 2
        async with asyncio.timeout(5):
            while True:
                async with controller._lock:
                    if (
                        controller.selection().target.model == "model-b"
                        and not controller._owner.pending()
                    ):
                        break
                await asyncio.sleep(0.01)
        assert (await app.session_store.load(result.session_id)).model == "model-a"
        source.model, source.revision = "unknown-model", 3
        await controller.poll_once()
        assert controller.selection().target.model == "model-b"
        async with cloud.credential_transaction(issued[1]) as (_grant, records):
            status = await read_application_report_status(
                records,
                scope=scope,
                incarnation_id="incarnation-1",
                incarnation_epoch=1,
                observed_at=datetime.now(UTC),
            )
        assert status["decision_status"] == "refused"
    finally:
        await app.stop_model_policy()
        await channel.aclose()
        await store.close()
