"""Operator fixture construction owns public trust without mounting private siblings."""

import asyncio
import ssl
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from pydantic import SecretStr
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy

from cayu import SQLiteSessionStore
from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
from cayu.evals.internal.browser_acceptance import (
    BrowserAcceptanceDeterministicProvider,
    _DeterministicScenarioExecutor,
    build,
)
from cayu.evals.internal.browser_acceptance_operator import OperatorFixtureBinding
from cayu.runtime.browser_control_config import BrowserControlConfig


def certificate_and_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "fixture.test")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .sign(key, hashes.SHA256())
    )
    return (
        certificate.public_bytes(serialization.Encoding.PEM),
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )


@pytest.mark.parametrize("combined", [False, True])
def test_builder_copies_only_public_trust_and_isolates_operator_authority(tmp_path, combined):
    certificate, private_key = certificate_and_key()
    source = tmp_path / "server.crt"
    source.write_bytes(certificate + (private_key if combined else b""))
    (tmp_path / "server.key").write_bytes(private_key)

    async def scenario(fixture):
        async with httpx.AsyncClient(base_url="https://127.0.0.1:8443") as client:
            binding = OperatorFixtureBinding(
                control=BrowserControlConfig(
                    policy=Policy(True),
                    purpose=operator_purpose(),
                    guest_endpoint="wss://cayu-control:8443/api/browser-control/guest",
                ),
                server_container_id="a" * 64,
                ca_certificate=source,
                client=client,
                tls=ssl.create_default_context(),
                operator_origin="https://operator.test",
                private_text=SecretStr("private-fixture-value"),
            )
            plan = await build(fixture, operator_fixture=binding)
            app = plan.eval_plan.app
            assert app is not None
            applications = {id(app): app, **{id(value): value for _, value in plan.case_apps}}
            try:
                provider = app.get_provider("browser-acceptance-scripted")
                assert isinstance(provider, BrowserAcceptanceDeterministicProvider)
                owned = provider.operator_fixture
                assert owned is not None and owned is not binding
                assert owned.ca_certificate.parent != source.parent
                assert list(owned.ca_certificate.parent.iterdir()) == [owned.ca_certificate]
                assert owned.ca_certificate.read_bytes() == certificate
                revision = owned.authority_revision
                source.write_bytes(b"not a certificate")
                assert owned.authority_revision == revision
                for _, ordinary in plan.case_apps:
                    assert ordinary is not app
                    assert ordinary._browser_control_runtime is None
                    assert not ordinary._secret_redactor.has_values
                assert isinstance(plan.scenario_executor, _DeterministicScenarioExecutor)
                assert plan.scenario_executor._control_server_container_id == "a" * 64
            finally:
                for application in applications.values():
                    store = application.session_store
                    assert isinstance(store, SQLiteSessionStore)
                    await store.close()

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))
