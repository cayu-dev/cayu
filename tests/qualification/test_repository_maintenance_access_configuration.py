"""Configured credentials cross the real ASGI authorization boundary."""

import asyncio
import importlib
import json
import traceback
import warnings

import pytest

from cayu.cli.project import project_context
from tests.qualification.test_repository_maintenance_application import project as project
from tests.qualification.test_repository_maintenance_http import client
from tests.qualification.test_repository_maintenance_http import host as host
from tests.qualification.test_repository_maintenance_request import consumer as consumer

_ACCESS = {
    "operator_token": "operator-token",
    "product_tokens": {
        "product-a": {"tenant_id": "tenant-a", "subject_id": "alice"},
        "product-b": {"tenant_id": "tenant-b", "subject_id": "bob"},
    },
}


def test_configured_credentials_enforce_real_routes_and_tenant_lookup(host, monkeypatch):
    _server, application, registry, provider, _auth = host
    monkeypatch.setenv("CAYU_MAINTENANCE_ACCESS_JSON", json.dumps(_ACCESS))
    configuration = importlib.import_module("configuration.maintenance")
    access = configuration.configured_maintenance_access()
    assert configuration.configured_maintenance_access() is not access
    api = importlib.import_module("operations.maintenance_http")
    server = api.build_maintenance_server(application, registry, access)

    async def scenario():
        async with client(server) as http:
            created = await http.post(
                "/runs",
                headers={"authorization": "Bearer product-a"},
                json={"instruction": "Fix upper endpoint", "idempotency_key": "config-1"},
            )
            assert created.status_code == 202
            path = "/runs/" + created.json()["id"]
            for token, expected in (
                ("product-a", 200),
                ("product-b", 404),
                ("operator-token", 401),
            ):
                response = await http.get(path, headers={"authorization": f"Bearer {token}"})
                assert response.status_code == expected
            for token, expected in (("product-a", 401), ("operator-token", 200)):
                response = await http.get(
                    "/internal/cayu/api/sessions", headers={"authorization": f"Bearer {token}"}
                )
                assert response.status_code == expected
        assert not provider.requests

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "invalid", ["missing", "duplicate", "oversize", "extra", "identity", "overlap", "empty"]
)
def test_invalid_access_configuration_is_not_echoed(project, monkeypatch, caplog, capsys, invalid):
    canary = "private-access-canary"
    value = {
        "operator_token": canary,
        "product_tokens": {"product": {"tenant_id": "tenant", "subject_id": "subject"}},
    }
    raw = json.dumps(value)
    if invalid == "missing":
        monkeypatch.delenv("CAYU_MAINTENANCE_ACCESS_JSON", raising=False)
    else:
        if invalid == "duplicate":
            raw = '{"operator_token":"' + canary + '","operator_token":"again"}'
        elif invalid == "oversize":
            raw = canary * 6000
        elif invalid == "extra":
            value["fallback"] = canary
            raw = json.dumps(value)
        elif invalid == "identity":
            raw = json.dumps(
                {
                    "operator_token": canary,
                    "product_tokens": {"product": {"tenant_id": True, "subject_id": "subject"}},
                }
            )
        elif invalid == "overlap":
            value["product_tokens"][canary] = value["product_tokens"].pop("product")
            raw = json.dumps(value)
        else:
            value["product_tokens"] = {}
            raw = json.dumps(value)
        monkeypatch.setenv("CAYU_MAINTENANCE_ACCESS_JSON", raw)
    with project_context(project), warnings.catch_warnings(record=True) as recorded:
        configuration = importlib.import_module("configuration.maintenance")
        with pytest.raises(ValueError, match="maintenance access configuration") as caught:
            configuration.configured_maintenance_access()
    assert canary not in "".join(traceback.format_exception(caught.value))
    output = capsys.readouterr()
    assert not recorded
    assert canary not in caplog.text + output.out + output.err
