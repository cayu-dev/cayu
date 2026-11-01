from __future__ import annotations

import pytest

from cayu.egress.proxy_exposure import ExposedProxy, HttpProxyEndpoint


def test_http_proxy_endpoint_centralizes_url_validation() -> None:
    endpoint = HttpProxyEndpoint.parse("http://cayu-egress.example:8443")
    assert endpoint.url == "http://cayu-egress.example:8443"
    assert endpoint.host == "cayu-egress.example"
    assert endpoint.port == 8443

    for invalid in (
        "https://cayu-egress.example:8443",
        "http://user:password@cayu-egress.example:8443",
        "http://cayu-egress.example/path",
        "http://cayu-egress.example:invalid",
    ):
        with pytest.raises(ValueError, match="HTTP proxy URL"):
            HttpProxyEndpoint.parse(invalid)


def test_exposed_proxy_requires_a_real_boolean_isolation_assertion() -> None:
    with pytest.raises(TypeError, match="credentialless_isolated"):
        ExposedProxy(
            proxy_url="http://203.0.113.10:8443",
            credentialless_isolated=1,  # type: ignore[arg-type]
        )
