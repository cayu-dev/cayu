from __future__ import annotations

import asyncio

import pytest

from cayu.egress.errors import UnsupportedEgressError
from cayu.egress.proxy_exposure import ExposedProxy, HttpProxyEndpoint, PodNetworkProxyExposure


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


@pytest.mark.parametrize(
    "pod_ip",
    ["10.8.0.12", "172.20.1.5", "192.168.3.4", "100.64.0.9", "100.127.255.254", "198.18.0.7"],
)
def test_pod_network_exposure_advertises_a_non_public_pod_ip(pod_ip: str) -> None:
    exposure = PodNetworkProxyExposure(pod_ip)

    exposed = asyncio.run(exposure.expose(local_host="0.0.0.0", local_port=40123))
    bound_to_pod = asyncio.run(exposure.expose(local_host=pod_ip, local_port=40123))

    assert exposed.proxy_url == bound_to_pod.proxy_url == f"http://{pod_ip}:40123"
    # The transport token, not the exposure, isolates the session.
    assert exposed.credentialless_isolated is False


@pytest.mark.parametrize(
    "pod_ip",
    [
        "8.8.8.8",
        "100.128.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "224.0.0.1",
        "0.0.0.0",
        "0.1.2.3",
        "240.0.0.1",
        "255.255.255.255",
        "fd00::1",
        "::1",
        "pod.example",
        "",
    ],
)
def test_pod_network_exposure_refuses_public_host_local_and_non_ipv4_addresses(
    pod_ip: str,
) -> None:
    with pytest.raises(ValueError, match="non-public IPv4 unicast"):
        PodNetworkProxyExposure(pod_ip)


def test_pod_network_exposure_refuses_a_listener_on_another_address() -> None:
    exposure = PodNetworkProxyExposure("100.64.0.9")

    with pytest.raises(UnsupportedEgressError, match="0.0.0.0 or the pod IP"):
        asyncio.run(exposure.expose(local_host="127.0.0.1", local_port=40123))
    with pytest.raises(ValueError, match="local_port"):
        asyncio.run(exposure.expose(local_host="0.0.0.0", local_port=0))


def test_pod_network_exposure_is_public() -> None:
    from cayu.egress import PodNetworkProxyExposure as exported

    assert exported is PodNetworkProxyExposure
