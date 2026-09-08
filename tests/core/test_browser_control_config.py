"""Application-owned browser control composition is opt-in and side-effect free."""

import pytest
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy

from cayu.runtime.app import CayuApp
from cayu.runtime.browser_control_config import BrowserControlConfig
from cayu.runtime.sessions import InMemorySessionStore
from cayu.tools._browser_control_transport import BrowserControlTransportUnavailable


def test_application_browser_control_owns_one_coordinator_and_service():
    store = InMemorySessionStore()
    config = BrowserControlConfig(
        purpose=operator_purpose(),
        policy=Policy(True),
        guest_endpoint="wss://control.test/browser-control/guest",
    )
    app = CayuApp(session_store=store, browser_control=config, enable_logging=False)
    runtime = app._browser_control_runtime
    assert runtime is not None
    assert runtime.coordinator._policy is config.policy
    assert app._tool_round_executor._browser_control_service is runtime.service
    assert not runtime.service._owners
    assert not runtime.service._viewers
    disabled = CayuApp(enable_logging=False)
    assert disabled._browser_control_runtime is None
    assert disabled._tool_round_executor._browser_control_service is None
    assert "control.test" not in repr(config)


@pytest.mark.parametrize(
    "endpoint",
    [
        "ws://control.test/guest",
        "wss://user:canary@control.test/guest",
        "wss://control.test/guest?token=canary",
    ],
)
def test_browser_control_config_rejects_insecure_or_credential_urls(endpoint):
    with pytest.raises(BrowserControlTransportUnavailable) as error:
        BrowserControlConfig(
            purpose=operator_purpose(), policy=Policy(True), guest_endpoint=endpoint
        )
    assert "canary" not in str(error.value)


def test_application_revalidates_mutated_browser_control_config():
    config = BrowserControlConfig(
        purpose=operator_purpose(), policy=Policy(True), guest_endpoint="wss://control.test/guest"
    )
    object.__setattr__(config, "guest_endpoint", "ws://control.test/guest")
    with pytest.raises(BrowserControlTransportUnavailable):
        CayuApp(browser_control=config, enable_logging=False)
