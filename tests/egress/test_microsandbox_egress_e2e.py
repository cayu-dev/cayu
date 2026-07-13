"""Opt-in adversarial virtual-egress test for a real Microsandbox microVM."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from tests.egress_e2e_support import (
    assert_brokered_provider_call,
    assert_direct_egress_blocked,
    assert_guest_non_possession,
    drive_adversarial_egress_contract,
)

from cayu.egress.microsandbox_adapter import MicrosandboxEgressAdapter

pytest.importorskip("cryptography")
pytest.importorskip("microsandbox")

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_RUN_MICROSANDBOX_EGRESS_E2E") != "1",
    reason="Set CAYU_RUN_MICROSANDBOX_EGRESS_E2E=1 to start a real microVM.",
)

REAL_SECRET = "sk_test_51MicrosandboxRealSecretNeverInGuest"


class _CapturingAdapter(MicrosandboxEgressAdapter):
    broker: Any = None
    grant: Any = None

    async def prepare(self, *, session_id, grants, broker):  # type: ignore[no-untyped-def]
        self.broker = broker
        self.grant = grants[0]
        return await super().prepare(session_id=session_id, grants=grants, broker=broker)


async def _drive() -> dict[str, Any]:
    adapter = _CapturingAdapter(bind_host="0.0.0.0")
    return await drive_adversarial_egress_contract(
        adapter=adapter,
        real_secret=REAL_SECRET,
        image=os.environ.get("CAYU_MICROSANDBOX_IMAGE", "python:3.13"),
        session_prefix="microsandbox-egress",
        search_roots=("/workspace", "/tmp", "/etc/cayu", "/root"),
        response_id="cus_microsandbox",
    )


@pytest.fixture(scope="module")
def e2e() -> dict[str, Any]:
    return asyncio.run(_drive())


def test_microsandbox_guest_possesses_only_virtual_credential(e2e: dict[str, Any]) -> None:
    assert e2e["env"]["HTTPS_PROXY"].startswith("http://host.microsandbox.internal:")
    assert_guest_non_possession(e2e, REAL_SECRET)


def test_microsandbox_allowed_call_uses_broker_secret(e2e: dict[str, Any]) -> None:
    assert_brokered_provider_call(
        e2e,
        real_secret=REAL_SECRET,
        response_id="cus_microsandbox",
    )


def test_microsandbox_direct_and_metadata_egress_are_blocked(e2e: dict[str, Any]) -> None:
    assert_direct_egress_blocked(e2e)
