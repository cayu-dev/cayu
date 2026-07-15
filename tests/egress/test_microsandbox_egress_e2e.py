"""Opt-in adversarial virtual-egress test for a real Microsandbox microVM."""

from __future__ import annotations

import asyncio
import os

import pytest
from tests.egress_conformance import (
    EgressScenarioEvidence,
    egress_nightly_failure_boundary,
    emit_egress_nightly_evidence,
    registration_for,
)
from tests.egress_e2e_support import CapturingEgressAdapter, drive_adversarial_egress_contract

from cayu.egress.microsandbox_adapter import MicrosandboxEgressAdapter
from cayu.workspaces import MicrosandboxWorkspace

pytest.importorskip("cryptography")
pytest.importorskip("microsandbox")

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_RUN_MICROSANDBOX_EGRESS_E2E") != "1",
    reason="Set CAYU_RUN_MICROSANDBOX_EGRESS_E2E=1 to start a real microVM.",
)

REAL_SECRET = "sk_test_51MicrosandboxRealSecretNeverInGuest"


async def _drive() -> tuple[EgressScenarioEvidence, ...]:
    adapter = CapturingEgressAdapter(MicrosandboxEgressAdapter())
    return await drive_adversarial_egress_contract(
        registration=registration_for("microsandbox"),
        adapter=adapter,
        real_secret=REAL_SECRET,
        image=os.environ.get("CAYU_MICROSANDBOX_IMAGE", "python:3.13"),
        search_roots=("/workspace", "/tmp", "/etc/cayu", "/root"),
        response_id="cus_microsandbox",
        workspace_factory=MicrosandboxWorkspace,
    )


@pytest.fixture(scope="module")
def e2e() -> tuple[EgressScenarioEvidence, ...]:
    with egress_nightly_failure_boundary("microsandbox"):
        return asyncio.run(_drive())


def test_microsandbox_shared_real_boundary_security_contract(
    e2e: tuple[EgressScenarioEvidence, ...],
) -> None:
    with egress_nightly_failure_boundary("microsandbox"):
        assert all(item.adapter == "microsandbox" for item in e2e)
        assert all(item.status == "verified" for item in e2e)
        assert REAL_SECRET not in repr(e2e)
        emit_egress_nightly_evidence(e2e)
