"""Application intervention scope stays bound across durable reconstruction."""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_publisher import publication_fixture

from cayu import BrowserControlConfig, BrowserOperatorPurpose, CayuApp
from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime._browser_control_channel import browser_allocation_digest
from cayu.runtime._browser_control_checkpoint import browser_control_checkpoint_read_scope
from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
from cayu.runtime._browser_control_model import browser_model_control_admission
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlConflict,
    BrowserControlPrincipal,
)
from cayu.vaults import SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "change", [{"code": "support"}, {"expected_origins": ("https://different.test",)}]
)
def test_reconstructed_purpose_drift_cannot_authorize_or_admit_model(tmp_path, backend, change):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            with browser_control_checkpoint_read_scope(record.identity.session_id):
                before = await store.load_checkpoint(record.identity.session_id)
            original = operator_purpose()
            changed = original.model_copy(update=change)
            policy = Policy(True)
            control = BrowserControlCoordinator(
                store=store,
                policy=policy,
                purpose=changed,
                redactor=SecretRedactor(),
                clock=lambda: datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC),
            )
            try:
                with pytest.raises(BrowserControlPermissionDenied):
                    await control.authorize_view(
                        principal=BrowserControlPrincipal(subject="operator"),
                        operator_session_id="continuity",
                        identity=record.identity,
                        expected_record_revision=record.revision,
                    )
                assert not policy.requests
                allocation = BrowserControlAllocation.model_validate(
                    record.identity.model_dump(exclude={"worker_instance_id"})
                )
                changed_allocation = allocation.model_copy(update={"operator_purpose": changed})
                assert browser_allocation_digest(changed_allocation) != browser_allocation_digest(
                    allocation
                )
                with pytest.raises(BrowserControlConflict):
                    browser_model_control_admission(
                        before, allocation=changed_allocation, operation_name="observe"
                    )
                admission = browser_model_control_admission(
                    before, allocation=allocation, operation_name="observe"
                )
                assert admission is not None
                assert admission.control_epoch == record.control_epoch
                with browser_control_checkpoint_read_scope(record.identity.session_id):
                    assert await store.load_checkpoint(record.identity.session_id) == before
            finally:
                assert await control.drain()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "origins",
    [
        (),
        ("https://site.test/private-canary",),
        ("https://private-canary@site.test",),
        ("http://site.test",),
        ("https://site.test", "https://site.test"),
    ],
)
def test_purpose_requires_nonempty_canonical_origin_set(origins, recwarn, caplog, capsys):
    with pytest.raises(ValidationError) as failure:
        BrowserOperatorPurpose(code="login", expected_origins=origins)
    captured = capsys.readouterr()
    assert "private-canary" not in str(failure.value) + repr(failure.value)
    assert "private-canary" not in captured.out + captured.err + caplog.text
    assert not recwarn


def test_config_owns_purpose_and_rejects_workload_secret_before_runtime_creation():
    original = operator_purpose()
    config = BrowserControlConfig(
        policy=Policy(True), guest_endpoint="wss://control.test/guest", purpose=original
    )
    object.__setattr__(original, "code", "changed")
    assert config.purpose.code == "login"
    with pytest.raises(BrowserControlPermissionDenied):
        CayuApp(
            enable_logging=False, browser_control=config, secret_redactor=SecretRedactor("login")
        )
