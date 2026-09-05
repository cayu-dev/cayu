"""Opt-in authenticated HTTPS observation; never creates a remote ref."""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path

import pytest
from tests.core.test_remote_git_delivery_recovery import _case

from cayu.remote_git_delivery import (
    RemoteGitDeliveryBroker,
    RemoteGitDeliveryConflictError,
    RemoteGitHttpCredentials,
    RemoteGitRemoteConfig,
)
from cayu.vaults import SecretRef, StaticVault


def test_authenticated_https_preparation_observes_base_without_writing(tmp_path: Path) -> None:
    if os.environ.get("CAYU_RUN_REMOTE_GIT_AUTH_TEST") != "1":
        pytest.skip("Requires explicit opt-in and a dedicated private HTTPS test repository.")
    names = (
        "CAYU_REMOTE_GIT_TEST_URL",
        "CAYU_REMOTE_GIT_TEST_BASE_COMMIT",
        "CAYU_REMOTE_GIT_TEST_USERNAME",
        "CAYU_REMOTE_GIT_TEST_PASSWORD",
    )
    if any(not os.environ.get(name) for name in names):
        pytest.fail("The authenticated Remote Git test configuration is incomplete.")
    _remote, workspace, product, original, request = _case(tmp_path, baseline_commit="0" * 40)
    credentials = RemoteGitHttpCredentials(
        credential_profile_id="dedicated-live-test",
        username=SecretRef(name="username"),
        password=SecretRef(name="password"),
        resolver=StaticVault(
            {
                "username": os.environ[names[2]],
                "password": os.environ[names[3]],
            }
        ),
    )
    configured = RemoteGitRemoteConfig(
        alias="origin",
        remote_identity=request.repository.remote_identity,
        url=os.environ[names[0]],
        default_branch_ref=request.repository.base_ref,
        credential_profile_id=credentials.credential_profile_id,
        egress_profile_id="dedicated-live-https",
        credentials=credentials,
    )
    broker = RemoteGitDeliveryBroker(
        replace(original.profile, remotes={"origin": configured}),
        repository=original.repository,
        coding_repository=original.coding_repository,
    )
    # The impossible expected base forces rejection immediately after the
    # authenticated read, before fetch, local commit, or any remote mutation.
    request = request.model_copy(
        update={
            "repository": request.repository.model_copy(update={"expected_base_commit": "0" * 40}),
            "security": request.security.model_copy(
                update={
                    "credential_profile_id": credentials.credential_profile_id,
                    "egress_profile_id": configured.egress_profile_id,
                }
            ),
        }
    )
    with pytest.raises(RemoteGitDeliveryConflictError) as caught:
        asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    assert caught.value.reason_code == "remote_base_changed_before_preparation"
    assert caught.value.observed_base == os.environ[names[1]]
    assert asyncio.run(broker.repository.load_prepared(request)) is None
    assert not (broker._delivery_root(request) / ".git").exists()
