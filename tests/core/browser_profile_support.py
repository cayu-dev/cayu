"""Reusable browser-profile binding and checkpoint runner test support."""

from __future__ import annotations

import json
from typing import Any

from tests.core.browser_session_support import _WireRunner

from cayu.browser_profiles import (
    AESGCMBrowserProfileKeyAuthority,
    BrowserProfileBinding,
    BrowserProfileCheckpointPolicy,
    BrowserProfileDestinationPolicy,
    BrowserProfileScope,
    BrowserProfileStore,
)
from cayu.runners.base import ExecCommand, ExecResult
from cayu.runners.workloads import PINNED_BROWSER_SESSION_WORKLOAD


class _ProfileWireRunner(_WireRunner):
    def __init__(self, *, revision: str = "br_profile_revision") -> None:
        self.operations: list[str] = []
        self.restored_state: dict[str, Any] | None = None
        self.checkpoint_state: dict[str, Any] = {"cookies": [], "origins": []}
        self.reject_restore = False
        self.revision = revision
        self.current_url = "https://example.test/"
        self.observation_snapshot = "- document"
        self.profile_output_protected = True
        self.session_id: str | None = None
        self.page_id: str | None = None
        self.control_epoch = 0
        self.operation_count = 0
        self.observation_count = 0

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        assert command.argv == list(PINNED_BROWSER_SESSION_WORKLOAD.command)
        request = json.loads(kwargs["stdin"])
        operation = request["operation"]
        self.operations.append(operation)
        if operation == "profile_restore":
            private_profile = request["browser_profile"]
            self.restored_state = private_profile["restore_state"]
            if self.reject_restore:
                return ExecResult(
                    stdout=json.dumps(
                        {
                            "protocol_version": "cayu.browser-session.v4",
                            "worker_version": "19",
                            "playwright_version": "1.62.0",
                            "kind": "error",
                            "allocation_disposition": "retired",
                            "error": "incompatible_browser",
                        }
                    )
                )
            return ExecResult(
                stdout=json.dumps(
                    {
                        "protocol_version": "cayu.browser-session.v4",
                        "worker_version": "19",
                        "playwright_version": "1.62.0",
                        "kind": "profile_restore",
                        "allocation_disposition": "live",
                        "profile_restored": True,
                    }
                )
            )
        if operation == "profile_checkpoint":
            return ExecResult(
                stdout=json.dumps(
                    {
                        "protocol_version": "cayu.browser-session.v4",
                        "worker_version": "19",
                        "playwright_version": "1.62.0",
                        "kind": "profile_checkpoint",
                        "allocation_disposition": "live",
                        "profile_state": self.checkpoint_state,
                    }
                )
            )
        if operation == "close":
            return ExecResult(
                stdout=json.dumps(
                    {
                        "protocol_version": "cayu.browser-session.v4",
                        "worker_version": "19",
                        "playwright_version": "1.62.0",
                        "kind": "closed",
                        "allocation_disposition": "retired",
                    }
                )
            )
        if operation == "navigate":
            self.current_url = request["url"]
            self.session_id = request["session_id"]
            self.page_id = request["page_id"]
            self.control_epoch = 1
            self.operation_count = 0
            self.observation_count = 0
        elif "url" not in request:
            request["url"] = self.current_url
            kwargs = {**kwargs, "stdin": json.dumps(request)}
        self.operation_count += 1
        if operation not in {"navigate", "observe"}:
            self.control_epoch += 1
        self.observation_count += 1
        result = await super().exec(command, **kwargs)
        payload = json.loads(result.stdout)
        revision = (
            self.revision
            if self.observation_count == 1
            else f"{self.revision}_{self.observation_count}"
        )
        payload["observation"]["revision"] = revision
        payload["observation"]["control_epoch"] = self.control_epoch
        payload["observation"]["snapshot"] = self.observation_snapshot
        page = payload["page_set"]["pages"][0]
        page["url"] = None
        page["title"] = None
        page["revision"] = revision
        page["last_observation_revision"] = revision
        page["control_epoch"] = self.control_epoch
        page["operation_count"] = self.operation_count
        page["observation_count"] = self.observation_count
        payload["page_set"]["total_operations"] = self.operation_count
        payload["page_set"]["total_observations"] = self.observation_count
        if operation != "navigate":
            payload["page_delta"] = {
                "created_page_ids": [],
                "admitted_page_ids": [],
                "closed_page_ids": [],
                "crashed_page_ids": [],
                "refused": [],
            }
        payload["profile_output_protected"] = self.profile_output_protected
        return result.model_copy(update={"stdout": json.dumps(payload)})

    async def _exec_private_browser_profile(
        self, command: ExecCommand, **kwargs: Any
    ) -> ExecResult:
        return await self.exec(command, **kwargs)


def _browser_profile_binding(
    store: BrowserProfileStore,
    *,
    checkpoint_policy: BrowserProfileCheckpointPolicy = BrowserProfileCheckpointPolicy.ON_CLOSE,
) -> BrowserProfileBinding:
    return BrowserProfileBinding.build(
        scope=BrowserProfileScope.build(
            application_id="browser-session-tests",
            tenant_id="tenant-one",
            sharing_scope="agent-release-one",
        ),
        destination_policy=BrowserProfileDestinationPolicy.build(("https://example.test",)),
        browser_protocol="cayu.browser-session.v4",
        browser_worker_version="19",
        store=store,
        key_authority=AESGCMBrowserProfileKeyAuthority(
            authority_id="browser-profile-test-key",
            key=b"p" * 32,
        ),
        profile_id="bprof_browser_session_test",
        checkpoint_policy=checkpoint_policy,
        lease_seconds=120,
    )
