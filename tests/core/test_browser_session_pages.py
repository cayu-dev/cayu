"""Browser-session page authority, popup receipts and page-set recovery contracts."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest
from tests.core.browser_session_support import (
    _IDENTITY,
    _context,
    _durable_context,
    _FakeBrowserBackend,
    _tool,
)

from cayu.tools.base import ToolContext
from cayu.tools.browser_session import (
    BrowserBackendFailure,
    BrowserBackendObservation,
    BrowserBackendResponse,
    BrowserElementRef,
    BrowserPageRefusal,
    BrowserPageSetDelta,
    BrowserPageSetState,
    BrowserPageSummary,
    BrowserPopupPolicy,
    BrowserSessionBackend,
    BrowserSessionTool,
)


class _MultipageBrowserBackend(BrowserSessionBackend):
    """Small stateful backend used to exercise the public page-authority boundary."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.session_id: str | None = None
        self.root_page_id: str | None = None
        self.popup_page_id = "bp_popup_page"
        self.active_page_id: str | None = None
        self.total_page_creations = 0
        self.total_operations = 0
        self.total_observations = 0
        self.cleanup_operations = 0
        self.pages: dict[str, dict[str, Any]] = {}

    def _revision(self, page_id: str) -> str:
        page = self.pages[page_id]
        return f"br_{page_id}_{page['control_epoch']}_{page['observation_count']}"

    def _summary(self, page_id: str) -> BrowserPageSummary:
        page = self.pages[page_id]
        return BrowserPageSummary(
            page_id=page_id,
            lifecycle=page["lifecycle"],
            creation_epoch=page["creation_epoch"],
            control_epoch=page["control_epoch"],
            opener_page_id=page["opener_page_id"],
            creating_operation_id_sha256=page["creating_operation_id_sha256"],
            revision=page["revision"],
            url=page["url"],
            title=page["title"],
            load_state="loaded",
            access_state="available",
            last_observation_revision=page.get("last_observation_revision"),
            last_operation_id_sha256=page["last_operation_id_sha256"],
            terminal_reason=page["terminal_reason"],
            operation_count=page["operation_count"],
            observation_count=page["observation_count"],
            ref_count=page["ref_count"],
            request_count=0,
            artifact_count=0,
        )

    def _page_set(self) -> BrowserPageSetState:
        assert self.session_id is not None
        return BrowserPageSetState(
            session_id=self.session_id,
            active_page_id=self.active_page_id,
            pages=tuple(
                self._summary(page_id)
                for page_id in sorted(
                    self.pages,
                    key=lambda candidate: self.pages[candidate]["creation_epoch"],
                )
            ),
            total_page_creations=self.total_page_creations,
            total_operations=self.total_operations,
            total_observations=self.total_observations,
            total_refs=sum(page["ref_count"] for page in self.pages.values()),
            total_requests=0,
            total_artifacts=0,
            cleanup_operation_count=self.cleanup_operations,
        )

    def _observe(self, page_id: str) -> BrowserBackendObservation:
        page = self.pages[page_id]
        page["observation_count"] += 1
        page["ref_count"] += 1
        self.total_observations += 1
        page["revision"] = self._revision(page_id)
        page["last_observation_revision"] = page["revision"]
        return BrowserBackendObservation(
            session_id=self.session_id,
            page_id=page_id,
            revision=page["revision"],
            creation_epoch=page["creation_epoch"],
            control_epoch=page["control_epoch"],
            url=page["url"],
            title=page["title"],
            snapshot=f'- button "{page["title"]}" [ref=internal]',
            refs=(
                BrowserElementRef(
                    ref="ref_root" if page_id == self.root_page_id else "ref_popup",
                    role="button",
                    name=page["title"],
                ),
            ),
            load_state="loaded",
            access_state="available",
            idle_timeout_seconds=60,
            truncation_reasons=(),
            backend_identity=_IDENTITY,
        )

    def _record_operation(self, page_id: str, operation_id: str) -> None:
        page = self.pages[page_id]
        page["operation_count"] += 1
        page["last_operation_id_sha256"] = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
        self.total_operations += 1

    async def execute(self, ctx: ToolContext, request: dict[str, Any]) -> BrowserBackendResponse:
        del ctx
        self.calls.append(dict(request))
        operation = request["operation"]
        if operation == "navigate":
            self.session_id = request["session_id"]
            self.root_page_id = request["page_id"]
            self.active_page_id = self.root_page_id
            self.total_page_creations = 1
            self.pages[self.root_page_id] = {
                "lifecycle": "active",
                "creation_epoch": 1,
                "control_epoch": 1,
                "opener_page_id": None,
                "creating_operation_id_sha256": None,
                "revision": None,
                "url": request["url"],
                "title": "Open popup",
                "last_operation_id_sha256": None,
                "terminal_reason": None,
                "operation_count": 0,
                "observation_count": 0,
                "ref_count": 0,
            }
            self._record_operation(self.root_page_id, request["operation_id"])
            observation = self._observe(self.root_page_id)
            return BrowserBackendResponse(
                observation=observation,
                page_set=self._page_set(),
                page_delta=BrowserPageSetDelta(
                    created_page_ids=(self.root_page_id,),
                    admitted_page_ids=(self.root_page_id,),
                ),
            )
        assert self.session_id is not None
        if operation == "close":
            return BrowserBackendResponse(closed=True)
        if operation == "list_pages":
            self.total_operations += 1
            return BrowserBackendResponse(page_set=self._page_set())
        page_id = request["page_id"]
        page = self.pages[page_id]
        if operation == "click":
            page["control_epoch"] += 1
            self._record_operation(page_id, request["operation_id"])
            self.total_page_creations += 1
            self.pages[self.popup_page_id] = {
                "lifecycle": "background",
                "creation_epoch": self.total_page_creations,
                "control_epoch": 1,
                "opener_page_id": page_id,
                "creating_operation_id_sha256": hashlib.sha256(
                    request["operation_id"].encode("utf-8")
                ).hexdigest(),
                "revision": "br_popup_initial",
                "url": "https://example.test/popup",
                "title": "Popup action",
                "last_operation_id_sha256": None,
                "terminal_reason": None,
                "operation_count": 0,
                "observation_count": 0,
                "ref_count": 0,
            }
            observation = self._observe(page_id)
            return BrowserBackendResponse(
                observation=observation,
                page_set=self._page_set(),
                page_delta=BrowserPageSetDelta(
                    created_page_ids=(self.popup_page_id,),
                    admitted_page_ids=(self.popup_page_id,),
                ),
            )
        if operation == "switch_page":
            previous = self.pages[self.active_page_id]
            if previous is not page:
                previous["lifecycle"] = "background"
                previous["control_epoch"] += 1
                previous["revision"] = self._revision(self.active_page_id)
            page["lifecycle"] = "active"
            page["control_epoch"] += 1
            self.active_page_id = page_id
            self._record_operation(page_id, request["operation_id"])
            observation = self._observe(page_id)
            return BrowserBackendResponse(
                observation=observation,
                page_set=self._page_set(),
            )
        if operation == "close_page":
            page["lifecycle"] = "closed"
            page["control_epoch"] += 1
            page["revision"] = None
            page["last_operation_id_sha256"] = hashlib.sha256(
                request["operation_id"].encode("utf-8")
            ).hexdigest()
            page["terminal_reason"] = "closed_by_model"
            self.cleanup_operations += 1
            if self.active_page_id == page_id:
                assert self.root_page_id is not None
                root = self.pages[self.root_page_id]
                root["lifecycle"] = "active"
                root["control_epoch"] += 1
                root["revision"] = self._revision(self.root_page_id)
                self.active_page_id = self.root_page_id
            return BrowserBackendResponse(
                page_set=self._page_set(),
                page_delta=BrowserPageSetDelta(closed_page_ids=(page_id,)),
            )
        if operation == "observe":
            self._record_operation(page_id, request["operation_id"])
            observation = self._observe(page_id)
            return BrowserBackendResponse(
                observation=observation,
                page_set=self._page_set(),
            )
        raise AssertionError(f"unexpected operation: {operation}")


def test_browser_session_multi_page_configuration_is_explicit_and_bounded() -> None:
    single = BrowserSessionTool(_backend=_FakeBrowserBackend())
    assert single.multi_page is False
    assert single.popup_policy == BrowserPopupPolicy()
    assert single.max_pages == 1
    assert single.max_total_page_creations == 1
    assert single.max_refs <= single.max_refs_per_page <= single.max_total_refs

    with pytest.raises(ValueError, match="Single-page mode"):
        BrowserSessionTool(
            popup_policy=BrowserPopupPolicy(
                mode="same_origin",
                allowed_operations=("click",),
            ),
            _backend=_FakeBrowserBackend(),
        )
    with pytest.raises(ValueError, match="max_pages"):
        BrowserSessionTool(multi_page=True, max_pages=1, _backend=_FakeBrowserBackend())
    with pytest.raises(ValueError, match="max_total_page_creations"):
        BrowserSessionTool(
            multi_page=True,
            max_pages=3,
            max_total_page_creations=2,
            _backend=_FakeBrowserBackend(),
        )
    with pytest.raises(ValueError, match="Background-page lifetime"):
        BrowserSessionTool(
            multi_page=True,
            idle_timeout_seconds=10,
            max_background_lifetime_seconds=11,
            _backend=_FakeBrowserBackend(),
        )
    with pytest.raises(ValueError, match="Per-observation refs"):
        BrowserSessionTool(
            max_refs=3,
            max_refs_per_page=2,
            _backend=_FakeBrowserBackend(),
        )
    with pytest.raises(ValueError, match="Per-page retained refs"):
        BrowserSessionTool(
            max_refs=1,
            max_refs_per_page=3,
            max_total_refs=2,
            _backend=_FakeBrowserBackend(),
        )


def test_browser_session_rejects_unproven_page_set_transition() -> None:
    class _MissingDeltaBackend(_FakeBrowserBackend):
        async def execute(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse:
            response = await super().execute(ctx, request)
            return BrowserBackendResponse(
                observation=response.observation,
                page_set=response.page_set,
                artifacts=response.artifacts,
                allocation_disposition=response.allocation_disposition,
            )

    async def scenario() -> None:
        backend = _MissingDeltaBackend()
        tool = BrowserSessionTool(_backend=backend)
        result = await tool.run(
            _context(Path(".")),
            {
                "operation": "navigate",
                "url": "https://example.test/start",
                "operation_id": "missing-page-delta",
            },
        )

        assert result.structured["error"] == "browser_crash"
        assert result.structured["allocation_disposition"] == "uncertain"
        assert len(backend.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "forgery",
    ["operation_count", "observation_control_epoch", "revision"],
)
def test_browser_session_rejects_forged_successful_page_authority(
    tmp_path: Path,
    forgery: str,
) -> None:
    class _ForgedTransitionBackend(_FakeBrowserBackend):
        async def execute(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse:
            response = await super().execute(ctx, request)
            if request["operation"] != "observe":
                return response
            assert response.observation is not None
            assert response.page_set is not None
            observation = response.observation
            page = response.page_set.pages[0]
            page_set = response.page_set
            if forgery == "operation_count":
                page_set = page_set.model_copy(
                    update={"total_operations": page_set.total_operations + 1}
                )
            elif forgery == "observation_control_epoch":
                observation = observation.model_copy(
                    update={"control_epoch": observation.control_epoch + 1}
                )
            else:
                page = page.model_copy(
                    update={
                        "revision": "br_revision_1",
                        "last_observation_revision": "br_revision_1",
                    }
                )
                observation = observation.model_copy(update={"revision": "br_revision_1"})
                page_set = page_set.model_copy(update={"pages": (page,)})
            return BrowserBackendResponse(
                observation=observation,
                page_set=page_set,
                page_delta=response.page_delta,
                artifacts=response.artifacts,
                allocation_disposition=response.allocation_disposition,
            )

    async def scenario() -> None:
        backend = _ForgedTransitionBackend()
        tool = _tool(backend)
        ctx = _context(tmp_path)
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/start",
                "operation_id": f"forged-{forgery}-open",
            },
        )
        forged = await tool.run(
            ctx,
            {
                "operation": "observe",
                "session_id": opened.structured["session_id"],
                "page_id": opened.structured["page_id"],
                "operation_id": f"forged-{forgery}-observe",
            },
        )

        assert forged.structured["error"] == "browser_crash"
        assert forged.structured["allocation_disposition"] == "uncertain"
        assert len(backend.calls) == 2

    asyncio.run(scenario())


def test_browser_session_initial_navigation_cannot_receive_popup_authority() -> None:
    class _InitialPopupBackend(_MultipageBrowserBackend):
        async def execute(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse:
            response = await super().execute(ctx, request)
            if request["operation"] != "navigate":
                return response
            assert self.root_page_id is not None
            self.total_page_creations = 2
            self.pages[self.popup_page_id] = {
                "lifecycle": "background",
                "creation_epoch": 2,
                "control_epoch": 1,
                "opener_page_id": self.root_page_id,
                "creating_operation_id_sha256": hashlib.sha256(
                    request["operation_id"].encode("utf-8")
                ).hexdigest(),
                "revision": "br_popup_initial",
                "url": "https://example.test/popup",
                "title": "Initial popup",
                "last_operation_id_sha256": None,
                "terminal_reason": None,
                "operation_count": 0,
                "observation_count": 0,
                "ref_count": 0,
            }
            return BrowserBackendResponse(
                observation=response.observation,
                page_set=self._page_set(),
                page_delta=BrowserPageSetDelta(
                    created_page_ids=(self.root_page_id, self.popup_page_id),
                    admitted_page_ids=(self.root_page_id, self.popup_page_id),
                ),
            )

    async def scenario() -> None:
        with pytest.raises(ValueError, match="allowed_operations"):
            BrowserPopupPolicy.model_validate(
                {"mode": "same_origin", "allowed_operations": ("navigate",)}
            )

        denied_backend = _InitialPopupBackend()
        denied = await BrowserSessionTool(
            multi_page=True,
            popup_policy=BrowserPopupPolicy(
                mode="same_origin",
                allowed_operations=("click",),
            ),
            max_pages=2,
            max_provisional_pages=2,
            max_page_creations_per_operation=2,
            max_total_page_creations=2,
            _backend=denied_backend,
        ).run(
            _context(Path(".")),
            {
                "operation": "navigate",
                "url": "https://example.test/start",
                "operation_id": "initial-popup-denied",
            },
        )
        assert denied.structured["error"] == "browser_crash"
        assert denied.structured["allocation_disposition"] == "uncertain"

    asyncio.run(scenario())


def test_browser_session_rejects_per_page_ref_overflow_below_aggregate_limit(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        tool = BrowserSessionTool(
            max_refs=2,
            max_refs_per_page=3,
            max_total_refs=8,
            _backend=backend,
        )
        opened = await tool.run(
            _context(tmp_path),
            {
                "operation": "navigate",
                "url": "https://example.test/start",
                "operation_id": "per-page-ref-open",
            },
        )

        refused = await tool.run(
            _context(tmp_path),
            {
                "operation": "observe",
                "session_id": opened.structured["session_id"],
                "page_id": opened.structured["page_id"],
                "operation_id": "per-page-ref-overflow",
            },
        )

        assert refused.structured["error"] == "browser_crash"
        assert refused.structured["allocation_disposition"] == "uncertain"
        assert len(backend.calls) == 2

    asyncio.run(scenario())


def test_browser_session_preserves_live_page_after_typed_action_failure(
    tmp_path: Path,
) -> None:
    class _TypedActionFailureBackend(_FakeBrowserBackend):
        async def execute(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse:
            if request["operation"] != "click":
                return await super().execute(ctx, request)
            del ctx
            self.calls.append(dict(request))
            assert self.session_id is not None
            assert self.page_id is not None
            self.control_epoch += 1
            self.operation_count += 1
            self.last_operation_id_sha256 = hashlib.sha256(
                request["operation_id"].encode("utf-8")
            ).hexdigest()
            previous_revision = f"br_revision_{self.revision_number}"
            page_set = BrowserPageSetState(
                session_id=self.session_id,
                active_page_id=self.page_id,
                pages=(
                    BrowserPageSummary(
                        page_id=self.page_id,
                        lifecycle="active",
                        creation_epoch=1,
                        control_epoch=self.control_epoch,
                        revision=None,
                        url="https://example.test/form",
                        title=self.title,
                        load_state="loaded",
                        access_state="available",
                        last_observation_revision=previous_revision,
                        last_operation_id_sha256=self.last_operation_id_sha256,
                        operation_count=self.operation_count,
                        observation_count=self.observation_count,
                        ref_count=self.ref_count,
                        request_count=0,
                        artifact_count=self.artifact_count,
                    ),
                ),
                total_page_creations=1,
                total_operations=self.operation_count,
                total_observations=self.observation_count,
                total_refs=self.ref_count,
                total_requests=0,
                total_artifacts=self.artifact_count,
                cleanup_operation_count=0,
            )
            return BrowserBackendResponse(
                failure=BrowserBackendFailure("actionability_failed"),
                page_set=page_set,
                allocation_disposition="live",
            )

    async def scenario() -> None:
        backend = _TypedActionFailureBackend()
        tool = _tool(backend)
        ctx = _context(tmp_path)
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/form",
                "operation_id": "typed-action-open",
            },
        )
        failed = await tool.run(
            ctx,
            {
                "operation": "click",
                "session_id": opened.structured["session_id"],
                "page_id": opened.structured["page_id"],
                "expected_revision": opened.structured["revision"],
                "expected_control_epoch": opened.structured["control_epoch"],
                "ref": "ref_save",
                "operation_id": "typed-action-failure",
            },
        )

        assert failed.structured["error"] == "actionability_failed"
        failed_page = failed.structured["page_set"]["pages"][0]
        assert failed_page["lifecycle"] == "active"
        assert failed_page["revision"] is None

        observed = await tool.run(
            ctx,
            {
                "operation": "observe",
                "session_id": opened.structured["session_id"],
                "page_id": opened.structured["page_id"],
                "operation_id": "typed-action-observe",
            },
        )

        assert observed.is_error is False
        assert [call["operation"] for call in backend.calls] == [
            "navigate",
            "click",
            "observe",
        ]

    asyncio.run(scenario())


def test_browser_session_close_page_retries_uncertain_cleanup_owner(
    tmp_path: Path,
) -> None:
    class _RetryingCloseBackend(_FakeBrowserBackend):
        close_attempts = 0

        async def execute(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse:
            if request["operation"] != "close_page":
                return await super().execute(ctx, request)
            del ctx
            self.calls.append(dict(request))
            assert self.session_id is not None
            assert self.page_id is not None
            self.close_attempts += 1
            self.control_epoch += 1
            closed = self.close_attempts == 2
            page_set = BrowserPageSetState(
                session_id=self.session_id,
                active_page_id=None,
                pages=(
                    BrowserPageSummary(
                        page_id=self.page_id,
                        lifecycle="closed" if closed else "uncertain",
                        creation_epoch=1,
                        control_epoch=self.control_epoch,
                        revision=None,
                        url="https://example.test/form",
                        title=self.title,
                        load_state="loaded" if closed else "failed",
                        access_state="available" if closed else "unknown",
                        last_observation_revision=f"br_revision_{self.revision_number}",
                        last_operation_id_sha256=hashlib.sha256(
                            request["operation_id"].encode("utf-8")
                        ).hexdigest(),
                        terminal_reason=("closed_by_model" if closed else "cleanup_failed"),
                        operation_count=self.operation_count,
                        observation_count=self.observation_count,
                        ref_count=self.ref_count,
                        request_count=0,
                        artifact_count=self.artifact_count,
                    ),
                ),
                total_page_creations=1,
                total_operations=self.operation_count,
                total_observations=self.observation_count,
                total_refs=self.ref_count,
                total_requests=0,
                total_artifacts=self.artifact_count,
                cleanup_operation_count=self.close_attempts,
            )
            delta = BrowserPageSetDelta(
                closed_page_ids=(self.page_id,) if closed else (),
            )
            if not closed:
                return BrowserBackendResponse(
                    failure=BrowserBackendFailure("cleanup_failed"),
                    page_set=page_set,
                    page_delta=delta,
                    allocation_disposition="uncertain",
                )
            return BrowserBackendResponse(page_set=page_set, page_delta=delta)

    async def scenario() -> None:
        backend = _RetryingCloseBackend()
        tool = _tool(backend)
        ctx = _context(tmp_path)
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/form",
                "operation_id": "retry-close-open",
            },
        )
        first = await tool.run(
            ctx,
            {
                "operation": "close_page",
                "session_id": opened.structured["session_id"],
                "page_id": opened.structured["page_id"],
                "operation_id": "retry-close-first",
            },
        )
        second = await tool.run(
            ctx,
            {
                "operation": "close_page",
                "session_id": opened.structured["session_id"],
                "page_id": opened.structured["page_id"],
                "operation_id": "retry-close-second",
            },
        )

        assert first.structured["error"] == "cleanup_failed"
        assert first.structured["page_set"]["pages"][0]["lifecycle"] == "uncertain"
        assert second.is_error is False
        assert second.structured["page_set"]["pages"][0]["lifecycle"] == "closed"
        assert backend.close_attempts == 2

    asyncio.run(scenario())


def test_browser_session_public_page_set_switch_close_and_cross_page_ref_fence(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _MultipageBrowserBackend()
        tool = BrowserSessionTool(
            multi_page=True,
            popup_policy=BrowserPopupPolicy(
                mode="same_origin",
                allowed_operations=("click",),
            ),
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
            _backend=backend,
        )
        ctx = _context(tmp_path)
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/start",
                "operation_id": "page-set-open",
            },
        )
        root_page_id = opened.structured["page_id"]
        popup_result = await tool.run(
            ctx,
            {
                "operation": "click",
                "session_id": opened.structured["session_id"],
                "page_id": root_page_id,
                "expected_revision": opened.structured["revision"],
                "expected_control_epoch": opened.structured["control_epoch"],
                "ref": "ref_root",
                "operation_id": "page-set-popup",
            },
        )
        assert popup_result.is_error is False
        assert popup_result.structured["page_delta"] == {
            "created_page_ids": [backend.popup_page_id],
            "admitted_page_ids": [backend.popup_page_id],
            "closed_page_ids": [],
            "crashed_page_ids": [],
            "refused": [],
        }
        assert len(popup_result.structured["pages"]) == 2
        calls_after_popup = len(backend.calls)

        exact_popup = await tool.run(
            ctx,
            {
                "operation": "click",
                "session_id": opened.structured["session_id"],
                "page_id": root_page_id,
                "expected_revision": opened.structured["revision"],
                "expected_control_epoch": opened.structured["control_epoch"],
                "ref": "ref_root",
                "operation_id": "page-set-popup",
            },
        )
        assert exact_popup == popup_result
        assert len(backend.calls) == calls_after_popup

        listed = await tool.run(
            ctx,
            {
                "operation": "list_pages",
                "session_id": opened.structured["session_id"],
                "operation_id": "page-set-list",
            },
        )
        assert listed.structured["active_page_id"] == root_page_id
        assert {item["page_id"] for item in listed.structured["pages"]} == {
            root_page_id,
            backend.popup_page_id,
        }

        switched = await tool.run(
            ctx,
            {
                "operation": "switch_page",
                "session_id": opened.structured["session_id"],
                "page_id": backend.popup_page_id,
                "operation_id": "page-set-switch",
            },
        )
        assert switched.structured["page_id"] == backend.popup_page_id
        assert switched.structured["active_page_id"] == backend.popup_page_id
        calls_before_cross_page_ref = len(backend.calls)
        refused = await tool.run(
            ctx,
            {
                "operation": "click",
                "session_id": opened.structured["session_id"],
                "page_id": backend.popup_page_id,
                "expected_revision": switched.structured["revision"],
                "expected_control_epoch": switched.structured["control_epoch"],
                "ref": "ref_root",
                "operation_id": "page-set-cross-page-ref",
            },
        )
        assert refused.structured["error"] == "unknown_element"
        assert refused.structured["execution"]["dispatch"] == "not_started"
        assert len(backend.calls) == calls_before_cross_page_ref

        closed = await tool.run(
            ctx,
            {
                "operation": "close_page",
                "session_id": opened.structured["session_id"],
                "page_id": backend.popup_page_id,
                "operation_id": "page-set-close",
            },
        )
        assert closed.is_error is False
        assert closed.structured["active_page_id"] == root_page_id
        assert closed.structured["page_delta"]["closed_page_ids"] == [backend.popup_page_id]
        popup = next(
            item for item in closed.structured["pages"] if item["page_id"] == backend.popup_page_id
        )
        assert popup["lifecycle"] == "closed"
        assert popup["terminal_reason"] == "closed_by_model"

    asyncio.run(scenario())


def test_browser_session_accepts_autonomous_navigation_epochs_for_fresh_page_operations(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _MultipageBrowserBackend()
        tool = BrowserSessionTool(
            multi_page=True,
            popup_policy=BrowserPopupPolicy(
                mode="same_origin",
                allowed_operations=("click",),
            ),
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
            _backend=backend,
        )
        ctx = _context(tmp_path)
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/start",
                "operation_id": "autonomous-page-open",
            },
        )
        root_page_id = opened.structured["page_id"]
        popup_result = await tool.run(
            ctx,
            {
                "operation": "click",
                "session_id": opened.structured["session_id"],
                "page_id": root_page_id,
                "expected_revision": opened.structured["revision"],
                "expected_control_epoch": opened.structured["control_epoch"],
                "ref": "ref_root",
                "operation_id": "autonomous-page-popup",
            },
        )
        assert popup_result.is_error is False

        # A real page can self-navigate between public operations. The guest
        # owns those epoch advances, while the host only sees the next exact
        # page-set snapshot.
        backend.pages[root_page_id]["control_epoch"] += 2
        backend.pages[root_page_id]["revision"] = None
        observed = await tool.run(
            ctx,
            {
                "operation": "observe",
                "session_id": opened.structured["session_id"],
                "page_id": root_page_id,
                "operation_id": "autonomous-page-observe",
            },
        )
        assert observed.is_error is False
        assert observed.structured["control_epoch"] == backend.pages[root_page_id]["control_epoch"]

        backend.pages[root_page_id]["control_epoch"] += 2
        backend.pages[root_page_id]["revision"] = None
        backend.pages[backend.popup_page_id]["control_epoch"] += 2
        backend.pages[backend.popup_page_id]["revision"] = None
        switched = await tool.run(
            ctx,
            {
                "operation": "switch_page",
                "session_id": opened.structured["session_id"],
                "page_id": backend.popup_page_id,
                "operation_id": "autonomous-page-switch",
            },
        )
        assert switched.is_error is False
        assert switched.structured["page_id"] == backend.popup_page_id

        backend.pages[backend.popup_page_id]["control_epoch"] += 2
        backend.pages[backend.popup_page_id]["revision"] = None
        closed = await tool.run(
            ctx,
            {
                "operation": "close_page",
                "session_id": opened.structured["session_id"],
                "page_id": backend.popup_page_id,
                "operation_id": "autonomous-page-close",
            },
        )
        assert closed.is_error is False
        closed_popup = next(
            page for page in closed.structured["pages"] if page["page_id"] == backend.popup_page_id
        )
        assert closed_popup["lifecycle"] == "closed"

    asyncio.run(scenario())


def test_browser_session_retains_popup_guard_failure_page_evidence(
    tmp_path: Path,
) -> None:
    class _PopupGuardFailureBackend(_MultipageBrowserBackend):
        async def execute(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse:
            if request["operation"] != "observe":
                return await super().execute(ctx, request)
            del ctx
            self.calls.append(dict(request))
            page = self.pages[request["page_id"]]
            page["lifecycle"] = "uncertain"
            page["control_epoch"] += 1
            page["revision"] = None
            page["terminal_reason"] = "popup_guard_failed"
            self.active_page_id = None
            return BrowserBackendResponse(
                failure=BrowserBackendFailure("browser_crash"),
                page_set=self._page_set(),
                allocation_disposition="uncertain",
            )

    async def scenario() -> None:
        backend = _PopupGuardFailureBackend()
        tool = BrowserSessionTool(_backend=backend)
        ctx = _context(tmp_path)
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/start",
                "operation_id": "popup-guard-page-open",
            },
        )
        failed = await tool.run(
            ctx,
            {
                "operation": "observe",
                "session_id": opened.structured["session_id"],
                "page_id": opened.structured["page_id"],
                "operation_id": "popup-guard-page-failure",
            },
        )

        assert failed.structured["error"] == "browser_crash"
        assert failed.structured["allocation_disposition"] == "uncertain"
        assert failed.structured["page_set"]["pages"][0]["lifecycle"] == "uncertain"
        assert failed.structured["page_set"]["pages"][0]["terminal_reason"] == "popup_guard_failed"

    asyncio.run(scenario())


def test_browser_session_durable_reconnect_restores_complete_page_authority(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _MultipageBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        configuration: dict[str, Any] = {
            "multi_page": True,
            "popup_policy": BrowserPopupPolicy(
                mode="same_origin",
                allowed_operations=("click",),
            ),
            "max_pages": 2,
            "max_provisional_pages": 1,
            "max_page_creations_per_operation": 1,
            "max_total_page_creations": 2,
            "_backend": backend,
        }
        opened_args = {
            "operation": "navigate",
            "url": "https://example.test/start",
            "operation_id": "durable-page-open",
        }
        first_tool = BrowserSessionTool(**configuration)
        opened = await first_tool.run(
            _durable_context(
                tmp_path,
                args=opened_args,
                records=records,
                tool_call_id="durable-page-open-call",
            ),
            opened_args,
        )
        popup_args = {
            "operation": "click",
            "session_id": opened.structured["session_id"],
            "page_id": opened.structured["page_id"],
            "expected_revision": opened.structured["revision"],
            "expected_control_epoch": opened.structured["control_epoch"],
            "ref": "ref_root",
            "operation_id": "durable-page-popup",
        }
        popup = await first_tool.run(
            _durable_context(
                tmp_path,
                args=popup_args,
                records=records,
                tool_call_id="durable-page-popup-call",
            ),
            popup_args,
        )
        assert popup.is_error is False

        list_args = {
            "operation": "list_pages",
            "session_id": opened.structured["session_id"],
            "operation_id": "durable-page-list",
        }
        reconnected = await BrowserSessionTool(**configuration).run(
            _durable_context(
                tmp_path,
                args=list_args,
                records=records,
                tool_call_id="durable-page-list-call",
            ),
            list_args,
        )
        assert reconnected.is_error is False
        assert reconnected.structured["page_set"]["total_page_creations"] == 2
        assert {item["page_id"] for item in reconnected.structured["pages"]} == {
            opened.structured["page_id"],
            backend.popup_page_id,
        }
        calls_after_list = len(backend.calls)
        exact_list = await BrowserSessionTool(**configuration).run(
            _durable_context(
                tmp_path,
                args=list_args,
                records=records,
                tool_call_id="durable-page-list-call",
            ),
            list_args,
        )
        assert exact_list == reconnected
        assert len(backend.calls) == calls_after_list
        session_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-session"
        )
        assert len(session_record["page_authorities"]) == 2
        assert all(
            "url" not in authority and "title" not in authority
            for authority in session_record["page_authorities"]
        )

    asyncio.run(scenario())


def test_browser_session_exact_durable_retry_retains_denied_popup_delta(
    tmp_path: Path,
) -> None:
    class _DeniedPopupBackend(_MultipageBrowserBackend):
        async def execute(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse:
            response = await super().execute(ctx, request)
            if request["operation"] != "click":
                return response
            assert response.page_set is not None
            assert self.root_page_id is not None
            pages = tuple(
                page.model_copy(
                    update={
                        "lifecycle": "closed",
                        "control_epoch": page.control_epoch + 1,
                        "revision": None,
                        "terminal_reason": "destination_denied",
                    }
                )
                if page.page_id == self.popup_page_id
                else page
                for page in response.page_set.pages
            )
            return BrowserBackendResponse(
                failure=BrowserBackendFailure("policy_denied"),
                page_set=response.page_set.model_copy(update={"pages": pages}),
                page_delta=BrowserPageSetDelta(
                    created_page_ids=(self.popup_page_id,),
                    closed_page_ids=(self.popup_page_id,),
                    refused=(
                        BrowserPageRefusal(
                            page_id=self.popup_page_id,
                            opener_page_id=self.root_page_id,
                            reason="destination_denied",
                        ),
                    ),
                ),
                allocation_disposition="live",
            )

    async def scenario() -> None:
        backend = _DeniedPopupBackend()
        records: dict[str, dict[str, Any]] = {}
        configuration: dict[str, Any] = {
            "multi_page": True,
            "popup_policy": BrowserPopupPolicy(
                mode="same_origin",
                allowed_operations=("click",),
            ),
            "max_pages": 2,
            "max_provisional_pages": 1,
            "max_page_creations_per_operation": 1,
            "max_total_page_creations": 2,
            "_backend": backend,
        }
        opened_args = {
            "operation": "navigate",
            "url": "https://example.test/start",
            "operation_id": "durable-denied-popup-open",
        }
        opened = await BrowserSessionTool(**configuration).run(
            _durable_context(
                tmp_path,
                args=opened_args,
                records=records,
                tool_call_id="durable-denied-popup-open-call",
            ),
            opened_args,
        )
        popup_args = {
            "operation": "click",
            "session_id": opened.structured["session_id"],
            "page_id": opened.structured["page_id"],
            "expected_revision": opened.structured["revision"],
            "expected_control_epoch": opened.structured["control_epoch"],
            "ref": "ref_root",
            "operation_id": "durable-denied-popup",
        }
        context = _durable_context(
            tmp_path,
            args=popup_args,
            records=records,
            tool_call_id="durable-denied-popup-call",
        )
        denied = await BrowserSessionTool(**configuration).run(context, popup_args)
        calls_after_denial = len(backend.calls)
        replayed = await BrowserSessionTool(**configuration).run(context, popup_args)

        assert denied.structured["error"] == "policy_denied"
        assert denied.structured["page_delta"]["refused"] == [
            {
                "page_id": backend.popup_page_id,
                "opener_page_id": opened.structured["page_id"],
                "reason": "destination_denied",
            }
        ]
        assert replayed == denied
        assert len(backend.calls) == calls_after_denial

    asyncio.run(scenario())


def test_browser_session_fresh_tool_reconciles_popup_receipt_without_reclick(
    tmp_path: Path,
) -> None:
    class _ReconcilingMultipageBackend(_MultipageBrowserBackend):
        def __init__(self) -> None:
            super().__init__()
            self.receipt: BrowserBackendResponse | None = None
            self.reconcile_calls = 0

        async def execute(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse:
            response = await super().execute(ctx, request)
            self.receipt = response
            return response

        async def reconcile(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse | None:
            del ctx, request
            self.reconcile_calls += 1
            return self.receipt

    async def scenario() -> None:
        backend = _ReconcilingMultipageBackend()
        records: dict[str, dict[str, Any]] = {}
        configuration = {
            "multi_page": True,
            "popup_policy": BrowserPopupPolicy(
                mode="same_origin",
                allowed_operations=("click",),
            ),
            "max_pages": 2,
            "max_provisional_pages": 1,
            "max_page_creations_per_operation": 1,
            "max_total_page_creations": 2,
            "_backend": backend,
        }
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/start",
            "operation_id": "popup-reconcile-navigate",
        }
        opened = await BrowserSessionTool(**configuration).run(
            _durable_context(
                tmp_path,
                args=navigate_args,
                records=records,
                tool_call_id="popup-reconcile-navigate-call",
            ),
            navigate_args,
        )
        click_args = {
            "operation": "click",
            "session_id": opened.structured["session_id"],
            "page_id": opened.structured["page_id"],
            "expected_revision": opened.structured["revision"],
            "expected_control_epoch": opened.structured["control_epoch"],
            "ref": "ref_root",
            "operation_id": "popup-reconcile-click",
        }
        ambiguous = await BrowserSessionTool(**configuration).run(
            _durable_context(
                tmp_path,
                args=click_args,
                records=records,
                tool_call_id="popup-reconcile-click-call",
                fail_before_state="terminal",
            ),
            click_args,
        )

        recovered = await BrowserSessionTool(**configuration).run(
            _durable_context(
                tmp_path,
                args=click_args,
                records=records,
                tool_call_id="popup-reconcile-click-call",
            ),
            click_args,
        )

        assert ambiguous.structured["error"] == "outcome_ambiguous"
        assert recovered.is_error is False
        assert recovered.structured["page_delta"]["created_page_ids"] == [backend.popup_page_id]
        assert backend.reconcile_calls == 1
        assert [call["operation"] for call in backend.calls] == ["navigate", "click"]

    asyncio.run(scenario())
