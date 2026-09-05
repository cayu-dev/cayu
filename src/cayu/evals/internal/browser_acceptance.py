"""Cayu-owned deterministic browser acceptance target.

The target deliberately uses the already-built pinned browser image.  It never
builds or pulls the image and never calls a live model provider.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from multiprocessing.process import BaseProcess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

from cayu import (
    AgentSpec,
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    CayuApp,
    EnvironmentSpec,
    EvalCase,
    EvalPlan,
    EvalSuite,
    LocalArtifactStore,
    Message,
    RunLimits,
    RunRequest,
    SessionCompleted,
    WebBridge,
)
from cayu._task_wait import (
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
)
from cayu.core.events import Event, event_durable_sequence
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.messages import TextPart, ToolResultPart
from cayu.egress import HttpxUpstream
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.evals._memory_attribution import (
    eval_memory_attribution_evidence_from_trajectory,
)
from cayu.evals.browser_acceptance import (
    BrowserAcceptanceFaultEvidenceV1,
    BrowserAcceptanceFaultScenario,
    BrowserAcceptancePlanV1,
    BrowserAcceptanceScenarioExecutionV1,
    _browser_dispatches_from_trial,
)
from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
from cayu.evals.browser_acceptance_manifests import (
    DETERMINISTIC_BROWSER_ACCEPTANCE_MAX_ARTIFACT_BYTES_PER_OPERATION,
    deterministic_browser_acceptance_manifest,
)
from cayu.evals.corpus import _content_revision
from cayu.evals.models import EvalStatus, EvalTrialResult
from cayu.evals.testing import ScriptedModelProvider
from cayu.evals.trajectory import _trajectory_from_terminal_evidence, trajectory_from_session
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD, ExecCommand, Runner
from cayu.runtime._event_projection import public_event_sequence
from cayu.runtime.egress import VirtualEgressEnvironmentFactory
from cayu.runtime.event_sinks import EventSink
from cayu.runtime.sessions import (
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS,
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES,
    EventQuery,
    IncompleteSessionRecoveryRequest,
    RunnerObservedEventIdentity,
    SessionOperationPublication,
    SessionStatus,
    TerminalSessionEvidenceLimits,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.browser_session import BrowserBackendFailure, BrowserBackendResponse

_AGENT = "browser-acceptance"
_MODEL = "browser-acceptance-deterministic-v1"
_POLICY = "browser-acceptance"
_PLANNER_REVISION = _content_revision(
    {
        "version": 1,
        "case_prompt": "browser acceptance case: <case-id>",
        "selection": "accessible-name exact first match",
        "terminal_output": "browser_acceptance:success",
    },
    "browser acceptance deterministic planner",
)
_SCENARIO_EXECUTOR_REVISION = _content_revision(
    {
        "version": 2,
        "entrance": "CayuApp.run",
        "recovery": "CayuApp.recover_incomplete_session",
        "process_store": "sqlite",
        "process_event_evidence": "fsynced-public-stream-plus-bounded-durable-recovery",
        "fault_phases": [item.value for item in BrowserAcceptanceFaultScenario],
    },
    "browser acceptance fault scenario executor",
)
_PROCESS_LOSS_EXIT_CODE = 86
_OBSERVED_EVENTS_FILENAME = "observed-events.jsonl"
_PROCESS_POLL_SECONDS = 0.02
_PROCESS_TERMINATE_GRACE_SECONDS = 10.0
_PROCESS_KILL_GRACE_SECONDS = 10.0
_BROWSER_DAEMON_SIGNAL_SCRIPT = """
import os
import signal
import sys

session_id = sys.argv[1].encode("utf-8")
signal_number = int(sys.argv[2])
needle = b"--interactive-daemon\\0" + session_id + b"\\0"
found = False
for entry in os.listdir("/proc"):
    if not entry.isdigit() or int(entry) == os.getpid():
        continue
    try:
        with open(f"/proc/{entry}/cmdline", "rb") as handle:
            command = handle.read(16_384)
    except OSError:
        continue
    if needle not in command:
        continue
    os.kill(int(entry), signal_number)
    found = True
raise SystemExit(0 if found else 3)
""".strip()
_BROWSER_PAGE_FAULT_CONTROL_SCRIPT = """
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

mode, session_id = sys.argv[1:3]
digest = hashlib.sha256(b"cayu-browser-session-socket-v1\\0" + session_id.encode()).hexdigest()
root = Path("/tmp/cayu-browser-sessions")
root.mkdir(mode=0o700, exist_ok=True)
control = root / (digest + ".page-fault.sock")
if mode == "launch":
    source = sys.stdin.buffer.read(32769)
    if len(source) > 32768:
        raise SystemExit(2)
    path = root / (digest + ".page-fault.py")
    with path.open("xb") as stream:
        stream.write(source)
    process = subprocess.Popen(
        [sys.executable, "-I", str(path), "--interactive-daemon", session_id],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True,
    )
    deadline = time.monotonic() + 10
    while not control.exists() or not (root / (digest + ".sock")).exists():
        if process.poll() is not None or time.monotonic() >= deadline:
            raise SystemExit(3)
        time.sleep(.02)
elif mode == "crash":
    page_id = sys.argv[3]
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(7)
        client.connect(str(control))
        client.sendall(json.dumps({"page_id": page_id}).encode() + b"\\n")
        response = b""
        while not response.endswith(b"\\n") and len(response) <= 1024:
            chunk = client.recv(1024)
            if not chunk:
                break
            response += chunk
    if json.loads(response) != {"page_id": page_id, "crashed": True}:
        raise SystemExit(4)
else:
    raise SystemExit(2)
""".strip()
_PROVIDER_EXECUTION_PROFILE_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="cayu:browser-acceptance-provider",
    behavior_version="1",
    implementation_version=_PLANNER_REVISION,
)
_ENVIRONMENT_EXECUTION_PROFILE_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="cayu:browser-acceptance-environment",
    behavior_version="1",
    implementation_version=_SCENARIO_EXECUTOR_REVISION,
)


def _scenario_stage(scenario: BrowserAcceptanceFaultScenario) -> tuple[str, str]:
    browser_stages = {
        BrowserAcceptanceFaultScenario.BROWSER_BEFORE_DISPATCH: (
            "browser",
            "before_dispatch",
        ),
        BrowserAcceptanceFaultScenario.BROWSER_ALLOCATION_LOSS: (
            "browser",
            "allocation_loss",
        ),
        BrowserAcceptanceFaultScenario.BROWSER_DURING_EXECUTION: (
            "browser",
            "during_execution",
        ),
        BrowserAcceptanceFaultScenario.BROWSER_AFTER_EFFECT: ("browser", "after_effect"),
        BrowserAcceptanceFaultScenario.BROWSER_DURING_CLEANUP: (
            "browser",
            "during_cleanup",
        ),
        BrowserAcceptanceFaultScenario.BROWSER_ACTIVE_PAGE_CRASH: ("browser", "active_page_crash"),
        BrowserAcceptanceFaultScenario.BROWSER_BACKGROUND_PAGE_CRASH: (
            "browser",
            "background_page_crash",
        ),
    }
    if scenario in browser_stages:
        return browser_stages[scenario]
    suffix = scenario.value.removeprefix("cancel_").removeprefix("process_")
    if scenario is BrowserAcceptanceFaultScenario.ACKNOWLEDGEMENT_LOSS:
        return "terminal", "after"
    if suffix == "before_terminal":
        return "terminal", "before"
    if suffix == "after_artifact":
        return "artifact", "after"
    return suffix.removeprefix("after_"), "after"


@dataclass
class _FaultControl:
    scenario: BrowserAcceptanceFaultScenario
    marker_path: Path
    target_operation_number: int
    observed: bool = False

    def trigger(self, stage: str, position: str, operation_number: int) -> None:
        expected_stage, expected_position = _scenario_stage(self.scenario)
        expected_number = 1 if stage == "artifact" else self.target_operation_number
        if (
            self.observed
            or operation_number != expected_number
            or (stage, position) != (expected_stage, expected_position)
        ):
            return
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.marker_path.write_text(
            f"{self.scenario.value}:{stage}:{position}\n",
            encoding="utf-8",
        )
        self.observed = True
        if self.scenario.value.startswith("process_"):
            os._exit(_PROCESS_LOSS_EXIT_CODE)
        if self.scenario.value.startswith("cancel_"):
            task = asyncio.current_task()
            if task is None:
                raise RuntimeError("Browser acceptance cancellation has no owning task.")
            task.cancel()
            raise asyncio.CancelledError
        if self.scenario is BrowserAcceptanceFaultScenario.ACKNOWLEDGEMENT_LOSS:
            raise ConnectionError("browser acceptance terminal acknowledgement lost")


class _FaultSQLiteSessionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version: ClassVar[int | None] = 1

    def __init__(self, path: Path, *, control: _FaultControl | None) -> None:
        super().__init__(path)
        self._acceptance_control = control
        self._acceptance_state_counts: dict[str, int] = {}

    async def publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform,
        events,
        expected_statuses=None,
        expected_run_epoch=None,
        expected_transcript_cursor=None,
    ):
        observed_state: str | None = None
        observed_number = 0

        def inspect(current_session, checkpoint, current):
            nonlocal observed_number, observed_state
            publication = operation_transform(current_session, checkpoint, current)
            if type(publication) is not SessionOperationPublication:
                return publication
            states = {
                record.get("state")
                for record in publication.operation_records.values()
                if record.get("record_type") == "cayu.browser-operation"
            }
            observed_state = next(
                (state for state in ("terminal", "dispatched", "intent") if state in states),
                None,
            )
            control = self._acceptance_control
            if control is not None and observed_state is not None:
                observed_number = self._acceptance_state_counts.get(observed_state, 0) + 1
                self._acceptance_state_counts[observed_state] = observed_number
                control.trigger(observed_state, "before", observed_number)
            return publication

        result = await super().publish_session_operation(
            session_id,
            idempotency_key=idempotency_key,
            operation_transform=inspect,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
        control = self._acceptance_control
        if control is not None and observed_state is not None:
            control.trigger(observed_state, "after", observed_number)
        return result


class _FaultArtifactStore(LocalArtifactStore):
    def __init__(self, root: Path, *, control: _FaultControl | None) -> None:
        super().__init__(root, store_id="browser-acceptance-artifacts-v1")
        self._acceptance_control = control

    async def put_bytes(self, *args: Any, **kwargs: Any):
        result = await super().put_bytes(*args, **kwargs)
        control = self._acceptance_control
        if control is not None:
            control.trigger("artifact", "after", 1)
        return result


class _AcceptanceEventJournalSink(EventSink):
    """Capture events delivered by the runtime without reading them back from storage."""

    def __init__(self, path: Path) -> None:
        self._path = path

    async def emit(self, event: Event) -> None:
        _append_delivered_event(self._path, event)


class BrowserAcceptanceDeterministicProvider(ScriptedModelProvider):
    """Closed transcript-driven provider for the checked-in browser corpus."""

    def __init__(self, cases: dict[str, Any]) -> None:
        super().__init__((), name="browser-acceptance-scripted")
        self._cases = dict(cases)
        self._acceptance_temporary_directory: TemporaryDirectory[str] | None = None
        self.execution_revision = _PLANNER_REVISION

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        """Return the checked-in planner identity used across restart recovery."""

        return _PROVIDER_EXECUTION_PROFILE_IDENTITY

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request.model_copy(deep=True))
        case_id = _case_id(request)
        case = self._cases.get(case_id)
        if case is None:
            raise RuntimeError("Browser acceptance request does not name a canonical case.")
        results = _browser_results(request)
        operation_index = len(results)
        if operation_index >= len(case.operations):
            yield ModelStreamEvent.text_delta("browser_acceptance:success")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        operation = case.operations[operation_index]
        arguments = _operation_arguments(
            case_id=case_id,
            operation=operation,
            operation_index=operation_index,
            fixture_route=case.fixture_route,
            results=results,
        )
        yield ModelStreamEvent.tool_call(
            id=f"{case_id}-{operation_index + 1}",
            name="browser_session",
            arguments=arguments,
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


async def _signal_browser_daemon(ctx: Any, session_id: str, signal_number: int) -> None:
    """Signal the exact guest daemon without adding a browser-operation dispatch event."""

    invocation_runner = ctx.runner
    raw_runner = getattr(invocation_runner, "_InvocationRunnerHandle__runner", None)
    if not isinstance(raw_runner, Runner):
        raise RuntimeError("Browser acceptance could not resolve the owned runner.")
    result = await raw_runner.exec_system(
        ExecCommand.process(
            "/usr/local/bin/python",
            "-I",
            "-c",
            _BROWSER_DAEMON_SIGNAL_SCRIPT,
            session_id,
            str(signal_number),
        ),
        timeout_s=5,
        output_limit_bytes=1_024,
    )
    if result.exit_code != 0 or result.timed_out or result.cancelled:
        raise RuntimeError("Browser acceptance could not signal the exact browser daemon.")


async def _control_browser_page_fault(
    ctx: Any, session_id: str, *, page_id: str | None = None
) -> None:
    raw_runner = getattr(ctx.runner, "_InvocationRunnerHandle__runner", None)
    if not isinstance(raw_runner, Runner):
        raise RuntimeError("Browser acceptance could not resolve the owned runner.")
    source = (
        Path(__file__).with_name("_browser_page_fault_guest.py").read_text(encoding="utf-8")
        if page_id is None
        else None
    )
    result = await raw_runner.exec_system(
        ExecCommand.process(
            "/usr/local/bin/python",
            "-I",
            "-c",
            _BROWSER_PAGE_FAULT_CONTROL_SCRIPT,
            "launch" if page_id is None else "crash",
            session_id,
            *(() if page_id is None else (page_id,)),
        ),
        stdin=source,
        timeout_s=15,
        output_limit_bytes=1024,
    )
    if result.exit_code != 0 or result.timed_out or result.cancelled:
        raise RuntimeError("Browser acceptance could not deliver the exact page crash.")


def _install_browser_crash_fault(bridge: WebBridge, control: _FaultControl) -> None:
    """Deliver a real page-target or daemon crash in the owned allocation."""

    browser_tool = bridge.tools[0]
    backend = getattr(browser_tool, "_backend", None)
    original_execute = getattr(backend, "execute", None)
    if not callable(original_execute):
        raise RuntimeError("Browser acceptance browser backend is unavailable.")
    operation_number = 0
    previous_response: BrowserBackendResponse | None = None
    page_fault = control.scenario in {
        BrowserAcceptanceFaultScenario.BROWSER_ACTIVE_PAGE_CRASH,
        BrowserAcceptanceFaultScenario.BROWSER_BACKGROUND_PAGE_CRASH,
    }

    async def execute_with_browser_fault(
        ctx: Any, request: dict[str, Any]
    ) -> BrowserBackendResponse:
        nonlocal operation_number, previous_response
        operation_number += 1
        if page_fault and operation_number == 1:
            await _control_browser_page_fault(ctx, request["session_id"])
        if operation_number != control.target_operation_number:
            previous_response = await original_execute(ctx, request)
            return previous_response
        scenario = control.scenario
        session_id = request["session_id"]
        if page_fault:
            before = None if previous_response is None else previous_response.page_set
            if before is None or len(before.pages) != 2:
                raise RuntimeError("Page crash acceptance requires two exact admitted pages.")
            lifecycle = (
                "active"
                if scenario is BrowserAcceptanceFaultScenario.BROWSER_ACTIVE_PAGE_CRASH
                else "background"
            )
            target = next(page for page in before.pages if page.lifecycle == lifecycle)
            survivor = next(page for page in before.pages if page.page_id != target.page_id)
            await _control_browser_page_fault(ctx, session_id, page_id=target.page_id)
            response = await original_execute(ctx, request)
            after = response.page_set
            if (
                response.failure is not None
                or response.allocation_disposition != "live"
                or after is None
            ):
                raise RuntimeError("Page crash did not preserve its browser allocation.")
            crashed = next(page for page in after.pages if page.page_id == target.page_id)
            remaining = next(page for page in after.pages if page.page_id == survivor.page_id)
            if (
                crashed.lifecycle != "crashed"
                or crashed.revision is not None
                or crashed.control_epoch <= target.control_epoch
                or after.active_page_id != survivor.page_id
                or remaining.lifecycle != "active"
                or (lifecycle == "background" and remaining != survivor)
                or (lifecycle == "active" and remaining.control_epoch <= survivor.control_epoch)
            ):
                raise RuntimeError("Page crash corrupted page lifecycle or reference authority.")
            control.trigger("browser", lifecycle + "_page_crash", operation_number)
            return response
        if scenario is BrowserAcceptanceFaultScenario.BROWSER_BEFORE_DISPATCH:
            await _signal_browser_daemon(ctx, session_id, 9)
            control.trigger("browser", "before_dispatch", operation_number)
            return BrowserBackendResponse(
                failure=BrowserBackendFailure("browser_crash"),
                allocation_disposition="uncertain",
            )
        if scenario is BrowserAcceptanceFaultScenario.BROWSER_ALLOCATION_LOSS:
            await _signal_browser_daemon(ctx, session_id, 9)
            control.trigger("browser", "allocation_loss", operation_number)
            return BrowserBackendResponse(
                failure=BrowserBackendFailure("allocation_lost"),
                allocation_disposition="retired",
            )
        if scenario is BrowserAcceptanceFaultScenario.BROWSER_AFTER_EFFECT:
            await original_execute(ctx, request)
            await _signal_browser_daemon(ctx, session_id, 9)
            control.trigger("browser", "after_effect", operation_number)
            raise RuntimeError("Browser acceptance crashed the browser after its effect.")
        if scenario is BrowserAcceptanceFaultScenario.BROWSER_DURING_EXECUTION:
            execution = asyncio.create_task(original_execute(ctx, request))
            await asyncio.sleep(0.05)
            await _signal_browser_daemon(ctx, session_id, 9)
            control.trigger("browser", "during_execution", operation_number)
            await execution
            raise RuntimeError("Browser acceptance browser survived its execution crash.")
        if scenario is BrowserAcceptanceFaultScenario.BROWSER_DURING_CLEANUP:
            await _signal_browser_daemon(ctx, session_id, 19)
            execution = asyncio.create_task(original_execute(ctx, request))
            await asyncio.sleep(0.05)
            await _signal_browser_daemon(ctx, session_id, 9)
            control.trigger("browser", "during_cleanup", operation_number)
            await execution
            raise RuntimeError("Browser acceptance browser survived its cleanup crash.")
        return await original_execute(ctx, request)

    object.__setattr__(backend, "execute", execute_with_browser_fault)


def _case_id(request: ModelRequest) -> str:
    prefix = "browser acceptance case: "
    for message in request.messages:
        if message.role != "user":
            continue
        for part in message.content:
            if type(part) is TextPart and part.text.startswith(prefix):
                return part.text.removeprefix(prefix)
    raise RuntimeError("Browser acceptance request has no canonical case identity.")


def _browser_results(request: ModelRequest) -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    for message in request.messages:
        for part in message.content:
            if type(part) is not ToolResultPart or part.tool_name != "browser_session":
                continue
            values.append(dict(part.structured or {}))
    return tuple(values)


def _latest_browser_state(results: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    for result in reversed(results):
        if all(type(result.get(key)) is str for key in ("session_id", "page_id", "revision")):
            return result
    raise RuntimeError("Browser acceptance operation has no prior browser observation.")


def _latest_page_set(results: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    for result in reversed(results):
        page_set = result.get("page_set")
        if isinstance(page_set, dict) and isinstance(page_set.get("pages"), list):
            return page_set
    raise RuntimeError("Browser acceptance operation has no prior page-set evidence.")


def _popup_page_id(results: tuple[dict[str, Any], ...]) -> str:
    page_set = _latest_page_set(results)
    pages = page_set["pages"]
    candidates = [
        page
        for page in pages
        if isinstance(page, dict)
        and type(page.get("page_id")) is str
        and type(page.get("opener_page_id")) is str
        and page.get("lifecycle") in {"admitted", "active", "background"}
    ]
    if len(candidates) != 1:
        raise RuntimeError("Browser acceptance page set lacks one admitted popup.")
    return candidates[0]["page_id"]


def _ref(state: dict[str, Any], names: tuple[str, ...]) -> str:
    refs = state.get("refs")
    if not isinstance(refs, list | tuple):
        raise RuntimeError("Browser acceptance observation has no element references.")
    for name in names:
        for item in refs:
            if isinstance(item, dict) and item.get("name") == name and type(item.get("ref")) is str:
                return item["ref"]
    raise RuntimeError("Browser acceptance observation lacks the required element reference.")


def _operation_arguments(
    *,
    case_id: str,
    operation: str,
    operation_index: int,
    fixture_route: str | None,
    results: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    operation_id = f"{case_id}:{operation_index + 1}:{operation}"
    if case_id in {
        "recovery-conflicting-operation-id",
        "recovery-exact-terminal-replay",
    }:
        operation_id = f"{case_id}:1:navigate"
    if case_id == "page-popup-exact-replay" and operation == "click":
        operation_id = f"{case_id}:2:click"
    if operation == "navigate":
        route = fixture_route or "/basic"
        if case_id == "recovery-conflicting-operation-id" and operation_index == 1:
            route = "/forms"
        url = route if route.startswith("https://") else f"https://docs.browser.test{route}"
        return {"operation": operation, "url": url, "operation_id": operation_id}
    state = (
        results[0]
        if case_id.startswith("revision-stale-ref-after-") and operation_index == 2 and results
        else results[0]
        if case_id == "page-popup-exact-replay" and operation_index == 2 and results
        else _latest_browser_state(results)
    )
    arguments: dict[str, Any] = {
        "operation": operation,
        "session_id": state["session_id"],
        "operation_id": operation_id,
    }
    if operation not in {"close", "list_pages"}:
        page_id = state["page_id"]
        if operation in {"switch_page", "close_page"} and case_id.startswith("page-"):
            page_id = _popup_page_id(results)
        arguments["page_id"] = page_id
        if operation not in {"switch_page", "close_page"}:
            arguments["expected_revision"] = state["revision"]
    if operation in {"click", "fill", "select", "press", "wait", "screenshot", "download"}:
        control_epoch = state.get("control_epoch")
        if type(control_epoch) is not int:
            raise RuntimeError("Browser acceptance observation lacks its control epoch.")
        arguments["expected_control_epoch"] = control_epoch
    if operation == "wait":
        arguments["wait_ms"] = 5_000 if case_id == "crash-during-execution" else 250
    elif operation == "screenshot":
        arguments["full_page"] = True
    elif operation in {"click", "fill", "select", "press", "download"}:
        names = (
            (("Download report",) if operation == "download" else ("Save",))
            if case_id.startswith("revision-stale-ref-after-") and operation_index == 2
            else {
                ("action-delayed-element", "click"): ("Continue",),
                ("action-disabled-control", "click"): ("Unavailable",),
                ("action-duplicate-labels", "click"): ("Continue",),
                ("action-form-controls", "fill"): ("Name",),
                ("action-form-controls", "select"): ("Region",),
                ("action-form-controls", "press"): ("Name",),
                ("action-form-controls", "click"): ("Save",),
                ("action-form-validation", "click"): ("Save",),
                ("action-hidden-control", "click"): ("Hidden action",),
                ("action-detached-control", "click"): ("Detach me",),
                ("action-occluded-control", "click"): ("Covered action",),
                ("action-replaced-element", "click"): ("Old", "New"),
                ("action-readonly-control", "fill"): ("Account",),
                ("navigation-scroll-dependent-control", "click"): ("Bottom action",),
                ("page-about-blank-popup-transition", "click"): ("Open blank popup",),
                ("page-active-page-crash", "click"): ("Open popup",),
                ("page-background-page-crash", "click"): ("Open popup",),
                ("page-complete-cleanup", "click"): ("Open popup",),
                ("page-cross-origin-popup", "click"): ("Open cross-origin popup",),
                ("page-cross-page-stale-ref", "click"): ("Open popup",),
                ("page-popup-burst", "click"): ("Open popup burst",),
                ("page-popup-exact-replay", "click"): ("Open popup",),
                ("page-popup-opener-navigation", "click"): ("Open navigating popup",),
                ("page-popup-process-loss-ambiguity", "click"): ("Open popup",),
                ("page-popup-redirect-pivot", "click"): ("Open redirecting popup",),
                ("page-multiple-popup-tab-switch-close", "click"): ("Open popup",),
                ("iframe-cross-origin", "fill"): ("Frame value",),
                ("iframe-cross-origin", "click"): ("Apply",),
                ("iframe-same-origin", "fill"): ("Frame value",),
                ("iframe-same-origin", "click"): ("Apply",),
                ("artifact-bounded-download", "download"): ("Download report",),
                ("limit-oversized-download", "download"): ("Download oversized file",),
                ("revision-stale-ref-after-click", "click"): ("Save",),
                ("revision-stale-ref-after-download", "download"): ("Download report",),
                ("revision-stale-ref-after-fill", "fill"): ("Name",),
                ("revision-stale-ref-after-press", "press"): ("Name",),
                ("revision-stale-ref-after-select", "select"): ("Region",),
            }.get((case_id, operation))
        )
        if names is None:
            raise RuntimeError("Browser acceptance planner lacks an operation target.")
        ref_state = (
            results[1] if case_id == "page-cross-page-stale-ref" and operation_index == 3 else state
        )
        arguments["ref"] = _ref(ref_state, names)
        if operation == "fill":
            arguments["value"] = "Cayu acceptance"
        elif operation == "select":
            arguments["value"] = "South"
        elif operation == "press":
            arguments["key"] = "Tab"
    return arguments


def _scenario_request(case: Any, *, session_id: str) -> RunRequest:
    return RunRequest(
        session_id=session_id,
        agent_name=_AGENT,
        messages=[Message.text("user", f"browser acceptance case: {case.case_id}")],
        max_steps=len(case.operations) + 1,
        limits=RunLimits(
            max_tool_calls=len(case.operations),
            max_elapsed_seconds=300,
        ),
    )


def _build_runtime(
    *,
    root: Path,
    upstream_routes: dict[str, str],
    hosts: tuple[str, ...],
    cases: dict[str, Any],
    seccomp_profile: Path,
    control: _FaultControl | None,
) -> tuple[CayuApp, WebBridge, BrowserAcceptanceDeterministicProvider]:
    root.mkdir(parents=True, exist_ok=True)
    provider = BrowserAcceptanceDeterministicProvider(cases)
    store = _FaultSQLiteSessionStore(root / "sessions.sqlite", control=control)
    factory = VirtualEgressEnvironmentFactory(
        policies={
            _POLICY: BrowserEgressPolicy(
                name=_POLICY,
                allowed_hosts=hosts,
                allowed_path_prefixes=("/",),
            )
        },
        approved_destinations=tuple(
            ApprovedEgressDestination(destination=host, policy_name=_POLICY) for host in hosts
        ),
        adapter=DockerEgressAdapter(seccomp_profile=str(seccomp_profile)),
        upstream=HttpxUpstream(routes=upstream_routes),
        image=PINNED_BROWSER_SESSION_WORKLOAD.image,
        artifact_store=_FaultArtifactStore(root / "artifacts", control=control),
        execution_profile_identity=_ENVIRONMENT_EXECUTION_PROFILE_IDENTITY,
    )
    bridge = WebBridge.sandboxed_browser(
        environment=factory,
        browser_image=PINNED_BROWSER_SESSION_WORKLOAD.image,
        interactive=True,
        interactive_options={
            "max_artifact_bytes": (
                DETERMINISTIC_BROWSER_ACCEPTANCE_MAX_ARTIFACT_BYTES_PER_OPERATION
            ),
            "max_operations": 16,
            "max_snapshot_bytes": 64 * 1024,
            "max_sessions": 1,
            "multi_page": True,
            "popup_policy": {
                "mode": "destination_policy",
                "allowed_operations": ["click"],
                "allowed_opener_origins": ["https://docs.browser.test/"],
                "allowed_destination_origins": [
                    "https://docs.browser.test/",
                    "https://static.browser.test/",
                ],
            },
            "max_pages": 4,
            "max_provisional_pages": 2,
            "max_page_creations_per_operation": 2,
            "max_total_page_creations": 8,
            "max_background_lifetime_seconds": 60,
            "max_operations_per_page": 16,
            "max_observations_per_page": 16,
            "max_total_observations": 32,
            "max_refs_per_page": 256,
            "max_total_refs": 512,
            "max_total_requests": 256,
            "max_artifacts_per_page": 4,
            "max_total_artifacts": 8,
            "max_page_cleanup_operations": 16,
        },
    )
    if control is not None and control.scenario.value.startswith("browser_"):
        _install_browser_crash_fault(bridge, control)
    app = CayuApp(
        session_store=store,
        event_sinks=[_AcceptanceEventJournalSink(root / _OBSERVED_EVENTS_FILENAME)],
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_environment_factory(
        EnvironmentSpec(
            name="browser",
            execution_profile_identity=_ENVIRONMENT_EXECUTION_PROFILE_IDENTITY,
        ),
        factory,
        default=True,
    )
    bridge.register_agent(app, AgentSpec(name=_AGENT, model=_MODEL))
    return app, bridge, provider


def _observed_event_identity(event: Event) -> RunnerObservedEventIdentity:
    sequence = event_durable_sequence(event)
    if sequence is None:
        raise RuntimeError("Browser acceptance received an event without a durable sequence.")
    return RunnerObservedEventIdentity(
        session_id=event.session_id,
        sequence=sequence,
        event_type=event.type,
    )


def _append_delivered_event(path: Path, event: Event) -> None:
    encoded = (
        json.dumps(
            {
                "event_id": event.id,
                "event_type": str(event.type),
                "session_id": event.session_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


async def _load_observed_events(
    app: CayuApp,
    path: Path,
    *,
    session_id: str,
) -> list[RunnerObservedEventIdentity]:
    if not path.is_file():
        raise RuntimeError("Browser acceptance process emitted no observable event journal.")
    size = path.stat().st_size
    if size > TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES:
        raise RuntimeError("Browser acceptance event journal exceeds its byte bound.")
    lines = path.read_bytes().splitlines()
    if not lines or len(lines) > TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS:
        raise RuntimeError("Browser acceptance event journal exceeds its event bound.")
    delivered: dict[int, str] = {}
    delivered_order: list[int] = []
    for line in lines:
        try:
            document = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Browser acceptance event journal is malformed.") from exc
        if (
            type(document) is not dict
            or set(document) != {"event_id", "event_type", "session_id"}
            or document["session_id"] != session_id
            or type(document["event_id"]) is not str
            or not document["event_id"]
            or type(document["event_type"]) is not str
        ):
            raise RuntimeError("Browser acceptance event journal is malformed.")
        sequence = public_event_sequence(document["event_id"])
        if sequence is None:
            raise RuntimeError("Browser acceptance event journal is malformed.")
        previous = delivered.get(sequence)
        if previous is None:
            delivered[sequence] = document["event_type"]
            delivered_order.append(sequence)
            continue
        if previous != document["event_type"]:
            raise RuntimeError("Browser acceptance event journal contains a conflict.")
    records = await app.session_store.query_events_bounded(
        EventQuery(session_id=session_id, limit=5000),
        max_bytes=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES,
    )
    if not records or len(records) >= 5000:
        raise RuntimeError("Browser acceptance interrupted event evidence exceeds its bound.")
    durable_sequences = tuple(record.sequence for record in records)
    if tuple(delivered_order) != durable_sequences or any(
        delivered[record.sequence] != str(record.event.type)
        or record.event.session_id != session_id
        for record in records
    ):
        raise RuntimeError(
            "Browser acceptance delivered event evidence conflicts with durable recovery state."
        )
    return [
        RunnerObservedEventIdentity(
            session_id=session_id,
            sequence=record.sequence,
            event_type=record.event.type,
        )
        for record in records
    ]


async def _consume_run(
    app: CayuApp,
    request: RunRequest,
) -> None:
    async for event in app.run(request):
        _observed_event_identity(event)


async def _recover_scenario(app: CayuApp, session_id: str):
    return await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_for_seconds=0,
            reason="browser_acceptance_fault_recovery",
        )
    )


async def _drain_acceptance_event_observations(app: CayuApp) -> None:
    for _ in range(8):
        recovered = await app.recover_persisted_event_side_effects(limit=1000)
        if not recovered:
            return
    raise RuntimeError("Browser acceptance event observation recovery exceeded its bound.")


async def _interrupted_trajectory(
    app: CayuApp,
    session_id: str,
    observed: list[RunnerObservedEventIdentity],
):
    ordered = sorted(observed, key=lambda item: item.sequence or 0)
    if any(
        left.sequence is None or right.sequence is None or left.sequence >= right.sequence
        for left, right in pairwise(ordered)
    ):
        raise RuntimeError("Browser acceptance observed conflicting event identities.")
    limits = TerminalSessionEvidenceLimits()
    evidence = await app.session_store.load_runner_owned_interrupted_evidence(
        session_id,
        observed_events=tuple(ordered),
        limits=limits,
    )
    revalidated = await app.session_store.load_runner_owned_interrupted_evidence(
        session_id,
        observed_events=tuple(ordered),
        limits=TerminalSessionEvidenceLimits(
            max_events=evidence.boundary.event_count,
            max_transcript_records=evidence.boundary.transcript_count,
            max_record_bytes=limits.max_record_bytes,
            max_total_bytes=limits.max_total_bytes,
        ),
    )
    if revalidated != evidence:
        raise RuntimeError("Browser acceptance interrupted evidence changed during capture.")
    return _trajectory_from_terminal_evidence(evidence)


async def _wait_for_process_exit(
    process: BaseProcess,
    timeout_seconds: float,
) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(timeout_seconds, 0.0)
    while process.is_alive():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(_PROCESS_POLL_SECONDS, remaining))
    process.join(timeout=0)
    return True


async def _settle_spawned_process(process: BaseProcess) -> None:
    if process.is_alive():
        process.terminate()
        exited = await _wait_for_process_exit(process, _PROCESS_TERMINATE_GRACE_SECONDS)
        if not exited:
            process.kill()
            exited = await _wait_for_process_exit(process, _PROCESS_KILL_GRACE_SECONDS)
            if not exited:
                raise RuntimeError("Browser acceptance child process did not quiesce.")
    else:
        process.join(timeout=0)


async def _settle_spawned_process_after_failure(
    process: BaseProcess,
    primary: BaseException,
) -> None:
    """Quiesce one child before preserving the owner's authoritative failure."""

    settlement = asyncio.create_task(
        _settle_spawned_process(process),
        name="cayu-browser-acceptance-process-settlement",
    )
    outcome = await await_shielded_task_outcome(
        settlement,
        cancellation=(primary if isinstance(primary, asyncio.CancelledError) else None),
    )
    cancellation = outcome.cancellation
    if cancellation is not None:
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed,
            cancellation=cancellation,
        )
        if outcome.error is not None:
            cancellation.add_note(
                "Browser acceptance child settlement failed while cancellation was pending."
            )
            raise cancellation from outcome.error
        if primary is not cancellation:
            raise cancellation from primary
        raise cancellation
    if outcome.error is not None:
        raise BaseExceptionGroup(
            "Browser acceptance execution and child settlement both failed.",
            [primary, outcome.error],
        ) from None


def _process_scenario_worker(
    root_value: str,
    upstream_routes: dict[str, str],
    hosts: tuple[str, ...],
    case_document: dict[str, Any],
    seccomp_value: str,
    scenario_value: str,
    session_id: str,
) -> None:
    async def execute() -> None:
        from cayu.evals.browser_acceptance import BrowserAcceptanceCaseV1

        root = Path(root_value)
        case = BrowserAcceptanceCaseV1.model_validate(case_document)
        control = _FaultControl(
            BrowserAcceptanceFaultScenario(scenario_value),
            root / "fault.marker",
            len(case.operations),
        )
        app, _, _ = _build_runtime(
            root=root,
            upstream_routes=upstream_routes,
            hosts=hosts,
            cases={case.case_id: case},
            seccomp_profile=Path(seccomp_value),
            control=control,
        )
        await _consume_run(
            app,
            _scenario_request(case, session_id=session_id),
        )
        raise RuntimeError("Browser acceptance process-loss boundary was not reached.")

    asyncio.run(execute())


_ProcessScenarioWorker = Callable[
    [str, dict[str, str], tuple[str, ...], dict[str, Any], str, str, str],
    None,
]


class _DeterministicScenarioExecutor:
    def __init__(
        self,
        *,
        root: Path,
        upstream_routes: dict[str, str],
        hosts: tuple[str, ...],
        cases: dict[str, Any],
        seccomp_profile: Path,
        process_worker: _ProcessScenarioWorker = _process_scenario_worker,
    ) -> None:
        self._root = root
        self._upstream_routes = dict(upstream_routes)
        self._hosts = hosts
        self._cases = cases
        self._seccomp_profile = seccomp_profile
        self._process_worker = process_worker

    async def __call__(
        self,
        case: Any,
        trial_number: int,
        attempt_number: int,
        timeout_seconds: float,
    ) -> BrowserAcceptanceScenarioExecutionV1:
        scenario = case.fault_scenario
        if not isinstance(scenario, BrowserAcceptanceFaultScenario):
            raise TypeError("Browser acceptance scenario is missing.")
        trial_root = self._root / f"{case.case_id}-{trial_number}-{attempt_number}"
        trial_root.mkdir(parents=True, exist_ok=False)
        marker_path = trial_root / "fault.marker"
        session_id = f"browser-acceptance-{case.case_id}-{trial_number}-{attempt_number}"
        started_at = datetime.now(UTC)
        process_loss = scenario.value.startswith("process_")
        interrupted = False
        if process_loss:
            process = multiprocessing.get_context("spawn").Process(
                target=self._process_worker,
                args=(
                    str(trial_root),
                    self._upstream_routes,
                    self._hosts,
                    case.model_dump(mode="json"),
                    str(self._seccomp_profile),
                    scenario.value,
                    session_id,
                ),
            )
            try:
                process.start()
                if not await _wait_for_process_exit(process, min(timeout_seconds, 300.0)):
                    raise TimeoutError("Browser acceptance process-loss scenario timed out.")
                if process.exitcode != _PROCESS_LOSS_EXIT_CODE or not marker_path.is_file():
                    raise RuntimeError("Browser acceptance process-loss boundary was not observed.")
            except BaseException as primary:
                if process.pid is not None:
                    await _settle_spawned_process_after_failure(process, primary)
                raise
            finally:
                if process.pid is not None and not process.is_alive():
                    process.close()
            app, _, _ = _build_runtime(
                root=trial_root,
                upstream_routes=self._upstream_routes,
                hosts=self._hosts,
                cases={case.case_id: case},
                seccomp_profile=self._seccomp_profile,
                control=None,
            )
            recovery = await _recover_scenario(app, session_id)
            interrupted = recovery.status is SessionStatus.INTERRUPTED
        else:
            control = _FaultControl(scenario, marker_path, len(case.operations))
            app, _, _ = _build_runtime(
                root=trial_root,
                upstream_routes=self._upstream_routes,
                hosts=self._hosts,
                cases={case.case_id: case},
                seccomp_profile=self._seccomp_profile,
                control=control,
            )
            run_task = asyncio.create_task(
                _consume_run(
                    app,
                    _scenario_request(case, session_id=session_id),
                )
            )
            try:
                await run_task
            except asyncio.CancelledError:
                if not scenario.value.startswith("cancel_"):
                    raise
                if run_task.cancelling() != 1 or not run_task.cancelled():
                    raise RuntimeError(
                        "Browser acceptance cancellation did not settle its owning task."
                    ) from None
            except ConnectionError:
                if scenario is not BrowserAcceptanceFaultScenario.ACKNOWLEDGEMENT_LOSS:
                    raise
            if not marker_path.is_file():
                raise RuntimeError("Browser acceptance fault boundary was not observed.")
            if scenario.value.startswith("cancel_") or scenario is (
                BrowserAcceptanceFaultScenario.ACKNOWLEDGEMENT_LOSS
            ):
                recovery = await _recover_scenario(app, session_id)
                interrupted = recovery.status is SessionStatus.INTERRUPTED
        await _drain_acceptance_event_observations(app)
        observed_events = await _load_observed_events(
            app,
            trial_root / _OBSERVED_EVENTS_FILENAME,
            session_id=session_id,
        )
        trajectory = (
            await _interrupted_trajectory(app, session_id, observed_events)
            if interrupted
            else await trajectory_from_session(app, session_id)
        )
        completed_at = datetime.now(UTC)
        usage_summary = (
            None
            if trajectory.usage_summary is None
            else trajectory.usage_summary.model_dump(mode="json")
        )
        trial = EvalTrialResult(
            trial_number=trial_number,
            status=EvalStatus.SKIPPED,
            session_id=session_id,
            score=0.0,
            final_output=trajectory.final_output,
            evidence_complete=True,
            events_count=len(trajectory.events),
            usage_summary=usage_summary,
            memory_attribution=eval_memory_attribution_evidence_from_trajectory(trajectory),
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=max(int((completed_at - started_at).total_seconds() * 1000), 0),
            trajectory=trajectory,
        )
        return BrowserAcceptanceScenarioExecutionV1(
            app=app,
            trial=trial,
            fault=BrowserAcceptanceFaultEvidenceV1(
                scenario=scenario,
                boundary_observed=True,
                cancellation_delivered=scenario.value.startswith("cancel_"),
                process_loss_observed=process_loss,
                recovered_in_fresh_app=process_loss,
                browser_dispatches=_browser_dispatches_from_trial(trial),
            ),
        )


async def build(fixture: BrowserAcceptanceFixtureV1) -> BrowserAcceptancePlanV1:
    """Build the exact local/Docker deterministic acceptance plan."""

    if type(fixture) is not BrowserAcceptanceFixtureV1:
        raise TypeError("fixture must be Cayu's exact BrowserAcceptanceFixtureV1.")
    manifest = deterministic_browser_acceptance_manifest()
    executable = tuple(
        case for case in manifest.cases if case.expected_state.value != "unsupported"
    )
    cases = {case.case_id: case for case in executable}
    temporary_directory = TemporaryDirectory(prefix="cayu-browser-acceptance-")
    temporary_root = Path(temporary_directory.name)
    root = Path(__file__).resolve().parents[4]
    seccomp_profile = root / "examples" / "browser_fetch" / "seccomp_profile.json"
    if not seccomp_profile.is_file():
        raise RuntimeError("Pinned browser seccomp profile is unavailable.")
    app, bridge, provider = _build_runtime(
        root=temporary_root / "ordinary",
        upstream_routes=fixture.upstream_routes,
        hosts=fixture.hosts,
        cases=cases,
        seccomp_profile=seccomp_profile,
        control=None,
    )
    provider._acceptance_temporary_directory = temporary_directory
    suite = EvalSuite(
        id=manifest.suite_id,
        cases=[
            EvalCase(
                id=case.case_id,
                request=RunRequest(
                    agent_name=_AGENT,
                    messages=[Message.text("user", f"browser acceptance case: {case.case_id}")],
                    max_steps=len(case.operations) + 1,
                    limits=RunLimits(
                        max_tool_calls=len(case.operations),
                        max_elapsed_seconds=max(
                            1, (manifest.limits.max_wall_time_ms + 999) // 1_000
                        ),
                    ),
                ),
                assertions=[SessionCompleted()],
                metadata={"browser_acceptance_case_revision": case.revision},
            )
            for case in executable
        ],
    )
    return BrowserAcceptancePlanV1(
        manifest=manifest,
        eval_plan=EvalPlan(app=app, suite=suite),
        bridge=bridge,
        scenario_executor_revision=_SCENARIO_EXECUTOR_REVISION,
        scenario_executor=_DeterministicScenarioExecutor(
            root=temporary_root / "scenarios",
            upstream_routes=fixture.upstream_routes,
            hosts=fixture.hosts,
            cases=cases,
            seccomp_profile=seccomp_profile,
        ),
    )


__all__ = ["BrowserAcceptanceDeterministicProvider", "build"]
