"""Per-operation overhead budgets, pinned as counts rather than wall-clock time.

Each tool call, sandbox setup, model step and tool round costs a number of sandbox
commands, broker requests, bytes sent and session-store calls. Those numbers are
deterministic, so CI can hold them to a budget without timing anything: a change
that adds work to a hot path fails here, and the failure lists what ran.

The brokered measurements use the in-memory fake broker, whose commands run as
real processes, and the real coding factory and binding. The session measurement
runs a scripted provider through three model steps against an in-memory store and
counts the store calls the Runtime makes between model and tool boundaries.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import shutil
from collections import Counter
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.environments import EnvironmentFactoryRequest
from cayu.environments import bindings as bindings_module
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.providers.base import ModelStreamEventType
from cayu.runtime import _session_control
from cayu.sandbox_broker import BrokeredSandboxCodingEnvironmentFactory
from cayu.sessions import InMemorySessionStore
from cayu.tools import (
    EditFileTool,
    ExecCommandTool,
    ListFilesTool,
    ReadFileTool,
    SearchTextTool,
    WriteFileTool,
)
from cayu.tools.base import ToolContext
from cayu.workspaces import LocalWorkspace
from tests.sandbox_broker.fake_broker import FakeBroker

if TYPE_CHECKING:
    import httpx

# Budgets -------------------------------------------------------------------
#
# Every number is a measured count and its ceiling. Lowering one is the goal: when
# a change makes an operation cheaper, lower its number here in the same change so
# the gain can't quietly come back. A count below its budget fails too, naming the
# number to lower it to. Raising a number needs a reason in the pull request.
#
# sandbox_commands  commands started in the sandbox
# broker_requests   HTTP requests to the broker (each is a round trip)
# command_kib       bytes of command start requests, argv included, in KiB rounded up
# archive_uploads   tar archives sent to the sandbox; upload_kib is their size
# target_cleans     passes that clear the target before copy-in
# store_calls       session-store calls the Runtime awaits (not the store's own
#                   nested calls)
# events            events the session stores
BUDGETS: dict[str, dict[str, int]] = {
    # One tool call on a bound brokered coding sandbox.
    "read_file whole": {"sandbox_commands": 1, "broker_requests": 2, "command_kib": 27},
    "read_file byte page": {"sandbox_commands": 1, "broker_requests": 2, "command_kib": 27},
    "read_file line range": {"sandbox_commands": 3, "broker_requests": 6, "command_kib": 81},
    "write_file create": {"sandbox_commands": 1, "broker_requests": 2, "command_kib": 27},
    "edit_file": {"sandbox_commands": 2, "broker_requests": 4, "command_kib": 56},
    "search_text": {"sandbox_commands": 1, "broker_requests": 2, "command_kib": 1},
    "search_text with path": {"sandbox_commands": 2, "broker_requests": 4, "command_kib": 2},
    "search_text with glob": {"sandbox_commands": 2, "broker_requests": 4, "command_kib": 2},
    "list_files": {"sandbox_commands": 1, "broker_requests": 2, "command_kib": 54},
    "exec_command": {"sandbox_commands": 1, "broker_requests": 2, "command_kib": 1},
    # Setup of the generated source tree (_SOURCE_FILES files, _SOURCE_BYTES bytes),
    # then copy-back of the two files the tool calls changed.
    "sandbox allocation": {"sandbox_commands": 0, "broker_requests": 2},
    "sandbox setup": {
        "sandbox_commands": 18,
        "broker_requests": 37,
        "command_kib": 114,
        "archive_uploads": 1,
        "upload_kib": 460,
        "target_cleans": 1,
    },
    "sandbox copy-back": {"sandbox_commands": 3, "broker_requests": 8, "command_kib": 161},
    # A session of three model steps: two tool calls, then one, then a reply.
    "session start to model request 1": {"store_calls": 41},
    "model reply 1 to first tool start": {"store_calls": 23},
    "between tool calls in round 1": {"store_calls": 15},
    "round 1 last tool result to model request 2": {"store_calls": 47},
    "model reply 2 to first tool start": {"store_calls": 23},
    "round 2 last tool result to model request 3": {"store_calls": 42},
    "model reply 3 to session end": {"store_calls": 37},
    "during the model calls": {"store_calls": 3},
    # Events stored, split where each model call starts (its model.started event).
    "events before model call 1": {"events": 4},
    "events from model call 1 to 2": {"events": 8},
    "events from model call 2 to 3": {"events": 6},
    "events from model call 3 to session end": {"events": 6},
}

_SOURCE_FILES = 301
_SOURCE_BYTES = 301_692


def _check_budgets(observed: Mapping[str, Mapping[str, int]], breakdown: Mapping[str, Any]) -> None:
    over: list[str] = []
    under: list[str] = []
    for name, counts in observed.items():
        budget = BUDGETS[name]
        assert set(counts) == set(budget), (name, sorted(counts), sorted(budget))
        for metric, ceiling in budget.items():
            if counts[metric] > ceiling:
                over.append(
                    f"  {name}: {metric} {counts[metric]} > {ceiling}\n"
                    f"    observed {dict(counts)}\n    breakdown {breakdown[name]}"
                )
            elif counts[metric] < ceiling:
                under.append(f"  {name}: {metric} is {counts[metric]}, budget {ceiling}")
    message = ""
    if over:
        message += (
            "Overhead above budget. If the extra work is needed, raise the number in "
            "BUDGETS and give the reason in the pull request:\n" + "\n".join(over) + "\n"
        )
    if under:
        message += (
            "Overhead below budget. Lower these numbers in BUDGETS to keep the gain:\n"
            + "\n".join(under)
        )
    assert not message, message


def _kib(size: int) -> int:
    return math.ceil(size / 1024)


# Brokered sandbox -----------------------------------------------------------


def _source_tree(root: Path) -> Path:
    """A fixed project: one 200-line text file and 300 small modules in three packages."""

    source = root / "source"
    source.mkdir()
    (source / "notes.txt").write_text("".join(f"line {number}\n" for number in range(1, 201)))
    for index in range(300):
        package = source / f"pkg{index % 3}"
        package.mkdir(exist_ok=True)
        (package / f"m{index:03}.py").write_bytes(b"x = %03d\n" % index + b"#" * 991 + b"\n")
    files = [path for path in source.rglob("*") if path.is_file()]
    assert (len(files), sum(path.stat().st_size for path in files)) == (
        _SOURCE_FILES,
        _SOURCE_BYTES,
    )
    return source


def _command_label(argv: list[str]) -> str:
    if len(argv) > 3 and argv[1] == "-c" and len(argv[2]) > 4096:
        # The workspace operations ship a whole Python program as an argument.
        return f"{Path(argv[0]).name} -c <{_kib(len(argv[2].encode()))} KiB program> {argv[3]}"
    return " ".join([Path(argv[0]).name, *(item[:24] for item in argv[1:3])])


def _request_label(request: httpx.Request) -> str:
    parts = request.url.path.split("/sandboxes", 1)[-1].strip("/").split("/")
    # Keep the action, without the sandbox and command ids.
    action = "/".join(part for part in parts[1:] if not part.startswith("cayu-"))
    return f"{request.method} {action or 'sandbox'}"


class _BrokerMeter:
    """What the broker saw between two points: commands, requests and uploads."""

    def __init__(self, broker: FakeBroker) -> None:
        self.broker = broker
        self.commands = 0
        self.requests = 0

    def mark(self) -> None:
        self.commands = len(self.broker.started)
        self.requests = len(self.broker.requests)

    def take(self, *, archives: bool = False) -> tuple[dict[str, int], dict[str, Any]]:
        commands = self.broker.started[self.commands :]
        requests = self.broker.requests[self.requests :]
        starts = [
            request
            for request in requests
            if request.method == "POST" and request.url.path.endswith("/commands")
        ]
        counts = {"sandbox_commands": len(commands), "broker_requests": len(requests)}
        if starts:
            counts["command_kib"] = _kib(sum(len(request.content) for request in starts))
        breakdown: dict[str, Any] = {
            "commands": dict(Counter(_command_label(argv) for argv in commands)),
            "requests": dict(Counter(_request_label(request) for request in requests)),
        }
        if archives:
            uploads = [request for request in requests if request.method == "PUT"]
            counts["archive_uploads"] = len(uploads)
            counts["upload_kib"] = _kib(sum(len(request.content) for request in uploads))
            breakdown["upload_bytes"] = [len(request.content) for request in uploads]
        self.mark()
        return counts, breakdown


def _tool_calls(*, with_search: bool) -> list[tuple[str, Tool, dict[str, Any]]]:
    calls: list[tuple[str, Tool, dict[str, Any]]] = [
        ("read_file whole", ReadFileTool(), {"path": "notes.txt"}),
        (
            "read_file byte page",
            ReadFileTool(),
            {"path": "notes.txt", "offset": 10, "max_bytes": 100},
        ),
        (
            "read_file line range",
            ReadFileTool(),
            {"path": "notes.txt", "start_line": 5, "end_line": 9},
        ),
        (
            "write_file create",
            WriteFileTool(),
            {"path": "added.txt", "content": "added\n", "mode": "create"},
        ),
        # The revision comes from the whole read above.
        (
            "edit_file",
            EditFileTool(),
            {"path": "notes.txt", "edits": [{"old_text": "line 7\n", "new_text": "line seven\n"}]},
        ),
    ]
    if with_search:
        calls += [
            ("search_text", SearchTextTool(), {"pattern": "line 1"}),
            ("search_text with path", SearchTextTool(), {"pattern": "x = 1", "path": "pkg1"}),
            ("search_text with glob", SearchTextTool(), {"pattern": "x = 1", "glob": "*.py"}),
        ]
    calls += [
        ("list_files", ListFilesTool(), {"pattern": "**"}),
        ("exec_command", ExecCommandTool(), {"kind": "process", "argv": ["echo", "ok"]}),
    ]
    return calls


async def _measure_brokered_session(
    root: Path, *, with_search: bool, clean_passes: list[int]
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    broker = FakeBroker(root / "cloud")
    source = _source_tree(root)
    factory = BrokeredSandboxCodingEnvironmentFactory(
        broker=broker.client(),
        source_workspace=LocalWorkspace(source, workspace_id="project"),
        egress="none",
    )
    observed: dict[str, dict[str, int]] = {}
    breakdown: dict[str, Any] = {}
    meter = _BrokerMeter(broker)

    created = await factory.create(
        EnvironmentFactoryRequest(
            session_id="coding", agent_name="coder", environment_name="sandbox"
        )
    )
    observed["sandbox allocation"], breakdown["sandbox allocation"] = meter.take()
    binding = created.environment.binding
    bound = await binding.bind(
        created.environment.workspace,
        created.environment.runner,
        session_id="coding",
        agent_name="coder",
        environment_name="sandbox",
    )
    setup, breakdown["sandbox setup"] = meter.take(archives=True)
    sync = bound.metadata["sync_binding"]
    assert (sync["copied_files"], sync["copied_bytes"]) == (_SOURCE_FILES, _SOURCE_BYTES)
    setup["target_cleans"] = len(clean_passes)
    breakdown["sandbox setup"]["clean"] = {
        "clean_target": sync["clean_target"],
        "cleaned_target_files": sync["cleaned_target_files"],
    }
    observed["sandbox setup"] = setup

    context = ToolContext(session_id="coding", runner=bound.runner, workspace=bound.workspace)
    revision = None
    for name, tool, arguments in _tool_calls(with_search=with_search):
        if name == "edit_file":
            arguments = {**arguments, "expected_revision": revision}
        result = await tool.run(context, arguments)
        assert not result.is_error, (name, result.content)
        if name == "read_file whole":
            revision = result.structured["revision"]
        observed[name], breakdown[name] = meter.take()

    snapshot = await binding.finalize(bound, outcome="completed")
    assert snapshot is not None and snapshot.metadata["copied_files"] == 2
    observed["sandbox copy-back"], breakdown["sandbox copy-back"] = meter.take()
    assert broker.live() == []
    return observed, breakdown


_HAS_RIPGREP = shutil.which("rg") is not None


def test_tool_calls_and_sandbox_setup_stay_within_their_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Count the passes that clear the target, not the policy that allows them.
    clean_passes: list[int] = []
    clear_workspace = bindings_module._clear_workspace

    async def counted_clear(*args: Any, **kwargs: Any) -> tuple[str, ...]:
        cleaned = await clear_workspace(*args, **kwargs)
        clean_passes.append(len(cleaned))
        return cleaned

    monkeypatch.setattr(bindings_module, "_clear_workspace", counted_clear)
    observed, breakdown = asyncio.run(
        _measure_brokered_session(tmp_path, with_search=_HAS_RIPGREP, clean_passes=clean_passes)
    )
    _check_budgets(observed, breakdown)
    if not _HAS_RIPGREP:
        pytest.skip("The search_text budgets need ripgrep; every other budget was checked.")


# Session store --------------------------------------------------------------

_MODEL_REQUEST = "model request"
_MODEL_REPLY = "model reply"
_TOOL = "tool"


class _TimedProvider(ScriptedModelProvider):
    """Marks when each model request starts and when its reply is complete."""

    def __init__(self, timeline: list[tuple[str, str]], batches: list[list[ModelStreamEvent]]):
        super().__init__(batches)
        self.timeline = timeline

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.timeline.append(("mark", _MODEL_REQUEST))
        async for event in super().stream(request):
            if event.type == ModelStreamEventType.COMPLETED:
                self.timeline.append(("mark", _MODEL_REPLY))
            yield event


class _MarkedTool(Tool):
    # Sequential, so each round's store calls come in one order.
    spec = ToolSpec(
        name="note",
        parallel_safe=False,
        max_terminal_payload_bytes=1024,
        input_schema={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
    )

    def __init__(self, timeline: list[tuple[str, str]]) -> None:
        super().__init__()
        self.timeline = timeline

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.timeline.append(("mark", _TOOL))
        return ToolResult(content=f"noted {args['index']}")


@contextmanager
def _record_store_calls(
    store: InMemorySessionStore,
    timeline: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Record each public store call the Runtime awaits, in order, on ``timeline``.

    The class is patched, as the store's own instance methods must stay as they are;
    calls the store makes to itself while serving one are not counted.
    """

    depth: ContextVar[int] = ContextVar("store_call_depth", default=0)

    def recorded(name: str, method: Any) -> Any:
        @wraps(method)
        async def call(self: Any, *args: Any, **kwargs: Any) -> Any:
            if self is not store:
                return await method(self, *args, **kwargs)
            if depth.get() == 0:
                timeline.append(("store", name))
            token = depth.set(depth.get() + 1)
            try:
                return await method(self, *args, **kwargs)
            finally:
                depth.reset(token)

        return call

    with monkeypatch.context() as patch:
        store_type = type(store)
        for name in dir(store_type):
            method = getattr(store_type, name)
            if not name.startswith("_") and inspect.iscoroutinefunction(method):
                patch.setattr(store_type, name, recorded(name, method))
        yield


_SEGMENTS = (
    "session start to model request 1",
    "model reply 1 to first tool start",
    "between tool calls in round 1",
    "round 1 last tool result to model request 2",
    "model reply 2 to first tool start",
    "round 2 last tool result to model request 3",
    "model reply 3 to session end",
)


def _store_segments(timeline: list[tuple[str, str]]) -> tuple[list[list[str]], list[str]]:
    """Split the store calls at the marks: the spans between steps, and those in model calls."""

    spans: list[tuple[str, str, list[str]]] = []
    calls: list[str] = []
    previous = "start"
    for kind, name in timeline:
        if kind == "store":
            calls.append(name)
        else:
            spans.append((previous, name, calls))
            calls, previous = [], name
    spans.append((previous, "end", calls))
    in_model = [
        call
        for start, end, calls in spans
        if (start, end) == (_MODEL_REQUEST, _MODEL_REPLY)
        for call in calls
    ]
    between = [
        calls for start, end, calls in spans if (start, end) != (_MODEL_REQUEST, _MODEL_REPLY)
    ]
    return between, in_model


def test_session_store_calls_per_model_step_and_tool_round_stay_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline: list[tuple[str, str]] = []
    batches = [
        [
            ModelStreamEvent.tool_call(id="call-1", name="note", arguments={"index": 1}),
            ModelStreamEvent.tool_call(id="call-2", name="note", arguments={"index": 2}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
        [
            ModelStreamEvent.tool_call(id="call-3", name="note", arguments={"index": 3}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
    ]

    async def scenario() -> list[Any]:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_TimedProvider(timeline, batches), default=True)
        app.register_agent(
            AgentSpec(name="worker", model="scripted-model"), tools=[_MarkedTool(timeline)]
        )
        request = RunRequest(
            session_id="budget",
            agent_name="worker",
            max_steps=3,
            messages=[Message.text("user", "take three notes")],
        )
        with _record_store_calls(store, timeline, monkeypatch):
            last = [event.type async for event in app.run(request)][-1]
        assert last == EventType.SESSION_COMPLETED
        return await store.load_events("budget")

    # The stream's interrupt poll reloads the session once its real-time interval
    # expires; a descheduled worker would add a load, so the interval never expires here.
    monkeypatch.setattr(_session_control, "STREAM_INTERRUPT_POLL_INTERVAL_S", float("inf"))
    events = asyncio.run(scenario())
    marks = [name for kind, name in timeline if kind == "mark"]
    assert marks == [
        *(_MODEL_REQUEST, _MODEL_REPLY, _TOOL, _TOOL),
        *(_MODEL_REQUEST, _MODEL_REPLY, _TOOL),
        *(_MODEL_REQUEST, _MODEL_REPLY),
    ]
    segments, in_model = _store_segments(timeline)
    observed: dict[str, dict[str, int]] = {"during the model calls": {"store_calls": len(in_model)}}
    breakdown: dict[str, Any] = {"during the model calls": dict(Counter(in_model))}
    for name, calls in zip(_SEGMENTS, segments, strict=True):
        observed[name] = {"store_calls": len(calls)}
        breakdown[name] = dict(Counter(calls))

    steps: list[list[str]] = [[]]
    for event in events:
        if event.type == EventType.MODEL_STARTED:
            steps.append([])
        steps[-1].append(str(event.type))
    step_names = (
        "events before model call 1",
        "events from model call 1 to 2",
        "events from model call 2 to 3",
        "events from model call 3 to session end",
    )
    for name, types in zip(step_names, steps, strict=True):
        observed[name] = {"events": len(types)}
        breakdown[name] = dict(Counter(types))
    _check_budgets(observed, breakdown)
