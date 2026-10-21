"""Built-in tools declare an honest terminal payload bound and keep results inside it."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable

import pytest

from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runners.local import LocalRunner
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tools._terminal_bounds import (
    MAX_TOOL_TERMINAL_PAYLOAD_BYTES,
    MIN_TOOL_TERMINAL_PAYLOAD_BYTES,
    TERMINAL_PAYLOAD_TRUNCATION_FIELD,
    bounded_terminal_result,
    fit_tool_result,
    tool_result_payload_bytes,
)
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.child_sessions import ChildSessionResultTool
from cayu.tools.commands import ExecCommandTool
from cayu.tools.discovery import SearchToolsTool
from cayu.tools.files import (
    DeleteFileTool,
    EditFileTool,
    ListArtifactsTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from cayu.tools.git import GitChangesTool
from cayu.tools.knowledge import (
    ListKnowledgeTool,
    ReadKnowledgeTool,
    RememberKnowledgeTool,
    SearchKnowledgeTool,
)
from cayu.tools.patches import ApplyPatchTool
from cayu.tools.search import SearchTextTool
from cayu.tools.subagents import (
    SubagentExecutionMode,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
)
from cayu.tools.user_input import UserInputTool
from cayu.tools.web import WebFetchTool, WebSearchTool
from cayu.vaults import SecretRedactor
from cayu.workspaces.local import LocalWorkspace

KIB = 1024


class _SearchAdapter:
    def __init__(self, result: ToolResult) -> None:
        self.result = result

    async def search(self, ctx, request) -> ToolResult:
        return self.result


def _subagent_tool(bound: int | None) -> SubagentTool:
    return SubagentTool(
        CayuApp(session_store=InMemorySessionStore(), enable_logging=False),
        agents={"child": SubagentSpec(agent_name="child", mode=SubagentExecutionMode.BACKGROUND)},
        max_terminal_payload_bytes=bound,
    )


FACTORIES: dict[str, Callable[[int | None], Tool]] = {
    "exec_command": lambda bound: ExecCommandTool(max_terminal_payload_bytes=bound),
    "read_file": lambda bound: ReadFileTool(max_terminal_payload_bytes=bound),
    "write_file": lambda bound: WriteFileTool(max_terminal_payload_bytes=bound),
    "edit_file": lambda bound: EditFileTool(max_terminal_payload_bytes=bound),
    "delete_file": lambda bound: DeleteFileTool(max_terminal_payload_bytes=bound),
    "list_files": lambda bound: ListFilesTool(max_terminal_payload_bytes=bound),
    "list_artifacts": lambda bound: ListArtifactsTool(max_terminal_payload_bytes=bound),
    "apply_patch": lambda bound: ApplyPatchTool(max_terminal_payload_bytes=bound),
    "git_changes": lambda bound: GitChangesTool(max_terminal_payload_bytes=bound),
    "search_text": lambda bound: SearchTextTool(max_terminal_payload_bytes=bound),
    "search_knowledge": lambda bound: SearchKnowledgeTool(max_terminal_payload_bytes=bound),
    "remember_knowledge": lambda bound: RememberKnowledgeTool(max_terminal_payload_bytes=bound),
    "list_knowledge": lambda bound: ListKnowledgeTool(max_terminal_payload_bytes=bound),
    "read_knowledge": lambda bound: ReadKnowledgeTool(max_terminal_payload_bytes=bound),
    "web_fetch": lambda bound: WebFetchTool(max_terminal_payload_bytes=bound),
    "web_search": lambda bound: WebSearchTool(
        adapter=_SearchAdapter(ToolResult()), max_terminal_payload_bytes=bound
    ),
    "subagent": _subagent_tool,
    "subagent_result": lambda bound: SubagentResultTool(
        InMemorySessionStore(), max_terminal_payload_bytes=bound
    ),
    "child_session_result": lambda bound: ChildSessionResultTool(
        InMemorySessionStore(), max_terminal_payload_bytes=bound
    ),
    "ask_user": lambda bound: UserInputTool(max_terminal_payload_bytes=bound),
}


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_builtin_tools_declare_a_storable_bound_by_default(name: str) -> None:
    bound = FACTORIES[name](None).spec.max_terminal_payload_bytes

    assert bound is not None
    assert MIN_TOOL_TERMINAL_PAYLOAD_BYTES <= bound <= MAX_TOOL_TERMINAL_PAYLOAD_BYTES


def test_framework_registered_search_tools_declares_a_storable_bound() -> None:
    bound = SearchToolsTool().spec.max_terminal_payload_bytes

    assert bound is not None and bound <= MAX_TOOL_TERMINAL_PAYLOAD_BYTES


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_applications_set_a_builtin_bound_at_construction(name: str) -> None:
    assert FACTORIES[name](96 * KIB).spec.max_terminal_payload_bytes == 96 * KIB


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (MIN_TOOL_TERMINAL_PAYLOAD_BYTES - 1, ValueError),
        (MAX_TOOL_TERMINAL_PAYLOAD_BYTES + 1, ValueError),
        (True, TypeError),
        (128 * KIB + 0.5, TypeError),
    ],
)
def test_builtin_bounds_outside_what_cayu_can_store_are_refused(value, error) -> None:
    with pytest.raises(error, match="max_terminal_payload_bytes"):
        ExecCommandTool(max_terminal_payload_bytes=value)


def test_a_bound_declared_on_an_explicit_spec_is_kept() -> None:
    spec = ExecCommandTool.spec.model_copy(update={"max_terminal_payload_bytes": 80 * KIB})

    assert ExecCommandTool(spec).spec.max_terminal_payload_bytes == 80 * KIB
    assert (
        ExecCommandTool(spec, max_terminal_payload_bytes=70 * KIB).spec.max_terminal_payload_bytes
        == 70 * KIB
    )


def test_fit_returns_a_result_within_its_bound_unchanged() -> None:
    result = ToolResult(content="ok", structured={"exit_code": 0})

    assert fit_tool_result(result, 64 * KIB) is result


def test_fit_cuts_the_longest_strings_and_keeps_every_field() -> None:
    output = "line\n" * 60_000 + "\x01" * 40_000
    result = ToolResult(
        content="Exit code 1\n" + output,
        structured={
            "exit_code": 1,
            "stdout": output,
            "output_sha256": "a" * 64,
            "paths": ["src/app.py", "tests/test_app.py"],
        },
        is_error=True,
    )

    fitted = fit_tool_result(result, 128 * KIB)

    assert tool_result_payload_bytes(fitted) <= 128 * KIB
    assert fitted.is_error is True
    assert fitted.structured["exit_code"] == 1
    assert fitted.structured["output_sha256"] == "a" * 64
    assert fitted.structured["paths"] == ["src/app.py", "tests/test_app.py"]
    assert fitted.structured["stdout"].startswith("line\n")
    assert "bytes truncated" in fitted.structured["stdout"]
    assert fitted.content.startswith("Exit code 1\nline\n")
    assert "[Result truncated: the complete result was" in fitted.content
    assert fitted.structured[TERMINAL_PAYLOAD_TRUNCATION_FIELD] == {
        "original_bytes": tool_result_payload_bytes(result),
        "max_bytes": 128 * KIB,
    }


def test_fit_reduces_structured_data_that_has_too_many_entries() -> None:
    result = ToolResult(
        content="100000 entries",
        structured={"entries": [{"path": f"f{index}"} for index in range(100_000)], "total": 7},
    )

    fitted = fit_tool_result(result, 64 * KIB)

    assert tool_result_payload_bytes(fitted) <= 64 * KIB
    assert fitted.content.startswith("100000 entries\n[Result truncated")
    assert fitted.structured["total"] == 7
    assert "entries" not in fitted.structured
    assert fitted.structured[TERMINAL_PAYLOAD_TRUNCATION_FIELD]["structured_fields_omitted"] == [
        "entries"
    ]


def test_bounded_terminal_result_fits_what_run_returns() -> None:
    class Loud(Tool):
        spec = ToolSpec(name="loud", max_terminal_payload_bytes=64 * KIB)

        @bounded_terminal_result
        async def run(self, ctx, args):
            return ToolResult(content="x" * (256 * KIB))

    result = asyncio.run(Loud().run(ToolContext(session_id="s"), {}))

    assert tool_result_payload_bytes(result) <= 64 * KIB
    assert "[Result truncated" in result.content


def test_adapter_backed_web_search_results_are_fitted() -> None:
    huge = ToolResult(content="snippet " * 200_000, structured={"query": "cayu"})
    tool = WebSearchTool(adapter=_SearchAdapter(huge), max_terminal_payload_bytes=64 * KIB)

    result = asyncio.run(tool.run(ToolContext(session_id="s"), {"query": "cayu"}))

    assert tool_result_payload_bytes(result) <= 64 * KIB
    assert result.structured["query"] == "cayu"


def test_read_file_reads_a_smaller_page_when_escaped_text_outgrows_its_bound(tmp_path) -> None:
    # Escape characters are allowed text and cost six bytes each in JSON.
    text = "a\x1b" * 120_000
    (tmp_path / "escaped.txt").write_text(text, encoding="utf-8")
    ctx = ToolContext(session_id="s", workspace=LocalWorkspace(tmp_path, workspace_id="local"))
    tool = ReadFileTool(max_terminal_payload_bytes=128 * KIB)

    first = asyncio.run(tool.run(ctx, {"path": "escaped.txt", "max_bytes": 200_000}))

    assert first.is_error is False
    assert tool_result_payload_bytes(first) <= 128 * KIB
    assert TERMINAL_PAYLOAD_TRUNCATION_FIELD not in first.structured
    assert first.structured["truncated"] is True
    next_offset = first.structured["next_offset"]
    assert 0 < next_offset == first.structured["bytes"]
    second = asyncio.run(
        tool.run(ctx, {"path": "escaped.txt", "max_bytes": 200_000, "offset": next_offset})
    )
    assert second.structured["offset"] == next_offset
    assert tool_result_payload_bytes(second) <= 128 * KIB


async def _seed_child(store, *, child_id: str, text: str, mode: str = "background") -> None:
    await store.create(
        RunRequest(
            agent_name="reviewer",
            session_id=child_id,
            parent_session_id="parent",
            messages=[Message.text("user", "child task")],
            metadata={"subagent": {"agent": "reviewer", "mode": mode}},
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    await store.append_transcript_messages(
        child_id, [Message.text("user", "child task"), Message.text("assistant", text)]
    )
    await store.append_events(
        child_id,
        [
            Event(type=EventType.SESSION_STARTED, session_id=child_id),
            Event(type=EventType.SESSION_COMPLETED, session_id=child_id),
        ],
    )
    await store.update_status(child_id, SessionStatus.COMPLETED)


async def _store_with_children(texts: dict[str, str]) -> InMemorySessionStore:
    store = InMemorySessionStore()
    await store.create(
        RunRequest(agent_name="parent", session_id="parent", messages=[Message.text("user", "go")]),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    for child_id, text in texts.items():
        await _seed_child(store, child_id=child_id, text=text)
    return store


def test_subagent_result_all_returns_what_fits_and_continues_from_next_offset() -> None:
    texts = {f"child-{index}": f"{index}" * 40_000 for index in range(5)}

    async def run() -> list[ToolResult]:
        store = await _store_with_children(texts)
        tool = SubagentResultTool(store, max_terminal_payload_bytes=128 * KIB)
        pages = []
        args = {"all": True, "wait": False, "max_chars": 50_000}
        while True:
            page = await tool.run(ToolContext(session_id="parent"), args)
            pages.append(page)
            next_offset = page.structured.get("next_offset")
            if next_offset is None:
                return pages
            args = {**args, "offset": next_offset}

    pages = asyncio.run(run())

    returned = [
        child["child_session_id"] for page in pages for child in page.structured["children"]
    ]
    assert returned == list(texts)
    assert len(pages) > 1
    for page in pages:
        assert page.is_error is False
        assert tool_result_payload_bytes(page) <= 128 * KIB
        assert TERMINAL_PAYLOAD_TRUNCATION_FIELD not in page.structured
    first = pages[0].structured
    assert first["offset"] == 0
    assert first["total_children"] == 5
    assert first["remaining_child_session_ids"] == returned[first["next_offset"] :]
    assert f"offset={first['next_offset']}" in pages[0].content
    for page in pages:
        for child in page.structured["children"]:
            assert child["result_text"] == texts[child["child_session_id"]]


def test_subagent_result_all_shrinks_a_single_child_larger_than_its_bound() -> None:
    texts = {"big": "z" * 150_000, "small": "done"}

    async def run() -> ToolResult:
        store = await _store_with_children(texts)
        tool = SubagentResultTool(store, max_terminal_payload_bytes=64 * KIB)
        return await tool.run(
            ToolContext(session_id="parent"), {"all": True, "wait": False, "max_chars": 200_000}
        )

    page = asyncio.run(run())

    assert tool_result_payload_bytes(page) <= 64 * KIB
    [child] = page.structured["children"]
    assert child["child_session_id"] == "big"
    assert child["result_truncated"] is True
    assert texts["big"].startswith(child["result_text"])
    assert page.structured["next_offset"] == 1
    assert page.structured["remaining_child_session_ids"] == ["small"]


def test_subagent_result_all_without_paging_keeps_its_original_shape() -> None:
    async def run() -> ToolResult:
        store = await _store_with_children({"a": "first", "b": "second"})
        return await SubagentResultTool(store).run(
            ToolContext(session_id="parent"), {"all": True, "wait": False}
        )

    page = asyncio.run(run())

    assert set(page.structured) == {"parent_session_id", "retrieval_status", "children"}
    assert page.content.splitlines()[1:] == [
        "- reviewer (a, completed): first",
        "- reviewer (b, completed): second",
    ]


def test_subagent_result_offset_requires_all() -> None:
    async def run() -> ToolResult:
        store = await _store_with_children({"a": "first"})
        return await SubagentResultTool(store).run(
            ToolContext(session_id="parent"), {"child_session_id": "a", "offset": 1}
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert "offset is only valid with all=true" in result.content


def test_fit_keeps_only_as_many_short_fields_as_the_budget_allows() -> None:
    adapter = _SearchAdapter(
        ToolResult(content="results", structured={f"item{i}": "x" * 400 for i in range(200)})
    )
    tool = WebSearchTool(adapter=adapter, max_terminal_payload_bytes=64 * KIB)

    result = asyncio.run(tool.run(ToolContext(session_id="s"), {"query": "test"}))

    assert tool_result_payload_bytes(result) <= 64 * KIB
    assert result.content.startswith("results\n[Result truncated")
    note = result.structured[TERMINAL_PAYLOAD_TRUNCATION_FIELD]
    assert len(note["structured_fields_omitted"]) == 64
    assert 0 < len(result.structured) - 1 < 200


def test_fitting_redacts_registered_secrets_before_cutting() -> None:
    secret = "test-api-key-0123456789abcdef0123456789abcdef"
    captured: list[int] = []
    # Pad so a plain cut would land inside the secret.
    text = "x" * 32_000 + secret + "y" * 200_000
    adapter = _SearchAdapter(ToolResult(content=text))
    tool = WebSearchTool(adapter=adapter, max_terminal_payload_bytes=64 * KIB)
    ctx = ToolContext(
        session_id="s",
        invocation_secret_redactor=lambda: SecretRedactor([secret]),
        invocation_secret_capture_observer=captured.append,
    )

    result = asyncio.run(tool.run(ctx, {"query": "test"}))

    assert tool_result_payload_bytes(result) <= 64 * KIB
    assert secret[:8] not in result.content
    assert captured == [0]


def test_exec_command_cuts_encoded_output_as_a_marked_preview(tmp_path) -> None:
    ctx = ToolContext(session_id="s", runner=LocalRunner(tmp_path))
    tool = ExecCommandTool(max_terminal_payload_bytes=128 * KIB)

    result = asyncio.run(
        tool.run(
            ctx,
            {
                "kind": "process",
                "argv": [sys.executable, "-c", "import sys; sys.stdout.write('\\0' * 80000)"],
                "max_output_bytes": 100_000,
            },
        )
    )

    assert tool_result_payload_bytes(result) <= 128 * KIB
    assert result.structured["stdout_encoding"] == "json-string-preview"
    assert result.structured["stdout_truncated"] is True
    assert result.structured["stdout"].startswith('"\\u0000')
    assert TERMINAL_PAYLOAD_TRUNCATION_FIELD not in result.structured


def test_exec_command_keeps_the_start_and_end_of_long_text_output(tmp_path) -> None:
    ctx = ToolContext(session_id="s", runner=LocalRunner(tmp_path))
    tool = ExecCommandTool(max_terminal_payload_bytes=64 * KIB)
    script = "print('first'); print('\"' * 150000); print('last')"

    result = asyncio.run(
        tool.run(
            ctx,
            {
                "kind": "process",
                "argv": [sys.executable, "-c", script],
                "max_output_bytes": 200_000,
            },
        )
    )

    assert tool_result_payload_bytes(result) <= 64 * KIB
    assert result.structured["stdout_encoding"] == "text"
    assert result.structured["stdout_truncated"] is True
    assert result.structured["stdout"].startswith("first\n")
    assert result.structured["stdout"].rstrip().endswith("last")
    assert "bytes truncated" in result.structured["stdout"]


def test_read_file_points_to_byte_paging_when_its_bound_caps_a_long_line(tmp_path) -> None:
    (tmp_path / "bundle.min.js").write_text("a" * 200_000, encoding="utf-8")
    ctx = ToolContext(session_id="s", workspace=LocalWorkspace(tmp_path, workspace_id="local"))
    tool = ReadFileTool(max_terminal_payload_bytes=128 * KIB)

    capped = asyncio.run(
        tool.run(ctx, {"path": "bundle.min.js", "start_line": 1, "max_bytes": 4_000_000})
    )
    uncapped = asyncio.run(
        tool.run(ctx, {"path": "bundle.min.js", "start_line": 1, "max_bytes": 1_000})
    )

    assert capped.structured["line_truncated"] is True
    assert "omit start_line and follow offset=next_offset" in capped.structured["continuation_hint"]
    assert "larger max_bytes" in uncapped.structured["continuation_hint"]


def test_recovered_foreground_subagent_result_is_omitted_rather_than_cut() -> None:
    async def run() -> ToolResult | None:
        store = await _store_with_children({})
        await _seed_child(store, child_id="child", text="r" * 190_000, mode="foreground")
        tool = SubagentTool(
            CayuApp(session_store=store, enable_logging=False),
            agents={"reviewer": SubagentSpec(agent_name="reviewer", result_max_chars=200_000)},
            max_terminal_payload_bytes=128 * KIB,
        )
        child = await store.load("child")
        assert child is not None
        return await tool.project_recoverable_child(child)

    result = asyncio.run(run())

    assert result is not None
    assert tool_result_payload_bytes(result) <= 128 * KIB
    assert result.is_error is False
    # Without the invocation's secret registry a cut could split a secret.
    assert "r" * 10 not in result.content
    assert "Child session child holds it." in result.content
    assert result.structured["result_omitted"] is True
    assert result.structured["child_session_id"] == "child"


def test_list_files_sizes_its_page_to_the_bound_so_next_offset_is_exact(tmp_path) -> None:
    for index in range(600):
        (tmp_path / f"{index:04d}{'n' * 176}.txt").write_text("x", encoding="utf-8")
    ctx = ToolContext(session_id="s", workspace=LocalWorkspace(tmp_path, workspace_id="local"))
    tool = ListFilesTool(max_terminal_payload_bytes=64 * KIB)

    result = asyncio.run(
        tool.run(ctx, {"pattern": "**/*", "limit": 500, "max_result_bytes": 200_000})
    )

    assert tool_result_payload_bytes(result) <= 64 * KIB
    assert TERMINAL_PAYLOAD_TRUNCATION_FIELD not in result.structured
    files = result.structured["files"]
    assert 0 < len(files) < 500
    assert result.structured["next_offset"] == len(files)


def test_fitting_judges_the_size_after_redaction_expands_short_secrets() -> None:
    secret = "s3cr3t"
    redactor = SecretRedactor([secret])
    text = (secret + " ") * 4_000
    adapter = _SearchAdapter(ToolResult(content=text, structured={"text": text}))
    tool = WebSearchTool(adapter=adapter, max_terminal_payload_bytes=64 * KIB)
    ctx = ToolContext(session_id="s", invocation_secret_redactor=lambda: redactor)

    result = asyncio.run(tool.run(ctx, {"query": "test"}))

    # The runtime redacts again before enforcing the bound; it must still fit.
    published = ToolResult(
        content=redactor.redact_text(result.content),
        structured=redactor.redact_json(result.model_dump(mode="json")["structured"]),
        is_error=result.is_error,
    )
    assert tool_result_payload_bytes(published) <= 64 * KIB
    assert secret not in result.content


def test_exec_command_judges_its_size_after_redaction(tmp_path) -> None:
    secret = "s3cr3t"
    redactor = SecretRedactor([secret])
    ctx = ToolContext(
        session_id="s",
        runner=LocalRunner(tmp_path),
        invocation_secret_redactor=lambda: redactor,
    )
    tool = ExecCommandTool(max_terminal_payload_bytes=64 * KIB)

    result = asyncio.run(
        tool.run(
            ctx,
            {
                "kind": "process",
                "argv": [sys.executable, "-c", f"print('{secret} ' * 4000)"],
                "max_output_bytes": 200_000,
            },
        )
    )

    assert tool_result_payload_bytes(result) <= 64 * KIB
    assert secret not in result.content


def test_list_files_sizes_its_page_as_published_after_redaction(tmp_path) -> None:
    secret = "s3cr3t"
    redactor = SecretRedactor([secret])
    for index in range(500):
        (tmp_path / (f"{index:04d}" + secret * 25 + ".txt")).touch()
    ctx = ToolContext(
        session_id="s",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
        invocation_secret_redactor=lambda: redactor,
    )
    tool = ListFilesTool(max_terminal_payload_bytes=64 * KIB)

    result = asyncio.run(tool.run(ctx, {"limit": 500, "max_result_bytes": 200_000}))

    published = ToolResult(
        content=redactor.redact_text(result.content),
        structured=redactor.redact_json(result.model_dump(mode="json")["structured"]),
    )
    assert tool_result_payload_bytes(published) <= 64 * KIB
    assert TERMINAL_PAYLOAD_TRUNCATION_FIELD not in result.structured
    files = result.structured["files"]
    assert 0 < len(files) < 500
    assert result.structured["next_offset"] == len(files)


def test_subagent_with_a_foreground_child_declares_no_bound_unless_asked() -> None:
    # A foreground child's rounds run inside the parent's round; only an
    # exclusive parent round always admits a child round whose tools declare
    # no bound, so the bound is the application's explicit choice there.
    def tool(bound: int | None) -> SubagentTool:
        return SubagentTool(
            CayuApp(session_store=InMemorySessionStore(), enable_logging=False),
            agents={
                "inline": SubagentSpec(agent_name="inline"),
                "queued": SubagentSpec(agent_name="queued", mode=SubagentExecutionMode.DURABLE),
            },
            max_terminal_payload_bytes=bound,
        )

    assert tool(None).spec.max_terminal_payload_bytes is None
    assert tool(96 * KIB).spec.max_terminal_payload_bytes == 96 * KIB


def test_read_file_reads_a_file_over_one_mebibyte_whole_for_its_revision(tmp_path) -> None:
    (tmp_path / "large.txt").write_text("a" * 1_100_000, encoding="utf-8")
    ctx = ToolContext(session_id="s", workspace=LocalWorkspace(tmp_path, workspace_id="local"))

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "large.txt", "max_bytes": 2_000_000}))

    assert result.structured["bytes"] == 1_100_000
    assert result.structured["truncated"] is False
    assert result.structured["revision"] is not None
    assert tool_result_payload_bytes(result) <= MAX_TOOL_TERMINAL_PAYLOAD_BYTES


def test_subagent_result_all_sizes_pages_as_published_after_redaction() -> None:
    secret = "abc"
    redactor = SecretRedactor([secret])
    texts = {f"child-{index:03d}": "abc " * 20 for index in range(600)}

    async def run() -> list[ToolResult]:
        store = await _store_with_children(texts)
        tool = SubagentResultTool(store, max_terminal_payload_bytes=256 * KIB)
        ctx = ToolContext(session_id="parent", invocation_secret_redactor=lambda: redactor)
        pages, args = [], {"all": True, "wait": False}
        while True:
            page = await tool.run(ctx, args)
            pages.append(page)
            next_offset = page.structured.get("next_offset")
            if next_offset is None:
                return pages
            args = {**args, "offset": next_offset}

    pages = asyncio.run(run())

    returned = [
        child["child_session_id"] for page in pages for child in page.structured["children"]
    ]
    assert returned == list(texts)
    assert len(pages) > 1
    for page in pages:
        assert TERMINAL_PAYLOAD_TRUNCATION_FIELD not in page.structured
