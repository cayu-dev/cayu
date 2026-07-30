from __future__ import annotations

import asyncio
import json
import shutil
import tracemalloc
from pathlib import Path

import pytest

import cayu.tools.search as search_module
from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventQuery,
    EventType,
    ExecCommand,
    ExecResult,
    InMemorySessionStore,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    Runner,
    RunRequest,
    ScriptedModelProvider,
    SearchTextTool,
    ToolContext,
)
from cayu.providers import ModelRequest, build_openai_payload
from cayu.runners import LocalRunner, RunnerUnavailableError
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._runner import InvocationRunnerHandle
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _ResultRunner(Runner):
    default_cwd = "/workspace"

    def __init__(self, result: ExecResult | list[ExecResult]) -> None:
        self.repeat_result = not isinstance(result, list)
        self.results = list(result) if isinstance(result, list) else [result]
        self.command: ExecCommand | None = None
        self.commands: list[ExecCommand] = []
        self.cwd: str | None = None
        self.timeout_s: int | None = None
        self.timeout_values: list[int | None] = []
        self.output_limit_bytes: int | None = None

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        self.command = command
        self.commands.append(command)
        self.cwd = cwd
        self.timeout_s = timeout_s
        self.timeout_values.append(timeout_s)
        self.output_limit_bytes = output_limit_bytes
        if not self.results:
            raise AssertionError("No configured result remains for Runner.exec().")
        return self.results[0] if self.repeat_result else self.results.pop(0)


def test_search_preview_bound_never_splits_redaction_marker() -> None:
    value = "prefix-" + REDACTED_SECRET + "-suffix"

    bounded, truncated = search_module._truncate_utf8(
        value,
        len("prefix-[REDA"),
        marker=" [match preview truncated]",
    )

    assert truncated is True
    assert "[REDA" not in bounded
    assert len(bounded.encode()) <= len("prefix-[REDA")


class _SwapToOutsideSymlinkRunner(LocalRunner):
    def __init__(
        self,
        root: Path,
        *,
        workspace_file: Path,
        outside_file: Path,
    ) -> None:
        super().__init__(root)
        self.workspace_file = workspace_file
        self.outside_file = outside_file
        self.swapped = False

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        argv = command.argv or []
        reads_file_contents = any(
            option in argv
            for option in ("--files-with-matches", "--line-number", "--count-matches")
        )
        if reads_file_contents and not self.swapped:
            self.workspace_file.unlink()
            self.workspace_file.symlink_to(self.outside_file)
            self.swapped = True
        return await super().exec(
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )


def test_search_text_files_mode_returns_a_bounded_page() -> None:
    runner = _ResultRunner(
        ExecResult(
            stdout="src/a.py\0src/b.py\0src/c.py\0",
            stdout_bytes=27,
            artifacts=[{"kind": "runner_log", "ref": "artifact://search/1"}],
        )
    )

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "needle", "mode": "files", "limit": 2},
        )
    )

    expected_content = (
        "src/a.py\nsrc/b.py\n\n"
        "[search metadata: returned=2; truncated=true; reasons=limit; next_offset=2]"
    )
    assert result.is_error is False
    assert result.content == expected_content
    assert result.artifacts == [{"kind": "runner_log", "ref": "artifact://search/1"}]
    assert result.structured is not None
    structured = dict(result.structured)
    assert structured.pop("duration_ms") >= 0
    assert structured == {
        "mode": "files",
        "pattern": "needle",
        "path": ".",
        "glob": None,
        "matches": [{"path": "src/a.py"}, {"path": "src/b.py"}],
        "returned": 2,
        "offset": 0,
        "limit": 2,
        "truncated": True,
        "truncation_reasons": ["limit"],
        "next_offset": 2,
        "stdout_bytes": 27,
        "projected_content_bytes": len(expected_content.encode("utf-8")),
        "projected_matches_bytes": 41,
    }


def test_search_text_uses_invocation_redactor_before_runner_capture_limit() -> None:
    secret = "search-capture-boundary-secret"

    def redactor_provider() -> SecretRedactor:
        return SecretRedactor(secret)

    complete_runner = _ResultRunner(
        ExecResult(
            stdout=f"src/a.py\0{1}\x1fprefix:{secret}:suffix\n",
            stdout_bytes=len(secret) + 30,
        )
    )
    context = ToolContext(
        session_id="sess_1",
        runner=InvocationRunnerHandle(
            complete_runner,
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0,
                redactor=redactor_provider(),
            ),
        ),
        invocation_secret_redactor=redactor_provider,
    )

    complete = asyncio.run(
        SearchTextTool().run(
            context,
            {"pattern": "prefix", "mode": "content"},
        )
    )

    assert secret not in complete.content
    assert REDACTED_SECRET in complete.content

    ambiguous_runner = _ResultRunner(
        ExecResult(
            stdout=f"src/a.py\0{1}\x1f{secret[:10]}",
            stdout_truncated=True,
            stdout_bytes=10_000,
        )
    )
    ambiguous_context = ToolContext(
        session_id="sess_2",
        runner=InvocationRunnerHandle(
            ambiguous_runner,
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0,
                redactor=redactor_provider(),
            ),
        ),
        invocation_secret_redactor=redactor_provider,
    )

    ambiguous = asyncio.run(
        SearchTextTool().run(
            ambiguous_context,
            {"pattern": "prefix", "mode": "content"},
        )
    )

    serialized = json.dumps(ambiguous.model_dump(mode="json"))
    assert secret not in serialized
    assert secret[:10] not in serialized


def test_search_text_content_mode_bounds_a_single_minified_line() -> None:
    raw_line = "MATCH" + ("x" * 1_000_000)
    runner = _ResultRunner(
        ExecResult(
            stdout=f"dist/app.js\0{1}\x1f{raw_line}\n",
            stdout_bytes=len(raw_line.encode("utf-8")) + 15,
        )
    )

    result = asyncio.run(
        SearchTextTool(max_preview_bytes=80, max_result_bytes=200).run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "MATCH", "mode": "content"},
        )
    )

    assert result.is_error is False
    assert len(result.content.encode("utf-8")) <= 200
    assert result.structured is not None
    assert result.structured["matches"][0]["path"] == "dist/app.js"
    assert result.structured["matches"][0]["line"] == 1
    assert len(result.structured["matches"][0]["preview"].encode("utf-8")) <= 80
    assert result.structured["matches"][0]["preview"].endswith("[match preview truncated]")
    assert result.structured["truncated"] is True
    assert result.structured["truncation_reasons"] == ["line"]
    assert result.structured["next_offset"] is None
    assert result.structured["projected_content_bytes"] == len(result.content.encode("utf-8"))


def test_search_text_exposes_truncation_metadata_to_the_model_payload() -> None:
    runner = _ResultRunner(ExecResult(stdout=f"src/app.js\0{1}\x1fMATCH{'x' * 1_000}\n"))
    result = asyncio.run(
        SearchTextTool(max_preview_bytes=80, max_result_bytes=200).run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "MATCH", "mode": "content"},
        )
    )

    payload = build_openai_payload(
        ModelRequest(
            model="gpt-test",
            messages=[
                Message.tool_result(
                    tool_call_id="call_search",
                    tool_name="search_text",
                    content=result.content,
                    structured=result.structured,
                )
            ],
        )
    )

    assert payload["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_search",
            "output": result.content,
        }
    ]
    assert "returned=1; truncated=true; reasons=line; next_offset=none" in result.content
    assert '"matches"' not in result.content


def test_search_text_does_not_infer_truncation_from_literal_source_text() -> None:
    preview = "literal source text [match preview truncated]"
    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(
                session_id="sess_1",
                runner=_ResultRunner(ExecResult(stdout=f"src/app.py\0{1}\x1f{preview}\n")),
            ),
            {"pattern": "literal", "mode": "content"},
        )
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["matches"] == [{"path": "src/app.py", "line": 1, "preview": preview}]
    assert result.structured["truncated"] is False
    assert result.structured["truncation_reasons"] == []


def test_search_text_parsing_memory_is_bounded_by_the_requested_page() -> None:
    output = "".join(f"src/file-{index:06}.py\0" for index in range(100_000))
    runner = _ResultRunner(ExecResult(stdout=output))
    tool = SearchTextTool()

    tracemalloc.start()
    try:
        result = asyncio.run(
            tool.run(
                ToolContext(session_id="sess_1", runner=runner),
                {"pattern": "needle", "mode": "files", "offset": 50_000, "limit": 1},
            )
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["matches"] == [{"path": "src/file-050000.py"}]
    assert peak_bytes < 8 * 1024 * 1024


def test_search_text_result_budget_returns_a_consistent_next_page_offset() -> None:
    output = "".join(
        f"src/file-{index}.py\0{index + 1}\x1fMATCH {'x' * 40}\n" for index in range(10)
    )
    runner = _ResultRunner(ExecResult(stdout=output))

    result = asyncio.run(
        SearchTextTool(max_preview_bytes=80, max_result_bytes=160).run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "MATCH", "mode": "content", "limit": 10},
        )
    )

    assert result.is_error is False
    assert result.structured is not None
    assert 0 < result.structured["returned"] < 10
    assert result.structured["next_offset"] == result.structured["returned"]
    assert result.structured["truncated"] is True
    assert result.structured["truncation_reasons"] == ["output"]
    assert len(result.content.encode("utf-8")) <= 160
    for match in result.structured["matches"]:
        assert match["path"] in result.content


def test_search_text_count_mode_pages_paths_containing_colons() -> None:
    runner = _ResultRunner(
        ExecResult(stdout="src/a:x.py\0" + "2\nsrc/b.py\0" + "5\nsrc/c.py\0" + "1\n")
    )

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "MATCH", "mode": "count", "offset": 1, "limit": 1},
        )
    )

    assert result.is_error is False
    assert result.content == (
        "src/b.py:5\n\n[search metadata: returned=1; truncated=true; reasons=limit; next_offset=2]"
    )
    assert result.structured is not None
    assert result.structured["matches"] == [{"path": "src/b.py", "count": 5}]
    assert result.structured["next_offset"] == 2
    assert result.structured["truncation_reasons"] == ["limit"]


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_search_text_real_runner_honors_project_search_scope(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "dist").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "src" / "main.py").write_text("needle\n")
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text("needle\n")
    (tmp_path / "node_modules" / "package.js").write_text("needle\n")
    (tmp_path / "dist" / "app.js").write_text("needle\n")
    (tmp_path / ".git" / "config").write_text("needle\n")
    (tmp_path / "ignored.txt").write_text("needle\n")

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path)),
            {"pattern": "needle", "mode": "files"},
        )
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["matches"] == [
        {"path": ".github/workflows/ci.yml"},
        {"path": "src/main.py"},
    ]


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
@pytest.mark.parametrize(
    ("scope", "expected_matches"),
    [
        ({"glob": "*.txt"}, [{"path": "visible.txt"}]),
        ({"path": "ignored.txt"}, []),
    ],
    ids=["glob", "explicit-path"],
)
def test_search_text_scope_filters_honor_ignore_rules_without_git_metadata(
    tmp_path,
    scope: dict[str, str],
    expected_matches: list[dict[str, str]],
) -> None:
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "ignored.txt").write_text("needle\n")
    (tmp_path / "visible.txt").write_text("needle\n")

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path)),
            {"pattern": "needle", **scope},
        )
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["matches"] == expected_matches


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_search_text_scoped_search_does_not_follow_a_file_replaced_by_symlink(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "app.py"
    source.parent.mkdir(parents=True)
    source.write_text("safe content\n")
    outside = tmp_path / "outside.py"
    outside.write_text("outside-only-needle\n")
    runner = _SwapToOutsideSymlinkRunner(
        workspace,
        workspace_file=source,
        outside_file=outside,
    )

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {
                "pattern": "outside-only-needle",
                "path": "src/app.py",
                "mode": "content",
            },
        )
    )

    assert runner.swapped is True
    assert result.is_error is False
    assert result.content == "No matches."
    assert result.structured is not None
    assert result.structured["matches"] == []


def test_search_text_returns_a_bounded_invalid_pattern_result() -> None:
    stderr = "regex parse error: invalid repetition" + ("x" * 10_000)
    runner = _ResultRunner(
        ExecResult(exit_code=2, stderr=stderr, stderr_bytes=len(stderr.encode("utf-8")))
    )

    result = asyncio.run(
        SearchTextTool(max_preview_bytes=80, max_result_bytes=200).run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "*", "mode": "files"},
        )
    )

    assert result.is_error is True
    assert len(result.content.encode("utf-8")) <= 200
    assert result.content == "Search pattern is invalid."
    assert "regex parse error" not in json.dumps(result.model_dump())
    assert result.artifacts == []
    assert result.structured is not None
    structured = dict(result.structured)
    assert structured.pop("duration_ms") >= 0
    assert structured == {
        "error": "invalid_pattern",
        "exit_code": 2,
        "stderr_bytes": len(stderr.encode("utf-8")),
    }


@pytest.mark.parametrize(
    ("exec_result", "error", "content"),
    [
        (
            ExecResult(exit_code=0, timed_out=True),
            "search_timed_out",
            "Search timed out after 30 seconds.",
        ),
        (
            ExecResult(exit_code=1, cancelled=True),
            "search_cancelled",
            "Search was cancelled.",
        ),
        (
            ExecResult(exit_code=127, stderr="SECRET: Command not found: rg"),
            "search_unavailable",
            "Text search is unavailable because ripgrep could not be started.",
        ),
    ],
)
def test_search_text_returns_typed_operational_failures(
    exec_result: ExecResult,
    error: str,
    content: str,
) -> None:
    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=_ResultRunner(exec_result)),
            {"pattern": "needle"},
        )
    )

    assert result.is_error is True
    assert result.content == content
    assert result.structured is not None
    assert result.structured["error"] == error


@pytest.mark.parametrize(
    "args",
    [
        {"pattern": "   "},
        {"pattern": "needle", "path": "../outside"},
        {"pattern": "needle", "path": "/outside"},
        {"pattern": "needle", "glob": "bad\0glob"},
        {"pattern": "needle", "mode": "raw"},
        {"pattern": "needle", "limit": 0},
        {"pattern": "needle", "limit": 501},
        {"pattern": "needle", "offset": -1},
        {"pattern": "needle", "offset": 10_000_001},
        {"pattern": "needle", "case_sensitive": "yes"},
    ],
)
def test_search_text_rejects_invalid_model_arguments(args: dict) -> None:
    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=_ResultRunner(ExecResult())),
            args,
        )
    )

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}


def test_search_text_runner_truncation_does_not_advertise_an_unreachable_offset() -> None:
    runner = _ResultRunner(
        ExecResult(
            stdout="src/a.py\0src/partial",
            stdout_truncated=True,
            stdout_bytes=50_000,
        )
    )

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "needle", "mode": "files"},
        )
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["matches"] == [{"path": "src/a.py"}]
    assert result.structured["truncation_reasons"] == ["runner_output"]
    assert result.structured["next_offset"] is None


def test_search_text_reports_when_capture_cannot_reach_requested_offset() -> None:
    runner = _ResultRunner(
        ExecResult(
            stdout="src/a.py\0src/partial",
            stdout_truncated=True,
            stdout_bytes=50_000,
        )
    )

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "needle", "mode": "files", "offset": 1},
        )
    )

    assert result.is_error is True
    assert result.content == (
        "Search output reached its capture limit before the requested offset. "
        "Use a narrower path, glob, or pattern."
    )
    assert result.structured is not None
    structured = dict(result.structured)
    assert structured.pop("duration_ms") >= 0
    assert structured == {
        "error": "search_capture_exhausted",
        "mode": "files",
        "offset": 1,
        "captured_entries": 1,
        "stdout_bytes": 50_000,
    }


def test_search_text_schema_and_registration_limits_are_finite() -> None:
    tool = SearchTextTool()

    assert tool.schema["properties"]["pattern"]["maxLength"] == 4096
    assert tool.schema["properties"]["path"]["maxLength"] == 4096
    assert tool.schema["properties"]["glob"]["maxLength"] == 4096
    assert tool.schema["properties"]["mode"]["enum"] == ["files", "content", "count"]
    assert tool.schema["properties"]["limit"]["default"] == 100
    assert tool.schema["properties"]["limit"]["maximum"] == 500
    assert tool.spec.parallel_safe is True
    assert tool.spec.effect.value == "none"

    with pytest.raises(ValueError, match="default_limit"):
        SearchTextTool(default_limit=10, max_limit=5)
    with pytest.raises(ValueError, match="max_result_bytes"):
        SearchTextTool(max_preview_bytes=500, max_result_bytes=540)


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_search_text_real_runner_bounds_a_one_megabyte_matching_line(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.js").write_text("MATCH" + ("x" * 1_000_000) + "\n")

    result = asyncio.run(
        SearchTextTool(max_preview_bytes=80, max_result_bytes=200).run(
            ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path)),
            {"pattern": "MATCH", "mode": "content"},
        )
    )

    assert result.is_error is False
    assert len(result.content.encode("utf-8")) <= 200
    assert result.structured is not None
    assert result.structured["matches"][0]["path"] == "src/app.js"
    assert result.structured["matches"][0]["line"] == 1
    assert result.structured["matches"][0]["preview"].endswith("[match preview truncated]")
    assert "x" * 1_000 not in json.dumps(result.model_dump())


def test_search_text_registration_controls_the_bounded_runner_request() -> None:
    runner = _ResultRunner(
        [
            ExecResult(stdout="src/a.py\0"),
            ExecResult(stdout="src/a.py\0"),
        ]
    )

    result = asyncio.run(
        SearchTextTool(
            timeout_s=7,
            capture_limit_bytes=12_345,
            max_file_size_bytes=54_321,
            exclude_directories=("generated",),
        ).run(
            ToolContext(session_id="sess_1", runner=runner),
            {
                "pattern": "-needle",
                "path": "src",
                "glob": "*.py",
                "case_sensitive": False,
            },
        )
    )

    assert result.is_error is False
    assert runner.timeout_values[0] == 7
    assert all(timeout is not None and timeout <= 7 for timeout in runner.timeout_values)
    assert runner.output_limit_bytes == 12_345
    assert runner.command is not None
    assert runner.command.kind == "process"
    assert runner.command.argv is not None
    assert "--no-config" in runner.command.argv
    assert "--no-require-git" in runner.command.argv
    assert "--ignore-case" in runner.command.argv
    assert runner.command.argv[-2:] == ["-needle", "."]
    option_end = runner.command.argv.index("--")
    assert runner.command.argv[option_end + 1 :] == ["-needle", "."]
    assert runner.command.argv[
        runner.command.argv.index("--max-filesize") : runner.command.argv.index("--max-filesize")
        + 2
    ] == ["--max-filesize", "54321"]
    assert "!generated/**" in runner.command.argv
    assert "!dist/**" not in runner.command.argv
    assert runner.command.argv[-1] == "."
    assert "src/**/*.py" in runner.command.argv
    assert "src/a.py" not in runner.command.argv


def test_search_text_scope_resolution_shares_one_timeout_budget(monkeypatch) -> None:
    clock = [100.0]

    class _AdvancingRunner(_ResultRunner):
        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            result = await super().exec(command, **kwargs)
            clock[0] += 2.0
            return result

    monkeypatch.setattr(search_module.time, "monotonic", lambda: clock[0])
    runner = _AdvancingRunner(
        [
            ExecResult(stdout="src/a.py\0"),
            ExecResult(stdout="src/a.py\0"),
        ]
    )

    result = asyncio.run(
        SearchTextTool(timeout_s=7).run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "needle", "path": "src"},
        )
    )

    assert result.is_error is False
    assert runner.timeout_values == [7, 5]


def test_search_text_scope_resolution_does_not_round_up_subsecond_budget(monkeypatch) -> None:
    clock = [100.0]

    class _NearlyExpiredRunner(_ResultRunner):
        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            result = await super().exec(command, **kwargs)
            clock[0] += 6.5
            return result

    monkeypatch.setattr(search_module.time, "monotonic", lambda: clock[0])
    runner = _NearlyExpiredRunner(ExecResult(stdout="src/a.py\0"))

    result = asyncio.run(
        SearchTextTool(timeout_s=7).run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "needle", "path": "src"},
        )
    )

    assert result.is_error is True
    assert result.structured is not None
    assert result.structured["error"] == "search_timed_out"
    assert runner.timeout_values == [7]


@pytest.mark.parametrize(
    ("mode", "stdout", "expected"),
    [
        (
            "files",
            "".join(f"src/{index}.py\0" for index in range(7)),
            [{"path": f"src/{index}.py"} for index in range(7)],
        ),
        (
            "content",
            "".join(f"src/{index}.py\0{index + 1}\x1fneedle {index}\n" for index in range(7)),
            [
                {
                    "path": f"src/{index}.py",
                    "line": index + 1,
                    "preview": f"needle {index}",
                }
                for index in range(7)
            ],
        ),
        (
            "count",
            "".join(f"src/{index}.py\0{index + 1}\n" for index in range(7)),
            [{"path": f"src/{index}.py", "count": index + 1} for index in range(7)],
        ),
    ],
)
def test_search_text_pagination_is_stable_in_every_mode(
    mode: str,
    stdout: str,
    expected: list[dict],
) -> None:
    runner = _ResultRunner(ExecResult(stdout=stdout))
    tool = SearchTextTool()
    actual: list[dict] = []
    offset = 0

    while True:
        result = asyncio.run(
            tool.run(
                ToolContext(session_id="sess_1", runner=runner),
                {
                    "pattern": "needle",
                    "mode": mode,
                    "limit": 3,
                    "offset": offset,
                },
            )
        )
        assert result.is_error is False
        assert result.structured is not None
        actual.extend(result.structured["matches"])
        next_offset = result.structured["next_offset"]
        if next_offset is None:
            break
        assert next_offset == offset + result.structured["returned"]
        offset = next_offset

    assert actual == expected
    beyond = asyncio.run(
        tool.run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "needle", "mode": mode, "offset": len(expected) + 10},
        )
    )
    assert beyond.content == "No matches."
    assert beyond.structured is not None
    assert beyond.structured["matches"] == []
    assert beyond.structured["next_offset"] is None


def test_search_text_pages_thousands_of_files_without_skips_or_duplicates() -> None:
    paths = [f"src/generated/file-{index:04}.py" for index in range(2_003)]
    runner = _ResultRunner(ExecResult(stdout="\0".join(paths) + "\0"))
    tool = SearchTextTool()
    actual: list[str] = []
    offset = 0

    while True:
        result = asyncio.run(
            tool.run(
                ToolContext(session_id="sess_1", runner=runner),
                {"pattern": "needle", "limit": 500, "offset": offset},
            )
        )
        assert result.is_error is False
        assert result.structured is not None
        assert result.structured["returned"] <= 500
        assert result.structured["projected_content_bytes"] <= 20_000
        assert result.structured["projected_matches_bytes"] <= 20_000
        actual.extend(match["path"] for match in result.structured["matches"])
        next_offset = result.structured["next_offset"]
        if next_offset is None:
            break
        offset = next_offset

    assert actual == paths
    assert len(set(actual)) == len(paths)


def test_search_text_returns_exactly_the_maximum_result_count_when_it_fits() -> None:
    paths = [f"f{index:03}.py" for index in range(501)]
    runner = _ResultRunner(ExecResult(stdout="\0".join(paths) + "\0"))

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "needle", "limit": 500},
        )
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["returned"] == 500
    assert result.structured["matches"] == [{"path": path} for path in paths[:500]]
    assert result.structured["truncation_reasons"] == ["limit"]
    assert result.structured["next_offset"] == 500


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_search_text_real_runner_handles_unicode_by_utf8_bytes(tmp_path) -> None:
    path = tmp_path / "src" / "日本語.py"
    path.parent.mkdir()
    path.write_text("照合🙂" + ("界" * 100) + "\n")

    result = asyncio.run(
        SearchTextTool(max_preview_bytes=48, max_result_bytes=256).run(
            ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path)),
            {"pattern": "照合🙂", "mode": "content"},
        )
    )

    assert result.is_error is False
    assert result.structured is not None
    match = result.structured["matches"][0]
    assert match["path"] == "src/日本語.py"
    assert len(match["preview"].encode("utf-8")) <= 48
    assert match["preview"].endswith("[match preview truncated]")


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_search_text_real_runner_skips_binary_large_symlink_and_generated_files(
    tmp_path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "src" / "visible.py").write_text("needle\n")
    (tmp_path / ".hidden.py").write_text("needle\n")
    (tmp_path / "src" / "binary.bin").write_bytes(b"\x00needle\x00")
    (tmp_path / "src" / "large.txt").write_bytes(b"needle" + (b"x" * (2 * 1024 * 1024)))
    generated = tmp_path / "node_modules" / "generated.js"
    generated.write_text("needle\n")
    (tmp_path / "src" / "generated-link.js").symlink_to(generated)

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path)),
            {"pattern": "needle", "mode": "files"},
        )
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["matches"] == [
        {"path": ".hidden.py"},
        {"path": "src/visible.py"},
    ]


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_search_text_generated_exclusions_cannot_be_reenabled_by_model_glob(
    tmp_path,
) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "generated.js").write_text("needle\n")

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path)),
            {"pattern": "needle", "glob": "node_modules/**"},
        )
    )

    assert result.is_error is False
    assert result.content == "No matches."


def test_search_text_rejects_an_explicit_path_inside_an_excluded_directory() -> None:
    runner = _ResultRunner(ExecResult())

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {
                "pattern": "needle",
                "path": "web/node_modules/package/index.js",
            },
        )
    )

    assert result.is_error is True
    assert result.structured == {
        "error": "search_path_excluded",
        "path": "web/node_modules/package/index.js",
        "excluded_directory": "node_modules",
    }
    assert runner.command is None


def test_search_text_bounds_an_excluded_path_message() -> None:
    directory = "generated-" + ("x" * 300)

    result = asyncio.run(
        SearchTextTool(
            exclude_directories=(directory,),
            max_preview_bytes=30,
            max_result_bytes=128,
        ).run(
            ToolContext(session_id="sess_1", runner=_ResultRunner(ExecResult())),
            {"pattern": "needle", "path": f"src/{directory}/bundle.js"},
        )
    )

    assert result.is_error is True
    assert len(result.content.encode("utf-8")) <= 128
    assert result.content.endswith("[search message truncated]")


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_search_text_explicit_file_path_still_honors_max_file_size(tmp_path) -> None:
    source = tmp_path / "src" / "large.py"
    source.parent.mkdir()
    source.write_text("needle" + ("x" * 2_000))
    context = ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path))

    skipped = asyncio.run(
        SearchTextTool(max_file_size_bytes=1_000).run(
            context,
            {"pattern": "needle", "path": "src/large.py"},
        )
    )
    included = asyncio.run(
        SearchTextTool(max_file_size_bytes=4_000).run(
            context,
            {"pattern": "needle", "path": "src/large.py"},
        )
    )

    assert skipped.is_error is False
    assert skipped.content == "No matches."
    assert included.is_error is False
    assert included.structured is not None
    assert included.structured["matches"] == [{"path": "src/large.py"}]


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_search_text_parallel_calls_share_a_read_only_workspace(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("alpha\n")
    (tmp_path / "src" / "b.py").write_text("beta\n")
    runner = LocalRunner(tmp_path)
    tool = SearchTextTool()

    async def search(pattern: str):
        return await tool.run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": pattern},
        )

    async def run_parallel():
        return await asyncio.gather(search("alpha"), search("beta"))

    alpha, beta = asyncio.run(run_parallel())

    assert alpha.structured is not None
    assert beta.structured is not None
    assert alpha.structured["matches"] == [{"path": "src/a.py"}]
    assert beta.structured["matches"] == [{"path": "src/b.py"}]


def test_search_text_returns_typed_missing_runner_result() -> None:
    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1"),
            {"pattern": "needle"},
        )
    )

    assert result.is_error is True
    assert result.structured == {"error": "runner_unavailable"}


class _UnavailableRunner(_ResultRunner):
    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        raise RunnerUnavailableError(
            "Runner is disconnected.",
            diagnostic={"kind": "runner_unavailable", "safe": True},
        )


class _LongUnavailableRunner(_ResultRunner):
    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        raise RunnerUnavailableError(
            "runner disconnected " + ("x" * 10_000),
            diagnostic={
                "kind": "runner_unavailable",
                "raw_stderr": "sensitive diagnostic " + ("x" * 100_000),
            },
        )


def test_search_text_preserves_runner_unavailable_diagnostic() -> None:
    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(
                session_id="sess_1",
                runner=_UnavailableRunner(ExecResult()),
            ),
            {"pattern": "needle"},
        )
    )

    assert result.is_error is True
    assert result.structured is not None
    assert result.structured["error"] == "runner_unavailable"
    assert result.structured["diagnostic"] == {
        "kind": "runner_unavailable",
        "safe": True,
    }
    assert result.structured["duration_ms"] >= 0
    assert result.artifacts == [{"kind": "runner_unavailable", "safe": True}]


def test_search_text_bounds_runner_unavailable_message() -> None:
    result = asyncio.run(
        SearchTextTool(max_preview_bytes=30, max_result_bytes=128).run(
            ToolContext(
                session_id="sess_1",
                runner=_LongUnavailableRunner(ExecResult()),
            ),
            {"pattern": "needle"},
        )
    )

    assert result.is_error is True
    assert result.content == "Runner is unavailable for text search."
    assert result.structured is not None
    assert result.structured["diagnostic"] == {
        "kind": "runner_unavailable",
        "raw_streams_omitted": True,
    }
    assert "diagnostic_truncated" not in result.structured
    assert result.artifacts == [{"kind": "runner_unavailable", "raw_streams_omitted": True}]
    assert "sensitive diagnostic" not in json.dumps(result.model_dump())


def test_search_text_bounds_generic_failure_stderr_and_preserves_artifacts() -> None:
    stderr = "SECRET unexpected ripgrep failure " + ("x" * 100_000)
    result = asyncio.run(
        SearchTextTool(max_preview_bytes=80, max_result_bytes=200).run(
            ToolContext(
                session_id="sess_1",
                runner=_ResultRunner(
                    ExecResult(
                        exit_code=9,
                        stderr=stderr,
                        stderr_bytes=len(stderr),
                        stderr_truncated=True,
                        artifacts=[{"kind": "cleanup", "status": "complete"}],
                    )
                ),
            ),
            {"pattern": "needle"},
        )
    )

    assert result.is_error is True
    assert len(result.content.encode("utf-8")) <= 200
    assert result.structured is not None
    assert result.structured["error"] == "search_failed"
    assert result.structured["stderr_bytes"] == len(stderr)
    assert result.artifacts == [{"kind": "cleanup", "status": "complete"}]
    assert result.content == "Text search failed."
    assert "SECRET" not in json.dumps(result.model_dump())


def test_search_text_bounds_oversized_runner_artifacts() -> None:
    result = asyncio.run(
        SearchTextTool(max_preview_bytes=30, max_result_bytes=128).run(
            ToolContext(
                session_id="sess_1",
                runner=_ResultRunner(
                    ExecResult(
                        artifacts=[
                            {
                                "details": "x" * 100_000,
                                "raw_stderr": "sensitive diagnostic",
                            }
                        ]
                    )
                ),
            ),
            {"pattern": "needle"},
        )
    )

    assert result.is_error is False
    assert result.artifacts == [{"type": "cayu.search_runner_artifacts.v1", "truncated": True}]
    assert result.structured is not None
    assert result.structured["artifacts_truncated"] is True
    assert "sensitive" not in json.dumps(result.model_dump())


def test_search_text_omits_small_raw_stream_fields_from_runner_artifacts() -> None:
    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(
                session_id="sess_1",
                runner=_ResultRunner(
                    ExecResult(
                        artifacts=[
                            {
                                "kind": "cleanup",
                                "raw_stdout": "SECRET stdout",
                                "stderr": "SECRET stderr",
                            }
                        ]
                    )
                ),
            ),
            {"pattern": "needle"},
        )
    )

    assert result.is_error is False
    assert result.artifacts == [{"kind": "cleanup", "raw_streams_omitted": True}]
    assert "SECRET" not in json.dumps(result.model_dump())


@pytest.mark.parametrize(
    "glob",
    [
        "!**/*.py",
        "../*.py",
        "/tmp/*.py",
        "[*.py",
        "*.py\\",
        "[z-a]",
        "[]",
        "[!]",
        "[^]",
    ],
)
def test_search_text_rejects_invalid_globs_before_runner_execution(glob: str) -> None:
    runner = _ResultRunner(ExecResult())

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "needle", "glob": glob},
        )
    )

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert runner.command is None


def test_search_text_normalizes_contained_relative_paths() -> None:
    runner = _ResultRunner(
        [
            ExecResult(stdout="tests/unit/test_app.py\0"),
            ExecResult(stdout="tests/unit/test_app.py\0"),
        ]
    )

    result = asyncio.run(
        SearchTextTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {"pattern": "needle", "path": "src/../tests//unit"},
        )
    )

    assert result.is_error is False
    assert runner.command is not None
    assert runner.command.argv is not None
    assert runner.command.argv[-1] == "."
    assert "tests/unit" in runner.command.argv
    assert "tests/unit/**" in runner.command.argv
    assert "tests/unit/test_app.py" not in runner.command.argv
    assert result.structured is not None
    assert result.structured["path"] == "tests/unit"


def test_search_text_bounds_structured_matches_independently_from_content() -> None:
    output = "".join(
        f"src/file-{index:03}.py\0{index + 1}\x1fneedle {'界' * 100}\n" for index in range(100)
    )
    result = asyncio.run(
        SearchTextTool(max_preview_bytes=120, max_result_bytes=2_000).run(
            ToolContext(
                session_id="sess_1",
                runner=_ResultRunner(ExecResult(stdout=output)),
            ),
            {"pattern": "needle", "mode": "content"},
        )
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["projected_content_bytes"] <= 2_000
    assert result.structured["projected_matches_bytes"] <= 2_000
    assert len(json.dumps(result.model_dump()).encode("utf-8")) < 7_000


def test_search_text_total_byte_budget_honors_exact_boundary() -> None:
    paths = [character * 80 for character in "abc"]
    output = "\0".join(paths) + "\0"
    exact_content = (
        f"{paths[0]}\n{paths[1]}\n\n"
        "[search metadata: returned=2; truncated=true; reasons=output; next_offset=2]"
    )
    exact_budget = len(exact_content.encode("utf-8"))
    context = ToolContext(
        session_id="sess_1",
        runner=_ResultRunner(ExecResult(stdout=output)),
    )

    exact = asyncio.run(
        SearchTextTool(max_preview_bytes=30, max_result_bytes=exact_budget).run(
            context,
            {"pattern": "needle", "limit": 3},
        )
    )
    one_under = asyncio.run(
        SearchTextTool(max_preview_bytes=30, max_result_bytes=exact_budget - 1).run(
            context,
            {"pattern": "needle", "limit": 3},
        )
    )

    assert exact.structured is not None
    assert exact.structured["returned"] == 2
    assert exact.structured["projected_content_bytes"] == exact_budget
    assert one_under.structured is not None
    assert one_under.structured["returned"] == 1
    assert one_under.structured["projected_content_bytes"] <= exact_budget - 1


def test_search_text_durable_event_and_transcript_payloads_remain_bounded(tmp_path) -> None:
    runner_output = "".join(
        f"src/generated.py\0{line}\x1fneedle {'界' * 100}\n" for line in range(1, 101)
    )
    store = InMemorySessionStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="search",
                    name="search_text",
                    arguments={"pattern": "needle", "mode": "content"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace(tmp_path, workspace_id="search-bounds"),
            runner=_ResultRunner(ExecResult(stdout=runner_output)),
        ),
        default=True,
    )
    tool = SearchTextTool(max_preview_bytes=120, max_result_bytes=2_000)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    async def run_and_read():
        emitted = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="durable_search_bounds",
                    messages=[Message.text("user", "search")],
                )
            )
        ]
        records = await store.query_events(
            EventQuery(
                session_id="durable_search_bounds",
                event_type=EventType.TOOL_CALL_COMPLETED,
            )
        )
        transcript = await store.load_transcript("durable_search_bounds")
        return emitted, records, transcript

    emitted, records, transcript = asyncio.run(run_and_read())
    emitted_tool = next(event for event in emitted if event.type == EventType.TOOL_CALL_COMPLETED)
    persisted_tool = records[0].event
    assert emitted_tool.payload == persisted_tool.payload
    result_payload = persisted_tool.payload["result"]
    assert result_payload["structured"]["projected_content_bytes"] <= 2_000
    assert result_payload["structured"]["projected_matches_bytes"] <= 2_000
    # Content, typed matches, bounded arguments, artifacts, and fixed metadata
    # each have independent finite ceilings; the durable envelope cannot recover
    # the runner's original 30 KB of matching line data.
    durable_result_bytes = len(json.dumps(result_payload, ensure_ascii=False).encode("utf-8"))
    assert durable_result_bytes < 24_000
    assert (
        len(
            json.dumps(
                [message.model_dump(mode="json") for message in transcript],
                ensure_ascii=False,
            ).encode("utf-8")
        )
        < 30_000
    )
    assert "界" * 1_000 not in json.dumps(
        {"event": persisted_tool.payload, "transcript": transcript},
        ensure_ascii=False,
        default=str,
    )


def test_search_text_reports_when_one_entry_cannot_fit_registration_budget() -> None:
    result = asyncio.run(
        SearchTextTool(max_preview_bytes=80, max_result_bytes=200).run(
            ToolContext(
                session_id="sess_1",
                runner=_ResultRunner(ExecResult(stdout=("x" * 500) + "\0")),
            ),
            {"pattern": "needle"},
        )
    )

    assert result.is_error is True
    assert result.structured is not None
    assert result.structured["error"] == "search_entry_too_large"
    assert len(result.content.encode("utf-8")) <= 200
