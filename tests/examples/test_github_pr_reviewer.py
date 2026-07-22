"""Composition guard for the flagship `examples/github_pr_reviewer/` recipe.

Loaded from disk via importlib (examples are not importable from ``cayu``). This
does not hit the network or a model — it asserts the recipe *wires up*, so that a
public-API drift (a renamed export, a changed ``register_agent`` signature) breaks
this test instead of silently breaking the featured example.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import shutil
from pathlib import Path

import pytest

from cayu import (
    EnvironmentFactoryRequest,
    PassthroughProxy,
    ScriptedModelProvider,
    Task,
    ToolContext,
    ToolEffect,
)
from cayu.vaults import StaticVault

_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "github_pr_reviewer" / "pr_reviewer.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("github_pr_reviewer_example", _EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_example()
demo_mod = importlib.import_module("examples.github_pr_reviewer.demo")
github_tools_mod = importlib.import_module("examples.github_pr_reviewer.github_tools")
worker_mod = importlib.import_module("examples.github_pr_reviewer.worker")

_EXPECTED_TOOLS = {"get_pr_diff", "post_pr_comment", "read_file", "list_files", "exec_command"}


def test_github_tools_classify_remote_reads_and_non_atomic_comment_creation() -> None:
    assert mod.GetPRDiffTool.spec.effect is ToolEffect.NONE
    assert mod.PostPRCommentTool.spec.effect is ToolEffect.EXTERNAL


class _GitHubResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def _pr_payload(
    *,
    title: str = "PR",
    changed_files: int = 1,
    head_ref: str = "feature",
    head_sha: str = "sha",
    base_ref: str = "main",
) -> dict:
    return {
        "title": title,
        "body": None,
        "changed_files": changed_files,
        "head": {"ref": head_ref, "sha": head_sha},
        "base": {"ref": base_ref},
    }


def _file_payload(index: int | str = 0, patch: str = "+ok\n") -> dict:
    return {
        "filename": f"file_{index}.py" if type(index) is int else index,
        "status": "modified",
        "additions": 1,
        "deletions": 0,
        "patch": patch,
    }


def test_build_app_composes_the_reviewer(tmp_path: Path) -> None:
    app, _task_store = mod.build_app(
        tmp_path / "data" / "cayu.db",
        tmp_path / "workspaces",
        provider=ScriptedModelProvider([]),
        model="scripted-model",
    )
    agent = app.get_agent("pr-reviewer")
    assert agent is not None
    assert set(agent.tools.keys()) == _EXPECTED_TOOLS


def test_build_provider_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        mod.build_provider()


def test_pr_workspace_fetches_github_pull_ref(tmp_path: Path) -> None:
    factory = mod.PRReviewWorkspaceFactory(tmp_path, with_credentials=False)

    result = asyncio.run(
        factory.create(
            EnvironmentFactoryRequest(
                session_id="review-42",
                agent_name="pr-reviewer",
                environment_name="pr-workspace",
                metadata={
                    "repo_url": "https://github.com/acme/app.git",
                    "head_ref": "feature-from-fork",
                    "base_ref": "main",
                    "pr_number": 42,
                    "head_sha": "abc123def456",
                },
            )
        )
    )

    binding = result.environment.binding
    assert binding is not None
    assert binding.ref == "abc123def456"
    assert binding.fetch_refspecs == ["+refs/pull/42/head:refs/cayu/pr-42"]


def test_pr_workspace_can_rebind_same_pull_ref(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required for this example")
    origin, pr_head = demo_mod._create_demo_origin(tmp_path)
    factory = mod.PRReviewWorkspaceFactory(tmp_path / "workspaces", with_credentials=False)
    result = asyncio.run(
        factory.create(
            EnvironmentFactoryRequest(
                session_id="review-1",
                agent_name="pr-reviewer",
                environment_name="pr-workspace",
                metadata={
                    "repo_url": str(origin),
                    "head_ref": "fixture-pr",
                    "base_ref": "main",
                    "pr_number": 1,
                    "head_sha": pr_head,
                },
            )
        )
    )
    environment = result.environment
    assert environment.binding is not None
    assert environment.binding.ref == pr_head
    assert environment.workspace is not None
    assert environment.runner is not None

    asyncio.run(
        environment.binding.bind(
            environment.workspace,
            environment.runner,
            session_id="review-1",
            agent_name="pr-reviewer",
            environment_name="pr-workspace",
        )
    )
    asyncio.run(
        environment.binding.bind(
            environment.workspace,
            environment.runner,
            session_id="review-1",
            agent_name="pr-reviewer",
            environment_name="pr-workspace",
        )
    )


def test_worker_once_uses_generic_task_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    async def fake_run_task_worker(app, task_store, handler, **kwargs):
        captured.update({"app": app, "task_store": task_store, "handler": handler, **kwargs})
        return 1

    monkeypatch.setattr(worker_mod, "run_task_worker", fake_run_task_worker)

    app = object()
    task_store = object()
    handled = asyncio.run(mod.run_pr_review_worker_once(app, task_store, worker_id="worker-9"))

    assert handled == 1
    assert captured["app"] is app
    assert captured["task_store"] is task_store
    assert captured["handler"] is mod._handle_pr_review_task
    assert captured["worker_id"] == "worker-9"
    assert captured["query"].type == "review_pr"
    assert captured["query"].assigned_agent_name == "pr-reviewer"
    assert captured["max_tasks"] == 1


def test_pr_review_session_identity_includes_head_sha() -> None:
    captured = {}

    class FakeApp:
        async def run(self, request):
            captured["request"] = request
            if False:
                yield None

    task = Task(
        id="task-1",
        type="review_pr",
        assigned_agent_name="pr-reviewer",
        input={
            "owner": "acme",
            "repo": "app",
            "pr_number": 7,
            "repo_url": "https://github.com/acme/app.git",
            "head_ref": "feature",
            "head_sha": "abcdef1234567890",
            "base_ref": "main",
        },
    )

    asyncio.run(mod._handle_pr_review_task(FakeApp(), task, "worker-1"))

    request = captured["request"]
    assert request.session_id == "pr-review-acme-app-7-abcdef123456"
    assert request.metadata["head_sha"] == "abcdef1234567890"
    assert "abcdef123456" in request.messages[0].content[0].text


def test_get_pr_diff_rejects_stale_head_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGitHubClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url, *, headers=None, params=None):
            if url.endswith("/pulls/3"):
                return _GitHubResponse(200, _pr_payload(title="Moved PR", head_sha="current-sha"))
            raise AssertionError("stale-head guard should not fetch PR files")

    monkeypatch.setattr(github_tools_mod.httpx, "AsyncClient", FakeGitHubClient)
    ctx = ToolContext(
        session_id="review-3",
        metadata={
            "repo_owner": "acme",
            "repo_name": "app",
            "pr_number": 3,
            "head_sha": "queued-sha",
        },
    )

    result = asyncio.run(mod.GetPRDiffTool().run(ctx, {}))

    assert result.is_error
    assert "PR head changed before review" in result.content
    assert result.structured == {
        "expected_head_sha": "queued-sha",
        "current_head_sha": "current-sha",
    }


def test_get_pr_diff_rechecks_head_after_fetching_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeGitHubClient:
        def __init__(self, *args, **kwargs) -> None:
            self._pr_calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url, *, headers=None, params=None):
            calls.append(url)
            if url.endswith("/pulls/3"):
                self._pr_calls += 1
                return _GitHubResponse(
                    200,
                    _pr_payload(
                        title="Moved PR",
                        head_sha="queued-sha" if self._pr_calls == 1 else "new-sha",
                    ),
                )
            if url.endswith("/pulls/3/files"):
                return _GitHubResponse(200, [_file_payload("file.py")])
            return _GitHubResponse(404, {"message": "not found"})

    monkeypatch.setattr(github_tools_mod.httpx, "AsyncClient", FakeGitHubClient)
    ctx = ToolContext(
        session_id="review-3",
        metadata={
            "repo_owner": "acme",
            "repo_name": "app",
            "pr_number": 3,
            "head_sha": "queued-sha",
        },
    )

    result = asyncio.run(mod.GetPRDiffTool().run(ctx, {}))

    assert result.is_error
    assert calls == [
        "https://api.github.com/repos/acme/app/pulls/3",
        "https://api.github.com/repos/acme/app/pulls/3/files",
        "https://api.github.com/repos/acme/app/pulls/3",
    ]
    assert result.structured == {
        "expected_head_sha": "queued-sha",
        "current_head_sha": "new-sha",
    }


def test_get_pr_diff_returns_results_when_queued_head_stays_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeGitHubClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url, *, headers=None, params=None):
            calls.append(url)
            if url.endswith("/pulls/3"):
                return _GitHubResponse(200, _pr_payload(title="Stable PR", head_sha="queued-sha"))
            if url.endswith("/pulls/3/files"):
                return _GitHubResponse(200, [_file_payload("stable.py")])
            return _GitHubResponse(404, {"message": "not found"})

    monkeypatch.setattr(github_tools_mod.httpx, "AsyncClient", FakeGitHubClient)
    ctx = ToolContext(
        session_id="review-3",
        metadata={
            "repo_owner": "acme",
            "repo_name": "app",
            "pr_number": 3,
            "head_sha": "queued-sha",
        },
    )

    result = asyncio.run(mod.GetPRDiffTool().run(ctx, {}))

    assert not result.is_error
    assert result.structured["head_sha"] == "queued-sha"
    assert result.structured["files"][0]["path"] == "stable.py"
    assert calls == [
        "https://api.github.com/repos/acme/app/pulls/3",
        "https://api.github.com/repos/acme/app/pulls/3/files",
        "https://api.github.com/repos/acme/app/pulls/3",
    ]


def test_get_pr_diff_does_not_apply_session_head_sha_to_explicit_other_pr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeGitHubClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url, *, headers=None, params=None):
            calls.append(url)
            if url.endswith("/pulls/99"):
                return _GitHubResponse(
                    200, _pr_payload(title="Other PR", head_ref="other", head_sha="other-sha")
                )
            if url.endswith("/pulls/99/files"):
                return _GitHubResponse(200, [_file_payload("other.py")])
            return _GitHubResponse(404, {"message": "not found"})

    monkeypatch.setattr(github_tools_mod.httpx, "AsyncClient", FakeGitHubClient)
    ctx = ToolContext(
        session_id="review-3",
        metadata={
            "repo_owner": "acme",
            "repo_name": "app",
            "pr_number": 3,
            "head_sha": "queued-sha",
        },
    )

    result = asyncio.run(
        mod.GetPRDiffTool().run(ctx, {"owner": "acme", "repo": "app", "pr_number": 99})
    )

    assert not result.is_error
    assert result.structured["head_sha"] == "other-sha"
    assert calls == [
        "https://api.github.com/repos/acme/app/pulls/99",
        "https://api.github.com/repos/acme/app/pulls/99/files",
    ]


def test_demo_runs_without_github_network(monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required for the PR-reviewer demo")

    class NoNetworkClient:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("demo must not open an HTTP client")

    monkeypatch.setattr(demo_mod.httpx, "AsyncClient", NoNetworkClient)

    asyncio.run(mod.demo())


def test_get_pr_diff_paginates_files_and_bounds_patch_previews(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Response:
        def __init__(self, status_code: int, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = str(payload)

        def json(self):
            return self._payload

    def file_payload(index: int, patch: str = "+ok\n") -> dict:
        return {
            "filename": f"file_{index}.py",
            "status": "modified",
            "additions": 1,
            "deletions": 0,
            "patch": patch,
        }

    class FakeGitHubClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url, *, headers=None, params=None):
            calls.append((url, params))
            if url.endswith("/pulls/3"):
                return Response(
                    200,
                    {
                        "title": "Large PR",
                        "body": None,
                        "changed_files": 101,
                        "head": {"ref": "feature", "sha": "abc123"},
                        "base": {"ref": "main"},
                    },
                )
            if url.endswith("/pulls/3/files"):
                assert params is not None
                page = params["page"]
                if page == 1:
                    return Response(200, [file_payload(i) for i in range(100)])
                if page == 2:
                    return Response(200, [file_payload(100, patch="+" + "x" * 5000)])
            return Response(404, {"message": "not found"})

    monkeypatch.setattr(github_tools_mod.httpx, "AsyncClient", FakeGitHubClient)
    ctx = ToolContext(
        session_id="review-3",
        metadata={"repo_owner": "acme", "repo_name": "app", "pr_number": 3},
    )

    result = asyncio.run(mod.GetPRDiffTool().run(ctx, {}))

    assert not result.is_error
    assert [call[1] for call in calls if call[0].endswith("/files")] == [
        {"per_page": 100, "page": 1},
        {"per_page": 100, "page": 2},
    ]
    assert result.structured["total_files"] == 101
    assert result.structured["files_returned"] == 101
    assert result.structured["files_truncated"] is False
    assert result.structured["patches_truncated"] is True
    assert result.structured["files"][-1]["patch_preview"] == "+" + "x" * 1199
    assert result.structured["files"][-1]["patch_preview_truncated"] is True
    assert "Output truncated" in result.content


def test_get_pr_diff_caps_returned_file_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __init__(self, status_code: int, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = str(payload)

        def json(self):
            return self._payload

    class FakeGitHubClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url, *, headers=None, params=None):
            if url.endswith("/pulls/4"):
                return Response(
                    200,
                    {
                        "title": "Huge PR",
                        "body": None,
                        "changed_files": 250,
                        "head": {"ref": "feature", "sha": "def456"},
                        "base": {"ref": "main"},
                    },
                )
            if url.endswith("/pulls/4/files"):
                assert params is not None
                page = params["page"]
                start = (page - 1) * 100
                return Response(
                    200,
                    [
                        {
                            "filename": f"file_{index}.py",
                            "status": "modified",
                            "additions": 1,
                            "deletions": 0,
                            "patch": "+ok\n",
                        }
                        for index in range(start, start + 100)
                    ],
                )
            return Response(404, {"message": "not found"})

    monkeypatch.setattr(github_tools_mod.httpx, "AsyncClient", FakeGitHubClient)
    ctx = ToolContext(
        session_id="review-4",
        metadata={"repo_owner": "acme", "repo_name": "app", "pr_number": 4},
    )

    result = asyncio.run(mod.GetPRDiffTool().run(ctx, {}))

    assert not result.is_error
    assert result.structured["total_files"] == 250
    assert result.structured["files_returned"] == 200
    assert result.structured["files_truncated"] is True
    assert len(result.structured["files"]) == 200
    assert "Showing 200 of 250 changed files" in result.content


def test_post_pr_comment_updates_existing_marked_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    comments = []

    class Response:
        def __init__(self, status_code: int, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = str(payload)

        def json(self):
            return self._payload

    class FakeGitHubClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url, *, headers=None, params=None):
            calls.append(("GET", url, params))
            return Response(200, list(comments))

        async def post(self, url, *, headers=None, json=None):
            assert json is not None
            calls.append(("POST", url, json))
            comment = {
                "id": 101,
                "url": "https://api.github.com/repos/acme/app/issues/comments/101",
                "html_url": "https://github.com/acme/app/pull/1#issuecomment-101",
                "body": json["body"],
            }
            comments.append(comment)
            return Response(201, comment)

        async def patch(self, url, *, headers=None, json=None):
            assert json is not None
            calls.append(("PATCH", url, json))
            comments[0] = {**comments[0], "body": json["body"]}
            return Response(200, comments[0])

    monkeypatch.setattr(github_tools_mod.httpx, "AsyncClient", FakeGitHubClient)
    proxy = PassthroughProxy(StaticVault({"github_token": "ghp_test"}))
    ctx = ToolContext(
        session_id="pr-review-acme-app-1",
        proxy=proxy,
        metadata={"repo_owner": "acme", "repo_name": "app", "pr_number": 1},
    )
    tool = mod.PostPRCommentTool()

    first = asyncio.run(tool.run(ctx, {"body": "Initial review"}))
    second = asyncio.run(tool.run(ctx, {"body": "Updated review"}))

    assert first.structured["operation"] == "posted"
    assert second.structured["operation"] == "updated"
    assert [call[0] for call in calls] == ["GET", "POST", "GET", "PATCH"]
    assert len(comments) == 1
    assert "Initial review" not in comments[0]["body"]
    assert "Updated review" in comments[0]["body"]
    assert "<!-- cayu-pr-reviewer:pr-review-acme-app-1 -->" in comments[0]["body"]


def test_qa_policy_denies_raw_shell_but_allows_allowlisted() -> None:
    # Pure-logic guard on the safety rail that lets the agent "QA end-to-end".
    assert "python3" in mod._ALLOWED_QA_COMMANDS
    assert "rm" in mod._DENYLISTED_TOKENS
