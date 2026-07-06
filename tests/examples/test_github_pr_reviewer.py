"""Composition guard for the flagship `examples/github_pr_reviewer/` recipe.

Loaded from disk via importlib (examples are not importable from ``cayu``). This
does not hit the network or a model — it asserts the recipe *wires up*, so that a
public-API drift (a renamed export, a changed ``register_agent`` signature) breaks
this test instead of silently breaking the featured example.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cayu import Event, EventType, ScriptedModelProvider, SQLiteTaskStore, TaskStatus

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

_EXPECTED_TOOLS = {"get_pr_diff", "post_pr_comment", "read_file", "list_files", "exec_command"}
_DEFAULT_HEAD_REPO = object()


def _webhook_client(tmp_path: Path, *, webhook_secret: str | None = None):
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi test client is not installed")

    store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    return store, TestClient(mod.build_webhook_app(store, webhook_secret=webhook_secret))


def _pull_request_payload(
    *,
    action: str = "opened",
    head_repo: dict | None | object = _DEFAULT_HEAD_REPO,
) -> dict:
    if head_repo is _DEFAULT_HEAD_REPO:
        head_repo = {"clone_url": "https://github.com/octo/repo.git"}
    return {
        "action": action,
        "repository": {
            "name": "repo",
            "full_name": "octo/repo",
            "owner": {"login": "octo"},
            "clone_url": "https://github.com/octo/repo.git",
        },
        "pull_request": {
            "number": 12,
            "head": {
                "ref": "feature",
                "sha": "abc123",
                "repo": head_repo,
            },
            "base": {"ref": "main"},
        },
    }


def test_build_app_composes_the_reviewer(tmp_path: Path) -> None:
    app, _task_store = mod.build_app(
        tmp_path / "tasks.sqlite",
        tmp_path / "sandboxes",
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


def test_qa_policy_denies_raw_shell_but_allows_allowlisted() -> None:
    # Pure-logic guard on the safety rail that lets the agent "QA end-to-end".
    assert "python3" in mod._ALLOWED_QA_COMMANDS
    assert "rm" in mod._DENYLISTED_TOKENS


def test_enqueue_pr_review_is_idempotent_with_delivery_task_id(tmp_path: Path) -> None:
    async def run_case() -> None:
        store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        task_id = mod._github_delivery_task_id("delivery-123")
        first = await mod.enqueue_pr_review(
            store,
            owner="octo",
            repo="repo",
            pr_number=12,
            repo_url="https://github.com/octo/repo.git",
            head_ref="feature",
            head_sha="abc123",
            base_ref="main",
            task_id=task_id,
            github_delivery_id="delivery-123",
        )
        second = await mod.enqueue_pr_review(
            store,
            owner="octo",
            repo="repo",
            pr_number=12,
            repo_url="https://github.com/octo/repo.git",
            head_ref="feature",
            head_sha="abc123",
            base_ref="main",
            task_id=task_id,
            github_delivery_id="delivery-123",
        )

        assert first.id == second.id == task_id
        assert second.metadata["github_delivery_id"] == "delivery-123"

    asyncio.run(run_case())


def test_enqueue_pr_review_rejects_conflicting_duplicate_task_id(tmp_path: Path) -> None:
    async def run_case() -> None:
        store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        task_id = mod._github_delivery_task_id("delivery-123")
        await mod.enqueue_pr_review(
            store,
            owner="octo",
            repo="repo",
            pr_number=12,
            repo_url="https://github.com/octo/repo.git",
            head_ref="feature",
            head_sha="abc123",
            base_ref="main",
            task_id=task_id,
            github_delivery_id="delivery-123",
        )

        with pytest.raises(
            mod.DeliveryTaskConflictError,
            match="GitHub delivery id already exists with different task data",
        ):
            await mod.enqueue_pr_review(
                store,
                owner="octo",
                repo="repo",
                pr_number=12,
                repo_url="https://github.com/octo/repo.git",
                head_ref="different-feature",
                head_sha="def456",
                base_ref="main",
                task_id=task_id,
                github_delivery_id="delivery-123",
            )

    asyncio.run(run_case())


def test_enqueue_pr_review_preserves_non_duplicate_store_errors() -> None:
    class FailingStore:
        async def create_task(self, request):
            raise ValueError("backend unavailable")

        async def load_task(self, task_id: str):
            return None

    async def run_case() -> None:
        with pytest.raises(ValueError, match="backend unavailable"):
            await mod.enqueue_pr_review(
                FailingStore(),
                owner="octo",
                repo="repo",
                pr_number=12,
                repo_url="https://github.com/octo/repo.git",
                head_ref="feature",
                head_sha="abc123",
                base_ref="main",
                task_id=mod._github_delivery_task_id("delivery-123"),
                github_delivery_id="delivery-123",
            )

    asyncio.run(run_case())


def test_review_session_id_is_stable_per_delivery_and_distinct_per_update() -> None:
    first = mod._review_session_id(
        owner="octo",
        repo="repo",
        pr_number=12,
        task_id="task-a",
        head_sha="aaaaaaaaaaaabbbb",
    )
    duplicate_delivery = mod._review_session_id(
        owner="octo",
        repo="repo",
        pr_number=12,
        task_id="task-a",
        head_sha="aaaaaaaaaaaabbbb",
    )
    same_sha_new_delivery = mod._review_session_id(
        owner="octo",
        repo="repo",
        pr_number=12,
        task_id="task-c",
        head_sha="aaaaaaaaaaaabbbb",
    )
    updated_pr = mod._review_session_id(
        owner="octo",
        repo="repo",
        pr_number=12,
        task_id="task-b",
        head_sha="bbbbbbbbbbbbcccc",
    )

    assert first == duplicate_delivery
    assert first != same_sha_new_delivery
    assert first != updated_pr


def test_terminalize_claimed_task_marks_unstarted_failed_session_failed(tmp_path: Path) -> None:
    async def run_case() -> None:
        store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        task = await store.create_task(mod.TaskCreate(task_id="task-a", type="review"))
        task = await store.claim_task("worker-a")
        assert task is not None
        assert task.status == TaskStatus.CLAIMED

        finished = await mod._terminalize_claimed_task_if_needed(
            store,
            task_id="task-a",
            worker_id="worker-a",
            terminal_event=Event(
                type=EventType.SESSION_FAILED,
                session_id="sess-a",
                agent_name="pr-reviewer",
                payload={"error": "checkout failed", "error_type": "RuntimeError"},
            ),
        )

        assert finished is not None
        assert finished.status == TaskStatus.FAILED
        assert finished.error == {
            "status": "failed",
            "error": "checkout failed",
            "error_type": "RuntimeError",
        }

    asyncio.run(run_case())


def test_terminalize_claimed_task_keeps_already_terminal_task(tmp_path: Path) -> None:
    async def run_case() -> None:
        store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        await store.create_task(mod.TaskCreate(task_id="task-a", type="review"))
        await store.claim_task("worker-a")
        await store.complete_task("task-a", {"ok": True}, worker_id="worker-a")

        finished = await mod._terminalize_claimed_task_if_needed(
            store,
            task_id="task-a",
            worker_id="worker-a",
            terminal_event=Event(
                type=EventType.SESSION_FAILED,
                session_id="sess-a",
                agent_name="pr-reviewer",
                payload={"error": "late failure", "error_type": "RuntimeError"},
            ),
        )

        assert finished is not None
        assert finished.status == TaskStatus.COMPLETED
        assert finished.result == {"ok": True}

    asyncio.run(run_case())


def test_terminalize_claimed_task_records_completed_session_result(tmp_path: Path) -> None:
    async def run_case() -> None:
        store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        await store.create_task(mod.TaskCreate(task_id="task-a", type="review"))
        await store.claim_task("worker-a")

        finished = await mod._terminalize_claimed_task_if_needed(
            store,
            task_id="task-a",
            worker_id="worker-a",
            terminal_event=Event(
                type=EventType.SESSION_COMPLETED,
                session_id="sess-a",
                agent_name="pr-reviewer",
                payload={},
            ),
        )

        assert finished is not None
        assert finished.status == TaskStatus.COMPLETED
        assert finished.result == {"status": "completed"}

    asyncio.run(run_case())


def test_terminalize_claimed_task_records_interrupted_session_result(tmp_path: Path) -> None:
    async def run_case() -> None:
        store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        await store.create_task(mod.TaskCreate(task_id="task-a", type="review"))
        await store.claim_task("worker-a")

        finished = await mod._terminalize_claimed_task_if_needed(
            store,
            task_id="task-a",
            worker_id="worker-a",
            terminal_event=Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id="sess-a",
                agent_name="pr-reviewer",
                payload={},
            ),
        )

        assert finished is not None
        assert finished.status == TaskStatus.COMPLETED
        assert finished.result == {"status": "interrupted"}

    asyncio.run(run_case())


def test_terminalize_claimed_task_respects_worker_lease(tmp_path: Path) -> None:
    async def run_case() -> None:
        store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        await store.create_task(mod.TaskCreate(task_id="task-a", type="review"))
        await store.claim_task("worker-a")

        with pytest.raises(ValueError, match="does not own task"):
            await mod._terminalize_claimed_task_if_needed(
                store,
                task_id="task-a",
                worker_id="worker-b",
                terminal_event=Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id="sess-a",
                    agent_name="pr-reviewer",
                    payload={},
                ),
            )

    asyncio.run(run_case())


def test_checkout_source_uses_head_repo_for_fork_prs() -> None:
    payload = {
        "repository": {
            "full_name": "base/repo",
            "clone_url": "https://github.com/base/repo.git",
        }
    }
    pr = {
        "number": 12,
        "head": {
            "ref": "contributor-branch",
            "repo": {
                "full_name": "fork/repo",
                "clone_url": "https://github.com/fork/repo.git",
            },
        },
    }

    assert mod._checkout_source_from_pr_payload(pr, payload) == (
        "https://github.com/fork/repo.git",
        "contributor-branch",
    )


def test_checkout_source_rejects_missing_head_repo() -> None:
    payload = {
        "repository": {
            "full_name": "base/repo",
            "clone_url": "https://github.com/base/repo.git",
        }
    }
    pr = {
        "number": 12,
        "head": {
            "ref": "deleted-fork-branch",
            "repo": None,
        },
    }

    with pytest.raises(ValueError, match="head repository is unavailable"):
        mod._checkout_source_from_pr_payload(pr, payload)


def test_checkout_source_rejects_malformed_head_payload() -> None:
    payload = {"repository": {"full_name": "base/repo"}}

    with pytest.raises(ValueError, match="head payload must be a JSON object"):
        mod._checkout_source_from_pr_payload({"number": 12, "head": "bad"}, payload)

    with pytest.raises(ValueError, match="head repository is unavailable"):
        mod._checkout_source_from_pr_payload(
            {"number": 12, "head": {"ref": "feature", "repo": "bad"}},
            payload,
        )

    with pytest.raises(ValueError, match="head ref is missing"):
        mod._checkout_source_from_pr_payload(
            {
                "number": 12,
                "head": {"repo": {"clone_url": "https://github.com/fork/repo.git"}},
            },
            payload,
        )


def test_webhook_rejects_empty_secret(tmp_path: Path) -> None:
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite")

    with pytest.raises(ValueError, match="webhook_secret must be non-empty"):
        mod.build_webhook_app(store, webhook_secret="")


def test_webhook_rejects_missing_delivery_header(tmp_path: Path) -> None:
    _store, client = _webhook_client(tmp_path)
    response = client.post(
        "/webhooks/github",
        headers={"X-GitHub-Event": "pull_request"},
        content=json.dumps(_pull_request_payload()).encode(),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "missing X-GitHub-Delivery header"


def test_webhook_rejects_bad_signature(tmp_path: Path) -> None:
    _store, client = _webhook_client(tmp_path, webhook_secret="secret")
    response = client.post(
        "/webhooks/github",
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": "sha256=bad",
        },
        content=b"{}",
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "bad signature"


def test_webhook_rejects_invalid_json_payload(tmp_path: Path) -> None:
    _store, client = _webhook_client(tmp_path)
    response = client.post(
        "/webhooks/github",
        headers={"X-GitHub-Event": "pull_request"},
        content=b"{not-json",
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid JSON webhook payload"


def test_webhook_rejects_non_object_json_payload(tmp_path: Path) -> None:
    _store, client = _webhook_client(tmp_path)
    response = client.post(
        "/webhooks/github",
        headers={"X-GitHub-Event": "pull_request"},
        content=b"[]",
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "webhook payload must be a JSON object"


def test_webhook_rejects_incomplete_pull_request_payload(tmp_path: Path) -> None:
    _store, client = _webhook_client(tmp_path)
    response = client.post(
        "/webhooks/github",
        headers={
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "delivery-123",
        },
        content=json.dumps({"action": "opened", "repository": {"name": "repo"}}).encode(),
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid pull_request webhook payload"


def test_webhook_rejects_malformed_head_payload(tmp_path: Path) -> None:
    _store, client = _webhook_client(tmp_path)
    payload = _pull_request_payload()
    payload["pull_request"]["head"] = "bad"
    response = client.post(
        "/webhooks/github",
        headers={
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "delivery-123",
        },
        content=json.dumps(payload).encode(),
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid pull_request webhook payload"


def test_webhook_enqueues_idempotent_review_task(tmp_path: Path) -> None:
    store, client = _webhook_client(tmp_path)
    payload = _pull_request_payload()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": "delivery-123",
    }

    first = client.post("/webhooks/github", headers=headers, content=json.dumps(payload).encode())
    second = client.post("/webhooks/github", headers=headers, content=json.dumps(payload).encode())
    task_id = mod._github_delivery_task_id("delivery-123")
    task = asyncio.run(store.load_task(task_id))

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json() == {"task_id": task_id}
    assert task is not None
    assert task.input["owner"] == "octo"
    assert task.input["repo"] == "repo"
    assert task.input["pr_number"] == 12
    assert task.input["repo_url"] == "https://github.com/octo/repo.git"
    assert task.input["head_ref"] == "feature"
    assert task.input["head_sha"] == "abc123"
    assert task.input["base_ref"] == "main"
    assert task.metadata["github_delivery_id"] == "delivery-123"


def test_webhook_rejects_conflicting_delivery_payload(tmp_path: Path) -> None:
    _store, client = _webhook_client(tmp_path)
    first_payload = _pull_request_payload()
    second_payload = _pull_request_payload()
    second_payload["pull_request"]["head"]["sha"] = "def456"
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": "delivery-123",
    }

    first = client.post(
        "/webhooks/github", headers=headers, content=json.dumps(first_payload).encode()
    )
    second = client.post(
        "/webhooks/github", headers=headers, content=json.dumps(second_payload).encode()
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"] == ("GitHub delivery id already exists with different task data")


def test_webhook_accepts_valid_signature(tmp_path: Path) -> None:
    _store, client = _webhook_client(tmp_path, webhook_secret="secret")
    body = json.dumps(_pull_request_payload()).encode()
    signature = "sha256=" + hmac.new(b"secret", body, hashlib.sha256).hexdigest()

    response = client.post(
        "/webhooks/github",
        headers={
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "delivery-123",
            "X-Hub-Signature-256": signature,
        },
        content=body,
    )

    assert response.status_code == 200
    assert response.json() == {"task_id": mod._github_delivery_task_id("delivery-123")}


def test_webhook_rejects_unavailable_head_repo(tmp_path: Path) -> None:
    _store, client = _webhook_client(tmp_path)
    response = client.post(
        "/webhooks/github",
        headers={
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "delivery-123",
        },
        content=json.dumps(_pull_request_payload(head_repo=None)).encode(),
    )

    assert response.status_code == 422
    assert "head repository is unavailable" in response.json()["detail"]


def test_get_pr_diff_fetches_all_file_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __init__(self, payload: object, status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code
            self.text = "ok"

        def json(self) -> object:
            return self._payload

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            self.calls: list[dict] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, *, headers: dict, params: dict | None = None):
            self.calls.append({"url": url, "params": params})
            if url.endswith("/files"):
                assert params is not None
                page = params["page"]
                count = mod.GITHUB_PAGE_SIZE if page == 1 else 1
                return FakeResponse(
                    [
                        {
                            "filename": f"file-{page}-{index}.py",
                            "status": "modified",
                            "additions": 1,
                            "deletions": 0,
                            "patch": "diff",
                        }
                        for index in range(count)
                    ]
                )
            return FakeResponse(
                {
                    "title": "Example",
                    "body": None,
                    "head": {"ref": "feature", "sha": "abc"},
                    "base": {"ref": "main"},
                }
            )

    async def run_case() -> None:
        monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)
        result = await mod.GetPRDiffTool().run(
            SimpleNamespace(
                metadata={"repo_owner": "octo", "repo_name": "repo", "pr_number": 12},
                proxy=None,
                session_id="session",
            ),
            {},
        )

        assert result.is_error is False
        assert len(result.structured["files"]) == mod.GITHUB_PAGE_SIZE + 1

    asyncio.run(run_case())
