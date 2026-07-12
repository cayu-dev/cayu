from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx


@dataclass
class FakeGitHubState:
    create_pull_requests: int = 0
    list_pull_requests: int = 0
    created_pulls: list[dict[str, Any]] = field(default_factory=list)


class FakeGitHubServer:
    """Small GitHub-shaped HTTP service used by the repo tournament.

    The example talks to this service through real HTTP. Production can point the
    same client at api.github.com without changing the orchestration boundary.
    """

    def __init__(self) -> None:
        self.state = FakeGitHubState()
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlsplit(self.path)
                path = parsed.path
                if path == "/repos/acme/calculator/pulls/1":
                    self._json(
                        200,
                        {
                            "number": 1,
                            "title": "Handle divide by zero",
                            "body": "divide(1, 0) should raise ValueError",
                            "changed_files": 1,
                            "head": {"ref": "bug", "sha": "fixture-head"},
                            "base": {"ref": "main"},
                        },
                    )
                    return
                if path == "/repos/acme/calculator/pulls/1/files":
                    self._json(
                        200,
                        [
                            {
                                "filename": "calculator.py",
                                "status": "modified",
                                "additions": 0,
                                "deletions": 0,
                                "patch": "def divide(a, b): return a / b",
                            }
                        ],
                    )
                    return
                if path == "/repos/acme/calculator/pulls":
                    state.list_pull_requests += 1
                    query = parse_qs(parsed.query)
                    requested_head = query.get("head", [""])[0].removeprefix("acme:")
                    requested_base = query.get("base", [""])[0]
                    matching = [
                        pull
                        for pull in state.created_pulls
                        if pull.get("state") == "open"
                        and pull.get("head") == requested_head
                        and pull.get("base") == requested_base
                    ]
                    self._json(200, matching)
                    return
                self._json(404, {"message": "Not Found"})

            def do_POST(self) -> None:
                path = urlsplit(self.path).path
                if path != "/repos/acme/calculator/pulls":
                    self._json(404, {"message": "Not Found"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                state.create_pull_requests += 1
                created = {
                    **payload,
                    "number": 101 + len(state.created_pulls),
                    "state": "open",
                    "html_url": "https://github.example/acme/calculator/pull/101",
                }
                state.created_pulls.append(created)
                self._json(201, created)

            def log_message(self, format: str, *args: object) -> None:
                return

            def _json(self, status: int, payload: object) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        address = self._server.server_address
        host, port = str(address[0]), int(address[1])
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeGitHubServer:
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class GitHubClient:
    def __init__(self, base_url: str, *, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get_pull(self, owner: str, repo: str, number: int) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo}/pulls/{number}",
                headers=self._headers(),
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("GitHub pull response was not an object.")
        return payload

    async def list_pull_files(
        self,
        owner: str,
        repo: str,
        number: int,
    ) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo}/pulls/{number}/files",
                headers=self._headers(),
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise RuntimeError("GitHub files response was not an object list.")
        return payload

    async def create_pull(
        self,
        owner: str,
        repo: str,
        *,
        title: str,
        head: str,
        base: str,
        body: str,
    ) -> dict[str, Any]:
        headers = self._headers()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.base_url}/repos/{owner}/{repo}/pulls",
                headers=headers,
                json={"title": title, "head": head, "base": base, "body": body},
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("GitHub create-pull response was not an object.")
        return payload

    async def list_open_pulls(
        self,
        owner: str,
        repo: str,
        *,
        head: str,
        base: str,
    ) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo}/pulls",
                headers=self._headers(),
                params={"state": "open", "head": f"{owner}:{head}", "base": base},
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise RuntimeError("GitHub pull-list response was not an object list.")
        return payload

    async def ensure_pull(
        self,
        owner: str,
        repo: str,
        *,
        title: str,
        head: str,
        base: str,
        body: str,
    ) -> dict[str, Any]:
        existing = await self.list_open_pulls(owner, repo, head=head, base=base)
        if existing:
            return existing[0]
        return await self.create_pull(
            owner,
            repo,
            title=title,
            head=head,
            base=base,
            body=body,
        )
