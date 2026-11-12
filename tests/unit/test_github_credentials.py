"""Local GitHub credentials: a token, a GitHub App's signed sign-in and its installation
tokens, and the choice between them from the environment."""

from __future__ import annotations

import asyncio
import base64
import gc
import json
import weakref
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from cayu.connections import CloudConnectionBroker, ConnectionCallError, LocalConnections
from cayu.connectors import github
from cayu.connectors.github import credentials

REPO = "acme/widgets"
NOW = 1_800_000_000.0
PERMISSIONS = {"pull_requests": "write", "issues": "write"}
_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PRIVATE_KEY = _KEY.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.TraditionalOpenSSL,
    serialization.NoEncryption(),
).decode()


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [NOW]
    monkeypatch.setattr(credentials, "_now", lambda: now[0])
    return now


def _decoded(part: str) -> Any:
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


class FakeGitHub:
    """GitHub's App endpoints and one repository route, recording each request."""

    def __init__(self, *, refuse_tokens: int = 0, installed: bool = True) -> None:
        self.requests: list[httpx.Request] = []
        self.minted = 0
        self.refuse_tokens = refuse_tokens
        self.installed = installed
        #: The App's installation on the repository; a reinstall gives it a new id.
        self.installation = 42
        self.transport = httpx.MockTransport(self._handle)

    def made(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if (r.method, r.url.path) == (method, path)]

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/installation"):
            if not self.installed:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json={"id": self.installation})
        if path.startswith("/app/installations/") and path != (
            f"/app/installations/{self.installation}/access_tokens"
        ):
            return httpx.Response(404, json={"message": "Not Found"})
        if path.endswith("/access_tokens"):
            await asyncio.sleep(0.01)
            self.minted += 1
            expires = datetime.fromtimestamp(credentials._now() + 3600, UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            return httpx.Response(201, json={"token": f"ghs_{self.minted}", "expires_at": expires})
        if path == "/app":
            return httpx.Response(200, json={"slug": "cayu-helper"})
        token = request.headers["authorization"].removeprefix("Bearer ")
        if int(token.removeprefix("ghs_")) <= self.refuse_tokens:
            return httpx.Response(401, json={"message": "Bad credentials"})
        return httpx.Response(200, json={"token": token})


def _app(fake: FakeGitHub, **kwargs: Any) -> github.GitHubAppConnections:
    return github.app_connections(
        "1234", PRIVATE_KEY, permissions=PERMISSIONS, transport=fake.transport, **kwargs
    )


def _get(app: github.GitHubAppConnections, repository: str = REPO) -> Any:
    response = asyncio.run(
        app.request(
            "github",
            "GET",
            f"/repos/{repository}/pulls",
            headers={"x-cayu-github-repository": repository},
        )
    )
    return json.loads(response.body)


def test_the_apps_sign_in_token_is_an_rs256_jwt_the_public_key_verifies() -> None:
    fake = FakeGitHub()
    _get(_app(fake))
    [lookup] = fake.made("GET", f"/repos/{REPO}/installation")
    header, claims, signature = lookup.headers["authorization"].removeprefix("Bearer ").split(".")

    _KEY.public_key().verify(
        base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4)),
        f"{header}.{claims}".encode(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    assert _decoded(header) == {"alg": "RS256", "typ": "JWT"}
    assert _decoded(claims) == {"iat": NOW - 60, "exp": NOW + 540, "iss": "1234"}


def test_an_installation_token_is_minted_for_the_repository_and_reused() -> None:
    fake = FakeGitHub()
    app = _app(fake)
    assert _get(app) == {"token": "ghs_1"}
    assert _get(app) == {"token": "ghs_1"}

    [mint] = fake.made("POST", "/app/installations/42/access_tokens")
    assert json.loads(mint.content) == {"repositories": ["widgets"], "permissions": PERMISSIONS}
    assert len(fake.made("GET", f"/repos/{REPO}/installation")) == 1
    calls = fake.made("GET", f"/repos/{REPO}/pulls")
    assert all("x-cayu-github-repository" not in call.headers for call in calls)


def test_a_token_is_replaced_inside_five_minutes_of_its_expiry(clock: list[float]) -> None:
    fake = FakeGitHub()
    app = _app(fake)
    _get(app)
    clock[0] = NOW + 3600 - 301
    assert _get(app) == {"token": "ghs_1"}
    clock[0] = NOW + 3600 - 299
    assert _get(app) == {"token": "ghs_2"}
    assert len(fake.made("GET", f"/repos/{REPO}/installation")) == 1


def test_concurrent_misses_mint_one_token() -> None:
    fake = FakeGitHub()
    app = _app(fake)

    async def five() -> list[Any]:
        return await asyncio.gather(
            *(
                app.request(
                    "github",
                    "GET",
                    f"/repos/{REPO}/pulls",
                    headers={"x-cayu-github-repository": REPO},
                )
                for _ in range(5)
            )
        )

    assert [json.loads(r.body) for r in asyncio.run(five())] == [{"token": "ghs_1"}] * 5
    assert fake.minted == 1


def test_a_refused_token_is_dropped_and_the_call_resent_once_with_a_new_one() -> None:
    fake = FakeGitHub(refuse_tokens=1)
    assert _get(_app(fake)) == {"token": "ghs_2"}
    assert len(fake.made("GET", f"/repos/{REPO}/installation")) == 2

    refused_twice = FakeGitHub(refuse_tokens=2)
    response = asyncio.run(
        _app(refused_twice).request(
            "github", "GET", f"/repos/{REPO}", headers={"x-cayu-github-repository": REPO}
        )
    )
    assert response.status == 401
    assert refused_twice.minted == 2


def test_an_app_not_installed_on_the_repository_says_so() -> None:
    fake = FakeGitHub(installed=False)
    with pytest.raises(github.GitHubAppNotInstalled) as raised:
        _get(_app(fake))
    assert raised.value.repository == REPO
    assert "isn't installed on acme/widgets" in str(raised.value)

    gh = github.GitHubClient(_app(fake))
    with pytest.raises(github.GitHubAppNotInstalled):
        asyncio.run(gh.list(f"/repos/{REPO}/pulls"))


def test_the_app_refuses_calls_with_no_repository_or_another_one() -> None:
    app = _app(FakeGitHub(), repositories=["acme/widgets"])

    def refused(path: str, headers: dict[str, str], slot: str = "github") -> str | None:
        with pytest.raises(ConnectionCallError) as raised:
            asyncio.run(app.request(slot, "GET", path, headers=headers))
        return raised.value.code

    assert refused("/rate_limit", {}) == "path_not_allowed"
    assert refused(f"/repos/{REPO}", {"x-cayu-github-repository": "acme/other"}) == (
        "path_not_allowed"
    )
    assert refused("/repos/acme/other", {"x-cayu-github-repository": "acme/other"}) == (
        "path_not_allowed"
    )
    assert refused(f"/repos/{REPO}", {"x-cayu-github-repository": REPO}, "other") == (
        "connection_unbound"
    )
    assert _get(app, "Acme/Widgets") == {"token": "ghs_1"}


def test_the_cache_keeps_a_bounded_number_of_repositories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credentials, "MAX_CACHED_REPOSITORIES", 2)
    fake = FakeGitHub()
    app = _app(fake)
    for repository in ("acme/a", "acme/b", "acme/c", "acme/a"):
        _get(app, repository)
    assert len(fake.made("GET", "/repos/acme/a/installation")) == 2
    assert fake.minted == 4


def test_login_is_the_apps_bot_name() -> None:
    fake = FakeGitHub()
    app = _app(fake)
    assert asyncio.run(app.login()) == "cayu-helper[bot]"
    assert asyncio.run(app.login()) == "cayu-helper[bot]"
    assert len(fake.made("GET", "/app")) == 1


def test_githubs_refusal_of_the_app_names_its_words_and_wait() -> None:
    def limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"message": "API rate limit exceeded for installation"},
            headers={"x-ratelimit-remaining": "0", "retry-after": "30"},
        )

    app = github.app_connections(
        "1234", PRIVATE_KEY, permissions=PERMISSIONS, transport=httpx.MockTransport(limited)
    )
    with pytest.raises(ConnectionCallError) as raised:
        _get(app)
    assert (raised.value.code, raised.value.retry_after) == ("too_many_calls", 30.0)
    assert "API rate limit exceeded for installation" in str(raised.value)


def test_the_private_key_never_shows_and_permissions_are_required() -> None:
    app = _app(FakeGitHub())
    assert "PRIVATE" not in repr(app) and "1234" in repr(app)
    with pytest.raises(ValueError, match="isn't an unencrypted PEM") as raised:
        github.app_connections("1234", "-----BEGIN junk", permissions=PERMISSIONS)
    assert "junk" not in str(raised.value)
    with pytest.raises(ValueError, match="Name the App permissions"):
        github.app_connections("1234", PRIVATE_KEY, permissions={})
    with pytest.raises(ValueError, match="read, write or admin"):
        github.app_connections("1234", PRIVATE_KEY, permissions={"issues": "all"})


def test_connections_from_the_environment_choose_one_credential() -> None:
    def choose(**environ: str) -> object:
        return github.connections_from_environment(environ, permissions=PERMISSIONS)

    assert choose() is None
    assert isinstance(choose(GITHUB_TOKEN="ghp_x"), LocalConnections)
    escaped = PRIVATE_KEY.replace("\n", "\\n")
    app = choose(GITHUB_APP_ID="1234", GITHUB_APP_PRIVATE_KEY=escaped)
    assert isinstance(app, github.GitHubAppConnections)
    assert app.permissions == PERMISSIONS
    broker = choose(
        CAYU_CONNECTIONS_URL="https://cloud.example/v1/agent-connections",
        CAYU_CONNECTIONS_TOKEN="cayu_conn_x",
        GITHUB_TOKEN="ghp_x",
    )
    assert isinstance(broker, CloudConnectionBroker)

    with pytest.raises(ValueError, match="not both"):
        choose(GITHUB_TOKEN="ghp_x", GITHUB_APP_ID="1234", GITHUB_APP_PRIVATE_KEY=PRIVATE_KEY)
    with pytest.raises(ValueError, match="must be set together"):
        choose(GITHUB_APP_ID="1234")


def test_a_workflow_on_the_app_lists_pulls_and_comments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/installation"):
            return httpx.Response(200, json={"id": 42})
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(
                201, json={"token": "ghs_1", "expires_at": "2099-01-01T00:00:00Z"}
            )
        assert request.headers["authorization"] == "Bearer ghs_1"
        if request.method == "GET":
            return httpx.Response(200, json=[{"number": 7}])
        return httpx.Response(201, json={"id": 1})

    app = github.app_connections(
        "1234", PRIVATE_KEY, permissions=PERMISSIONS, transport=httpx.MockTransport(handler)
    )
    gh = github.GitHubClient(app)

    async def workflow() -> Any:
        pulls = await gh.list(f"/repos/{REPO}/pulls", query={"state": "open"})
        return await gh.post(
            f"/repos/{REPO}/issues/{pulls.items[0]['number']}/comments", json={"body": "Hi"}
        )

    assert asyncio.run(workflow()) == {"id": 1}


def test_tokens_never_show_in_a_repr() -> None:
    app = _app(FakeGitHub())
    _get(app)
    assert "ghs_1" not in repr(app._entries)


def test_the_app_calls_only_a_repositorys_api_and_graphql() -> None:
    app = _app(FakeGitHub())
    with pytest.raises(ConnectionCallError, match="only a repository's own API") as raised:
        asyncio.run(
            app.request(
                "github", "GET", "/orgs/acme/members", headers={"x-cayu-github-repository": REPO}
            )
        )
    assert raised.value.code == "path_not_allowed"
    response = asyncio.run(
        app.request(
            "github", "POST", "/graphql", headers={"x-cayu-github-repository": REPO}, body=b"{}"
        )
    )
    assert json.loads(response.body) == {"token": "ghs_1"}


def _gathered(app: github.GitHubAppConnections, repositories: list[str]) -> list[Any]:
    async def calls() -> list[Any]:
        responses = await asyncio.gather(
            *(
                app.request(
                    "github",
                    "GET",
                    f"/repos/{repository}/pulls",
                    headers={"x-cayu-github-repository": repository},
                )
                for repository in repositories
            )
        )
        return [json.loads(response.body) for response in responses]

    return asyncio.run(calls())


def test_one_app_client_serves_one_event_loop_after_another(clock: list[float]) -> None:
    fake = FakeGitHub()
    app = _app(fake)
    assert _gathered(app, [REPO] * 5) == [{"token": "ghs_1"}] * 5
    clock[0] = NOW + 3600
    assert _gathered(app, [REPO] * 5) == [{"token": "ghs_2"}] * 5
    assert fake.minted == 2


def test_a_repository_being_minted_for_is_never_evicted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials, "MAX_CACHED_REPOSITORIES", 1)
    fake = FakeGitHub()
    app = _app(fake)
    assert _gathered(app, ["acme/a", "acme/b", "acme/a"]) == [
        {"token": "ghs_1"},
        {"token": "ghs_2"},
        {"token": "ghs_1"},
    ]
    assert fake.minted == 2


def test_the_app_reads_its_rate_limit_for_a_repository() -> None:
    app = _app(FakeGitHub())
    response = asyncio.run(
        app.request("github", "GET", "/rate_limit", headers={"x-cayu-github-repository": REPO})
    )
    assert json.loads(response.body) == {"token": "ghs_1"}
    with pytest.raises(ConnectionCallError, match="only for a repository named"):
        asyncio.run(app.request("github", "GET", "/rate_limit"))


def test_an_app_client_keeps_no_closed_event_loop_alive(clock: list[float]) -> None:
    app = _app(FakeGitHub())
    loops: list[weakref.ref[asyncio.AbstractEventLoop]] = []

    async def five() -> None:
        loops.append(weakref.ref(asyncio.get_running_loop()))
        await asyncio.gather(
            *(
                app.request(
                    "github", "GET", f"/repos/{REPO}", headers={"x-cayu-github-repository": REPO}
                )
                for _ in range(5)
            )
        )

    for step in range(3):
        clock[0] = NOW + step * 3600
        asyncio.run(five())
    gc.collect()
    assert [loop() is None for loop in loops[:-1]] == [True, True]


def test_a_removed_installation_is_looked_up_again_when_its_token_is_refreshed(
    clock: list[float],
) -> None:
    fake = FakeGitHub()
    app = _app(fake)
    assert _get(app) == {"token": "ghs_1"}

    fake.installation = 99
    clock[0] = NOW + 3600
    assert _get(app) == {"token": "ghs_2"}
    assert _get(app) == {"token": "ghs_2"}
    assert [(r.method, r.url.path) for r in fake.requests[-5:-2]] == [
        ("POST", "/app/installations/42/access_tokens"),
        ("GET", f"/repos/{REPO}/installation"),
        ("POST", "/app/installations/99/access_tokens"),
    ]


def test_a_freshly_looked_up_installation_that_is_gone_raises_once() -> None:
    def gone(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/installation"):
            return httpx.Response(200, json={"id": 7})
        return httpx.Response(404, json={"message": "Not Found"})

    app = github.app_connections(
        "1234", PRIVATE_KEY, permissions=PERMISSIONS, transport=httpx.MockTransport(gone)
    )
    with pytest.raises(github.GitHubAppNotInstalled):
        _get(app)


def test_the_client_cant_read_the_rate_limit_through_an_app_slot_yet() -> None:
    gh = github.GitHubClient(_app(FakeGitHub()))
    with pytest.raises(github.GitHubConnectionRefused, match="doesn't send for it yet") as raised:
        asyncio.run(gh.get("/rate_limit"))
    assert raised.value.code == "path_not_allowed"
