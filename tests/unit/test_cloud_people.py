import asyncio

import httpx
import pytest

from cayu.cloud_people import (
    CloudPeopleDirectory,
    PeopleDirectoryUnavailable,
    PeopleDirectoryUnsupported,
    Person,
    StaticPeopleDirectory,
    agent_people,
    find_person,
    people_directory,
)

URL = "https://cloud.example.test/v1/agent/people"
ENVIRON = {"CAYU_PEOPLE_URL": URL, "CAYU_ATTENTION_HANDOFF_TOKEN": "agent-token"}


def _transport(status: int, body: object, seen: list[httpx.Request] | None = None):
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handle)


def test_cloud_directory_reads_people_with_the_agent_credential():
    seen: list[httpx.Request] = []
    body = {
        "application_id": "invoices",
        "items": [
            {
                "user_id": "u1",
                "role": "admin",
                "name": "Ada",
                "email": "a@example.com",
                "on_agent": True,
            },
            {"user_id": "u2", "role": "member", "name": None, "email": None, "on_agent": False},
        ],
    }
    directory = people_directory(ENVIRON, transport=_transport(200, body, seen))
    assert isinstance(directory, CloudPeopleDirectory)
    people = asyncio.run(directory.list_people())
    assert people == (
        Person("u1", "admin", "Ada", "a@example.com", on_agent=True),
        Person("u2", "member"),
    )
    assert seen[0].headers["authorization"] == "Bearer agent-token"
    assert str(seen[0].url) == URL
    assert [p.user_id for p in asyncio.run(agent_people(directory))] == ["u1"]
    assert asyncio.run(find_person(directory, "u2")) == Person("u2", "member")
    assert asyncio.run(find_person(directory, "nobody")) is None


def test_cloud_without_the_read_is_reported_as_unsupported():
    directory = CloudPeopleDirectory(URL, "agent-token", transport=_transport(404, {}))
    with pytest.raises(PeopleDirectoryUnsupported, match="needs a Cayu Cloud"):
        asyncio.run(directory.list_people())
    on_cloud = {"CAYU_ATTENTION_HANDOFF_TOKEN": "agent-token"}
    with pytest.raises(PeopleDirectoryUnsupported, match="CAYU_PEOPLE_URL"):
        people_directory(on_cloud, local=StaticPeopleDirectory(()))
    with pytest.raises(PeopleDirectoryUnsupported, match="CAYU_PEOPLE_URL"):
        CloudPeopleDirectory.from_environment({})


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (401, {"detail": {"code": "agent_credential_invalid"}}, "agent_credential_invalid"),
        (
            503,
            {"detail": {"code": "organization_people_unavailable"}},
            "organization_people_unavailable",
        ),
        (200, {"items": [{"role": "admin"}]}, None),
        (200, {"items": "nope"}, None),
    ],
)
def test_cloud_errors_are_unavailable_and_never_carry_the_token(status, body, code):
    directory = CloudPeopleDirectory(URL, "agent-token", transport=_transport(status, body))
    with pytest.raises(PeopleDirectoryUnavailable) as raised:
        asyncio.run(directory.list_people())
    assert raised.value.code == code
    assert "agent-token" not in str(raised.value)


def test_network_failure_is_unavailable():
    def fail(request):
        raise httpx.ConnectError("refused", request=request)

    directory = CloudPeopleDirectory(URL, "agent-token", transport=httpx.MockTransport(fail))
    with pytest.raises(PeopleDirectoryUnavailable, match="ConnectError"):
        asyncio.run(directory.list_people())


def test_local_stand_in_reads_the_development_people(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.cayu.serve.identity_proxy]
default = "alice"
people = [
  { subject = "alice", role = "admin", name = "Alice", email = "alice@example.com" },
  { subject = "bob" },
]
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    directory = people_directory({})
    assert isinstance(directory, StaticPeopleDirectory)
    assert asyncio.run(directory.list_people()) == (
        Person("alice", "admin", "Alice", "alice@example.com", on_agent=True),
        Person("bob", "member", on_agent=True),
    )
    assert (
        asyncio.run(StaticPeopleDirectory.from_pyproject(tmp_path / "missing").list_people()) == ()
    )
    given = StaticPeopleDirectory([Person("x", "member")])
    assert people_directory({}, local=given) is given
    with pytest.raises(ValueError, match="twice"):
        StaticPeopleDirectory([Person("x", "member"), Person("x", "admin")])
