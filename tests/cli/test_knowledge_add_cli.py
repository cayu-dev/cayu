"""`cayu knowledge add`: one command adds a knowledge entry through the agent's store rules."""

from __future__ import annotations

import asyncio
import io
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.cli import main
from cayu.knowledge.records import KnowledgeStatus
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.search import KnowledgeQuery
from cayu.server import ServerConfig, create_server
from cayu.storage.memory import InMemoryKnowledgeStore

_NAMESPACE = "acme-agent"
_ORGANIZATIONS = KnowledgeAccessScope(
    allowed_namespaces=[_NAMESPACE],
    allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    session_label_keys=("organization",),
)


@pytest.fixture
def agent(monkeypatch):
    """A running agent with organization-scoped knowledge, which the CLI reaches."""

    from cayu.cli import _control_plane_actions, knowledge
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.cli._knowledge_readers import ApiKnowledgeReader

    store = InMemoryKnowledgeStore()
    app = CayuApp(
        knowledge_store=store, knowledge_access_scope=_ORGANIZATIONS, enable_logging=False
    )
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        monkeypatch.setattr(
            _control_plane_actions,
            "open_action_api",
            lambda args: CayuApi("http://testserver", http=client),
        )
        monkeypatch.setattr(
            knowledge,
            "open_knowledge_reader",
            lambda target: ApiKnowledgeReader(
                CayuApi("http://testserver", http=client), source={"kind": "server"}
            ),
        )
        yield store


def _run(capsys, *args: str) -> tuple[int, dict]:
    code = main(["knowledge", *args])
    return code, json.loads(capsys.readouterr().out)


def test_add_keeps_a_labelled_entry_that_its_organization_recalls(agent, capsys):
    code, added = _run(
        capsys,
        "add",
        "Invoices over 10,000 need a second approver.",
        "--kind",
        "policy",
        "--title",
        "Invoice approvals",
        "--label",
        "organization=acme",
    )
    assert code == 0
    result = added["result"]
    assert result["created"] is True
    assert result["status"] == "active"
    assert result["labels"] == {"organization": "acme"}
    assert result["namespace"] == _NAMESPACE

    code, listed = _run(
        capsys, "list", "--label", "organization=acme", "--server-url", "http://localhost", "--json"
    )
    assert code == 0
    assert [entry["entry_id"] for entry in listed["entries"]] == [result["entry_id"]]

    def recall(organization: str) -> list[str]:
        scope = KnowledgeAccessScope(
            allowed_namespaces=[_NAMESPACE], required_labels={"organization": organization}
        )
        found = asyncio.run(
            agent.search(
                KnowledgeQuery(text="second approver", namespace=_NAMESPACE), access_scope=scope
            )
        )
        return [hit.entry.id for hit in found.hits]

    assert recall("acme") == [result["entry_id"]]
    assert recall("globex") == []

    # The same entry again changes nothing.
    code, again = _run(
        capsys,
        "add",
        "Invoices over 10,000 need a second approver.",
        "--kind",
        "policy",
        "--title",
        "Invoice approvals",
        "--label",
        "organization=acme",
    )
    assert code == 0
    assert again["result"]["created"] is False
    assert again["result"]["entry_id"] == result["entry_id"]


def test_add_refuses_an_entry_without_the_organization_label(agent, capsys):
    code, refused = _run(capsys, "add", "Acme pays net 30.")
    assert code == 1
    message = refused["error"]["message"]
    assert "HTTP 422" in message
    assert "no 'organization' label" in message
    assert "session_label_keys" in message
    code, listed = _run(capsys, "list", "--server-url", "http://localhost", "--json")
    assert code == 0 and listed["entries"] == []


def test_add_reads_a_file_or_stdin(agent, capsys, tmp_path, monkeypatch):
    policy = tmp_path / "policy.md"
    policy.write_text("# Refunds\n\nRefunds within 30 days are automatic.\n", encoding="utf-8")
    code, added = _run(capsys, "add", "--file", str(policy), "--label", "organization=acme")
    assert code == 0
    assert added["result"]["text_preview"].startswith("# Refunds")

    stdin = io.TextIOWrapper(io.BytesIO(b"Globex ships on Fridays."), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", stdin)
    code, piped = _run(capsys, "add", "--file", "-", "--label", "organization=globex")
    assert code == 0
    assert piped["result"]["labels"] == {"organization": "globex"}


def test_add_pending_without_authentication_is_refused_with_why(agent, capsys):
    # Keeping refuses the one who added a proposal, and an unauthenticated server has one caller.
    code, refused = _run(capsys, "add", "Proposal", "--label", "organization=acme", "--pending")
    assert code == 1
    assert "could never be kept" in refused["error"]["message"]


def test_add_reports_an_oversized_file_as_too_long(tmp_path, monkeypatch, capsys):
    from cayu.cli import _control_plane_actions

    monkeypatch.setattr(
        _control_plane_actions, "open_action_api", lambda args: pytest.fail("nothing is sent")
    )
    large = tmp_path / "large.md"
    # A multi-byte character straddles the read bound, which a decode would trip on.
    large.write_text("a" + "\u898f\u5247" * 100_000, encoding="utf-8")
    code, refused = _run(capsys, "add", "--file", str(large), "--json")
    assert code == 1
    assert "longer than one entry holds" in refused["error"]["message"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["   "], "text"),
        (["text", "--label", "organization"], "KEY=VALUE"),
        (["text", "--kind", " "], "--kind"),
    ],
)
def test_add_refuses_bad_input_before_sending(args, message, monkeypatch, capsys):
    from cayu.cli import _control_plane_actions

    def refuse(_args):
        raise AssertionError("nothing is sent")

    monkeypatch.setattr(_control_plane_actions, "open_action_api", refuse)
    code, refused = _run(capsys, "add", *args, "--json")
    assert code == 1
    assert message in refused["error"]["message"]


def test_add_needs_text_or_a_file(capsys):
    with pytest.raises(SystemExit):
        main(["knowledge", "add"])
    assert "required" in capsys.readouterr().err
