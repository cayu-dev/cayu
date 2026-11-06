"""Knowledge listings preview each entry's text, redacted whole before the cut."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.knowledge._retrieval_results import _knowledge_text_excerpt
from cayu.server import AuthContext, ServerConfig, create_server
from cayu.storage.memory import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeStatus,
)
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

_SECRET = "canary phrase alpha bravo charlie delta"
_INVOICE_TEXT = (
    "Retry failed invoice exports three times with a five minute backoff, then page the "
    "billing on-call. Exports that fail on a closed accounting period must not be retried "
    "at all: they need the controller to reopen the period first."
)
_PAYROLL_TEXT = "Retry failed payroll exports once, then open a ticket for the payroll team."
# The secret straddles the excerpt's 200-character cut, so cutting before redacting
# would leave its first words where the redactor can't recognize them.
_SECRET_TEXT = "word " * 37 + _SECRET + " and the rest of the note."


def _client(status: KnowledgeStatus) -> TestClient:
    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="invoice",
                title="Retry policy",
                text=_INVOICE_TEXT,
                importance=0.9,
                status=status,
            ),
            KnowledgeEntry(
                id="payroll",
                title="Retry policy",
                text=_PAYROLL_TEXT,
                importance=0.8,
                status=status,
            ),
            KnowledgeEntry(id="secret", text=_SECRET_TEXT, importance=0.1, status=status),
        ],
        access_scope=scope,
    )
    app = CayuApp(
        knowledge_store=store,
        knowledge_access_scope=scope,
        enable_logging=False,
        secret_redactor=SecretRedactor(_SECRET),
    )
    return TestClient(
        create_server(app, config=ServerConfig.protected(lambda _: AuthContext(subject="reviewer")))
    )


@pytest.mark.parametrize(
    ("status", "path"),
    [
        (KnowledgeStatus.PENDING, "/api/knowledge/pending"),
        (KnowledgeStatus.ACTIVE, "/api/knowledge"),
    ],
)
def test_listing_previews_the_text_of_entries_with_similar_titles(
    status: KnowledgeStatus, path: str
) -> None:
    with _client(status) as client:
        response = client.get(path)

    assert response.status_code == 200, response.text
    rows = {row["entry_id"]: row for row in response.json()["entries"]}
    assert rows["invoice"]["title"] == rows["payroll"]["title"] == "Retry policy"
    assert rows["invoice"]["text_preview"] == _knowledge_text_excerpt(_INVOICE_TEXT)
    assert rows["invoice"]["text_preview"].startswith("Retry failed invoice exports")
    assert rows["invoice"]["text_preview"].endswith("…")
    assert len(rows["invoice"]["text_preview"]) <= 200
    assert rows["payroll"]["text_preview"] == _PAYROLL_TEXT


@pytest.mark.parametrize(
    ("status", "path"),
    [
        (KnowledgeStatus.PENDING, "/api/knowledge/pending"),
        (KnowledgeStatus.ACTIVE, "/api/knowledge"),
    ],
)
def test_listing_preview_redacts_the_whole_text_before_cutting_it(
    status: KnowledgeStatus, path: str
) -> None:
    with _client(status) as client:
        response = client.get(path)

    assert _SECRET[:9] in _knowledge_text_excerpt(_SECRET_TEXT)
    assert response.status_code == 200, response.text
    assert _SECRET[:9] not in response.text
    [row] = [row for row in response.json()["entries"] if row["entry_id"] == "secret"]
    redacted = _SECRET_TEXT.replace(_SECRET, REDACTED_SECRET)
    assert row["text_preview"] == _knowledge_text_excerpt(redacted)


def test_inbox_titles_an_untitled_proposal_with_the_start_of_its_text() -> None:
    with _client(KnowledgeStatus.PENDING) as client:
        response = client.get("/api/inbox")

    assert response.status_code == 200, response.text
    assert _SECRET[:9] not in response.text
    items = {item["knowledge"]["entry_id"]: item for item in response.json()["items"]}
    assert items["invoice"]["title"] == "Retry policy"
    assert items["invoice"]["detail"] == _INVOICE_TEXT
    redacted = _SECRET_TEXT.replace(_SECRET, REDACTED_SECRET)
    assert items["secret"]["title"] == _knowledge_text_excerpt(redacted)
    assert items["secret"]["detail"] == redacted.strip()
    assert _SECRET[:9] not in json.dumps(items)
