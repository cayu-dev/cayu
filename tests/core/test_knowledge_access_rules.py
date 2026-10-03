"""Shared authorization rules retain their boundaries independently of a backend."""

import inspect
import os
import pickle
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_type_hints

import pytest

import cayu
from cayu.knowledge import _access_rules as rules
from cayu.knowledge.activation_contracts import (
    KnowledgeActivationConflict,
    _knowledge_activation_retirement,
)
from cayu.knowledge.changes import KnowledgeChange, KnowledgeChangeKind
from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus
from cayu.knowledge.scopes import (
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    _knowledge_access_snapshot,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _entry(**updates):
    return KnowledgeEntry(
        id="entry", namespace="example", text="content", labels={"team": "sales"}
    ).model_copy(update=updates)


def _relation(entry):
    return rules._knowledge_relation_access_snapshot(
        subject_exact=entry, subject_current=entry, object_exact=entry, object_current=entry
    )


def _change(kind, *, revision=1):
    return KnowledgeChange(
        id="change",
        sequence=1,
        kind=kind,
        entry_id="entry",
        entry_revision=revision,
        committed_at=_NOW,
        relation_id="relation" if kind is KnowledgeChangeKind.RELATION_PUBLISHED else None,
    )


def test_knowledge_access_rules_compose_without_storage_implementations():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from cayu.knowledge import _access_rules as rules
from cayu.knowledge.records import KnowledgeEntry
from cayu.knowledge.scopes import KnowledgeAccessScope

entry = KnowledgeEntry(id="entry", namespace="example", text="content")
scope = KnowledgeAccessScope.for_namespace("example")
relation = rules._knowledge_relation_access_snapshot(
    subject_exact=entry, subject_current=entry, object_exact=entry, object_current=entry,
)
maintenance = rules._knowledge_maintenance_access_snapshot([entry])
assert rules._knowledge_scope_allows_entry(scope, entry)
assert rules._knowledge_scope_allows_relation_access_snapshot(scope, relation)
assert rules._knowledge_scope_allows_maintenance_access_snapshot(scope, maintenance)
assert not rules._knowledge_scope_allows_entry(KnowledgeAccessScope.for_namespace("foreign"), entry)
assert not {
    "cayu.storage.memory", "cayu.storage.knowledge_sqlite", "cayu.storage.postgres",
    "cayu.knowledge.maintenance_persistence", "cayu.knowledge.maintenance_governance",
}.intersection(sys.modules)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_access_rule_legacy_imports_pickles_and_annotations_remain_compatible():
    from cayu.storage import memory

    assert memory._KNOWLEDGE_RETIREMENT_STATUSES is rules._KNOWLEDGE_RETIREMENT_STATUSES
    owned = {
        name: value
        for name, value in vars(rules).items()
        if getattr(value, "__module__", None) == rules.__name__
    }
    for name, canonical in owned.items():
        assert getattr(memory, name) is canonical
        assert pickle.loads(f"ccayu.storage.memory\n{name}\n.".encode()) is canonical
        get_type_hints(canonical)
        if inspect.isclass(canonical):
            for value in vars(canonical).values():
                if isinstance(value, classmethod):
                    get_type_hints(value.__func__)


def test_access_snapshots_copy_inputs_and_preserve_json_and_pickle_round_trips():
    entry = _entry()
    relation = _relation(entry)
    maintenance = rules._knowledge_maintenance_access_snapshot([entry])
    audience = rules._KnowledgeChangeAudience(
        kind="after", snapshot=_knowledge_access_snapshot(entry)
    )
    entry.labels["team"] = "changed"
    assert relation.subject_exact.labels == maintenance.entries[0].labels == {"team": "sales"}
    assert audience.snapshot.labels == {"team": "sales"}
    relation.subject_exact.labels["team"] = "detached"
    assert relation.subject_current.labels == {"team": "sales"}
    for family, value in (("relation", relation), ("maintenance", maintenance)):
        encoded = getattr(rules, f"_knowledge_{family}_access_snapshot_json")(value)
        decoded = getattr(rules, f"_parse_knowledge_{family}_access_snapshot_json")(encoded)
        assert decoded == value and decoded is not value
        decoded = pickle.loads(pickle.dumps(value))
        assert type(decoded) is type(value) and decoded == value
    with pytest.raises(ValueError, match="requires reviewed entries"):
        rules._knowledge_maintenance_access_snapshot([])
    with pytest.raises(ValueError, match="boolean"):
        rules._KnowledgeChangeAudience(
            kind="after", snapshot=audience.snapshot, requires_include_expired=1
        )


def test_shared_access_rules_preserve_scope_and_resource_predicates():
    from cayu.resource_access import encode_scope
    from cayu.sessions.access import SessionAccessRule, SessionAccessScope, SessionAccessSelector

    entry = _entry()
    scope = KnowledgeAccessScope.for_namespace("example")
    assert rules._knowledge_scope_allows_entry(scope, entry)
    for restriction in (
        {"allowed_namespaces": ["foreign"]},
        {"required_labels": {"team": "support"}},
        {"allowed_source_types": ["foreign"]},
        {"allowed_source_ids": ["foreign"]},
        {"allowed_statuses": [KnowledgeStatus.PENDING]},
    ):
        denied = scope.model_copy(update=restriction)
        assert not rules._knowledge_scope_allows_entry(denied, entry)
        with pytest.raises(KnowledgeAccessDenied, match="write"):
            rules._require_knowledge_entry_access(denied, entry, operation="write")
    resource = SessionAccessScope(
        read=(
            SessionAccessRule(selectors=(SessionAccessSelector(key="team", values=("support",)),)),
        )
    )
    restricted_admin = KnowledgeAccessScope.privileged().model_copy(
        update={"resource_constraints": (encode_scope(resource),)}
    )
    assert not rules._knowledge_scope_allows_entry(restricted_admin, entry)


def test_relation_and_maintenance_checks_share_one_expiration_cutoff(monkeypatch):
    class Clock(datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            return _NOW

    monkeypatch.setattr(rules, "datetime", Clock)
    scope = KnowledgeAccessScope.for_namespace("example")
    entry = _entry(expires_at=_NOW + timedelta(microseconds=1))
    assert rules._knowledge_scope_allows_relation_access_snapshot(scope, _relation(entry))
    assert Clock.calls == 1
    assert rules._knowledge_scope_allows_maintenance_access_snapshot(
        scope, rules._knowledge_maintenance_access_snapshot([entry, entry])
    )
    assert Clock.calls == 2
    expired = entry.model_copy(update={"expires_at": _NOW})
    assert not rules._knowledge_scope_allows_entry(scope, expired, now=_NOW)
    assert rules._knowledge_scope_allows_entry(KnowledgeAccessScope.privileged(), expired, now=_NOW)


def test_retirement_and_lineage_do_not_grant_access_to_archived_content():
    entry = _entry()
    scope = KnowledgeAccessScope.for_namespace("example")
    archived = entry.model_copy(update={"status": KnowledgeStatus.ARCHIVED, "revision": 2})
    rules._require_knowledge_successor_access(scope, archived, operation="archive")
    assert rules._knowledge_scope_allows_lineage_endpoint(scope, entry, archived, now=_NOW)
    assert not rules._knowledge_scope_allows_entry(scope, archived, now=_NOW)
    foreign = archived.model_copy(update={"namespace": "foreign"})
    with pytest.raises(KnowledgeAccessDenied):
        rules._require_knowledge_successor_access(scope, foreign, operation="archive")
    assert not rules._knowledge_scope_allows_lineage_endpoint(scope, entry, foreign, now=_NOW)
    for status in (KnowledgeStatus.DELETED, KnowledgeStatus.PENDING):
        current = archived.model_copy(update={"status": status})
        assert not rules._knowledge_scope_allows_lineage_endpoint(scope, entry, current, now=_NOW)


def test_change_audiences_preserve_removal_visibility_and_require_every_relation_endpoint():
    entry = _entry()
    scope = KnowledgeAccessScope.for_namespace("example")
    hidden = entry.model_copy(update={"namespace": "foreign", "revision": 2})
    change = _change(KnowledgeChangeKind.REVISION_APPENDED, revision=2)
    audiences = rules._knowledge_change_audiences(change, before_entry=entry, after_entry=hidden)
    assert rules._knowledge_scope_allows_change(scope, change, audiences)
    assert not rules._knowledge_scope_allows_change(scope, change, ())

    expired = entry.model_copy(update={"expires_at": _NOW - timedelta(seconds=1)})
    removal = _change(KnowledgeChangeKind.EXPIRED)
    captured = rules._knowledge_change_audiences(
        removal, before_entry=expired, after_entry=None, before_requires_include_expired=False
    )
    uncaptured = rules._knowledge_change_audiences(removal, before_entry=expired, after_entry=None)
    assert rules._knowledge_scope_allows_change(scope, removal, captured)
    assert not rules._knowledge_scope_allows_change(scope, removal, uncaptured)

    relation = _change(KnowledgeChangeKind.RELATION_PUBLISHED)
    audiences = rules._knowledge_relation_change_audiences(
        relation, access_snapshot=_relation(entry)
    )
    assert rules._knowledge_scope_allows_change(scope, relation, audiences)
    assert not rules._knowledge_scope_allows_change(scope, relation, audiences[:1])
    assert not rules._knowledge_scope_allows_change(scope, relation, (audiences[0],) * 4)
    hidden_endpoint = audiences[0].model_copy(
        update={"snapshot": _knowledge_access_snapshot(hidden)}
    )
    assert not rules._knowledge_scope_allows_change(
        scope, relation, (hidden_endpoint, *audiences[1:])
    )


def test_activation_history_requires_current_or_explicit_retirement_authority():
    entry = _entry()
    snapshot = _knowledge_access_snapshot(entry)
    retirement = _knowledge_activation_retirement(entry, retired_at=_NOW)
    scope = KnowledgeAccessScope.for_namespace("example")
    common = dict(entry_id=entry.id, entry_revision=entry.revision, now=_NOW)
    assert rules._knowledge_scope_allows_activation_receipt(
        scope, snapshot, entry, retirement=None, **common
    )
    with pytest.raises(KnowledgeActivationConflict) as missing:
        rules._knowledge_scope_allows_activation_receipt(
            scope, snapshot, None, retirement=None, **common
        )
    assert missing.value.reason == "malformed_receipt"
    with pytest.raises(KnowledgeActivationConflict) as contradictory:
        rules._knowledge_scope_allows_activation_receipt(
            scope, snapshot, entry, retirement=retirement, **common
        )
    assert contradictory.value.reason == "malformed_retirement"
    assert not rules._knowledge_scope_allows_activation_receipt(
        scope, snapshot, None, retirement=retirement, **common
    )
    assert rules._knowledge_scope_allows_activation_receipt(
        KnowledgeAccessScope.privileged(), snapshot, None, retirement=retirement, **common
    )
    with pytest.raises(KnowledgeAccessDenied):
        rules._require_knowledge_activation_retirement_access(scope, retirement, operation="purge")
